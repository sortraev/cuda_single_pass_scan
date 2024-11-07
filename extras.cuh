#pragma once
#include <cstdint>
#include <tuple>

#define WARPSIZE       (32)
#define SHFL_FULL_MASK (0xffffffff)

typedef char flag_t;
enum {
  flag_A = (flag_t) 0,  // 00
  flag_P = (flag_t) 1,  // 01
  flag_X = (flag_t) 2   // 10
};

__device__ __host__ __forceinline__
bool flag_is_A(flag_t f) {
  return f == flag_A;
}
__device__ __host__ __forceinline__
bool flag_is_P(flag_t f) {
  return f == flag_P;
}
__device__ __host__ __forceinline__
bool flag_is_X(flag_t f) {
  return f >= flag_X;
}

template <class OP, int32_t B, uint8_t Q>
__device__ void
copy_gmem_to_smem(volatile typename OP::ElTp *g_xs,
                  volatile typename OP::ElTp *s_xs,
                  size_t tblock_offset,
                  size_t N
                 ) {
  #pragma unroll
  for (int i = 0; i < Q; i++) {
    size_t loc_ind = i * B + threadIdx.x;
    size_t glb_ind = tblock_offset + loc_ind;
    s_xs[loc_ind] = glb_ind < N ? g_xs[glb_ind] : OP::ne();
  }
}

template <typename OP, int32_t B, uint8_t Q>
__device__ void
copy_smem_to_gmem(volatile typename OP::ElTp *g_xs,
                  volatile typename OP::ElTp *s_xs,
                  size_t tblock_offset,
                  size_t N
                 ) {
  #pragma unroll
  for (int i = 0; i < Q; i++) {
    size_t loc_ind = i * B + threadIdx.x;
    size_t glb_ind = tblock_offset + loc_ind;
    if (glb_ind < N)
      g_xs[glb_ind] = s_xs[loc_ind];
  }
}

template <class OP>
__device__
typename OP::ElTp warpscan_smem(volatile typename OP::ElTp *s_xs) {
  // this warpscan only works with WARPSIZE == 32.
  static_assert((WARPSIZE & (WARPSIZE - 1)) == 0,
      "warpscan_shfl specialized to power of 2 WARPSIZEs");

  const int32_t i    = threadIdx.x;
  const uint8_t lane = i & (WARPSIZE - 1);
  #pragma unroll
  for (int p = 1; p <= WARPSIZE / 2; p <<= 1) {
    if (lane >= p)
      s_xs[i] = OP::apply(s_xs[i - p], s_xs[i]);
    __syncwarp(); // TODO: necessary?
  }

  return s_xs[i];
}

template <class OP>
__device__
typename OP::ElTp warpscan_shfl(volatile typename OP::ElTp *s_xs) {
  // this warpscan only works with WARPSIZE == 32.
  static_assert((WARPSIZE & (WARPSIZE - 1)) == 0,
      "warpscan_shfl specialized to power of 2 WARPSIZEs");

  const int32_t i    = threadIdx.x;
  const uint8_t lane = i & (WARPSIZE - 1);

  typename OP::ElTp x = s_xs[i];
  #pragma unroll
  for (int p = 1; p <= WARPSIZE / 2; p <<= 1) {
    typename OP::ElTp other = __shfl_down_sync(SHFL_FULL_MASK, x, p);
    if (lane >= p)
      x = OP::apply(other, x);
  }
  s_xs[i] = x;

  return x;
}


template <class T>
class FV_pair {
  public:
    flag_t f;
    T v;
    __device__ __host__ inline FV_pair() { f = flag_P; }
    __device__ __host__ inline FV_pair(flag_t f, T v) { f = f; v = v;}
    __device__ __host__ inline
    FV_pair& operator = (const FV_pair &other) {
      f = other.f;
      v = other.v;
      return *this;
    }

    static __device__ __host__ inline
    FV_pair remVolatile(volatile FV_pair &other) {
      FV_pair out;
      out.v = other.v;
      out.f = other.f;
      return out;
    }
};

template <class OP>
__device__
void warpreduce_FV_smem(volatile flag_t *s_f, volatile typename OP::ElTp *s_v) {
  /*
   * NOTE: result stored in s_f[31] and s_v[31] after call!
   * NOTE: actually computes a warp scan of s_f and s_v.
   */
  const int32_t tid  = threadIdx.x;
  const uint8_t lane = tid & 31;

  #pragma unroll
  for (int p = 1; p < WARPSIZE; p <<= 1) {
    if (lane >= p) {
      if (s_f[tid] == flag_A)
        s_v[tid] = OP::apply(s_v[tid - p], s_v[tid]);
      s_f[tid] |= s_f[tid - p];
    }
  }
}


template <class OP>
__device__ __forceinline__
void warpreduce_FV_shfl(flag_t *_f, typename OP::ElTp *_v) {
  typedef typename OP::ElTp ElTp;
  /*
   * NOTE: result stored in *_f and *_v after call!
   */
  flag_t f = *_f;
  ElTp   v = *_v;
  #pragma unroll
  for (int p = 1; p <= WARPSIZE / 2; p <<= 1) {
    ElTp   other_v = __shfl_xor_sync(SHFL_FULL_MASK, v, p);
    flag_t other_f = __shfl_xor_sync(SHFL_FULL_MASK, f, p);
    if (f == flag_A)
      v = OP::apply(other_v, v);
    f |= other_f;
  }
  *_f = __shfl_sync(SHFL_FULL_MASK, f, WARPSIZE - 1);
  *_v = __shfl_sync(SHFL_FULL_MASK, v, WARPSIZE - 1);
}

template <typename OP, int32_t B>
__device__
typename OP::ElTp blockscan_smem(volatile typename OP::ElTp *s_xs) {
  static_assert(WARPSIZE == 32,
                "blockscan_smem specialized to WARPSIZE == 32 (TODO!)");

  typename OP::ElTp res = warpscan_shfl<OP>(s_xs);

  if constexpr (B <= WARPSIZE)
    return res;

  __syncthreads();

  const uint8_t warp_idx = threadIdx.x >> 5;

  if (warp_idx == 0) {
    int32_t idx_gather = threadIdx.x << 5 | 31;
    if (idx_gather < B)
      s_xs[threadIdx.x] = s_xs[idx_gather];
    warpscan_shfl<OP>(s_xs);
  }
  __syncthreads();

  if (warp_idx > 0)
    res = OP::apply(s_xs[warp_idx - 1], res);

  return res;
}

template <class _ElTp>
class Add {
  static_assert(std::is_arithmetic<_ElTp>::value,
                "Class Add<ElTp> expects an arithmetic ElTp");
public:
  typedef _ElTp ElTp; // make ElTp visible to the outside.

  static __device__ __host__ inline
  ElTp apply(const ElTp t1, const ElTp t2) {
    return t1 + t2;
  }
  static __device__ __host__ inline 
  ElTp ne() {
    return (ElTp) 0;
  }
};


bool eq(int a, int b) {
  return a == b;
}
bool eq(float a, float b) {
  constexpr float EPS = 0.00001;
  return (std::isnan(a) && std::isnan(b)) || std::abs(a - b) <= EPS;
}

template <typename ElTp>
void validate(ElTp *h_expected, ElTp *h_actual, ssize_t N) {

  ssize_t first_invalid_index = -1;
  for (ssize_t i = 0; i < N; i++) {
    if (!eq(h_expected[i], h_actual[i])) {
      first_invalid_index = i;
      break;
    }
  }

  bool valid = first_invalid_index == -1;
  if (valid) {
    std::cout << ">> Valid!" << std::endl;
  }
  else {
    std::cout
      << ">> Invalid at index "
      << first_invalid_index
      << "! Next 10 values are:"
      << std::endl;
    for (int i = first_invalid_index; i < first_invalid_index + 10; i++) {
      std::cout
        << "Index: "
        << i
        << ": "
        << h_expected[i]
        << " vs "
        << h_actual[i]
        << std::endl;
    }
  }
}
