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
  static_assert(WARPSIZE == 32, "warpscan_smem specialized to WARPSIZE == 32");

  const int32_t i    = threadIdx.x;
  const uint8_t lane = i & 31;

  if (lane >=  1) s_xs[i] = OP::apply(s_xs[i -  1], s_xs[i]);
  if (lane >=  2) s_xs[i] = OP::apply(s_xs[i -  2], s_xs[i]);
  if (lane >=  4) s_xs[i] = OP::apply(s_xs[i -  4], s_xs[i]);
  if (lane >=  8) s_xs[i] = OP::apply(s_xs[i -  8], s_xs[i]);
  if (lane >= 16) s_xs[i] = OP::apply(s_xs[i - 16], s_xs[i]);

  return s_xs[i];
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
  /*
   * NOTE: result stored in *_f and *_v after call!
   */
  flag_t f            = *_f;
  typename OP::ElTp v = *_v;
  #pragma unroll
  for (int i = 1; i < WARPSIZE; i *= 2) {
    typename OP::ElTp w = __shfl_xor_sync(SHFL_FULL_MASK, v, i);
    v = f == flag_A ? OP::apply(w, v) : v;
    f |= __shfl_xor_sync(SHFL_FULL_MASK, f, i);
  }
  *_f = __shfl_sync(SHFL_FULL_MASK, f, WARPSIZE - 1);
  *_v = __shfl_sync(SHFL_FULL_MASK, v, WARPSIZE - 1);
}

template <typename OP, int32_t B>
__device__
typename OP::ElTp blockscan_smem(volatile typename OP::ElTp *s_xs) {
  static_assert(WARPSIZE == 32,
                "blockscan_smem specialized to WARPSIZE == 32 (TODO!)");

  typename OP::ElTp res = warpscan_smem<OP>(s_xs);

  if constexpr (B <= WARPSIZE)
    return res;

  __syncthreads();

  const uint8_t warp_idx = threadIdx.x >> 5;

  if (warp_idx == 0) {
    int32_t idx_gather = threadIdx.x << 5 | 31;
    if (idx_gather < B)
      s_xs[threadIdx.x] = s_xs[idx_gather];
    warpscan_smem<OP>(s_xs);
  }
  __syncthreads();

  if (warp_idx > 0)
    res = OP::apply(s_xs[warp_idx - 1], res);

  return res;
}

template<class OP>
__device__ inline typename OP::ElTp
scanIncWarp(typename OP::ElTp* ptr, const unsigned int idx) {
    const unsigned int lane = idx & 31;
    #pragma unroll
    for(uint32_t i=0; i<5; i++) {
        const uint32_t p = (1<<i);
        if(lane >= p) ptr[idx] = OP::apply(ptr[idx-p], ptr[idx]);
        // __syncwarp();
    }
    return ptr[idx];//OP::remVolatile(ptr[idx]);
}


template<class OP>
__device__ inline typename OP::ElTp
scanIncBlock(volatile typename OP::ElTp* ptr) {
    const unsigned int idx = threadIdx.x;
    const unsigned int warpid = idx >> 5;

    // 1. perform scan at warp level
    typename OP::ElTp res = warpscan_smem<OP>(ptr);
    // typename OP::ElTp res = scanIncWarp<OP>(ptr,idx);
    __syncthreads();

#if 1
    const unsigned int lane = idx & (32-1);
    // 2. place the end-of-warp results in
    //   the first warp. This works because
    //   warp size = 32, and 
    //   max block size = 32^2 = 1024
    if (lane == (32-1)) { ptr[warpid] = res; } 
    __syncthreads();

    // 3. scan again the first warp
    if (warpid == 0) warpscan_smem<OP>(ptr);
    // if (warpid == 0) scanIncWarp<OP>(ptr, idx);
    __syncthreads();
#else
    // Alternative solution, combining steps 2 and 3:
    // let warp 0 gather elements from threads with lane == 31 and then scan
    // them. Since the first warp executes in lockstep this eliminates a sync.
    if (warpid == 0) {
      int32_t idx_gather = threadIdx.x << 5 | 31;
      if (idx_gather < blockDim.x)
        ptr[threadIdx.x] = OP::remVolatile(ptr[idx_gather]);
      scanIncWarp<OP>(ptr, idx);
    }
    __syncthreads();
#endif

    // 4. accumulate results from previous step;
    if (warpid > 0) {
        res = OP::apply(ptr[warpid-1], res);
    }

    __syncthreads();
    ptr[idx] = res;

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

template <typename ElTp>
void validate(ElTp *h_expected, ElTp *h_actual, ssize_t N) {

  ssize_t first_invalid_index = -1;
  for (ssize_t i = 0; i < N; i++) {
    if (h_expected[i] != h_expected[i]) {
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
