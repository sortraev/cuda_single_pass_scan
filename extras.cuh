#pragma once
#include <cstdint>

#define WARPSIZE 32
#define SHFL_MASK 0xffffffff


typedef char flag_t;
enum {
  flag_A = (flag_t) 0,  // 00
  flag_P = (flag_t) 1,  // 01
  flag_X = (flag_t) 2   // 11
};

template <int32_t B = -1>
__device__ void
syncthreads() {
  if constexpr (B != -1 && B > 32)
    __syncthreads();
}

template <typename ElTp, int32_t B, uint8_t Q>
__device__ void
copy_gmem_to_smem(ElTp  *g_xs,
                  ElTp  *s_xs,
                  size_t tblock_offset,
                  size_t N
                 ) {
  #pragma unroll
  for (int i = 0; i < Q; i++) {
    size_t loc_ind = i * B + threadIdx.x;
    size_t glb_ind = tblock_offset + loc_ind;
    s_xs[loc_ind] = glb_ind < N ? g_xs[glb_ind] : 0.0f;
  }
}

template <typename ElTp, int32_t B, uint8_t Q>
__device__ void
copy_gmem_to_smem_(ElTp  *g_xs,
                   ElTp  *s_xs,
                   size_t tblock_offset,
                   size_t N
                  ) {
  s_xs += threadIdx.x;
  g_xs += tblock_offset;
  #pragma unroll
  for (int i = 0; i < Q; i++) {
    size_t loc_ind = i * B;
    s_xs[loc_ind] = tblock_offset + loc_ind < N ? g_xs[loc_ind] : 0.0f;
  }
}

template <typename ElTp, int32_t B, uint8_t Q>
__device__ void
copy_smem_to_gmem(ElTp *g_xs,
                  ElTp *s_xs,
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
float warpscan_smem(typename OP::ElTp *s_xs) {
  const int32_t i    = threadIdx.x;
  const uint8_t lane = i & (WARPSIZE - 1); // assumes WARPSIZE is mult of 32.

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
    char f;
    T v;
    __device__ __host__ inline FV_pair() { f = flag_P; }
    __device__ __host__ inline FV_pair(char f, T v) { f = f; v = v;}
};

template <class OP>
__device__
FV_pair<typename OP::ElTp> warpreduce_FV_smem(char *s_f, typename OP::ElTp *s_v) {
  const int32_t tid  = threadIdx.x;
  const uint8_t lane = tid & 31;

  // char flag = s_f[tid];
  // typename OP::ElTp acc = s_v[tid];

  #pragma unroll
  for (int p = 1; p < 32; p <<= 1) {
    if (lane >= p) {
      if (s_f[tid] != flag_A)
        s_v[tid] = OP::apply(s_v[tid - p], s_v[tid]);
      s_f[tid] |= s_f[tid - p];
    }
  }
  // return FV_pair(flag, acc);
  return FV_pair(s_f[31], s_v[31]);
}

template <typename OP, int32_t B>
__device__
typename OP::ElTp blockscan_smem(typename OP::ElTp *s_xs) {

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


template <class _ElTp>
class Add {
  static_assert(std::is_arithmetic<_ElTp>::value);

public:

  typedef _ElTp ElTp;

  static __device__ __host__ inline
  ElTp apply(const ElTp t1, const ElTp t2) { return t1 + t2; }

  static __device__ __host__ inline 
  ElTp ne() { return (ElTp) 0.0f; }
};
