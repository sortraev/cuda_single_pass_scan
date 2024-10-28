#pragma once
#include <cstdint>

#include "extras.cuh"


// TODO: document the rest of the steps in the kernel.
template <class OP, int32_t B, uint8_t Q>
__global__ void
__launch_bounds__(B)
scan_kernel(
    typename OP::ElTp *g_in,
    typename OP::ElTp *g_out,
    int32_t N,
    volatile char *g_flags,
    volatile typename OP::ElTp *g_aggregates,
    volatile typename OP::ElTp *g_prefixes,
    int32_t *g_dynid_counter = NULL
) {

  typedef typename OP::ElTp ElTp;

  static_assert(B >= WARPSIZE && B <= 1024 && B % WARPSIZE == 0,
                "B must be a multiple of WARPSIZE between WARPSIZE and 1024");
  static_assert(Q > 0, "Q must be positive");

  extern __shared__ char smem_ext[];
  ElTp *s_xs = (ElTp*) smem_ext;

  ElTp *s_lookback_v = (ElTp*) smem_ext;
  char  *s_lookback_f = (char*)  (smem_ext + WARPSIZE * sizeof(ElTp));

  ElTp r_chunk[Q];

  const int dyn_blockIdx = blockIdx.x; // TODO: dynamic block indexing!
  const int tblock_offset = dyn_blockIdx * B * Q;

  /* 
   * 1) each thread copies and scans a Q-sized chunk, placing the per-thread
   *    results in smem.
   */
  copy_gmem_to_smem<ElTp, B, Q>(g_in, s_xs, tblock_offset, N);
  __syncthreads();

  ElTp acc = OP::ne();
  #pragma unroll
  for (int i = 1; i < Q; i++)
    r_chunk[i] = OP::apply(r_chunk[i - 1], s_xs[threadIdx.x * Q + i]);
  __syncthreads();

  // store per-thread accumulators.
  s_xs[threadIdx.x] = r_chunk[Q - 1];
  __syncthreads();

  /*
   * 2) scan per-thread accumulators. Note that `block_aggregate` is only
   *    actually a block aggregate for the last thread in each block.
   */
  ElTp block_aggregate = blockscan_smem<OP, B>(s_xs);

  /*
   * 3) immediately publish aggregate/prefix!
   */
  if (threadIdx.x == B - 1) {
    volatile ElTp *which = dyn_blockIdx == 0 ? g_prefixes : g_aggregates;
    which[dyn_blockIdx] = block_aggregate;
    __threadfence();
    g_flags[dyn_blockIdx] = dyn_blockIdx == 0 ? flag_P : flag_A;
  }

  __syncthreads();
  s_xs[threadIdx.x] = block_aggregate;
  __syncthreads();

  ElTp chunk_exc_prefix =
    threadIdx.x > 0 ? s_xs[threadIdx.x - 1] : OP::ne();

  ElTp block_exc_prefix = OP::ne();
  if (dyn_blockIdx > 0) {

    // TODO: is this method for lookback correct, or should only one thread read
    // the flag and publish do_lookback to smem?
    bool do_lookback = g_flags[dyn_blockIdx - 1] != flag_P;

    if (do_lookback) {
      // first WARPSIZE threads perform the lookback.
      if (threadIdx.x < WARPSIZE) {
        int32_t lookback_idx = dyn_blockIdx + threadIdx.x;
        while (1) {
          lookback_idx -= WARPSIZE;

          // spin until no flag read is X.
          char f;
          do {
            f = lookback_idx >= 0 ? g_flags[lookback_idx] : flag_P;
          } while (__any_sync(SHFL_MASK, f == flag_X));

          // then, scan a chunk of aggregates and prefixes!
          ElTp v = lookback_idx >= 0
                  ? (f == flag_P ? g_prefixes : g_aggregates)[lookback_idx]
                  : OP::ne();
          s_lookback_f[threadIdx.x] = f;
          s_lookback_v[threadIdx.x] = v;
          FV_pair res = warpreduce_FV_smem<OP>(s_lookback_f, s_lookback_v);

          block_exc_prefix = OP::apply(res.v, block_exc_prefix);
          if (res.f == flag_P)
            break;
        }
      }

      // broadcast the computed block_exc_prefix to the other warps.
      if (threadIdx.x == 0)
        s_xs[0] = block_exc_prefix;
      __syncthreads();
      block_exc_prefix = s_xs[0];
    }
    else // we already have the prefix, so skip the lookback!
      block_exc_prefix = g_prefixes[dyn_blockIdx - 1];

    if (threadIdx.x == B - 1) {
      g_prefixes[dyn_blockIdx] = OP::apply(block_exc_prefix, block_aggregate);
      __threadfence();
      g_flags[dyn_blockIdx] = flag_P;
    }
  }
  // TODO: do we need a sync here? are we missing some logic?

  ElTp thread_exc_prefix = OP::apply(block_exc_prefix, chunk_exc_prefix);
  __syncthreads();

  #pragma unroll
  for (int i = 0; i < Q; i++)
    s_xs[threadIdx.x * Q + i] = OP::apply(thread_exc_prefix, r_chunk[i]);

  __syncthreads();
  copy_gmem_to_smem<ElTp, B, Q>(g_out, s_xs, tblock_offset, N);
}
