#pragma once
#include <cstdint>
#include <cstdio>

#include "extras.cuh"

// TODO: document the rest of the steps in the kernel.
template <class OP, int32_t B, uint8_t Q>
__global__ void
__launch_bounds__(B) // TODO
scan_kernel(
    volatile typename OP::ElTp *g_in,
    volatile typename OP::ElTp *g_out,
    int32_t N,
    volatile flag_t *g_flags,
    volatile typename OP::ElTp *g_aggregates,
    volatile typename OP::ElTp *g_prefixes,
    uint32_t *g_dynid_counter
) {
  static_assert(B >= WARPSIZE && B <= 1024 && B % WARPSIZE == 0,
                "B must be a multiple of WARPSIZE between WARPSIZE and 1024");
  static_assert(Q > 0, "Q must be positive");

  typedef typename OP::ElTp ElTp;

  extern __shared__ char smem_ext[];

  uint32_t      *s_dynid_counter    = (uint32_t*) smem_ext;
  volatile ElTp *s_copy_buf         = (volatile ElTp*) smem_ext;
  volatile ElTp *s_blockscan_buf    = (volatile ElTp*) smem_ext;
  volatile ElTp *s_block_exc_prefix = (volatile ElTp*) smem_ext;

  ElTp r_chunk[Q];
  // volatile ElTp *r_chunk = s_copy_buf + threadIdx.x * Q;

  if (threadIdx.x == 0) {
    uint32_t tmp = atomicAdd(g_dynid_counter, 1);
    *s_dynid_counter = tmp;
    // TODO: only works as long as num physical tblocks equals num virt tblocks.
    if (tmp == gridDim.x - 1)
      g_dynid_counter = 0;
  }

  __syncthreads();
  const int dyn_blockIdx = *s_dynid_counter; // TODO: dynamic block indexing!
  __syncthreads();

  const int tblock_offset = dyn_blockIdx * B * Q;

  /* 
   * 1) each thread copies and scans a Q-sized chunk, placing the per-thread
   *    results in smem.
   */
  copy_gmem_to_smem<OP, B, Q>(g_in, s_copy_buf, tblock_offset, N);
  __syncthreads();

  ElTp acc = OP::ne();
  #pragma unroll
  for (int i = 0; i < Q; i++)
    r_chunk[i] = acc = OP::apply(acc, s_copy_buf[threadIdx.x * Q + i]);
  __syncthreads();

  // store per-thread accumulators.
  s_blockscan_buf[threadIdx.x] = acc;


  /*
   * 2) scan per-thread accumulators. Note that `block_aggregate` is only
   *    actually a block aggregate for the last thread in each block.
   */
  ElTp block_aggregate = blockscan_smem<OP, B>(s_blockscan_buf);

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
  s_blockscan_buf[threadIdx.x] = block_aggregate;
  __syncthreads();

  ElTp chunk_exc_prefix =
    threadIdx.x > 0 ? s_blockscan_buf[threadIdx.x - 1] : OP::ne();

  ElTp block_exc_prefix = OP::ne();
  if (dyn_blockIdx > 0) {

    if (threadIdx.x < WARPSIZE) {
      bool do_lookback = g_flags[dyn_blockIdx - 1] != flag_P;
      if (do_lookback) {
        // first WARPSIZE threads perform the lookback.
        int32_t lookback_idx = dyn_blockIdx + threadIdx.x - WARPSIZE;
        while (1) {

          // load a flag and, if flag not X, an aggregate/prefix.
          flag_t f = lookback_idx >= 0 ? g_flags[lookback_idx] : flag_P;
          ElTp v = lookback_idx >= 0 && !flag_is_X(f)
                 ? (flag_is_P(f) ? g_prefixes : g_aggregates)[lookback_idx]
                 : OP::ne();

          // find number of valid lookback values; this is how much lookback
          // window should be shifted back.
          // use ballot_sync to find threads whose flag is X, then use clz
          // to compute number of leading zeroes. if no flag is X, then
          // shift_amt will be WARPSIZE (this is safe because, unlike in e.g.
          // GNU, __clz(0) is defined in CUDA).
          int shift_amt = __clz(__ballot_sync(SHFL_FULL_MASK, flag_is_X(f)));
          lookback_idx -= shift_amt;

          warpreduce_FV_shfl<OP>(&f, &v);

          block_exc_prefix = OP::apply(v, block_exc_prefix);
          if (flag_is_P(f)) break;
        }
      }
      else {
        block_exc_prefix = g_prefixes[dyn_blockIdx - 1];
      }

      // broadcast computed block_exc_prefix to the other warps.
      // TODO: is it OK to have all threads in warp write to s_block_exc_prefix?
      if (threadIdx.x == 0)
        *s_block_exc_prefix = block_exc_prefix;
    }
    __syncthreads();
    block_exc_prefix = *s_block_exc_prefix;

    if (threadIdx.x == B - 1) {
      g_prefixes[dyn_blockIdx] = OP::apply(block_exc_prefix, block_aggregate);
      __threadfence();
      g_flags[dyn_blockIdx] = flag_P;
    }
  }

  // TODO: this sync redundant if we keep s_block_exc_prefix separate from
  // s_copy_buf.
  __syncthreads();

  ElTp thread_exc_prefix = OP::apply(block_exc_prefix, chunk_exc_prefix);

  #pragma unroll
  for (int i = 0; i < Q; i++)
    s_copy_buf[threadIdx.x * Q + i] = OP::apply(thread_exc_prefix, r_chunk[i]);

  __syncthreads();
  copy_smem_to_gmem<OP, B, Q>(g_out, s_copy_buf, tblock_offset, N);
}
