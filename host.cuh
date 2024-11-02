#include <algorithm>

#include "util.cuh"
#include "kernel.cuh"

template <class OP, int B = 256, int Q = 9>
int __device_scan(typename OP::ElTp *g_in, typename OP::ElTp *g_out, size_t N) {
  typedef typename OP::ElTp ElTp;

  ElTp *g_aggregates, *g_prefixes;
  flag_t *g_flags;
  uint32_t *g_dynid_counter;

  size_t num_tblocks = CEIL_DIV(N, B * Q);
  CUDASSERT(cudaMalloc(&g_aggregates,    num_tblocks * sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&g_prefixes,      num_tblocks * sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&g_flags,         num_tblocks * sizeof(flag_t)));
  CUDASSERT(cudaMalloc(&g_dynid_counter, 1 * sizeof(uint32_t)));

  CUDASSERT(cudaMemset(g_dynid_counter, 0,      1 * sizeof(uint32_t)));
  CUDASSERT(cudaMemset(g_flags,         flag_X, num_tblocks * sizeof(flag_t)));

  CUDASSERT(cudaPeekAtLastError());

  // TODO: this.
  constexpr int smem_size = std::max({
      B * Q * sizeof(ElTp), // scan buffer
      sizeof(uint32_t),     // dynamic tblock id
      WARPSIZE * sizeof(ElTp) + WARPSIZE * sizeof(flag_t), // s_lookback
  });

  scan_kernel
    <OP, B, Q>
    <<<num_tblocks, B, smem_size>>>
    (g_in,
     g_out,
     N,
     g_flags,
     g_aggregates,
     g_prefixes,
     g_dynid_counter
    );

  CUDASSERT(cudaPeekAtLastError());

  CUDASSERT(cudaFree(g_aggregates));
  CUDASSERT(cudaFree(g_prefixes));
  CUDASSERT(cudaFree(g_flags));

  return 0;
}


template <class OP, int B = 256, int Q = 9>
int device_scan(typename OP::ElTp *h_in, typename OP::ElTp *h_out, size_t N) {
  typedef typename OP::ElTp ElTp;

  ElTp *g_in, *g_out;
  cudaMalloc(&g_in,  N * sizeof(ElTp));
  cudaMalloc(&g_out, N * sizeof(ElTp));
  cudaMemcpy(h_in, g_in, N * sizeof(ElTp), cudaMemcpyHostToDevice);

  __device_scan<OP, B, Q>(g_in, g_out, N);
  cudaMemcpy(h_out, g_out, N * sizeof(ElTp), cudaMemcpyDeviceToHost);

  cudaFree(g_in);
  cudaFree(g_out);

  CUDASSERT(cudaPeekAtLastError());
  return 0;
}
