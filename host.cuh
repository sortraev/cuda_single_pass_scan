#include <algorithm>

#include "util.cuh"
#include "kernels/kernel.cuh"

template <class OP, int B = 256, int Q = 25>
void __device_scan_prealloc(typename OP::ElTp *g_in,
                            typename OP::ElTp *g_out,
                            size_t N,
                            flag_t *g_flags,
                            typename OP::ElTp *g_aggregates,
                            typename OP::ElTp *g_prefixes,
                            uint32_t *g_dynid_counter) {
  /*
   * device scan of g_in.
   * manages only initialization of flags and dynamic ID counter in global mem.
   */

  typedef typename OP::ElTp ElTp;

  const size_t num_tblocks = CEIL_DIV(N, B * Q);

  CUDASSERT(cudaMemset(g_flags,         flag_X, num_tblocks * sizeof(flag_t)));
  CUDASSERT(cudaMemset(g_dynid_counter, 0,      1 * sizeof(uint32_t)));

  constexpr int smem_size = get_smem_size<ElTp, B, Q>();

  if constexpr(smem_size > 49152)
    cudaFuncSetAttribute(
        scan_kernel<OP, B, Q>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

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
}

template <class OP, int B = 256, int Q = 25>
void __device_scan(typename OP::ElTp *g_in,
                   typename OP::ElTp *g_out,
                   size_t N) {
  /*
   * device scan of g_in.
   * manages everything to do with temporary device arrays needed by the single
   * pass kernel.
   */

  typedef typename OP::ElTp ElTp;

  ElTp *g_aggregates, *g_prefixes;
  flag_t *g_flags;
  uint32_t *g_dynid_counter;

  size_t num_tblocks = CEIL_DIV(N, B * Q);

  CUDASSERT(cudaMalloc(&g_aggregates,    num_tblocks * sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&g_prefixes,      num_tblocks * sizeof(ElTp)));
  CUDASSERT(cudaMalloc(&g_flags,         num_tblocks * sizeof(flag_t)));
  CUDASSERT(cudaMalloc(&g_dynid_counter, 1 * sizeof(uint32_t)));

  __device_scan_prealloc
    <OP, B, Q>
    (g_in,
     g_out,
     N,
     g_flags,
     g_aggregates,
     g_prefixes,
     g_dynid_counter
    );
  CUDASSERT(cudaPeekAtLastError());

  CUDACHECK(cudaFree(g_aggregates));
  CUDACHECK(cudaFree(g_prefixes));
  CUDACHECK(cudaFree(g_flags));
}

template <class OP, int B = 256, int Q = 25>
void device_scan(typename OP::ElTp *h_in,
                 typename OP::ElTp *h_out,
                 size_t N) {
  /*
   * h_out = scan(h_in, OP::apply, OP::ne()).
   * h_out may alias h_in.
   * manages everything to do with device memory.
   */
  typedef typename OP::ElTp ElTp;

  // if (N < (1 << 20)) {
  //   host_scan(h_in, h_out, N);
  // }
  // else
  {

    ElTp *g_in, *g_out;
    CUDASSERT(cudaMalloc(&g_in,  N * sizeof(ElTp)));
    CUDASSERT(cudaMalloc(&g_out, N * sizeof(ElTp)));
    CUDASSERT(cudaMemcpy(g_in, h_in, N * sizeof(ElTp), cudaMemcpyHostToDevice));

    __device_scan<OP, B, Q>(g_in, g_out, N);

    CUDASSERT(cudaMemcpy(h_out, g_out, N * sizeof(ElTp), cudaMemcpyDeviceToHost));

    CUDACHECK(cudaFree(g_in));
    CUDACHECK(cudaFree(g_out));
  }
}
