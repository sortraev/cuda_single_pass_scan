#include "kernel.cuh"
#include "util.cuh"

#define GPU_RUNS 1

typedef float ElTp;

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
    return 1;
  }

  size_t N = atol(argv[1]);

  constexpr int32_t B = 256;
  constexpr int32_t Q = 15;

  constexpr int32_t ELEMS_PER_THREAD = B * Q; 

  const int32_t num_tblocks = CEIL_DIV(N, ELEMS_PER_THREAD);

  ElTp *g_in, *g_out;
  ElTp *g_aggregates, *g_prefixes;
  char *g_flags;
  int32_t *g_dynid_counter;

  cudaMalloc(&g_in,  N * sizeof(ElTp));
  cudaMalloc(&g_out, N * sizeof(ElTp));

  cudaMalloc(&g_flags,      num_tblocks * sizeof(char));
  cudaMalloc(&g_aggregates, num_tblocks * sizeof(ElTp));
  cudaMalloc(&g_prefixes,   num_tblocks * sizeof(ElTp));

  cudaMalloc(&g_dynid_counter, sizeof(int32_t));

  cudaMemset(g_flags,         flag_X, num_tblocks * sizeof(char));
  cudaMemset(g_dynid_counter, 0,      sizeof(int32_t));

  CUDASSERT(cudaPeekAtLastError());

  const int32_t smem_size = 0; // TODO: set this one.

  for (int i = 0; i < GPU_RUNS; i++) {
    scan_kernel
      <Add<ElTp>, B, Q>
      <<<num_tblocks, B, smem_size>>>
      (g_in, g_out, N, g_flags, g_aggregates, g_prefixes, g_dynid_counter);
  }

  CUDASSERT(cudaDeviceSynchronize());
  CUDASSERT(cudaPeekAtLastError());

  cudaFree(g_in);
  cudaFree(g_out);
  cudaFree(g_flags);
  cudaFree(g_aggregates);
  cudaFree(g_prefixes);
  cudaFree(g_dynid_counter);
}
