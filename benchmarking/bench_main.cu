#include <cstdint>

#include "util.cuh"
#include "host.cuh"

typedef float ElTp;

#define GPU_RUNS 200


#if defined __B && defined __Q
  constexpr int B = __B;
  constexpr int Q = __Q;
#else
  constexpr int B = 256;
  constexpr int Q = 7;
#endif

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
    return 0;
  }


  int N = atoi(argv[1]);

  const uint64_t num_tblocks = CEIL_DIV(N, B * Q);

  ElTp *g_in, *g_out;
  ElTp *g_aggregates, *g_prefixes;
  flag_t *g_flags;

  uint32_t *g_dynid_counter;

  CUDASSERT(cudaMalloc((void**)&g_in,  N * sizeof(ElTp)));
  CUDASSERT(cudaMalloc((void**)&g_out, N * sizeof(ElTp)));

  CUDASSERT(cudaMalloc((void**)&g_flags,      num_tblocks * sizeof(flag_t)));
  CUDASSERT(cudaMalloc((void**)&g_aggregates, num_tblocks * sizeof(ElTp)));
  CUDASSERT(cudaMalloc((void**)&g_prefixes,   num_tblocks * sizeof(ElTp)));

  CUDASSERT(cudaMalloc((void**)&g_dynid_counter, 1 * sizeof(uint32_t)));

  device_init_random(g_out, N, 42);

  __device_scan_prealloc
    <Add<ElTp>, B, Q>
    (g_in,
     g_out,
     N,
     g_flags,
     g_aggregates,
     g_prefixes,
     g_dynid_counter
    );

  cudaDeviceSynchronize();

  // TODO: timing.
  for (int i = 0; i < GPU_RUNS; i++) {
    __device_scan_prealloc
      <Add<ElTp>, B, Q>
      (g_in,
       g_out,
       N,
       g_flags,
       g_aggregates,
       g_prefixes,
       g_dynid_counter
      );
  }
  cudaDeviceSynchronize();
  // TODO: timing.

  cudaFree(g_in);
  cudaFree(g_out);
}
