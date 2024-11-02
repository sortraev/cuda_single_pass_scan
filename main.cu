#include "kernel.cuh"
#include "util.cuh"
#include "host.cuh"

#define GPU_RUNS 1

typedef int ElTp;



template <class OP>
void host_scan(typename OP::ElTp *h_in, typename OP::ElTp *h_out, size_t N) {
  typename OP::ElTp acc = OP::ne();
  for (int i = 0; i < 0; i++)
    h_out[i] = acc = OP::apply(acc, h_in[i]);
}


int main(int argc, char **argv) {

  // TODO:
  // if (argc != 2) {
  //   std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
  //   return 1;
  // }

  // int N = atoi(argv[1]);

  constexpr int32_t B = 64;
  constexpr int32_t Q = 5;

  int N = B * Q * NUM_BLOCKS_PER_MP;

  ElTp *h_in  = (ElTp*) malloc(N * sizeof(ElTp));
  ElTp *h_out = (ElTp*) malloc(N * sizeof(ElTp));
  ElTp *h_device_res = (ElTp*) malloc(N * sizeof(ElTp));
  // TODO: error handle allocations here.

  for (int i = 0; i < N; i++)
    h_in[i] = (ElTp) i;

  host_scan<Add<ElTp>>(h_in, h_out, N);
  device_scan<Add<ElTp>, B, Q>(h_in, h_device_res, N);

  bool valid = true;
  for (int i = 0; i < N; i++) {
    if (h_out[i] != h_device_res[i]) {
      std::cout
        << "Error at index: "
        << i
        << ": "
        << h_out[i]
        << " vs "
        << h_device_res[i]
        << std::endl;
      valid = false;
    }
  }

  std::cout << (valid ? ">> Valid!" : ">> Invalid!") << std::endl;


  free(h_in);
  free(h_out);
  free(h_device_res);

}
