#include "util.cuh"
#include "host.cuh"

#define GPU_RUNS 1

typedef int ElTp;

template <class OP>
void host_scan(typename OP::ElTp *h_in, typename OP::ElTp *h_out, size_t N) {
  typename OP::ElTp acc = OP::ne();
  for (size_t i = 0; i < N; i++)
    h_out[i] = acc = OP::apply(acc, h_in[i]);
}


int main(int argc, char **argv) {

  // TODO:
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <N>" << std::endl;
    return 0;
  }

  int N = atoi(argv[1]);


  ElTp *h_in  = (ElTp*) malloc(N * sizeof(ElTp));
  ElTp *h_out = (ElTp*) malloc(N * sizeof(ElTp));
  ElTp *h_device_res = (ElTp*) malloc(N * sizeof(ElTp));
  if (!(h_in && h_out && h_device_res)) {
    printf("host alloc error\n");
    return 1;
  }

  for (int i = 0; i < N; i++)
    h_in[i] = (ElTp) i;

  host_scan<Add<ElTp>>(h_in, h_out, N);

  device_scan<Add<ElTp>>(h_in, h_device_res, N);

  validate<ElTp>(h_out, h_device_res, N);

  free(h_in);
  free(h_out);
  free(h_device_res);
}
