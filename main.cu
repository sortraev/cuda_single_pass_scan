#include <cstdlib> // std::rand

#include "util.cuh"
#include "host.cuh"

typedef int ElTp;

template <class OP>
void host_scan(typename OP::ElTp *h_in, typename OP::ElTp *h_out, size_t N) {
  typename OP::ElTp acc = OP::ne();
  for (size_t i = 0; i < N; i++)
    h_out[i] = acc = OP::apply(acc, h_in[i]);
}

void init_array(int *xs, size_t N) {
  for (size_t i = 0; i < N; i++)
    xs[i] = i + 1;
}
void init_array(float *xs, size_t N) {
  for (size_t i = 0; i < N; i++)
    xs[i] = std::rand() / RAND_MAX;
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

  init_array(h_in, N);

  host_scan<Add<ElTp>>(h_in, h_out, N);

  device_scan<Add<ElTp>>(h_in, h_device_res, N);

  validate<ElTp>(h_out, h_device_res, N);

  free(h_in);
  free(h_out);
  free(h_device_res);
}
