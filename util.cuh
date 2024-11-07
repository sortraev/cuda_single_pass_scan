#pragma once
#include <iostream>
#include <cstdint>
#include <cstdlib>

#define CEIL_DIV(m, n) (((m) + (n) - 1) / (n))

#define CUDASSERT(code) { __cudassert((code), __FILE__, __LINE__); }
#define CUDACHECK(code) { __cudassert((code), __FILE__, __LINE__, false); }

cudaError_t __cudassert(cudaError_t code,
                        const char *file,
                        int line,
                        bool do_abort_on_err = true) {
  if (code != cudaSuccess) {
    std::cerr
      << "GPU Error in "
      << file
      << " (line "
      << line
      << "): "
      << cudaGetErrorString(code)
      << std::endl;
    if (do_abort_on_err)
        exit(1);
  }
  return code;
}

void init_array(int *xs, uint64_t N) {
  for (uint64_t i = 0; i < N; i++)
    xs[i] = i + 1;
}
void init_array(float *xs, uint64_t N) {
  for (uint64_t i = 0; i < N; i++)
    xs[i] = std::rand() / RAND_MAX;
}

bool eq(int a, int b) {
  return a == b;
}
bool eq(float a, float b) {
  constexpr float EPS = 0.00001;
  return (std::isnan(a) && std::isnan(b)) || std::abs(a - b) <= EPS;
}

template <typename ElTp>
void validate(ElTp *h_expected, ElTp *h_actual, int64_t N) {

  int64_t first_invalid_index = -1;
  for (int64_t i = 0; i < N; i++) {
    if (!eq(h_expected[i], h_actual[i])) {
      first_invalid_index = i;
      break;
    }
  }

  bool valid = first_invalid_index == -1;
  if (valid) {
    std::cout << ">> Valid!" << std::endl;
  }
  else {
    std::cout
      << ">> Invalid at index "
      << first_invalid_index
      << "! Next 10 values are:"
      << std::endl;
    for (int i = first_invalid_index; i < first_invalid_index + 10; i++) {
      std::cout
        << "Index: "
        << i
        << ": "
        << h_expected[i]
        << " vs "
        << h_actual[i]
        << std::endl;
    }
  }
}

template <class OP>
void host_scan(typename OP::ElTp *h_in, typename OP::ElTp *h_out, uint64_t N) {
  typename OP::ElTp acc = OP::ne();
  for (uint64_t i = 0; i < N; i++)
    h_out[i] = acc = OP::apply(acc, h_in[i]);
}



__device__ __forceinline__
void set_random(float *x, uint64_t seed) {
  constexpr uint64_t m = 20000000UL;
  *x = float((seed * 214741UL) % m - m/2) / 100000.0f;
}
__device__ __forceinline__
void set_random(int *x, uint64_t seed) {
  constexpr uint64_t m = 200UL;
  *x = (seed * 214741UL) % m - m/2;
}

template <typename ElTp>
__global__ void
init_random_kernel(ElTp *g_out, uint64_t N, uint64_t seed = 1) {
  uint64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    set_random(g_out + gid, seed * gid);
  }
}

template <typename ElTp>
void device_init_random(ElTp *g_out, uint64_t N, uint64_t seed = 1) {
  constexpr int B = 256;
  uint64_t num_tblocks = CEIL_DIV(N, B);
  init_random_kernel<ElTp><<<num_tblocks, B>>>(g_out, N, seed);
}
