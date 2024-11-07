#pragma once
#include <iostream>

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

void init_array(int *xs, size_t N) {
  for (size_t i = 0; i < N; i++)
    xs[i] = i + 1;
}
void init_array(float *xs, size_t N) {
  for (size_t i = 0; i < N; i++)
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
void validate(ElTp *h_expected, ElTp *h_actual, ssize_t N) {

  ssize_t first_invalid_index = -1;
  for (ssize_t i = 0; i < N; i++) {
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
void host_scan(typename OP::ElTp *h_in, typename OP::ElTp *h_out, size_t N) {
  typename OP::ElTp acc = OP::ne();
  for (size_t i = 0; i < N; i++)
    h_out[i] = acc = OP::apply(acc, h_in[i]);
}
