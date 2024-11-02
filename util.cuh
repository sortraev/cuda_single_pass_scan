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
