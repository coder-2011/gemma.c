#pragma once

#include <cuda_runtime.h>
#include <cstdio>

#define C10_CUDA_CHECK(EXPR) \
  do {                                                            \
    cudaError_t c10_cuda_check_status = (EXPR);                   \
    if (c10_cuda_check_status != cudaSuccess) {                   \
      std::fprintf(stderr, "C10_CUDA_CHECK failed: %s: %s\n",     \
                   #EXPR, cudaGetErrorString(c10_cuda_check_status)); \
    }                                                            \
  } while (0)

#define C10_CUDA_KERNEL_LAUNCH_CHECK() \
  do {                                                            \
    cudaError_t c10_cuda_launch_status = cudaPeekAtLastError();   \
    if (c10_cuda_launch_status != cudaSuccess) {                  \
      std::fprintf(stderr, "C10_CUDA_KERNEL_LAUNCH_CHECK failed: %s\n", \
                   cudaGetErrorString(c10_cuda_launch_status));   \
    }                                                            \
  } while (0)
