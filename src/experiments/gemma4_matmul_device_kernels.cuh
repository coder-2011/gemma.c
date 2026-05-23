#ifndef GEMMA4_MATMUL_DEVICE_KERNELS_CUH
#define GEMMA4_MATMUL_DEVICE_KERNELS_CUH

#include "gemma4_matmul_kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

cudaError_t gemma4_projection_decode_device_wrapped(
    Gemma4Projection projection,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream);

cudaError_t gemma4_projection_decode_device_wrapped_swizzled(
    Gemma4Projection projection,
    Gemma4DecodeSwizzle swizzle,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream);

#endif
