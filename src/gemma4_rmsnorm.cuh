#pragma once

#include "gemma4_cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Learned-weight RMSNorm:
// y = x * rsqrt(mean(x * x) + eps) * weight.
cudaError_t gemma4_rmsnorm_bf16(__nv_bfloat16 *out,
                                const __nv_bfloat16 *inp,
                                const __nv_bfloat16 *__restrict__ weight,
                                int rows,
                                int width,
                                float eps,
                                cudaStream_t stream);

// Adds two row-major BF16 tensors.
cudaError_t gemma4_residual_add_bf16(__nv_bfloat16 *out,
                                     const __nv_bfloat16 *inp1,
                                     const __nv_bfloat16 *inp2,
                                     int count,
                                     cudaStream_t stream);

// Writes rounded BF16 residual = inp1 + inp2, then RMSNorms that residual.
cudaError_t gemma4_residual_add_rmsnorm_bf16(__nv_bfloat16 *residual,
                                             __nv_bfloat16 *normed,
                                             const __nv_bfloat16 *inp1,
                                             const __nv_bfloat16 *inp2,
                                             const __nv_bfloat16 *__restrict__ weight,
                                             int rows,
                                             int width,
                                             float eps,
                                             cudaStream_t stream);

// Runs one hidden-width learned RMSNorm row from a caller-owned CUDA block.
extern "C" __device__ void gemma4_rmsnorm_hidden_row_512_bf16_device(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ in,
    const __nv_bfloat16 *__restrict__ weight,
    float eps,
    Bf16Packed128 *__restrict__ cached_row,
    float *__restrict__ warp_sums,
    float *__restrict__ scale,
    int thread_idx);
