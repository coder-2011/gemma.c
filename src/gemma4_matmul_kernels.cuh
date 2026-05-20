#ifndef GEMMA4_MATMUL_KERNELS_CUH
#define GEMMA4_MATMUL_KERNELS_CUH

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef enum {
  GEMMA4_PROJECTION_FFN_GATE_UP = 0,
  GEMMA4_PROJECTION_FFN_DOWN,
  GEMMA4_PROJECTION_SLIDING_QKV,
  GEMMA4_PROJECTION_SLIDING_O,
  GEMMA4_PROJECTION_GLOBAL_Q,
  GEMMA4_PROJECTION_GLOBAL_K,
  GEMMA4_PROJECTION_GLOBAL_O,
  GEMMA4_PROJECTION_FINAL_LOGITS,
} Gemma4Projection;

typedef enum {
  GEMMA4_DECODE_SWIZZLE_IDENTITY = 0,
  GEMMA4_DECODE_SWIZZLE_INTERLEAVE_16 = 1,
} Gemma4DecodeSwizzle;

cublasStatus_t gemma4_projection_prefill(
    Gemma4Projection projection,
    cublasHandle_t handle,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int m,
    cudaStream_t stream);

cudaError_t gemma4_projection_decode(
    Gemma4Projection projection,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream);

cudaError_t gemma4_projection_decode_swizzled(
    Gemma4Projection projection,
    Gemma4DecodeSwizzle swizzle,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream);

#endif
