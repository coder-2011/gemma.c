#ifndef GEMMA4_MATMUL_KERNELS_CUH
#define GEMMA4_MATMUL_KERNELS_CUH

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" {

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

cublasStatus_t gemma4_projection_prefill(
    Gemma4Projection projection,
    cublasHandle_t handle,
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major,
    __nv_bfloat16 *y,
    int m,
    cudaStream_t stream);

cudaError_t gemma4_projection_decode(
    Gemma4Projection projection,
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major,
    __nv_bfloat16 *y,
    cudaStream_t stream);

}

#endif
