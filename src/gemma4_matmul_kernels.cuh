#pragma once

// Public projection API plus the decode FFN gate/up device helper.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

typedef enum {
  GEMMA4_PROJECTION_FFN_GATE_UP = 0,
  GEMMA4_PROJECTION_FFN_DOWN,
  GEMMA4_PROJECTION_SLIDING_Q,
  GEMMA4_PROJECTION_SLIDING_KV,
  GEMMA4_PROJECTION_SLIDING_QKV,
  GEMMA4_PROJECTION_SLIDING_O,
  GEMMA4_PROJECTION_GLOBAL_Q,
  GEMMA4_PROJECTION_GLOBAL_K,
  GEMMA4_PROJECTION_GLOBAL_O,
  GEMMA4_PROJECTION_FINAL_LOGITS,
  GEMMA4_PROJECTION_GLOBAL_QK,
} Gemma4Projection;

cudaError_t gemma4_projection_decode(
    Gemma4Projection projection,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream);

// Runs BF16 prefill GEMM: Y[M,N] = X[M,K] * W[N,K]^T.
cudaError_t gemma4_prefill_gemm_bf16(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int rows,
    int k,
    int n,
    cudaStream_t stream);

namespace gemma4_matmul_kernel_impl {

// Computes one fixed Gemma 4 FFN gate/up decode tile inside a caller kernel.
extern "C" __device__ void gemma4_ffn_gate_up_tile_bf16_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_interleaved_row_major,
    int col0,
    int thread_idx,
    float *__restrict__ warp_sums,
    float *__restrict__ gate,
    float *__restrict__ up);

}  // namespace gemma4_matmul_kernel_impl
