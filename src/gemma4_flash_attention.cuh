#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>

// Sliding prefill APIs take total live causal keys, including the current key.
cudaError_t gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
    __nv_bfloat16 *__restrict__ d_out,
    __nv_bfloat16 *__restrict__ d_q_prepared,
    __nv_bfloat16 *__restrict__ d_k_prepared,
    __nv_bfloat16 *__restrict__ d_v_prepared,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    const __nv_bfloat16 *__restrict__ d_q_norm_weight,
    const __nv_bfloat16 *__restrict__ d_k_norm_weight,
    const float *__restrict__ d_cos,
    const float *__restrict__ d_sin,
    const int32_t *__restrict__ d_token_position,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int window_size,
    float softmax_scale,
    cudaStream_t stream);

cudaError_t gemma4_flash_attention_global_fwd_bf16_norm_rope(
    __nv_bfloat16 *__restrict__ d_out,
    __nv_bfloat16 *__restrict__ d_q_prepared,
    __nv_bfloat16 *__restrict__ d_k_prepared,
    __nv_bfloat16 *__restrict__ d_v_prepared,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_q_norm_weight,
    const __nv_bfloat16 *__restrict__ d_k_norm_weight,
    const float *__restrict__ d_cos,
    const float *__restrict__ d_sin,
    const int32_t *__restrict__ d_token_position,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    float softmax_scale,
    cudaStream_t stream);
