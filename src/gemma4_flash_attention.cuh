#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stddef.h>

#include "gemma4_kv_cache.cuh"

struct Gemma4AttentionProjectionWeights {
  const __nv_bfloat16 *d_q_col_major;
  const __nv_bfloat16 *d_k_col_major;
  const __nv_bfloat16 *d_v_col_major;
  int32_t q_col_base;
  int32_t k_col_base;
  int32_t v_col_base;
};

// Sliding prefill APIs take total live causal keys, including the current key.
extern "C" cudaError_t gemma4_flash_attention_sliding_fwd_bf16(
    __nv_bfloat16 *__restrict__ d_out,
    // Optional. Pass nullptr for inference paths that do not need LSE.
    float *__restrict__ d_softmax_lse,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int window_size,
    float softmax_scale,
    cudaStream_t stream);

extern "C" cudaError_t gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
    __nv_bfloat16 *__restrict__ d_out,
    // Optional. Pass nullptr for inference paths that do not need LSE.
    float *__restrict__ d_softmax_lse,
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

extern "C" cudaError_t gemma4_flash_attention_global_fwd_bf16_norm_rope(
    __nv_bfloat16 *__restrict__ d_out,
    // Optional. Pass nullptr for inference paths that do not need LSE.
    float *__restrict__ d_softmax_lse,
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

extern "C" cudaError_t gemma4_flash_attention_sliding_decode_prepare_q_paged_kv_bf16(
    __nv_bfloat16 *__restrict__ d_q_prepared,
    __nv_bfloat16 *__restrict__ d_cache_k,
    __nv_bfloat16 *__restrict__ d_cache_v,
    Gemma4KvCacheConfig cache_config,
    const int32_t *__restrict__ d_page_table,
    const int32_t *__restrict__ d_token_position,
    int32_t batch_size,
    int32_t cache_layer,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    const __nv_bfloat16 *__restrict__ d_q_norm_weight,
    const __nv_bfloat16 *__restrict__ d_k_norm_weight,
    const float *__restrict__ d_cos,
    const float *__restrict__ d_sin,
    cudaStream_t stream);

// Prepares already-projected decode Q/K/V for either sliding or global attention.
extern "C" cudaError_t gemma4_flash_attention_decode_prepare_q_paged_kv_bf16(
    __nv_bfloat16 *__restrict__ d_q_prepared,
    __nv_bfloat16 *__restrict__ d_cache_k,
    __nv_bfloat16 *__restrict__ d_cache_v,
    Gemma4KvCacheConfig cache_config,
    const int32_t *__restrict__ d_page_table,
    const int32_t *__restrict__ d_token_position,
    int32_t batch_size,
    int32_t cache_layer,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    const __nv_bfloat16 *__restrict__ d_q_norm_weight,
    const __nv_bfloat16 *__restrict__ d_k_norm_weight,
    const float *__restrict__ d_cos,
    const float *__restrict__ d_sin,
    cudaStream_t stream);

extern "C" cudaError_t gemma4_flash_attention_decode_norm_project_prepare_paged_kv_bf16(
    __nv_bfloat16 *__restrict__ d_q_prepared,
    __nv_bfloat16 *__restrict__ d_cache_k,
    __nv_bfloat16 *__restrict__ d_cache_v,
    Gemma4KvCacheConfig cache_config,
    const int32_t *__restrict__ d_page_table,
    const int32_t *__restrict__ d_token_position,
    int32_t batch_size,
    int32_t cache_layer,
    const __nv_bfloat16 *__restrict__ d_x,
    const __nv_bfloat16 *__restrict__ d_input_norm_weight,
    Gemma4AttentionProjectionWeights weights,
    const __nv_bfloat16 *__restrict__ d_q_norm_weight,
    const __nv_bfloat16 *__restrict__ d_k_norm_weight,
    const float *__restrict__ d_cos,
    const float *__restrict__ d_sin,
    cudaStream_t stream);

extern "C" cudaError_t gemma4_flash_attention_sliding_decode_paged_bf16(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_partial_m,
    float *__restrict__ d_partial_l,
    float *__restrict__ d_partial_acc,
    const __nv_bfloat16 *__restrict__ d_q_prepared,
    const __nv_bfloat16 *__restrict__ d_cache_k,
    const __nv_bfloat16 *__restrict__ d_cache_v,
    const int32_t *__restrict__ d_page_table,
    const int32_t *__restrict__ d_seq_lengths,
    Gemma4KvCacheConfig cache_config,
    int32_t cache_layer,
    int32_t batch_size,
    float softmax_scale,
    int32_t split_size,
    int32_t num_splits,
    cudaStream_t stream);

extern "C" cudaError_t gemma4_flash_attention_decode_paged_bf16(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_partial_m,
    float *__restrict__ d_partial_l,
    float *__restrict__ d_partial_acc,
    const __nv_bfloat16 *__restrict__ d_q_prepared,
    const __nv_bfloat16 *__restrict__ d_cache_k,
    const __nv_bfloat16 *__restrict__ d_cache_v,
    const int32_t *__restrict__ d_page_table,
    const int32_t *__restrict__ d_seq_lengths,
    Gemma4KvCacheConfig cache_config,
    int32_t cache_layer,
    int32_t batch_size,
    float softmax_scale,
    int32_t split_size,
    int32_t num_splits,
    cudaStream_t stream);

extern "C" size_t gemma4_flash_attention_sliding_decode_persistent_scratch_i32(
    int32_t batch_size,
    int32_t num_splits);

// Currently returns cudaErrorNotSupported for otherwise-valid calls.
extern "C" cudaError_t gemma4_flash_attention_sliding_decode_paged_persistent_bf16(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_partial_m,
    float *__restrict__ d_partial_l,
    float *__restrict__ d_partial_acc,
    int32_t *__restrict__ d_work_scratch,
    int32_t work_scratch_i32,
    const __nv_bfloat16 *__restrict__ d_q_prepared,
    const __nv_bfloat16 *__restrict__ d_cache_k,
    const __nv_bfloat16 *__restrict__ d_cache_v,
    const int32_t *__restrict__ d_page_table,
    const int32_t *__restrict__ d_seq_lengths,
    Gemma4KvCacheConfig cache_config,
    int32_t cache_layer,
    int32_t batch_size,
    float softmax_scale,
    int32_t split_size,
    int32_t num_splits,
    int32_t persistent_blocks,
    cudaStream_t stream);

extern "C" size_t gemma4_flash_attention_sliding_smem_bytes();
extern "C" int gemma4_flash_attention_sliding_threads_per_block();
extern "C" cudaError_t gemma4_flash_attention_sliding_kernel_attributes(
    long long *out,
    int len);
