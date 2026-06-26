#pragma once

#include "gemma4_flash_attention.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

static constexpr uint32_t GEMMA4_DECODE_MEGAKERNEL_FLAG_FLASH_ATTENTION = 1u;

struct Gemma4DecodeMegakernelSpineArgs {
  __nv_bfloat16 *state = nullptr;
  __nv_bfloat16 *next_hidden = nullptr;
  int32_t *next_token = nullptr;
  const __nv_bfloat16 *final_norm_weight = nullptr;
  const __nv_bfloat16 *lm_head_col_major = nullptr;
};

struct Gemma4DecodeMegakernelFfnTailArgs {
  __nv_bfloat16 *residual_out = nullptr;
  __nv_bfloat16 *normed_out = nullptr;
  __nv_bfloat16 *next_hidden = nullptr;
  int32_t *next_token = nullptr;
  __nv_bfloat16 *ffn_x = nullptr;
  __nv_bfloat16 *ffn_residual = nullptr;
  const __nv_bfloat16 *ffn_norm_weight = nullptr;
  const __nv_bfloat16 *ffn_gate_up_decode = nullptr;
  const __nv_bfloat16 *ffn_down_decode = nullptr;
  const __nv_bfloat16 *layer_scalar = nullptr;
  const __nv_bfloat16 *final_norm_weight = nullptr;
  const __nv_bfloat16 *lm_head_col_major = nullptr;
  uint32_t flags = 0;

  __nv_bfloat16 *attention_q = nullptr;
  __nv_bfloat16 *attention_out = nullptr;
  float *attention_partial_m = nullptr;
  float *attention_partial_l = nullptr;
  float *attention_partial_acc = nullptr;
  __nv_bfloat16 *attention_cache_k = nullptr;
  __nv_bfloat16 *attention_cache_v = nullptr;
  Gemma4KvCacheConfig attention_cache_config = {};
  const int32_t *attention_page_table = nullptr;
  const int32_t *attention_token_position = nullptr;
  const int32_t *attention_seq_lengths = nullptr;
  int32_t attention_cache_layer = 0;
  int32_t attention_split_size = 0;
  int32_t attention_num_splits = 0;
  float attention_softmax_scale = 0.0f;
  const __nv_bfloat16 *attention_x = nullptr;
  const __nv_bfloat16 *attention_input_norm_weight = nullptr;
  Gemma4AttentionProjectionWeights attention_weights = {};
  const __nv_bfloat16 *attention_o_proj_col_major = nullptr;
  const __nv_bfloat16 *attention_post_norm_weight = nullptr;
  const __nv_bfloat16 *attention_pre_ffn_norm_weight = nullptr;
  const __nv_bfloat16 *attention_q_norm_weight = nullptr;
  const __nv_bfloat16 *attention_k_norm_weight = nullptr;
  const float *attention_cos = nullptr;
  const float *attention_sin = nullptr;
};

size_t gemma4_decode_megakernel_spine_scratch_bytes(void);

size_t gemma4_decode_megakernel_ffn_tail_scratch_bytes(void);

// Runs the first cooperative decode megakernel spine tail: final RMSNorm,
// LM-head projection, token selection, and tied embedding gather.
cudaError_t gemma4_decode_megakernel_spine_bf16(
    const Gemma4DecodeMegakernelSpineArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream);

// Runs optional attention preparation and the CUTLASS FFN tail, stopping after residual_out.
cudaError_t gemma4_decode_megakernel_attention_ffn_bf16(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream);

// Runs one CUTLASS FFN phase followed by the final sampling tail. When
// requested by flags, FlashAttention runs before the FFN phase.
cudaError_t gemma4_decode_megakernel_ffn_tail_bf16(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream);
