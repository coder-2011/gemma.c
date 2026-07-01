#pragma once

#include "gemma4_checkpoint.cuh"
#include "gemma4_runtime.cuh"
#include "gemma4_sampling.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

struct Gemma4FfnDecodeScratch;

enum class Gemma4MegakernelPrepMode {
  kPrefill,
  kDecode,
};

struct Gemma4PrefillMegakernelArgs {
  __nv_bfloat16 *hidden_a = nullptr;
  __nv_bfloat16 *hidden_b = nullptr;
  __nv_bfloat16 **final_hidden = nullptr;
  __nv_bfloat16 *scratch = nullptr;
  size_t scratch_elements = 0;
  const Gemma4TextWeightsDevice *weights = nullptr;
  Gemma4RuntimeState *runtime = nullptr;
  int32_t seq_len = 0;
  cudaStream_t stream = nullptr;
};

struct Gemma4DecodeMegakernelLayerArgs {
  __nv_bfloat16 *residual_out = nullptr;
  __nv_bfloat16 *normed_out = nullptr;
  __nv_bfloat16 *ffn_x = nullptr;
  __nv_bfloat16 *ffn_residual = nullptr;
  const __nv_bfloat16 *ffn_norm_weight = nullptr;
  const __nv_bfloat16 *ffn_gate_up_decode = nullptr;
  const __nv_bfloat16 *ffn_down_decode = nullptr;
  Gemma4FfnDecodeScratch *ffn_scratch = nullptr;
  const __nv_bfloat16 *layer_scalar = nullptr;
  // Folded pre-FFN norm scale slot in the partial_acc tail scratch.
  float *pre_ffn_scale = nullptr;
  // Folded input norm: normed_out already holds gamma_in * x staged by the
  // previous layer's FFN tail, so the input-norm phase and its sync are skipped.
  bool attention_input_staged = false;
  // Next layer's input-norm gamma and the staging row the FFN tail writes.
  const __nv_bfloat16 *ffn_next_input_norm_weight = nullptr;
  __nv_bfloat16 *ffn_staged_next_input = nullptr;

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
  const __nv_bfloat16 *attention_qkv_proj_col_major = nullptr;
  const __nv_bfloat16 *attention_o_proj_col_major = nullptr;
  const __nv_bfloat16 *attention_post_norm_weight = nullptr;
  const __nv_bfloat16 *attention_pre_ffn_norm_weight = nullptr;
  const __nv_bfloat16 *attention_q_norm_weight = nullptr;
  const __nv_bfloat16 *attention_k_norm_weight = nullptr;
  const float *attention_cos = nullptr;
  const float *attention_sin = nullptr;
};

struct Gemma4DecodeMegakernelArgs {
  __nv_bfloat16 *hidden_a = nullptr;
  __nv_bfloat16 *hidden_b = nullptr;
  __nv_bfloat16 *normed = nullptr;
  __nv_bfloat16 *sampled_hidden = nullptr;
  int32_t *next_token = nullptr;
  __nv_bfloat16 *attention_q = nullptr;
  __nv_bfloat16 *attention_out = nullptr;
  float *partial_m = nullptr;
  float *partial_l = nullptr;
  float *partial_acc = nullptr;
  void *scratch = nullptr;
  size_t scratch_bytes = 0;
  const Gemma4TextWeightsDevice *weights = nullptr;
  Gemma4RuntimeState *runtime = nullptr;
  int32_t split_size = GEMMA4_SLIDING_DECODE_SPLIT_SIZE;
  int32_t sliding_splits = 0;
  int32_t global_splits = 0;
  cudaStream_t stream = nullptr;
};

// Prepares shared runtime metadata for either prompt prefill or one decode append.
cudaError_t gemma4_megakernel_prepare_runtime(
    Gemma4RuntimeState *runtime,
    Gemma4MegakernelPrepMode mode,
    int32_t prefill_seq_len,
    cudaStream_t stream);

// Returns caller-owned BF16 scratch elements needed by the full prefill path.
size_t gemma4_prefill_megakernel_scratch_elements(int32_t rows);

// Runs all model prefill layers over caller-provided prompt embeddings.
cudaError_t gemma4_prefill_megakernel(const Gemma4PrefillMegakernelArgs &args);

// Returns caller-owned scratch bytes needed by one full decode step.
size_t gemma4_decode_megakernel_scratch_bytes(void);

// Applies final RMSNorm, softcapped sampling, and tied embedding gather.
cudaError_t gemma4_megakernel_sample_final_bf16(
    __nv_bfloat16 *__restrict__ next_hidden,
    int32_t *__restrict__ next_token,
    __nv_bfloat16 *__restrict__ normed_hidden,
    const __nv_bfloat16 *__restrict__ state,
    const Gemma4TextWeightsDevice *weights,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    Gemma4SamplingStage stage,
    cudaStream_t stream);

// Runs one batch-1 model decode step and writes the next token embedding.
cudaError_t gemma4_decode_megakernel(const Gemma4DecodeMegakernelArgs &args);
