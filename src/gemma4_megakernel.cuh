#pragma once

#include "gemma4_checkpoint.cuh"
#include "gemma4_flash_attention.cuh"
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
  bool attention_inputs_prepared = false;
  bool attention_direct_output_ingress = false;
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
  // Scratch for raw packed QKV/QK rows and attention output.
  // Callers must allocate GEMMA4_DECODE_ATTENTION_OUT_SCRATCH_SIZE elements.
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
  // Debug fallback for comparing against the old three-launch attention ingress.
  bool use_split_attention_ingress = false;
  // Experimental no-raw-QKV ingress path. Uses the direct projection+prep kernel.
  bool use_direct_attention_ingress = false;
  // Experimental path that runs all decode layers in one cooperative launch.
  bool use_token_megakernel = false;
  cudaStream_t stream = nullptr;
};

struct Gemma4DecodeTokenMegakernelArgs {
  __nv_bfloat16 *hidden_a = nullptr;
  __nv_bfloat16 *hidden_b = nullptr;
  __nv_bfloat16 *normed = nullptr;
  int32_t *next_token = nullptr;
  __nv_bfloat16 *attention_q = nullptr;
  __nv_bfloat16 *attention_out = nullptr;
  float *partial_m = nullptr;
  float *partial_l = nullptr;
  float *partial_acc = nullptr;
  Gemma4FfnDecodeScratch *ffn_scratch = nullptr;
  Gemma4SampleCandidate *sample_candidates = nullptr;
  uint32_t *sample_ready_flags = nullptr;
  Gemma4TextWeightsDevice weights = {};

  __nv_bfloat16 *sliding_cache_k = nullptr;
  __nv_bfloat16 *sliding_cache_v = nullptr;
  __nv_bfloat16 *global_cache_k = nullptr;
  __nv_bfloat16 *global_cache_v = nullptr;
  Gemma4KvCacheConfig sliding_cache_config = {};
  Gemma4KvCacheConfig global_cache_config = {};
  int32_t *sliding_page_table = nullptr;
  int32_t *global_page_table = nullptr;
  int32_t *seq_lengths = nullptr;
  int32_t *token_position = nullptr;
  const float *sliding_cos = nullptr;
  const float *sliding_sin = nullptr;
  const float *global_cos = nullptr;
  const float *global_sin = nullptr;

  int32_t split_size = GEMMA4_SLIDING_DECODE_SPLIT_SIZE;
  int32_t sliding_splits = 0;
  int32_t global_splits = 0;
  int32_t max_seq_len = 0;
};

// Prepares shared runtime metadata for either prompt prefill or one decode append.
cudaError_t gemma4_megakernel_prepare_runtime(
    Gemma4RuntimeState *runtime,
    Gemma4MegakernelPrepMode mode,
    int32_t prefill_seq_len,
    cudaStream_t stream);

// Returns caller-owned BF16 scratch elements needed by the full prefill path.
size_t gemma4_prefill_megakernel_scratch_elements(int32_t rows);

// Runs all 48 prefill layers over caller-provided prompt embeddings.
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

// Runs the batch-1 48-layer decode step and writes the next token embedding.
cudaError_t gemma4_decode_megakernel(const Gemma4DecodeMegakernelArgs &args);

// Runs all 48 decode layers in one cooperative kernel, leaving sampling to host.
cudaError_t gemma4_decode_token_megakernel_bf16(
    const Gemma4DecodeTokenMegakernelArgs &args,
    cudaStream_t stream);
