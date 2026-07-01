#include "gemma4_megakernel.cuh"

#include "gemma4_cuda_utils.cuh"
#include "gemma4_ffn.cuh"
#include "gemma4_rmsnorm.cuh"

#include <math.h>
#include <stdint.h>

cudaError_t gemma4_decode_megakernel_flash_attention_layer_bf16(
    const Gemma4DecodeMegakernelLayerArgs &args,
    cudaStream_t stream);

namespace {

struct DecodeScratch {
  Gemma4FfnDecodeScratch *ffn = nullptr;
  void *sample = nullptr;
  size_t sample_bytes = 0;
#if GEMMA4_DECODE_ATTENTION_READY_HANDOFF
  uint32_t *attention_ready = nullptr;
#endif
};

#if GEMMA4_DECODE_ATTENTION_READY_HANDOFF
// Returns the scratch byte offset of the layer/KV-head readiness table.
size_t decode_attention_ready_offset(void) {
  const size_t sample_end =
      sizeof(Gemma4FfnDecodeScratch) + gemma4_sample_next_scratch_bytes();
  return (sample_end + alignof(uint32_t) - 1) & ~(alignof(uint32_t) - 1);
}

// Counts one readiness tag per model layer and possible sliding KV head.
size_t decode_attention_ready_bytes(void) {
  return static_cast<size_t>(GEMMA4_NUM_LAYERS) *
         GEMMA4_SLIDING_KV_HEADS * sizeof(uint32_t);
}
#endif

// Splits caller-owned decode scratch into FFN and sampling regions.
DecodeScratch decode_scratch_from_buffer(void *scratch) {
  auto *ffn = reinterpret_cast<Gemma4FfnDecodeScratch *>(scratch);
  void *sample = reinterpret_cast<void *>(ffn + 1);
  const size_t sample_bytes = gemma4_sample_next_scratch_bytes();
#if GEMMA4_DECODE_ATTENTION_READY_HANDOFF
  const size_t ready_offset = decode_attention_ready_offset();
  auto *ready = reinterpret_cast<uint32_t *>(
      static_cast<unsigned char *>(scratch) + ready_offset);
  return {ffn, sample, sample_bytes, ready};
#else
  return {ffn, sample, sample_bytes};
#endif
}

// Fills the per-layer cooperative FlashAttention+FFN argument block.
Gemma4DecodeMegakernelLayerArgs decode_layer_args(
    const Gemma4DecodeMegakernelArgs &args,
    Gemma4FfnDecodeScratch *ffn_scratch,
    int32_t layer,
    __nv_bfloat16 *hidden_in,
    __nv_bfloat16 *hidden_out) {
  const bool global = gemma4_is_global_layer(layer);
  const int32_t head_dim =
      global ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_SLIDING_HEAD_DIM;
  const Gemma4TextLayerWeightsDevice &w = args.weights->layers[layer];

  Gemma4DecodeMegakernelLayerArgs layer_args = {};
  layer_args.residual_out = hidden_out;
  layer_args.normed_out = args.normed;
  layer_args.ffn_x = args.attention_out;
  layer_args.ffn_residual = hidden_out;
  layer_args.ffn_norm_weight = w.post_feedforward_norm_weight;
  layer_args.ffn_gate_up_decode = w.ffn_gate_up_decode;
  layer_args.ffn_down_decode = w.ffn_down_decode;
  layer_args.ffn_scratch = ffn_scratch;
  layer_args.layer_scalar = w.layer_scalar;
  layer_args.attention_q = args.attention_q;
  layer_args.attention_out = args.attention_out;
  layer_args.attention_partial_m = args.partial_m;
  layer_args.attention_partial_l = args.partial_l;
  layer_args.attention_partial_acc = args.partial_acc;
  layer_args.attention_cache_k =
      global ? args.runtime->global_cache_k : args.runtime->sliding_cache_k;
  layer_args.attention_cache_v =
      global ? args.runtime->global_cache_v : args.runtime->sliding_cache_v;
  layer_args.attention_cache_config =
      global ? args.runtime->global_cache_config
             : args.runtime->sliding_cache_config;
  layer_args.attention_page_table =
      global ? args.runtime->global_page_table
             : args.runtime->sliding_page_table;
  layer_args.attention_token_position = args.runtime->token_position;
  layer_args.attention_seq_lengths = args.runtime->seq_lengths;
  layer_args.attention_cache_layer = gemma4_kv_cache_layer_index(layer, global);
  layer_args.attention_split_size = args.split_size;
  layer_args.attention_num_splits =
      global ? args.global_splits : args.sliding_splits;
  layer_args.attention_softmax_scale = 1.0f / sqrtf(float(head_dim));
  layer_args.attention_x = hidden_in;
  layer_args.attention_input_norm_weight = w.input_norm_weight;
  layer_args.attention_qkv_proj_col_major = w.qkv_proj_col_major;
  layer_args.attention_o_proj_col_major = w.o_proj_col_major;
  layer_args.attention_post_norm_weight = w.post_attention_norm_weight;
  layer_args.attention_pre_ffn_norm_weight = w.pre_feedforward_norm_weight;
  layer_args.attention_q_norm_weight = w.q_norm_weight;
  layer_args.attention_k_norm_weight = w.k_norm_weight;
  layer_args.attention_cos =
      global ? args.runtime->global_cos : args.runtime->sliding_cos;
  layer_args.attention_sin =
      global ? args.runtime->global_sin : args.runtime->sliding_sin;
#if GEMMA4_DECODE_ATTENTION_READY_HANDOFF
  if (args.attention_ready != nullptr) {
    const size_t layer_ready_offset =
        static_cast<size_t>(layer) * GEMMA4_SLIDING_KV_HEADS;
    layer_args.attention_ready = args.attention_ready + layer_ready_offset;
    layer_args.attention_ready_tag =
        static_cast<uint32_t>(args.runtime->h_token_position[0] + 1);
  }
#endif
  return layer_args;
}

}  // namespace

// Prepares shared runtime metadata for either prompt prefill or one decode append.
cudaError_t gemma4_megakernel_prepare_runtime(
    Gemma4RuntimeState *runtime,
    Gemma4MegakernelPrepMode mode,
    int32_t prefill_seq_len,
    cudaStream_t stream) {
  if (runtime == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (mode == Gemma4MegakernelPrepMode::kPrefill) {
    return gemma4_runtime_prepare_prefill(runtime, prefill_seq_len, stream);
  }
  if (mode == Gemma4MegakernelPrepMode::kDecode) {
    return gemma4_runtime_prepare_decode_step(runtime, stream);
  }
  return cudaErrorInvalidValue;
}

// Returns caller-owned scratch bytes needed by one full decode step.
size_t gemma4_decode_megakernel_scratch_bytes(void) {
#if GEMMA4_DECODE_ATTENTION_READY_HANDOFF
  return decode_attention_ready_offset() + decode_attention_ready_bytes();
#else
  return sizeof(Gemma4FfnDecodeScratch) + gemma4_sample_next_scratch_bytes();
#endif
}

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
    cudaStream_t stream) {
  if (next_hidden == nullptr || next_token == nullptr ||
      normed_hidden == nullptr || state == nullptr || weights == nullptr ||
      weights->final_norm_weight == nullptr || weights->token_embedding == nullptr) {
    return cudaErrorInvalidValue;
  }
  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_rmsnorm_bf16(
      normed_hidden, state, weights->final_norm_weight, 1, GEMMA4_HIDDEN_SIZE,
      GEMMA4_RMS_NORM_EPS, stream));
  return gemma4_sample_next_bf16(
      next_hidden, next_token, scratch, scratch_bytes, normed_hidden,
      weights->token_embedding, stage, stream);
}

// Runs the batch-1 48-layer decode step and writes the next token embedding.
cudaError_t gemma4_decode_megakernel(const Gemma4DecodeMegakernelArgs &args) {
  if (args.hidden_a == nullptr || args.hidden_b == nullptr ||
      args.normed == nullptr || args.sampled_hidden == nullptr ||
      args.next_token == nullptr || args.weights == nullptr ||
      args.runtime == nullptr || args.runtime->batch_size != 1 ||
      args.scratch == nullptr ||
      args.scratch_bytes < gemma4_decode_megakernel_scratch_bytes()) {
    return cudaErrorInvalidValue;
  }
  if (args.attention_q == nullptr || args.attention_out == nullptr ||
      args.partial_m == nullptr || args.partial_l == nullptr ||
      args.partial_acc == nullptr || args.split_size <= 0 ||
      args.sliding_splits <= 0 || args.global_splits <= 0) {
    return cudaErrorInvalidValue;
  }

  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_megakernel_prepare_runtime(
      args.runtime, Gemma4MegakernelPrepMode::kDecode, 0, args.stream));

  const int32_t seq_len = args.runtime->h_seq_lengths[0];
  const int32_t window = args.runtime->sliding_cache_config.window_size;
  const int32_t sliding_keys = seq_len < window ? seq_len : window;
  Gemma4DecodeMegakernelArgs active_args = args;
  active_args.sliding_splits = div_up(sliding_keys, args.split_size);
  active_args.global_splits = div_up(seq_len, args.split_size);
  if (active_args.sliding_splits > args.sliding_splits ||
      active_args.global_splits > args.global_splits) {
    return cudaErrorInvalidValue;
  }

  DecodeScratch scratch_parts = decode_scratch_from_buffer(active_args.scratch);
#if GEMMA4_DECODE_ATTENTION_READY_HANDOFF
  if (active_args.attention_ready == nullptr) {
    active_args.attention_ready = scratch_parts.attention_ready;
  }
#endif
  __nv_bfloat16 *hidden_in = active_args.hidden_a;
  __nv_bfloat16 *hidden_out = active_args.hidden_b;
  for (int32_t layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
    Gemma4DecodeMegakernelLayerArgs layer_args =
        decode_layer_args(
            active_args, scratch_parts.ffn, layer, hidden_in, hidden_out);
    GEMMA4_RETURN_IF_CUDA_ERROR(
        gemma4_decode_megakernel_flash_attention_layer_bf16(
            layer_args, active_args.stream));

    __nv_bfloat16 *tmp = hidden_in;
    hidden_in = hidden_out;
    hidden_out = tmp;
  }

  const __nv_bfloat16 *final_hidden = hidden_in;
  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_megakernel_sample_final_bf16(
      active_args.sampled_hidden, active_args.next_token, active_args.normed,
      final_hidden, active_args.weights, scratch_parts.sample,
      scratch_parts.sample_bytes, Gemma4SamplingStage::kDecode,
      active_args.stream));
  if (active_args.hidden_a == active_args.sampled_hidden) {
    return cudaSuccess;
  }
  const size_t row_bytes =
      static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * sizeof(__nv_bfloat16);
  return cudaMemcpyAsync(
      active_args.hidden_a, active_args.sampled_hidden, row_bytes,
      cudaMemcpyDeviceToDevice, active_args.stream);
}
