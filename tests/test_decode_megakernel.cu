#include "gemma4_megakernel.cuh"
#include "gemma4_ffn.cuh"
#include "gemma4_matmul_kernels.cuh"
#include "gemma4_rmsnorm.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <stdio.h>

cudaError_t gemma4_decode_megakernel_flash_attention_layer_bf16(
    const Gemma4DecodeMegakernelLayerArgs &args,
    cudaStream_t stream);

// Returns the raw CUDA pointer owned by a Thrust device vector.
template <typename T>
T *raw_ptr(thrust::device_vector<T> &v) {
  return thrust::raw_pointer_cast(v.data());
}

// Returns the raw const CUDA pointer owned by a Thrust device vector.
template <typename T>
const T *raw_ptr(const thrust::device_vector<T> &v) {
  return thrust::raw_pointer_cast(v.data());
}

// Prints one failed condition and keeps the sentinel test compact.
bool expect_true(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message);
  }
  return condition;
}

// Converts BF16 test values back to float for comparisons.
float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

// Generates deterministic nonzero BF16 values for compact CUDA fixtures.
__nv_bfloat16 make_value(int seed, float scale) {
  const int centered = ((seed * 37 + 17) % 257) - 128;
  return __float2bfloat16_rn(static_cast<float>(centered) * scale);
}

// Fills a host vector with deterministic BF16 values.
void fill_values(std::vector<__nv_bfloat16> &dst, int seed, float scale) {
  for (size_t i = 0; i < dst.size(); ++i) {
    dst[i] = make_value(seed + static_cast<int>(i), scale);
  }
}

// Copies a host vector into an already-sized Thrust device vector.
template <typename T>
bool copy_to_device(thrust::device_vector<T> &dst, const std::vector<T> &src) {
  const cudaError_t status = cudaMemcpy(
      raw_ptr(dst), src.data(), src.size() * sizeof(T),
      cudaMemcpyHostToDevice);
  return expect_true(status == cudaSuccess, "host vector copies to device");
}

// Returns the total BF16 slots in a Layout-A KV cache allocation.
int64_t cache_elements(const Gemma4KvCacheConfig &config) {
  return int64_t(config.num_layers) * config.num_pages * config.page_size *
         config.num_heads * config.head_dim;
}

// Compares two BF16 device buffers with an absolute tolerance.
bool compare_device_bf16(
    const thrust::device_vector<__nv_bfloat16> &actual,
    const thrust::device_vector<__nv_bfloat16> &expected,
    int count,
    float tolerance,
    const char *label) {
  std::vector<__nv_bfloat16> h_actual(count);
  std::vector<__nv_bfloat16> h_expected(count);
  cudaError_t status = cudaMemcpy(
      h_actual.data(), raw_ptr(actual), count * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost);
  if (!expect_true(status == cudaSuccess, "actual buffer copies to host")) {
    return false;
  }
  status = cudaMemcpy(
      h_expected.data(), raw_ptr(expected), count * sizeof(__nv_bfloat16),
      cudaMemcpyDeviceToHost);
  if (!expect_true(status == cudaSuccess, "expected buffer copies to host")) {
    return false;
  }

  float max_abs = 0.0f;
  int max_index = 0;
  for (int i = 0; i < count; ++i) {
    const float diff =
        std::fabs(bf16_to_float(h_actual[i]) - bf16_to_float(h_expected[i]));
    if (diff > max_abs) {
      max_abs = diff;
      max_index = i;
    }
  }
  if (max_abs <= tolerance) {
    return true;
  }
  fprintf(stderr,
          "FAIL: %s max_abs=%g index=%d actual=%g expected=%g\n",
          label, max_abs, max_index, bf16_to_float(h_actual[max_index]),
          bf16_to_float(h_expected[max_index]));
  return false;
}

// Checks the fused layer attention path against standalone phase references.
bool test_decode_layer_attention_pipeline(bool global) {
  const int head_dim =
      global ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_SLIDING_HEAD_DIM;
  const int kv_heads =
      global ? GEMMA4_GLOBAL_KV_HEADS : GEMMA4_SLIDING_KV_HEADS;
  const int q_width = GEMMA4_NUM_QUERY_HEADS * head_dim;
  const int kv_width = kv_heads * head_dim;
  const int projection_width =
      global ? GEMMA4_GLOBAL_QK_PROJ_SIZE : GEMMA4_SLIDING_QKV_SIZE;
  const int rotary_half =
      global ? GEMMA4_GLOBAL_HEAD_DIM / 8 : GEMMA4_SLIDING_HEAD_DIM / 2;
  const float softmax_scale = 1.0f / std::sqrt(float(head_dim));
  const Gemma4Projection projection =
      global ? GEMMA4_PROJECTION_GLOBAL_QK : GEMMA4_PROJECTION_SLIDING_QKV;
  const Gemma4Projection o_projection =
      global ? GEMMA4_PROJECTION_GLOBAL_O : GEMMA4_PROJECTION_SLIDING_O;

  Gemma4KvCacheConfig cache_config = {};
  cache_config.num_layers = 1;
  cache_config.num_pages = 1;
  cache_config.page_size = 1;
  cache_config.max_pages_per_seq = 1;
  cache_config.batch_size = 1;
  cache_config.num_heads = kv_heads;
  cache_config.head_dim = head_dim;
  cache_config.window_size = global ? 0 : 1;

  std::vector<__nv_bfloat16> hidden(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> input_norm(GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> qkv_weights(
      size_t(projection_width) * GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> o_proj(size_t(q_width) * GEMMA4_HIDDEN_SIZE);
  std::vector<__nv_bfloat16> head_norm(head_dim);
  std::vector<__nv_bfloat16> hidden_norm(
      GEMMA4_HIDDEN_SIZE, __float2bfloat16_rn(1.0f));
  std::vector<float> cos(rotary_half, 1.0f);
  std::vector<float> sin(rotary_half, 0.0f);
  fill_values(hidden, 1000 + projection_width, 1.0f / 128.0f);
  fill_values(input_norm, 2000 + projection_width, 1.0f / 64.0f);
  fill_values(qkv_weights, 3000 + projection_width, 1.0f / 512.0f);
  fill_values(head_norm, 4000 + head_dim, 1.0f / 64.0f);
  fill_values(o_proj, 5000 + q_width, 1.0f / 4096.0f);

  thrust::device_vector<__nv_bfloat16> d_hidden(hidden.size());
  thrust::device_vector<__nv_bfloat16> d_input_norm(input_norm.size());
  thrust::device_vector<__nv_bfloat16> d_qkv_weights(qkv_weights.size());
  thrust::device_vector<__nv_bfloat16> d_o_proj(o_proj.size());
  thrust::device_vector<__nv_bfloat16> d_head_norm(head_norm.size());
  thrust::device_vector<__nv_bfloat16> d_hidden_norm(hidden_norm.size());
  thrust::device_vector<float> d_cos(cos.size());
  thrust::device_vector<float> d_sin(sin.size());
  thrust::device_vector<int32_t> d_page_table(1, 0);
  thrust::device_vector<int32_t> d_token_position(1, 0);
  thrust::device_vector<int32_t> d_seq_lengths(1, 1);
  bool ok = true;
  ok &= copy_to_device(d_hidden, hidden);
  ok &= copy_to_device(d_input_norm, input_norm);
  ok &= copy_to_device(d_qkv_weights, qkv_weights);
  ok &= copy_to_device(d_o_proj, o_proj);
  ok &= copy_to_device(d_head_norm, head_norm);
  ok &= copy_to_device(d_hidden_norm, hidden_norm);
  ok &= copy_to_device(d_cos, cos);
  ok &= copy_to_device(d_sin, sin);
  if (!ok) return false;

  const int64_t cache_count = cache_elements(cache_config);
  thrust::device_vector<__nv_bfloat16> d_ref_normed_hidden(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_ref_raw(projection_width);
  thrust::device_vector<__nv_bfloat16> d_ref_q(q_width);
  thrust::device_vector<__nv_bfloat16> d_ref_attention_out(q_width);
  thrust::device_vector<__nv_bfloat16> d_ref_o(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_ref_post_norm(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_ref_ffn_residual(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_ref_ffn_x(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<float> d_ref_partial_m(GEMMA4_NUM_QUERY_HEADS);
  thrust::device_vector<float> d_ref_partial_l(GEMMA4_NUM_QUERY_HEADS);
  thrust::device_vector<float> d_ref_partial_acc(
      size_t(GEMMA4_NUM_QUERY_HEADS) * head_dim);
  thrust::device_vector<__nv_bfloat16> d_ref_cache_k(cache_count);
  thrust::device_vector<__nv_bfloat16> d_ref_cache_v(cache_count);
  thrust::device_vector<__nv_bfloat16> d_layer_q(
      global ? GEMMA4_GLOBAL_QK_PROJ_SIZE : q_width);
  thrust::device_vector<__nv_bfloat16> d_layer_attention_out(
      std::max(projection_width, q_width));
  const size_t attention_tail_scratch_count = GEMMA4_HIDDEN_SIZE + 1;
  thrust::device_vector<float> d_layer_partial_acc(
      attention_tail_scratch_count);
  thrust::device_vector<__nv_bfloat16> d_layer_cache_k(cache_count);
  thrust::device_vector<__nv_bfloat16> d_layer_cache_v(cache_count);
#if GEMMA4_DECODE_ATTENTION_READY_HANDOFF
  thrust::device_vector<uint32_t> d_attention_ready(
      GEMMA4_SLIDING_KV_HEADS, 0);
#endif

  cudaError_t status = gemma4_rmsnorm_bf16(
      raw_ptr(d_ref_normed_hidden), raw_ptr(d_hidden), raw_ptr(d_input_norm), 1,
      GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_NORM_EPS, nullptr);
  ok &= expect_true(status == cudaSuccess, "reference input norm succeeds");
  status = gemma4_projection_decode(
      projection, raw_ptr(d_ref_normed_hidden), raw_ptr(d_qkv_weights),
      raw_ptr(d_ref_raw), nullptr);
  ok &= expect_true(status == cudaSuccess, "reference QKV projection succeeds");
  const __nv_bfloat16 *raw_q = raw_ptr(d_ref_raw);
  const __nv_bfloat16 *raw_k = raw_q + q_width;
  const __nv_bfloat16 *raw_v = global ? nullptr : raw_k + kv_width;
  status = gemma4_flash_attention_decode_prepare_q_paged_kv_bf16(
      raw_ptr(d_ref_q), raw_ptr(d_ref_cache_k), raw_ptr(d_ref_cache_v),
      cache_config, raw_ptr(d_page_table), raw_ptr(d_token_position), 1, 0,
      raw_q, raw_k, raw_v, raw_ptr(d_head_norm), raw_ptr(d_head_norm),
      raw_ptr(d_cos), raw_ptr(d_sin), nullptr);
  ok &= expect_true(status == cudaSuccess, "reference decode prep succeeds");
  status = gemma4_flash_attention_decode_paged_bf16(
      raw_ptr(d_ref_attention_out), raw_ptr(d_ref_partial_m),
      raw_ptr(d_ref_partial_l), raw_ptr(d_ref_partial_acc), raw_ptr(d_ref_q),
      raw_ptr(d_ref_cache_k), raw_ptr(d_ref_cache_v), raw_ptr(d_page_table),
      raw_ptr(d_seq_lengths), cache_config, 0, 1, softmax_scale, 1, 1,
      nullptr);
  ok &= expect_true(status == cudaSuccess, "reference decode attention succeeds");
  status = gemma4_projection_decode(
      o_projection, raw_ptr(d_ref_attention_out), raw_ptr(d_o_proj),
      raw_ptr(d_ref_o), nullptr);
  ok &= expect_true(status == cudaSuccess, "reference O projection succeeds");
  status = gemma4_rmsnorm_bf16(
      raw_ptr(d_ref_post_norm), raw_ptr(d_ref_o), raw_ptr(d_hidden_norm), 1,
      GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_NORM_EPS, nullptr);
  ok &= expect_true(status == cudaSuccess, "reference post-attention norm succeeds");
  status = gemma4_residual_add_bf16(
      raw_ptr(d_ref_ffn_residual), raw_ptr(d_hidden), raw_ptr(d_ref_post_norm),
      GEMMA4_HIDDEN_SIZE, nullptr);
  ok &= expect_true(status == cudaSuccess, "reference attention residual add succeeds");
  status = gemma4_rmsnorm_bf16(
      raw_ptr(d_ref_ffn_x), raw_ptr(d_ref_ffn_residual),
      raw_ptr(d_hidden_norm), 1, GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_NORM_EPS,
      nullptr);
  ok &= expect_true(status == cudaSuccess, "reference pre-FFN norm succeeds");
  status = cudaDeviceSynchronize();
  ok &= expect_true(status == cudaSuccess, "reference ingress synchronizes");
  if (!ok) return false;

  thrust::device_vector<__nv_bfloat16> d_residual_out(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_normed_out(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_ffn_x(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_ffn_residual(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_ffn_gate_up(
      size_t(GEMMA4_PACKED_FFN_SIZE) * GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_ffn_down(
      size_t(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<Gemma4FfnDecodeScratch> d_ffn_scratch(1);
  status = cudaMemset(
      raw_ptr(d_ffn_gate_up), 0, d_ffn_gate_up.size() * sizeof(__nv_bfloat16));
  ok &= expect_true(status == cudaSuccess, "zero FFN gate/up");
  status = cudaMemset(
      raw_ptr(d_ffn_down), 0, d_ffn_down.size() * sizeof(__nv_bfloat16));
  ok &= expect_true(status == cudaSuccess, "zero FFN down");
  if (!ok) return false;

  Gemma4DecodeMegakernelLayerArgs args = {};
  args.residual_out = raw_ptr(d_residual_out);
  args.normed_out = raw_ptr(d_normed_out);
  args.ffn_x = raw_ptr(d_ffn_x);
  args.ffn_residual = raw_ptr(d_ffn_residual);
  args.ffn_norm_weight = raw_ptr(d_hidden_norm);
  args.ffn_gate_up_decode = raw_ptr(d_ffn_gate_up);
  args.ffn_down_decode = raw_ptr(d_ffn_down);
  args.ffn_scratch = raw_ptr(d_ffn_scratch);
  args.attention_q = raw_ptr(d_layer_q);
  args.attention_out = raw_ptr(d_layer_attention_out);
  args.attention_partial_acc = raw_ptr(d_layer_partial_acc);
  args.attention_cache_k = raw_ptr(d_layer_cache_k);
  args.attention_cache_v = raw_ptr(d_layer_cache_v);
  args.attention_cache_config = cache_config;
  args.attention_page_table = raw_ptr(d_page_table);
  args.attention_token_position = raw_ptr(d_token_position);
  args.attention_seq_lengths = raw_ptr(d_seq_lengths);
  args.attention_cache_layer = 0;
  args.attention_split_size = 1;
  args.attention_num_splits = 1;
  args.attention_softmax_scale = softmax_scale;
  args.attention_x = raw_ptr(d_hidden);
  args.attention_input_norm_weight = raw_ptr(d_input_norm);
  args.attention_qkv_proj_col_major = raw_ptr(d_qkv_weights);
  args.attention_o_proj_col_major = raw_ptr(d_o_proj);
  args.attention_post_norm_weight = raw_ptr(d_hidden_norm);
  args.attention_pre_ffn_norm_weight = raw_ptr(d_hidden_norm);
  args.attention_q_norm_weight = raw_ptr(d_head_norm);
  args.attention_k_norm_weight = raw_ptr(d_head_norm);
  args.attention_cos = raw_ptr(d_cos);
  args.attention_sin = raw_ptr(d_sin);
#if GEMMA4_DECODE_ATTENTION_READY_HANDOFF
  args.attention_ready = raw_ptr(d_attention_ready);
  args.attention_ready_tag = 1;
#endif
  status = gemma4_decode_megakernel_flash_attention_layer_bf16(args, nullptr);
  ok &= expect_true(status == cudaSuccess, "fused layer launch succeeds");
  status = cudaDeviceSynchronize();
  ok &= expect_true(status == cudaSuccess, "fused layer synchronizes");
  if (!ok) return false;

  const char *label = global ? "global fused layer attention"
                             : "sliding fused layer attention";
  ok &= compare_device_bf16(d_layer_q, d_ref_q, q_width, 0.0f, label);
  ok &= compare_device_bf16(
      d_layer_cache_k, d_ref_cache_k, int(cache_count), 0.0f, label);
  ok &= compare_device_bf16(
      d_layer_cache_v, d_ref_cache_v, int(cache_count), 0.0f, label);
  ok &= compare_device_bf16(
      d_normed_out, d_ref_post_norm, GEMMA4_HIDDEN_SIZE, 1.0f / 32.0f,
      label);
  ok &= compare_device_bf16(
      d_ffn_residual, d_ref_ffn_residual, GEMMA4_HIDDEN_SIZE, 1.0f / 32.0f,
      label);
  ok &= compare_device_bf16(
      d_ffn_x, d_ref_ffn_x, GEMMA4_HIDDEN_SIZE, 1.0f / 32.0f, label);
  ok &= compare_device_bf16(
      d_residual_out, d_ref_ffn_residual, GEMMA4_HIDDEN_SIZE, 1.0f / 32.0f,
      label);
  return ok;
}

// Exercises the shared runtime-prep wrapper on a tiny owned runtime state.
bool test_runtime_prep_smoke() {
  Gemma4RuntimeState runtime = {};
  cudaError_t status = gemma4_runtime_state_init(&runtime, 1, 4, 2, nullptr);
  if (status != cudaSuccess) {
    fprintf(stderr, "FAIL: runtime init: %s\n", cudaGetErrorString(status));
    return false;
  }

  bool ok = true;
  status = gemma4_megakernel_prepare_runtime(
      &runtime, Gemma4MegakernelPrepMode::kPrefill, 2, nullptr);
  ok &= expect_true(status == cudaSuccess, "prefill runtime prep succeeds");
  ok &= expect_true(runtime.token_count == 2, "prefill token count is prompt rows");
  ok &= expect_true(runtime.h_seq_lengths[0] == 2, "prefill sequence length is set");

  status = gemma4_megakernel_prepare_runtime(
      &runtime, Gemma4MegakernelPrepMode::kDecode, 0, nullptr);
  ok &= expect_true(status == cudaSuccess, "decode runtime prep succeeds");
  ok &= expect_true(runtime.token_count == 1, "decode prep uploads one live row");
  ok &= expect_true(runtime.h_seq_lengths[0] == 3, "decode sequence length advances");

  gemma4_runtime_state_free(&runtime);
  return ok;
}

// Checks that the rewritten decode owner exposes only the full-step API surface.
int main() {
  bool ok = true;
  ok &= expect_true(
      gemma4_decode_megakernel_scratch_bytes() >
          gemma4_sample_next_scratch_bytes(),
      "decode scratch includes FFN state plus final sampling state");

  const cudaError_t prep_status = gemma4_megakernel_prepare_runtime(
      nullptr, Gemma4MegakernelPrepMode::kDecode, 0, nullptr);
  ok &= expect_true(
      prep_status == cudaErrorInvalidValue,
      "null runtime is rejected by shared prep");

  Gemma4DecodeMegakernelArgs args = {};
  const cudaError_t decode_status = gemma4_decode_megakernel(args);
  ok &= expect_true(
      decode_status == cudaErrorInvalidValue,
      "empty decode args are rejected at the public boundary");
  ok &= test_runtime_prep_smoke();
  ok &= test_decode_layer_attention_pipeline(false);
  ok &= test_decode_layer_attention_pipeline(true);

  if (!ok) {
    return 1;
  }
  printf("test_decode_megakernel passed\n");
  return 0;
}
