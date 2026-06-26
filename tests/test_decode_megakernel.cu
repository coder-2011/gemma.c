#include "gemma4_decode_megakernel.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Fails the test immediately when a CUDA API call returns an error.
void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

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

// Converts BF16 to float for small host-side assertions.
float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
void copy_to_device(thrust::device_vector<T> &dst, const std::vector<T> &src) {
  CHECK_CUDA(cudaMemcpy(raw_ptr(dst), src.data(), src.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
}

template <typename T>
void copy_to_host(std::vector<T> &dst, const thrust::device_vector<T> &src) {
  CHECK_CUDA(cudaMemcpy(dst.data(), raw_ptr(src), dst.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

// Compares BF16 rows bit-exactly after tied-embedding gather.
void compare_hidden_bits(const std::vector<__nv_bfloat16> &actual,
                         const std::vector<__nv_bfloat16> &expected,
                         const char *label) {
  const auto *actual_bits = reinterpret_cast<const uint16_t *>(actual.data());
  const auto *expected_bits = reinterpret_cast<const uint16_t *>(expected.data());
  for (int channel = 0; channel < GEMMA4_HIDDEN_SIZE; ++channel) {
    if (actual_bits[channel] != expected_bits[channel]) {
      std::fprintf(stderr,
                   "%s mismatch channel=%d actual=0x%04x expected=0x%04x\n",
                   label, channel, actual_bits[channel], expected_bits[channel]);
      std::exit(1);
    }
  }
}

// Creates a small nonzero hidden row for deterministic token scoring.
std::vector<__nv_bfloat16> make_hidden(float first_value) {
  std::vector<__nv_bfloat16> hidden(GEMMA4_HIDDEN_SIZE);
  hidden[0] = __float2bfloat16_rn(first_value);
  hidden[7] = __float2bfloat16_rn(0.25f);
  return hidden;
}

// Mirrors the scaled tied-embedding row returned as the next decode input.
std::vector<__nv_bfloat16> scaled_embedding_row(
    const std::vector<__nv_bfloat16> &row) {
  std::vector<__nv_bfloat16> scaled(row.size());
  for (size_t i = 0; i < row.size(); ++i) {
    scaled[i] =
        __float2bfloat16_rn(bf16_to_float(row[i]) * GEMMA4_EMBEDDING_SCALE);
  }
  return scaled;
}

// Creates an all-ones final RMSNorm weight vector.
std::vector<__nv_bfloat16> make_norm_weight() {
  return std::vector<__nv_bfloat16>(
      GEMMA4_HIDDEN_SIZE, __float2bfloat16_rn(1.0f));
}

// Clears the LM head and installs the only row that should win token selection.
void install_target_row(thrust::device_vector<__nv_bfloat16> &d_lm_head,
                        int32_t token_id,
                        const std::vector<__nv_bfloat16> &row) {
  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  CHECK_CUDA(cudaMemset(raw_ptr(d_lm_head), 0,
                        lm_head_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemcpy(raw_ptr(d_lm_head) +
                            static_cast<size_t>(token_id) *
                                GEMMA4_HIDDEN_SIZE,
                        row.data(), row.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
}

// Writes one BF16 value into a device array.
void write_device_bf16(__nv_bfloat16 *dst, size_t index, float value) {
  const __nv_bfloat16 bf16 = __float2bfloat16_rn(value);
  CHECK_CUDA(cudaMemcpy(dst + index, &bf16, sizeof(bf16),
                        cudaMemcpyHostToDevice));
}

// Runs one cooperative FFN-tail step and returns the selected token id.
int32_t run_ffn_tail_once(
    thrust::device_vector<__nv_bfloat16> &d_residual_out,
    thrust::device_vector<__nv_bfloat16> &d_normed_out,
    thrust::device_vector<__nv_bfloat16> &d_next_hidden,
    thrust::device_vector<int32_t> &d_next_token,
    thrust::device_vector<unsigned char> &d_scratch,
    thrust::device_vector<__nv_bfloat16> &d_x,
    thrust::device_vector<__nv_bfloat16> &d_residual,
    thrust::device_vector<__nv_bfloat16> &d_gamma,
    thrust::device_vector<__nv_bfloat16> &d_gate_up,
    thrust::device_vector<__nv_bfloat16> &d_down,
    thrust::device_vector<__nv_bfloat16> &d_layer_scalar,
    thrust::device_vector<__nv_bfloat16> &d_final_norm,
    thrust::device_vector<__nv_bfloat16> &d_lm_head) {
  Gemma4DecodeMegakernelFfnTailArgs args = {};
  args.residual_out = raw_ptr(d_residual_out);
  args.normed_out = raw_ptr(d_normed_out);
  args.next_hidden = raw_ptr(d_next_hidden);
  args.next_token = raw_ptr(d_next_token);
  args.ffn_x = raw_ptr(d_x);
  args.ffn_residual = raw_ptr(d_residual);
  args.ffn_norm_weight = raw_ptr(d_gamma);
  args.ffn_gate_up_decode = raw_ptr(d_gate_up);
  args.ffn_down_decode = raw_ptr(d_down);
  args.layer_scalar = raw_ptr(d_layer_scalar);
  args.final_norm_weight = raw_ptr(d_final_norm);
  args.lm_head_col_major = raw_ptr(d_lm_head);
  CHECK_CUDA(gemma4_decode_megakernel_ffn_tail_bf16(
      args, raw_ptr(d_scratch), gemma4_decode_megakernel_ffn_tail_scratch_bytes(),
      0));
  CHECK_CUDA(cudaDeviceSynchronize());

  int32_t token = -1;
  CHECK_CUDA(cudaMemcpy(&token, raw_ptr(d_next_token), sizeof(token),
                        cudaMemcpyDeviceToHost));
  return token;
}

// Verifies the cooperative FFN-tail launch reaches final sampling and gather.
void run_ffn_tail_zero_weight_case() {
  constexpr int kTargetToken = 6789;
  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t gate_up_elems =
      static_cast<size_t>(GEMMA4_PACKED_FFN_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t down_elems =
      static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE;

  const std::vector<__nv_bfloat16> x = make_hidden(0.5f);
  const std::vector<__nv_bfloat16> residual = make_hidden(1.0f);
  const std::vector<__nv_bfloat16> norm_weight = make_norm_weight();
  const std::vector<__nv_bfloat16> layer_scalar(
      1, __float2bfloat16_rn(0.5f));

  thrust::device_vector<__nv_bfloat16> d_x(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_residual(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_gamma(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_residual_out(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_normed_out(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_next_hidden(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_gate_up(gate_up_elems);
  thrust::device_vector<__nv_bfloat16> d_down(down_elems);
  thrust::device_vector<__nv_bfloat16> d_layer_scalar(1);
  thrust::device_vector<__nv_bfloat16> d_final_norm(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_lm_head(lm_head_elems);
  thrust::device_vector<int32_t> d_next_token(1);
  thrust::device_vector<unsigned char> d_scratch(
      gemma4_decode_megakernel_ffn_tail_scratch_bytes());

  copy_to_device(d_x, x);
  copy_to_device(d_residual, residual);
  copy_to_device(d_gamma, norm_weight);
  copy_to_device(d_layer_scalar, layer_scalar);
  copy_to_device(d_final_norm, norm_weight);
  CHECK_CUDA(cudaMemset(raw_ptr(d_gate_up), 0,
                        gate_up_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(raw_ptr(d_down), 0,
                        down_elems * sizeof(__nv_bfloat16)));

  install_target_row(d_lm_head, kTargetToken, residual);
  const int32_t token =
      run_ffn_tail_once(d_residual_out, d_normed_out, d_next_hidden,
                        d_next_token, d_scratch, d_x, d_residual, d_gamma,
                        d_gate_up, d_down, d_layer_scalar, d_final_norm,
                        d_lm_head);
  if (token != kTargetToken) {
    std::fprintf(stderr, "ffn tail token actual=%d expected=%d\n",
                 token, kTargetToken);
    std::exit(1);
  }

  std::vector<__nv_bfloat16> actual_next_hidden(GEMMA4_HIDDEN_SIZE);
  copy_to_host(actual_next_hidden, d_next_hidden);
  compare_hidden_bits(actual_next_hidden, scaled_embedding_row(residual),
                      "ffn tail next hidden");

  std::vector<__nv_bfloat16> expected_residual = residual;
  for (int channel = 0; channel < GEMMA4_HIDDEN_SIZE; ++channel) {
    expected_residual[channel] =
        __float2bfloat16_rn(bf16_to_float(residual[channel]) * 0.5f);
  }
  std::vector<__nv_bfloat16> actual_residual(GEMMA4_HIDDEN_SIZE);
  copy_to_host(actual_residual, d_residual_out);
  compare_hidden_bits(actual_residual, expected_residual,
                      "ffn tail layer scalar");
}

// Verifies the optional FlashAttention phase runs inside the megakernel tail.
void run_flash_attention_flag_case() {
  constexpr int kTargetToken = 6789;
  constexpr int kRotaryHalf = GEMMA4_SLIDING_HEAD_DIM / 2;
  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t gate_up_elems =
      static_cast<size_t>(GEMMA4_PACKED_FFN_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t down_elems =
      static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t q_weight_elems =
      static_cast<size_t>(GEMMA4_SLIDING_Q_PROJ_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t kv_weight_elems =
      static_cast<size_t>(GEMMA4_SLIDING_KV_PROJ_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t o_weight_elems =
      static_cast<size_t>(GEMMA4_HIDDEN_SIZE) *
      GEMMA4_SLIDING_ATTENTION_OUT_SIZE;
  Gemma4KvCacheConfig cache_config = {
      1,
      1,
      1,
      1,
      GEMMA4_SLIDING_KV_HEADS,
      GEMMA4_SLIDING_HEAD_DIM,
      1,
  };

  const std::vector<__nv_bfloat16> x = make_hidden(0.5f);
  const std::vector<__nv_bfloat16> residual = make_hidden(1.0f);
  const std::vector<__nv_bfloat16> norm_weight = make_norm_weight();
  const std::vector<__nv_bfloat16> layer_scalar(
      1, __float2bfloat16_rn(1.0f));
  const std::vector<__nv_bfloat16> head_weight(
      GEMMA4_SLIDING_HEAD_DIM, __float2bfloat16_rn(1.0f));
  const std::vector<float> cos(kRotaryHalf, 1.0f);
  const std::vector<float> sin(kRotaryHalf, 0.0f);
  const std::vector<int32_t> page_table = {0};
  const std::vector<int32_t> token_position = {0};
  const std::vector<int32_t> seq_lengths = {1};

  thrust::device_vector<__nv_bfloat16> d_x(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_residual(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_gamma(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_residual_out(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_split_residual_out(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_normed_out(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_split_normed_out(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_next_hidden(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_split_next_hidden(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_gate_up(gate_up_elems);
  thrust::device_vector<__nv_bfloat16> d_down(down_elems);
  thrust::device_vector<__nv_bfloat16> d_layer_scalar(1);
  thrust::device_vector<__nv_bfloat16> d_final_norm(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_lm_head(lm_head_elems);
  thrust::device_vector<int32_t> d_next_token(1);
  thrust::device_vector<int32_t> d_split_next_token(1);
  thrust::device_vector<unsigned char> d_scratch(
      gemma4_decode_megakernel_ffn_tail_scratch_bytes());
  thrust::device_vector<unsigned char> d_spine_scratch(
      gemma4_decode_megakernel_spine_scratch_bytes());

  thrust::device_vector<__nv_bfloat16> d_attention_x(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_input_norm(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_q_norm(GEMMA4_SLIDING_HEAD_DIM);
  thrust::device_vector<__nv_bfloat16> d_k_norm(GEMMA4_SLIDING_HEAD_DIM);
  thrust::device_vector<__nv_bfloat16> d_w_q(q_weight_elems);
  thrust::device_vector<__nv_bfloat16> d_w_k(kv_weight_elems);
  thrust::device_vector<__nv_bfloat16> d_w_v(kv_weight_elems);
  thrust::device_vector<__nv_bfloat16> d_w_o(o_weight_elems);
  thrust::device_vector<__nv_bfloat16> d_attention_q(GEMMA4_SLIDING_Q_PROJ_SIZE);
  thrust::device_vector<__nv_bfloat16> d_attention_out(
      GEMMA4_SLIDING_ATTENTION_OUT_SIZE);
  thrust::device_vector<__nv_bfloat16> d_cache_k(
      GEMMA4_SLIDING_KV_HEADS * GEMMA4_SLIDING_HEAD_DIM);
  thrust::device_vector<__nv_bfloat16> d_cache_v(
      GEMMA4_SLIDING_KV_HEADS * GEMMA4_SLIDING_HEAD_DIM);
  thrust::device_vector<float> d_partial_m(GEMMA4_NUM_QUERY_HEADS);
  thrust::device_vector<float> d_partial_l(GEMMA4_NUM_QUERY_HEADS);
  thrust::device_vector<float> d_partial_acc(
      GEMMA4_NUM_QUERY_HEADS * GEMMA4_SLIDING_HEAD_DIM);
  thrust::device_vector<float> d_cos(cos.size());
  thrust::device_vector<float> d_sin(sin.size());
  thrust::device_vector<int32_t> d_page_table(page_table.size());
  thrust::device_vector<int32_t> d_token_position(token_position.size());
  thrust::device_vector<int32_t> d_seq_lengths(seq_lengths.size());

  copy_to_device(d_x, x);
  copy_to_device(d_residual, residual);
  copy_to_device(d_gamma, norm_weight);
  copy_to_device(d_layer_scalar, layer_scalar);
  copy_to_device(d_final_norm, norm_weight);
  copy_to_device(d_attention_x, x);
  copy_to_device(d_input_norm, norm_weight);
  copy_to_device(d_q_norm, head_weight);
  copy_to_device(d_k_norm, head_weight);
  copy_to_device(d_cos, cos);
  copy_to_device(d_sin, sin);
  copy_to_device(d_page_table, page_table);
  copy_to_device(d_token_position, token_position);
  copy_to_device(d_seq_lengths, seq_lengths);

  CHECK_CUDA(cudaMemset(raw_ptr(d_gate_up), 0,
                        gate_up_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(raw_ptr(d_down), 0,
                        down_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(raw_ptr(d_w_q), 0,
                        q_weight_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(raw_ptr(d_w_k), 0,
                        kv_weight_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(raw_ptr(d_w_v), 0,
                        kv_weight_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(raw_ptr(d_w_o), 0,
                        o_weight_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(raw_ptr(d_attention_out), 0,
                        GEMMA4_SLIDING_ATTENTION_OUT_SIZE *
                            sizeof(__nv_bfloat16)));
  write_device_bf16(raw_ptr(d_w_q), 0, 1.0f);
  write_device_bf16(raw_ptr(d_w_k), 0, 1.0f);
  write_device_bf16(raw_ptr(d_w_v), 0, 1.0f);

  install_target_row(d_lm_head, kTargetToken, x);

  Gemma4DecodeMegakernelFfnTailArgs args = {};
  args.residual_out = raw_ptr(d_residual_out);
  args.normed_out = raw_ptr(d_normed_out);
  args.next_hidden = raw_ptr(d_next_hidden);
  args.next_token = raw_ptr(d_next_token);
  args.ffn_x = raw_ptr(d_x);
  args.ffn_residual = raw_ptr(d_residual);
  args.ffn_norm_weight = raw_ptr(d_gamma);
  args.ffn_gate_up_decode = raw_ptr(d_gate_up);
  args.ffn_down_decode = raw_ptr(d_down);
  args.layer_scalar = raw_ptr(d_layer_scalar);
  args.final_norm_weight = raw_ptr(d_final_norm);
  args.lm_head_col_major = raw_ptr(d_lm_head);
  args.flags = GEMMA4_DECODE_MEGAKERNEL_FLAG_FLASH_ATTENTION;
  args.attention_q = raw_ptr(d_attention_q);
  args.attention_out = raw_ptr(d_attention_out);
  args.attention_partial_m = raw_ptr(d_partial_m);
  args.attention_partial_l = raw_ptr(d_partial_l);
  args.attention_partial_acc = raw_ptr(d_partial_acc);
  args.attention_cache_k = raw_ptr(d_cache_k);
  args.attention_cache_v = raw_ptr(d_cache_v);
  args.attention_cache_config = cache_config;
  args.attention_page_table = raw_ptr(d_page_table);
  args.attention_token_position = raw_ptr(d_token_position);
  args.attention_seq_lengths = raw_ptr(d_seq_lengths);
  args.attention_cache_layer = 0;
  args.attention_split_size = 1;
  args.attention_num_splits = 1;
  args.attention_softmax_scale =
      1.0f / std::sqrt(float(GEMMA4_SLIDING_HEAD_DIM));
  args.attention_x = raw_ptr(d_attention_x);
  args.attention_input_norm_weight = raw_ptr(d_input_norm);
  args.attention_weights = {raw_ptr(d_w_q), raw_ptr(d_w_k), raw_ptr(d_w_v), 0, 0, 0};
  args.attention_o_proj_col_major = raw_ptr(d_w_o);
  args.attention_post_norm_weight = raw_ptr(d_gamma);
  args.attention_pre_ffn_norm_weight = raw_ptr(d_gamma);
  args.attention_q_norm_weight = raw_ptr(d_q_norm);
  args.attention_k_norm_weight = raw_ptr(d_k_norm);
  args.attention_cos = raw_ptr(d_cos);
  args.attention_sin = raw_ptr(d_sin);
  CHECK_CUDA(gemma4_decode_megakernel_ffn_tail_bf16(
      args, raw_ptr(d_scratch), gemma4_decode_megakernel_ffn_tail_scratch_bytes(),
      0));
  CHECK_CUDA(cudaDeviceSynchronize());

  int32_t current_token = -1;
  CHECK_CUDA(cudaMemcpy(&current_token, raw_ptr(d_next_token),
                        sizeof(current_token), cudaMemcpyDeviceToHost));
  if (current_token != kTargetToken) {
    std::fprintf(stderr, "flash attention tail token actual=%d expected=%d\n",
                 current_token, kTargetToken);
    std::exit(1);
  }

  std::vector<__nv_bfloat16> attention_out(GEMMA4_SLIDING_ATTENTION_OUT_SIZE);
  copy_to_host(attention_out, d_attention_out);
  if (bf16_to_float(attention_out[0]) <= 0.0f) {
    std::fprintf(stderr, "flash attention flag did not produce attention\n");
    std::exit(1);
  }

  CHECK_CUDA(cudaMemset(raw_ptr(d_cache_k), 0,
                        GEMMA4_SLIDING_KV_HEADS *
                            GEMMA4_SLIDING_HEAD_DIM *
                            sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(raw_ptr(d_cache_v), 0,
                        GEMMA4_SLIDING_KV_HEADS *
                            GEMMA4_SLIDING_HEAD_DIM *
                            sizeof(__nv_bfloat16)));
  args.residual_out = raw_ptr(d_split_residual_out);
  args.normed_out = raw_ptr(d_split_normed_out);
  args.next_hidden = raw_ptr(d_split_next_hidden);
  args.next_token = raw_ptr(d_split_next_token);
  CHECK_CUDA(gemma4_decode_megakernel_attention_ffn_bf16(
      args, raw_ptr(d_scratch), gemma4_decode_megakernel_ffn_tail_scratch_bytes(),
      0));

  Gemma4DecodeMegakernelSpineArgs spine_args = {};
  spine_args.state = raw_ptr(d_split_residual_out);
  spine_args.next_hidden = raw_ptr(d_split_next_hidden);
  spine_args.next_token = raw_ptr(d_split_next_token);
  spine_args.final_norm_weight = raw_ptr(d_final_norm);
  spine_args.lm_head_col_major = raw_ptr(d_lm_head);
  CHECK_CUDA(gemma4_decode_megakernel_spine_bf16(
      spine_args, raw_ptr(d_spine_scratch),
      gemma4_decode_megakernel_spine_scratch_bytes(), 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  int32_t split_token = -1;
  CHECK_CUDA(cudaMemcpy(&split_token, raw_ptr(d_split_next_token),
                        sizeof(split_token), cudaMemcpyDeviceToHost));
  if (split_token != current_token) {
    std::fprintf(stderr, "split token actual=%d expected=%d\n",
                 split_token, current_token);
    std::exit(1);
  }
}

}  // namespace

// Runs the focused decode megakernel regression suite.
int main() {
  run_ffn_tail_zero_weight_case();
  run_flash_attention_flag_case();
  std::printf("decode megakernel tests passed\n");
  return 0;
}
