#include "gemma4_decode_megakernel.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

// Converts BF16 to float for small host-side assertions.
float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

// Minimal RAII wrapper for device buffers owned by this test.
template <typename T>
class DeviceBuffer {
 public:
  // Allocates `count` elements on the current CUDA device.
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ > 0) {
      CHECK_CUDA(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }

  // Frees the device buffer; tests intentionally ignore cleanup errors.
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  // Returns the mutable device pointer.
  T *get() { return ptr_; }

  // Returns the const device pointer.
  const T *get() const { return ptr_; }

  // Copies a host vector into the full device allocation.
  void copy_from(const std::vector<T> &src) {
    CHECK_CUDA(cudaMemcpy(ptr_, src.data(), count_ * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  // Copies the full device allocation into a host vector.
  void copy_to(std::vector<T> &dst) const {
    CHECK_CUDA(cudaMemcpy(dst.data(), ptr_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

 private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

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
void install_target_row(DeviceBuffer<__nv_bfloat16> &d_lm_head,
                        int32_t token_id,
                        const std::vector<__nv_bfloat16> &row) {
  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  CHECK_CUDA(cudaMemset(d_lm_head.get(), 0,
                        lm_head_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemcpy(d_lm_head.get() +
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
    DeviceBuffer<__nv_bfloat16> &d_residual_out,
    DeviceBuffer<__nv_bfloat16> &d_normed_out,
    DeviceBuffer<__nv_bfloat16> &d_next_hidden,
    DeviceBuffer<int32_t> &d_next_token,
    DeviceBuffer<unsigned char> &d_scratch,
    DeviceBuffer<__nv_bfloat16> &d_x,
    DeviceBuffer<__nv_bfloat16> &d_residual,
    DeviceBuffer<__nv_bfloat16> &d_gamma,
    DeviceBuffer<__nv_bfloat16> &d_gate_up,
    DeviceBuffer<__nv_bfloat16> &d_down,
    DeviceBuffer<__nv_bfloat16> &d_layer_scalar,
    DeviceBuffer<__nv_bfloat16> &d_final_norm,
    DeviceBuffer<__nv_bfloat16> &d_lm_head) {
  Gemma4DecodeMegakernelFfnTailArgs args = {};
  args.residual_out = d_residual_out.get();
  args.normed_out = d_normed_out.get();
  args.next_hidden = d_next_hidden.get();
  args.next_token = d_next_token.get();
  args.ffn_x = d_x.get();
  args.ffn_residual = d_residual.get();
  args.ffn_norm_weight = d_gamma.get();
  args.ffn_gate_up_decode = d_gate_up.get();
  args.ffn_down_decode = d_down.get();
  args.layer_scalar = d_layer_scalar.get();
  args.final_norm_weight = d_final_norm.get();
  args.lm_head_col_major = d_lm_head.get();
  CHECK_CUDA(gemma4_decode_megakernel_ffn_tail_bf16(
      args, d_scratch.get(), gemma4_decode_megakernel_ffn_tail_scratch_bytes(),
      0));
  CHECK_CUDA(cudaDeviceSynchronize());

  int32_t token = -1;
  CHECK_CUDA(cudaMemcpy(&token, d_next_token.get(), sizeof(token),
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

  DeviceBuffer<__nv_bfloat16> d_x(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_residual(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_gamma(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_residual_out(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_normed_out(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_next_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_gate_up(gate_up_elems);
  DeviceBuffer<__nv_bfloat16> d_down(down_elems);
  DeviceBuffer<__nv_bfloat16> d_layer_scalar(1);
  DeviceBuffer<__nv_bfloat16> d_final_norm(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_lm_head(lm_head_elems);
  DeviceBuffer<int32_t> d_next_token(1);
  DeviceBuffer<unsigned char> d_scratch(
      gemma4_decode_megakernel_ffn_tail_scratch_bytes());

  d_x.copy_from(x);
  d_residual.copy_from(residual);
  d_gamma.copy_from(norm_weight);
  d_layer_scalar.copy_from(layer_scalar);
  d_final_norm.copy_from(norm_weight);
  CHECK_CUDA(cudaMemset(d_gate_up.get(), 0,
                        gate_up_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_down.get(), 0,
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
  d_next_hidden.copy_to(actual_next_hidden);
  compare_hidden_bits(actual_next_hidden, scaled_embedding_row(residual),
                      "ffn tail next hidden");

  std::vector<__nv_bfloat16> expected_residual = residual;
  for (int channel = 0; channel < GEMMA4_HIDDEN_SIZE; ++channel) {
    expected_residual[channel] =
        __float2bfloat16_rn(bf16_to_float(residual[channel]) * 0.5f);
  }
  std::vector<__nv_bfloat16> actual_residual(GEMMA4_HIDDEN_SIZE);
  d_residual_out.copy_to(actual_residual);
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

  DeviceBuffer<__nv_bfloat16> d_x(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_residual(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_gamma(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_residual_out(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_split_residual_out(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_normed_out(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_split_normed_out(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_next_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_split_next_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_gate_up(gate_up_elems);
  DeviceBuffer<__nv_bfloat16> d_down(down_elems);
  DeviceBuffer<__nv_bfloat16> d_layer_scalar(1);
  DeviceBuffer<__nv_bfloat16> d_final_norm(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_lm_head(lm_head_elems);
  DeviceBuffer<int32_t> d_next_token(1);
  DeviceBuffer<int32_t> d_split_next_token(1);
  DeviceBuffer<unsigned char> d_scratch(
      gemma4_decode_megakernel_ffn_tail_scratch_bytes());
  DeviceBuffer<unsigned char> d_spine_scratch(
      gemma4_decode_megakernel_spine_scratch_bytes());

  DeviceBuffer<__nv_bfloat16> d_attention_x(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_input_norm(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_q_norm(GEMMA4_SLIDING_HEAD_DIM);
  DeviceBuffer<__nv_bfloat16> d_k_norm(GEMMA4_SLIDING_HEAD_DIM);
  DeviceBuffer<__nv_bfloat16> d_w_q(q_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_k(kv_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_v(kv_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_o(o_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_attention_q(GEMMA4_SLIDING_Q_PROJ_SIZE);
  DeviceBuffer<__nv_bfloat16> d_attention_out(
      GEMMA4_SLIDING_ATTENTION_OUT_SIZE);
  DeviceBuffer<__nv_bfloat16> d_cache_k(
      GEMMA4_SLIDING_KV_HEADS * GEMMA4_SLIDING_HEAD_DIM);
  DeviceBuffer<__nv_bfloat16> d_cache_v(
      GEMMA4_SLIDING_KV_HEADS * GEMMA4_SLIDING_HEAD_DIM);
  DeviceBuffer<float> d_partial_m(GEMMA4_NUM_QUERY_HEADS);
  DeviceBuffer<float> d_partial_l(GEMMA4_NUM_QUERY_HEADS);
  DeviceBuffer<float> d_partial_acc(
      GEMMA4_NUM_QUERY_HEADS * GEMMA4_SLIDING_HEAD_DIM);
  DeviceBuffer<float> d_cos(cos.size());
  DeviceBuffer<float> d_sin(sin.size());
  DeviceBuffer<int32_t> d_page_table(page_table.size());
  DeviceBuffer<int32_t> d_token_position(token_position.size());
  DeviceBuffer<int32_t> d_seq_lengths(seq_lengths.size());

  d_x.copy_from(x);
  d_residual.copy_from(residual);
  d_gamma.copy_from(norm_weight);
  d_layer_scalar.copy_from(layer_scalar);
  d_final_norm.copy_from(norm_weight);
  d_attention_x.copy_from(x);
  d_input_norm.copy_from(norm_weight);
  d_q_norm.copy_from(head_weight);
  d_k_norm.copy_from(head_weight);
  d_cos.copy_from(cos);
  d_sin.copy_from(sin);
  d_page_table.copy_from(page_table);
  d_token_position.copy_from(token_position);
  d_seq_lengths.copy_from(seq_lengths);

  CHECK_CUDA(cudaMemset(d_gate_up.get(), 0,
                        gate_up_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_down.get(), 0,
                        down_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_w_q.get(), 0,
                        q_weight_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_w_k.get(), 0,
                        kv_weight_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_w_v.get(), 0,
                        kv_weight_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_w_o.get(), 0,
                        o_weight_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_attention_out.get(), 0,
                        GEMMA4_SLIDING_ATTENTION_OUT_SIZE *
                            sizeof(__nv_bfloat16)));
  write_device_bf16(d_w_q.get(), 0, 1.0f);
  write_device_bf16(d_w_k.get(), 0, 1.0f);
  write_device_bf16(d_w_v.get(), 0, 1.0f);

  install_target_row(d_lm_head, kTargetToken, x);

  Gemma4DecodeMegakernelFfnTailArgs args = {};
  args.residual_out = d_residual_out.get();
  args.normed_out = d_normed_out.get();
  args.next_hidden = d_next_hidden.get();
  args.next_token = d_next_token.get();
  args.ffn_x = d_x.get();
  args.ffn_residual = d_residual.get();
  args.ffn_norm_weight = d_gamma.get();
  args.ffn_gate_up_decode = d_gate_up.get();
  args.ffn_down_decode = d_down.get();
  args.layer_scalar = d_layer_scalar.get();
  args.final_norm_weight = d_final_norm.get();
  args.lm_head_col_major = d_lm_head.get();
  args.flags = GEMMA4_DECODE_MEGAKERNEL_FLAG_FLASH_ATTENTION;
  args.attention_q = d_attention_q.get();
  args.attention_out = d_attention_out.get();
  args.attention_partial_m = d_partial_m.get();
  args.attention_partial_l = d_partial_l.get();
  args.attention_partial_acc = d_partial_acc.get();
  args.attention_cache_k = d_cache_k.get();
  args.attention_cache_v = d_cache_v.get();
  args.attention_cache_config = cache_config;
  args.attention_page_table = d_page_table.get();
  args.attention_token_position = d_token_position.get();
  args.attention_seq_lengths = d_seq_lengths.get();
  args.attention_cache_layer = 0;
  args.attention_split_size = 1;
  args.attention_num_splits = 1;
  args.attention_softmax_scale =
      1.0f / std::sqrt(float(GEMMA4_SLIDING_HEAD_DIM));
  args.attention_x = d_attention_x.get();
  args.attention_input_norm_weight = d_input_norm.get();
  args.attention_weights = {d_w_q.get(), d_w_k.get(), d_w_v.get(), 0, 0, 0};
  args.attention_o_proj_col_major = d_w_o.get();
  args.attention_post_norm_weight = d_gamma.get();
  args.attention_pre_ffn_norm_weight = d_gamma.get();
  args.attention_q_norm_weight = d_q_norm.get();
  args.attention_k_norm_weight = d_k_norm.get();
  args.attention_cos = d_cos.get();
  args.attention_sin = d_sin.get();
  CHECK_CUDA(gemma4_decode_megakernel_ffn_tail_bf16(
      args, d_scratch.get(), gemma4_decode_megakernel_ffn_tail_scratch_bytes(),
      0));
  CHECK_CUDA(cudaDeviceSynchronize());

  int32_t current_token = -1;
  CHECK_CUDA(cudaMemcpy(&current_token, d_next_token.get(),
                        sizeof(current_token), cudaMemcpyDeviceToHost));
  if (current_token != kTargetToken) {
    std::fprintf(stderr, "flash attention tail token actual=%d expected=%d\n",
                 current_token, kTargetToken);
    std::exit(1);
  }

  std::vector<__nv_bfloat16> attention_out(GEMMA4_SLIDING_ATTENTION_OUT_SIZE);
  d_attention_out.copy_to(attention_out);
  if (bf16_to_float(attention_out[0]) <= 0.0f) {
    std::fprintf(stderr, "flash attention flag did not produce attention\n");
    std::exit(1);
  }

  CHECK_CUDA(cudaMemset(d_cache_k.get(), 0,
                        GEMMA4_SLIDING_KV_HEADS *
                            GEMMA4_SLIDING_HEAD_DIM *
                            sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_cache_v.get(), 0,
                        GEMMA4_SLIDING_KV_HEADS *
                            GEMMA4_SLIDING_HEAD_DIM *
                            sizeof(__nv_bfloat16)));
  args.residual_out = d_split_residual_out.get();
  args.normed_out = d_split_normed_out.get();
  args.next_hidden = d_split_next_hidden.get();
  args.next_token = d_split_next_token.get();
  CHECK_CUDA(gemma4_decode_megakernel_attention_ffn_bf16(
      args, d_scratch.get(), gemma4_decode_megakernel_ffn_tail_scratch_bytes(),
      0));

  Gemma4DecodeMegakernelSpineArgs spine_args = {};
  spine_args.state = d_split_residual_out.get();
  spine_args.next_hidden = d_split_next_hidden.get();
  spine_args.next_token = d_split_next_token.get();
  spine_args.final_norm_weight = d_final_norm.get();
  spine_args.lm_head_col_major = d_lm_head.get();
  CHECK_CUDA(gemma4_decode_megakernel_spine_bf16(
      spine_args, d_spine_scratch.get(),
      gemma4_decode_megakernel_spine_scratch_bytes(), 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  int32_t split_token = -1;
  CHECK_CUDA(cudaMemcpy(&split_token, d_split_next_token.get(),
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
