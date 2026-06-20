#include "gemma4_decode_megakernel.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

// Creates a small nonzero hidden row for deterministic greedy scoring.
std::vector<__nv_bfloat16> make_hidden(float first_value) {
  std::vector<__nv_bfloat16> hidden(GEMMA4_HIDDEN_SIZE);
  hidden[0] = __float2bfloat16_rn(first_value);
  hidden[7] = __float2bfloat16_rn(0.25f);
  return hidden;
}

// Creates an all-ones final RMSNorm weight vector.
std::vector<__nv_bfloat16> make_norm_weight() {
  return std::vector<__nv_bfloat16>(
      GEMMA4_HIDDEN_SIZE, __float2bfloat16_rn(1.0f));
}

// Clears the LM head and installs the only row that should win greedy decode.
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
  args.ffn_gate_up_col_major = d_gate_up.get();
  args.ffn_down_row_major = d_down.get();
  args.final_norm_weight = d_final_norm.get();
  args.lm_head_col_major = d_lm_head.get();
  args.eps = GEMMA4_RMS_NORM_EPS;

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

  DeviceBuffer<__nv_bfloat16> d_x(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_residual(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_gamma(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_residual_out(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_normed_out(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_next_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_gate_up(gate_up_elems);
  DeviceBuffer<__nv_bfloat16> d_down(down_elems);
  DeviceBuffer<__nv_bfloat16> d_final_norm(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_lm_head(lm_head_elems);
  DeviceBuffer<int32_t> d_next_token(1);
  DeviceBuffer<unsigned char> d_scratch(
      gemma4_decode_megakernel_ffn_tail_scratch_bytes());

  d_x.copy_from(x);
  d_residual.copy_from(residual);
  d_gamma.copy_from(norm_weight);
  d_final_norm.copy_from(norm_weight);
  CHECK_CUDA(cudaMemset(d_gate_up.get(), 0,
                        gate_up_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_down.get(), 0,
                        down_elems * sizeof(__nv_bfloat16)));

  install_target_row(d_lm_head, kTargetToken, residual);
  const int32_t token =
      run_ffn_tail_once(d_residual_out, d_normed_out, d_next_hidden,
                        d_next_token, d_scratch, d_x, d_residual, d_gamma,
                        d_gate_up, d_down, d_final_norm, d_lm_head);
  if (token != kTargetToken) {
    std::fprintf(stderr, "ffn tail token actual=%d expected=%d\n",
                 token, kTargetToken);
    std::exit(1);
  }

  std::vector<__nv_bfloat16> actual_next_hidden(GEMMA4_HIDDEN_SIZE);
  d_next_hidden.copy_to(actual_next_hidden);
  compare_hidden_bits(actual_next_hidden, residual, "ffn tail next hidden");
}

}  // namespace

// Runs the focused decode megakernel regression suite.
int main() {
  run_ffn_tail_zero_weight_case();
  std::printf("decode megakernel tests passed\n");
  return 0;
}
