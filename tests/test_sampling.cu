#include "gemma4_sampling.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ > 0) {
      CHECK_CUDA(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *get() { return ptr_; }
  const T *get() const { return ptr_; }

  void copy_from(const std::vector<T> &src) {
    CHECK_CUDA(cudaMemcpy(ptr_, src.data(), count_ * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  void copy_to(std::vector<T> &dst) const {
    CHECK_CUDA(cudaMemcpy(dst.data(), ptr_, count_ * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

 private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

std::vector<__nv_bfloat16> make_hidden() {
  std::vector<__nv_bfloat16> hidden(GEMMA4_HIDDEN_SIZE);
  hidden[0] = __float2bfloat16_rn(1.0f);
  hidden[1] = __float2bfloat16_rn(0.5f);
  return hidden;
}

std::vector<__nv_bfloat16> make_target_row() {
  std::vector<__nv_bfloat16> row(GEMMA4_HIDDEN_SIZE);
  row[0] = __float2bfloat16_rn(2.0f);
  row[1] = __float2bfloat16_rn(0.25f);
  row[17] = __float2bfloat16_rn(-1.0f);
  return row;
}

// Mirrors the scaled embedding row returned as the next decode input.
std::vector<__nv_bfloat16> scaled_embedding_row(
    const std::vector<__nv_bfloat16> &row) {
  std::vector<__nv_bfloat16> scaled(row.size());
  for (size_t i = 0; i < row.size(); ++i) {
    scaled[i] =
        __float2bfloat16_rn(__bfloat162float(row[i]) * GEMMA4_EMBEDDING_SCALE);
  }
  return scaled;
}

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

void run_case(DeviceBuffer<__nv_bfloat16> &d_lm_head,
              DeviceBuffer<__nv_bfloat16> &d_hidden,
              DeviceBuffer<__nv_bfloat16> &d_next_hidden,
              DeviceBuffer<int32_t> &d_next_token,
              DeviceBuffer<unsigned char> &d_scratch,
              size_t scratch_bytes,
              const std::vector<__nv_bfloat16> &expected_row,
              int32_t expected_token,
              const char *label) {
  CHECK_CUDA(gemma4_sample_next_decode_bf16(
      d_next_hidden.get(), d_next_token.get(), d_scratch.get(), scratch_bytes,
      d_hidden.get(), d_lm_head.get(), 0));

  int32_t actual_token = -1;
  CHECK_CUDA(cudaMemcpy(&actual_token, d_next_token.get(), sizeof(actual_token),
                        cudaMemcpyDeviceToHost));
  if (actual_token != expected_token) {
    std::fprintf(stderr, "%s actual_token=%d expected_token=%d\n", label,
                 actual_token, expected_token);
    std::exit(1);
  }

  std::vector<__nv_bfloat16> actual_hidden(GEMMA4_HIDDEN_SIZE);
  d_next_hidden.copy_to(actual_hidden);
  compare_hidden_bits(actual_hidden, expected_row, label);
}

void run_sample_cases() {
  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t scratch_bytes =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE / 8) *
      (sizeof(Gemma4SampleCandidate) + sizeof(uint32_t));

  DeviceBuffer<__nv_bfloat16> d_lm_head(lm_head_elems);
  DeviceBuffer<__nv_bfloat16> d_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_next_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<int32_t> d_next_token(1);
  DeviceBuffer<unsigned char> d_scratch(scratch_bytes);

  const std::vector<__nv_bfloat16> hidden = make_hidden();
  d_hidden.copy_from(hidden);

  constexpr int32_t kTargetToken = 12345;
  const std::vector<__nv_bfloat16> target_row = make_target_row();
  CHECK_CUDA(cudaMemset(d_lm_head.get(), 0,
                        lm_head_elems * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemcpy(d_lm_head.get() +
                            static_cast<size_t>(kTargetToken) *
                                GEMMA4_HIDDEN_SIZE,
                        target_row.data(),
                        target_row.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  run_case(d_lm_head, d_hidden, d_next_hidden, d_next_token, d_scratch,
           scratch_bytes,
           scaled_embedding_row(target_row), kTargetToken, "single target");
}

}  // namespace

int main() {
  run_sample_cases();
  std::printf("sampling tests passed\n");
  return 0;
}
