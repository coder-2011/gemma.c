#include "gemma4_sampling.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
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

bool better_candidate_cpu(float score,
                          int32_t token_id,
                          float best_score,
                          int32_t best_token_id) {
  return score > best_score ||
         (score == best_score && token_id < best_token_id);
}

uint64_t splitmix64_cpu(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// Mirrors the device per-token open-interval uniform generator.
float gumbel_uniform01_cpu(uint64_t seed, uint64_t step, int32_t token_id) {
  uint64_t state = seed;
  state ^= (step + 0x9e3779b97f4a7c15ull) * 0xbf58476d1ce4e5b9ull;
  state ^= (uint64_t(token_id) + 0x94d049bb133111ebull) *
           0x9e3779b97f4a7c15ull;
  const uint32_t bits = uint32_t(splitmix64_cpu(state) >> 40);
  return (float(bits) + 0.5f) * (1.0f / 16777216.0f);
}

// Converts the CPU reference uniform into the same standard Gumbel noise.
float gumbel_noise_cpu(uint64_t seed, uint64_t step, int32_t token_id) {
  const float u = gumbel_uniform01_cpu(seed, step, token_id);
  return -std::log(-std::log(u));
}

// Applies the final Gemma softcap and temperature in the CPU reference path.
float transformed_lm_head_score_cpu(float raw_logit, float temperature) {
  const float capped =
      std::tanh(raw_logit / GEMMA4_FINAL_LOGIT_SOFTCAPPING) *
      GEMMA4_FINAL_LOGIT_SOFTCAPPING;
  return capped / temperature;
}

// Computes the exact expected token for a sparse LM-head test fixture.
int32_t reference_gumbel_sparse_lm_head(
    int32_t hot_token_id,
    float hot_raw_logit,
    Gemma4GumbelSamplingParams params) {
  float best_score = -std::numeric_limits<float>::infinity();
  int32_t best_token_id = GEMMA4_VOCAB_SIZE;
  for (int32_t token_id = 0; token_id < GEMMA4_VOCAB_SIZE; ++token_id) {
    const float raw_logit =
        token_id == hot_token_id ? hot_raw_logit : 0.0f;
    const float score =
        transformed_lm_head_score_cpu(raw_logit, params.temperature) +
        gumbel_noise_cpu(params.seed, params.step, token_id);
    if (better_candidate_cpu(score, token_id, best_score, best_token_id)) {
      best_score = score;
      best_token_id = token_id;
    }
  }
  return best_token_id;
}

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
              const std::vector<__nv_bfloat16> &expected_row,
              int32_t expected_token,
              const char *label) {
  CHECK_CUDA(gemma4_greedy_sample_next_decode_bf16(
      d_next_hidden.get(), d_next_token.get(), d_scratch.get(),
      gemma4_greedy_sample_next_scratch_bytes(), d_hidden.get(),
      d_lm_head.get(), 0));

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

void run_greedy_cases() {
  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t scratch_bytes = gemma4_greedy_sample_next_scratch_bytes();

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
           target_row, kTargetToken, "single target");
}

void run_invalid_args_case() {
  cudaError_t status = gemma4_greedy_sample_next_decode_bf16(
      nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected cudaErrorInvalidValue for null arguments\n");
    std::exit(1);
  }

  Gemma4GumbelSamplingParams gumbel_params = {1.0f, 1234u, 0u};
  status = gemma4_gumbel_sample_next_decode_bf16(
      nullptr, nullptr, nullptr, 0, nullptr, nullptr, gumbel_params, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr,
                 "expected cudaErrorInvalidValue for gumbel null args\n");
    std::exit(1);
  }
}

// Runs one fused Gumbel sampling case and checks token plus gathered embedding.
void run_gumbel_case(DeviceBuffer<__nv_bfloat16> &d_lm_head,
                     DeviceBuffer<__nv_bfloat16> &d_hidden,
                     DeviceBuffer<__nv_bfloat16> &d_next_hidden,
                     DeviceBuffer<int32_t> &d_next_token,
                     DeviceBuffer<unsigned char> &d_scratch,
                     int32_t hot_token_id,
                     const std::vector<__nv_bfloat16> &hot_row,
                     float hot_raw_logit,
                     Gemma4GumbelSamplingParams params,
                     const char *label) {
  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  CHECK_CUDA(cudaMemset(d_lm_head.get(), 0,
                        lm_head_elems * sizeof(__nv_bfloat16)));
  if (hot_token_id >= 0) {
    CHECK_CUDA(cudaMemcpy(d_lm_head.get() +
                              static_cast<size_t>(hot_token_id) *
                                  GEMMA4_HIDDEN_SIZE,
                          hot_row.data(),
                          hot_row.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
  }

  CHECK_CUDA(gemma4_gumbel_sample_next_decode_bf16(
      d_next_hidden.get(), d_next_token.get(), d_scratch.get(),
      gemma4_gumbel_sample_next_scratch_bytes(), d_hidden.get(),
      d_lm_head.get(), params, 0));

  const int32_t expected_token =
      reference_gumbel_sparse_lm_head(hot_token_id, hot_raw_logit, params);
  int32_t actual_token = -1;
  CHECK_CUDA(cudaMemcpy(&actual_token, d_next_token.get(), sizeof(actual_token),
                        cudaMemcpyDeviceToHost));
  if (actual_token != expected_token) {
    std::fprintf(stderr, "%s actual_token=%d expected_token=%d\n", label,
                 actual_token, expected_token);
    std::exit(1);
  }

  std::vector<__nv_bfloat16> expected_row(GEMMA4_HIDDEN_SIZE);
  if (actual_token == hot_token_id) {
    expected_row = hot_row;
  }
  std::vector<__nv_bfloat16> actual_hidden(GEMMA4_HIDDEN_SIZE);
  d_next_hidden.copy_to(actual_hidden);
  compare_hidden_bits(actual_hidden, expected_row, label);
}

// Covers full-vocab Gumbel reduction and selected-row gather.
void run_gumbel_cases() {
  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t scratch_bytes = gemma4_gumbel_sample_next_scratch_bytes();

  DeviceBuffer<__nv_bfloat16> d_lm_head(lm_head_elems);
  DeviceBuffer<__nv_bfloat16> d_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_next_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<int32_t> d_next_token(1);
  DeviceBuffer<unsigned char> d_scratch(scratch_bytes);

  const std::vector<__nv_bfloat16> hidden = make_hidden();
  d_hidden.copy_from(hidden);

  const std::vector<__nv_bfloat16> zero_row(GEMMA4_HIDDEN_SIZE);
  Gemma4GumbelSamplingParams zero_params = {1.0f, 0xabcdu, 7u};
  run_gumbel_case(d_lm_head, d_hidden, d_next_hidden, d_next_token, d_scratch,
                  -1, zero_row, 0.0f, zero_params, "gumbel zero logits");

  constexpr int32_t kHotToken = 54321;
  std::vector<__nv_bfloat16> hot_row(GEMMA4_HIDDEN_SIZE);
  hot_row[0] = __float2bfloat16_rn(256.0f);
  Gemma4GumbelSamplingParams hot_params = {0.7f, 0x123456u, 11u};
  run_gumbel_case(d_lm_head, d_hidden, d_next_hidden, d_next_token, d_scratch,
                  kHotToken, hot_row, 256.0f, hot_params,
                  "gumbel hot token");
}

}  // namespace

int main() {
  run_invalid_args_case();
  run_greedy_cases();
  run_gumbel_cases();
  std::printf("sampling tests passed\n");
  return 0;
}
