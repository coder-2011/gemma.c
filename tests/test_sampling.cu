#include "gemma4_sampling.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

struct Candidate {
  float score;
  int32_t token_id;
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

float sampling_uniform01_cpu(uint64_t seed, uint64_t step, int32_t batch_row) {
  uint64_t state = seed;
  state ^= (step + 0x9e3779b97f4a7c15ull) * 0xbf58476d1ce4e5b9ull;
  state ^= (uint64_t(batch_row) + 0x94d049bb133111ebull) *
           0x9e3779b97f4a7c15ull;
  const uint32_t bits = uint32_t(splitmix64_cpu(state) >> 40);
  return float(bits) * (1.0f / 16777216.0f);
}

int32_t reference_sample_from_logits(
    const std::vector<__nv_bfloat16> &logits,
    int32_t batch_row,
    Gemma4SamplingParams params) {
  std::vector<Candidate> candidates(GEMMA4_VOCAB_SIZE);
  const float inv_temperature = 1.0f / params.temperature;
  const size_t row_offset =
      static_cast<size_t>(batch_row) * GEMMA4_VOCAB_SIZE;
  for (int token_id = 0; token_id < GEMMA4_VOCAB_SIZE; ++token_id) {
    const float raw = bf16_to_float(logits[row_offset + token_id]);
    const float score = std::tanh(raw / GEMMA4_FINAL_LOGIT_SOFTCAPPING) *
                        GEMMA4_FINAL_LOGIT_SOFTCAPPING * inv_temperature;
    candidates[token_id] = {score, token_id};
  }
  auto better = [](Candidate a, Candidate b) {
    return better_candidate_cpu(a.score, a.token_id, b.score, b.token_id);
  };
  std::partial_sort(candidates.begin(), candidates.begin() + params.top_k,
                    candidates.end(), better);

  const float max_score = candidates[0].score;
  std::vector<float> weights(params.top_k);
  float total_weight = 0.0f;
  for (int i = 0; i < params.top_k; ++i) {
    weights[i] = std::exp(candidates[i].score - max_score);
    total_weight += weights[i];
  }

  const float threshold = params.top_p * total_weight;
  float nucleus_weight = 0.0f;
  int nucleus_size = 0;
  for (int i = 0; i < params.top_k; ++i) {
    nucleus_weight += weights[i];
    nucleus_size = i + 1;
    if (nucleus_weight >= threshold) {
      break;
    }
  }

  const float target =
      sampling_uniform01_cpu(params.seed, params.step, batch_row) *
      nucleus_weight;
  float cumulative = 0.0f;
  int32_t selected_token_id = candidates[nucleus_size - 1].token_id;
  for (int i = 0; i < nucleus_size; ++i) {
    cumulative += weights[i];
    if (cumulative > target) {
      selected_token_id = candidates[i].token_id;
      break;
    }
  }
  return selected_token_id;
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

std::vector<__nv_bfloat16> make_embedding_row(int32_t token_id) {
  std::vector<__nv_bfloat16> row(GEMMA4_HIDDEN_SIZE);
  for (int channel = 0; channel < GEMMA4_HIDDEN_SIZE; ++channel) {
    const float value =
        float((token_id % 97) - 48) * 0.01f + float(channel % 13) * 0.001f;
    row[channel] = __float2bfloat16_rn(value);
  }
  return row;
}

std::vector<__nv_bfloat16> make_zero_logits(int32_t batch_size) {
  return std::vector<__nv_bfloat16>(
      static_cast<size_t>(batch_size) * GEMMA4_VOCAB_SIZE,
      __float2bfloat16_rn(0.0f));
}

void set_logit(std::vector<__nv_bfloat16> &logits,
               int32_t batch_row,
               int32_t token_id,
               float value) {
  logits[static_cast<size_t>(batch_row) * GEMMA4_VOCAB_SIZE + token_id] =
      __float2bfloat16_rn(value);
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

  Gemma4SamplingParams params = {1.0f, 1.0f, 64, 1234u, 0u};
  status = gemma4_sample_from_logits_decode_bf16(
      nullptr, nullptr, nullptr, 0, nullptr, nullptr, 1, params, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr,
                 "expected cudaErrorInvalidValue for probabilistic null args\n");
    std::exit(1);
  }
}

void install_embedding_rows(
    DeviceBuffer<__nv_bfloat16> &d_lm_head,
    const std::vector<int32_t> &token_ids) {
  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  CHECK_CUDA(cudaMemset(d_lm_head.get(), 0,
                        lm_head_elems * sizeof(__nv_bfloat16)));

  std::vector<int32_t> unique_tokens = token_ids;
  std::sort(unique_tokens.begin(), unique_tokens.end());
  unique_tokens.erase(std::unique(unique_tokens.begin(), unique_tokens.end()),
                      unique_tokens.end());
  for (int32_t token_id : unique_tokens) {
    const std::vector<__nv_bfloat16> row = make_embedding_row(token_id);
    CHECK_CUDA(cudaMemcpy(d_lm_head.get() +
                              static_cast<size_t>(token_id) *
                                  GEMMA4_HIDDEN_SIZE,
                          row.data(), row.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
  }
}

void run_probabilistic_case(DeviceBuffer<__nv_bfloat16> &d_lm_head,
                            const std::vector<__nv_bfloat16> &logits,
                            int32_t batch_size,
                            Gemma4SamplingParams params,
                            const char *label) {
  std::vector<int32_t> expected_tokens(batch_size);
  for (int32_t row = 0; row < batch_size; ++row) {
    expected_tokens[row] = reference_sample_from_logits(logits, row, params);
  }
  install_embedding_rows(d_lm_head, expected_tokens);

  DeviceBuffer<__nv_bfloat16> d_logits(logits.size());
  DeviceBuffer<__nv_bfloat16> d_next_hidden(
      static_cast<size_t>(batch_size) * GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<int32_t> d_next_token(batch_size);
  d_logits.copy_from(logits);

  CHECK_CUDA(gemma4_sample_from_logits_decode_bf16(
      d_next_hidden.get(), d_next_token.get(), nullptr, 0,
      d_logits.get(), d_lm_head.get(), batch_size, params, 0));

  std::vector<int32_t> actual_tokens(batch_size);
  d_next_token.copy_to(actual_tokens);
  if (actual_tokens != expected_tokens) {
    for (int32_t row = 0; row < batch_size; ++row) {
      std::fprintf(stderr,
                   "%s token row=%d actual=%d expected=%d\n",
                   label, row, actual_tokens[row], expected_tokens[row]);
    }
    std::exit(1);
  }

  std::vector<__nv_bfloat16> actual_hidden(
      static_cast<size_t>(batch_size) * GEMMA4_HIDDEN_SIZE);
  d_next_hidden.copy_to(actual_hidden);
  for (int32_t row = 0; row < batch_size; ++row) {
    const std::vector<__nv_bfloat16> expected_row =
        make_embedding_row(expected_tokens[row]);
    std::vector<__nv_bfloat16> actual_row(
        actual_hidden.begin() + static_cast<size_t>(row) * GEMMA4_HIDDEN_SIZE,
        actual_hidden.begin() + static_cast<size_t>(row + 1) *
                                  GEMMA4_HIDDEN_SIZE);
    compare_hidden_bits(actual_row, expected_row, label);
  }
}

void run_probabilistic_cases() {
  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  DeviceBuffer<__nv_bfloat16> d_lm_head(lm_head_elems);

  std::vector<__nv_bfloat16> logits = make_zero_logits(2);
  set_logit(logits, 0, 17, 12.0f);
  set_logit(logits, 0, 18, 12.0f);
  set_logit(logits, 1, 23, 18.0f);
  set_logit(logits, 1, 24, 6.0f);
  Gemma4SamplingParams params = {0.7f, 0.8f, 64, 0x12345678abcdef00ull, 7u};
  run_probabilistic_case(d_lm_head, logits, 2, params, "probabilistic");
}

}  // namespace

int main() {
  run_invalid_args_case();
  run_greedy_cases();
  run_probabilistic_cases();
  std::printf("sampling tests passed\n");
  return 0;
}
