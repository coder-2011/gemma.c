#include "gemma4_kv_cache.cuh"
#include "gemma4_flash_attention.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Abort the test at the first CUDA error so the failing call site is explicit.
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

// Convert BF16 test values back to float for comparisons.
float bf16_to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

// Generate deterministic nontrivial BF16 values without storing fixtures.
__nv_bfloat16 make_value(int seed) {
  int centered = ((seed * 37 + 17) % 257) - 128;
  return __float2bfloat16_rn(static_cast<float>(centered) / 64.0f);
}

// Generate smaller BF16 values so CPU references stay stable across BF16 MMA.
__nv_bfloat16 make_prefill_value(int seed) {
  int centered = ((seed * 29 + 11) % 257) - 128;
  return __float2bfloat16_rn(static_cast<float>(centered) / 512.0f);
}

// Return total BF16 slots in the paged K/V cache layout.
int64_t cache_elements(const Gemma4KvCacheConfig &config) {
  return int64_t(config.num_layers) * config.num_pages * config.page_size *
         config.num_heads * config.head_dim;
}

// Build the one-page cache configs used by tiny attention tests.
Gemma4KvCacheConfig single_page_cache_config(int heads,
                                             int head_dim,
                                             int window_size) {
  return {1, 1, 1, 1, heads, head_dim, window_size};
}

// Return the row-major packed-token offset used by CPU references.
int64_t token_offset(int batch,
                     int position,
                     int head,
                     int dim,
                     int max_seq,
                     int heads,
                     int head_dim) {
  return (((int64_t)batch * max_seq + position) * heads + head) * head_dim +
         dim;
}

template <typename T>
void copy_to_device(thrust::device_vector<T> &dst, const std::vector<T> &src) {
  CHECK_CUDA(cudaMemcpy(raw_ptr(dst), src.data(), src.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
}

template <typename T>
std::vector<T> copy_to_host(const thrust::device_vector<T> &src) {
  std::vector<T> dst(src.size());
  CHECK_CUDA(cudaMemcpy(dst.data(), raw_ptr(src), dst.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  return dst;
}

template <typename T>
void memset_device(thrust::device_vector<T> &dst, int value) {
  CHECK_CUDA(cudaMemset(raw_ptr(dst), value, dst.size() * sizeof(T)));
}

// Compare BF16 vectors with an absolute tolerance.
void compare_bf16(const std::vector<__nv_bfloat16> &actual,
                  const std::vector<__nv_bfloat16> &expected,
                  float tolerance,
                  const char *label) {
  float max_abs = 0.0f;
  int max_index = 0;
  for (int i = 0; i < static_cast<int>(actual.size()); ++i) {
    float diff = std::fabs(bf16_to_float(actual[i]) -
                           bf16_to_float(expected[i]));
    if (diff > max_abs) {
      max_abs = diff;
      max_index = i;
    }
  }
  if (max_abs > tolerance) {
    std::fprintf(stderr,
                 "%s max_abs=%g index=%d actual=%g expected=%g tolerance=%g\n",
                 label, max_abs, max_index, bf16_to_float(actual[max_index]),
                 bf16_to_float(expected[max_index]), tolerance);
    std::exit(1);
  }
}

// Compare float vectors with an absolute tolerance.
void compare_float(const std::vector<float> &actual,
                   const std::vector<float> &expected,
                   float tolerance,
                   const char *label) {
  float max_abs = 0.0f;
  int max_index = 0;
  for (int i = 0; i < static_cast<int>(actual.size()); ++i) {
    float diff = std::fabs(actual[i] - expected[i]);
    if (diff > max_abs) {
      max_abs = diff;
      max_index = i;
    }
  }
  if (max_abs > tolerance) {
    std::fprintf(stderr,
                 "%s max_abs=%g index=%d actual=%g expected=%g tolerance=%g\n",
                 label, max_abs, max_index, actual[max_index],
                 expected[max_index], tolerance);
    std::exit(1);
  }
}

// Compare a full BF16 device buffer after copying it back to host memory.
void compare_device_bf16(const thrust::device_vector<__nv_bfloat16> &actual,
                         const std::vector<__nv_bfloat16> &expected,
                         float tolerance,
                         const char *label) {
  compare_bf16(copy_to_host(actual), expected, tolerance, label);
}

// Return the scalar RMSNorm multiplier for one row.
float reference_rms_scale(float sum_sq, int width) {
  return 1.0f / std::sqrt(sum_sq / float(width) + GEMMA4_RMS_NORM_EPS);
}

// Reproduce hidden RMSNorm for the sparse fixture where only element 0 is nonzero.
__nv_bfloat16 reference_sparse_hidden_rmsnorm_scalar(float x0, float weight0) {
  float scale = reference_rms_scale(x0 * x0, GEMMA4_HIDDEN_SIZE);
  return __float2bfloat16_rn(x0 * scale * weight0);
}

// Reproduce the one-term BF16 projection rounding used by the CUDA path.
float reference_project_scalar(__nv_bfloat16 x, float weight) {
  __nv_bfloat16 bf16_weight = __float2bfloat16_rn(weight);
  float product = bf16_to_float(x) * bf16_to_float(bf16_weight);
  return bf16_to_float(__float2bfloat16_rn(product));
}

// Reproduce per-head RMSNorm for a sparse head with only dim 0 nonzero.
__nv_bfloat16 reference_head_rms_scalar(float value, int head_dim) {
  float scale = reference_rms_scale(value * value, head_dim);
  return __float2bfloat16_rn(value * scale);
}

// Write one BF16 value into a large zeroed device matrix.
void write_device_bf16(thrust::device_vector<__nv_bfloat16> &dst,
                       int64_t index,
                       float value) {
  __nv_bfloat16 bf16 = __float2bfloat16_rn(value);
  CHECK_CUDA(cudaMemcpy(raw_ptr(dst) + index, &bf16, sizeof(bf16),
                        cudaMemcpyHostToDevice));
}

// Compute the same causal GQA attention as the raw prefill kernels on CPU.
void reference_attention(std::vector<__nv_bfloat16> &expected,
                         const std::vector<__nv_bfloat16> &q,
                         const std::vector<__nv_bfloat16> &k,
                         const std::vector<__nv_bfloat16> &v,
                         int seq_len,
                         int q_heads,
                         int kv_heads,
                         int head_dim,
                         int window_size,
                         float softmax_scale,
                         std::vector<float> *expected_lse = nullptr) {
  const int group = q_heads / kv_heads;
  std::vector<float> scores(seq_len);
  for (int row = 0; row < seq_len; ++row) {
    const int left = window_size > 0 ? std::max(0, row - window_size + 1) : 0;
    for (int qh = 0; qh < q_heads; ++qh) {
      const int kvh = qh / group;
      float max_score = -INFINITY;
      for (int col = left; col <= row; ++col) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
          const int64_t q_idx = token_offset(0, row, qh, d, seq_len, q_heads,
                                             head_dim);
          const int64_t k_idx = token_offset(0, col, kvh, d, seq_len, kv_heads,
                                             head_dim);
          score += bf16_to_float(q[q_idx]) * bf16_to_float(k[k_idx]);
        }
        scores[col] = score * softmax_scale;
        max_score = std::max(max_score, scores[col]);
      }

      float denom = 0.0f;
      for (int col = left; col <= row; ++col) {
        scores[col] = std::exp(scores[col] - max_score);
        denom += scores[col];
      }
      // LSE is the scaled row max plus the log of the shifted denominator.
      if (expected_lse != nullptr) {
        (*expected_lse)[qh * seq_len + row] = max_score + std::log(denom);
      }

      for (int d = 0; d < head_dim; ++d) {
        float out = 0.0f;
        for (int col = left; col <= row; ++col) {
          const int64_t v_idx = token_offset(0, col, kvh, d, seq_len, kv_heads,
                                             head_dim);
          out += (scores[col] / denom) * bf16_to_float(v[v_idx]);
        }
        const int64_t out_idx = token_offset(0, row, qh, d, seq_len, q_heads,
                                             head_dim);
        expected[out_idx] = __float2bfloat16_rn(out);
      }
    }
  }
}

// Compare raw sliding prefill with a CPU reference for one sequence/window shape.
void run_sliding_prefill_reference_case(int seq_len,
                                        int window_size,
                                        const char *label) {
  const int q_count = seq_len * GEMMA4_SLIDING_Q_PROJ_SIZE;
  const int kv_count = seq_len * GEMMA4_SLIDING_KV_PROJ_SIZE;
  std::vector<__nv_bfloat16> q(q_count);
  std::vector<__nv_bfloat16> k(kv_count);
  std::vector<__nv_bfloat16> v(kv_count);
  for (int i = 0; i < q_count; ++i) q[i] = make_prefill_value(1000 + i);
  for (int i = 0; i < kv_count; ++i) {
    k[i] = make_prefill_value(2000 + i);
    v[i] = make_prefill_value(3000 + i);
  }

  thrust::device_vector<__nv_bfloat16> d_q(q.size());
  thrust::device_vector<__nv_bfloat16> d_k(k.size());
  thrust::device_vector<__nv_bfloat16> d_v(v.size());
  thrust::device_vector<__nv_bfloat16> d_out(q_count);
  thrust::device_vector<float> d_lse(GEMMA4_NUM_QUERY_HEADS * seq_len);
  copy_to_device(d_q, q);
  copy_to_device(d_k, k);
  copy_to_device(d_v, v);

  const float scale = 1.0f / std::sqrt(float(GEMMA4_SLIDING_HEAD_DIM));
  CHECK_CUDA(gemma4_flash_attention_sliding_fwd_bf16(
      raw_ptr(d_out), raw_ptr(d_lse), raw_ptr(d_q), raw_ptr(d_k), raw_ptr(d_v), 1, seq_len,
      seq_len, window_size, scale, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> expected(q_count);
  std::vector<float> expected_lse(GEMMA4_NUM_QUERY_HEADS * seq_len);
  reference_attention(expected, q, k, v, seq_len, GEMMA4_NUM_QUERY_HEADS,
                      GEMMA4_SLIDING_KV_HEADS, GEMMA4_SLIDING_HEAD_DIM,
                      window_size, scale, &expected_lse);
  compare_device_bf16(d_out, expected, 0.03125f, label);
  compare_float(copy_to_host(d_lse), expected_lse, 0.03125f,
                "sliding prefill LSE");
}

// Compare global D=512 prefill with norm/RoPE prep against a CPU reference.
void run_global_prefill_reference_case() {
  constexpr int batch_size = 1;
  constexpr int seq_len = 33;
  constexpr int q_count = seq_len * GEMMA4_GLOBAL_Q_PROJ_SIZE;
  constexpr int kv_count = seq_len * GEMMA4_GLOBAL_K_PROJ_SIZE;
  constexpr int rotary_half = GEMMA4_GLOBAL_HEAD_DIM / 8;

  std::vector<__nv_bfloat16> q(q_count);
  std::vector<__nv_bfloat16> k(kv_count);
  std::vector<__nv_bfloat16> norm_weight(
      GEMMA4_GLOBAL_HEAD_DIM, __float2bfloat16_rn(1.0f));
  std::vector<float> cos(seq_len * rotary_half, 1.0f);
  std::vector<float> sin(seq_len * rotary_half, 0.0f);
  std::vector<int32_t> token_position(seq_len);
  for (int i = 0; i < q_count; ++i) q[i] = make_prefill_value(4000 + i);
  for (int i = 0; i < kv_count; ++i) k[i] = make_prefill_value(5000 + i);
  for (int i = 0; i < seq_len; ++i) token_position[i] = i;

  thrust::device_vector<__nv_bfloat16> d_q(q.size());
  thrust::device_vector<__nv_bfloat16> d_k(k.size());
  thrust::device_vector<__nv_bfloat16> d_q_prepared(q_count);
  thrust::device_vector<__nv_bfloat16> d_k_prepared(kv_count);
  thrust::device_vector<__nv_bfloat16> d_v_prepared(kv_count);
  thrust::device_vector<__nv_bfloat16> d_out(q_count);
  thrust::device_vector<float> d_lse(GEMMA4_NUM_QUERY_HEADS * seq_len);
  thrust::device_vector<__nv_bfloat16> d_norm_weight(
      norm_weight.size());
  thrust::device_vector<float> d_cos(cos.size());
  thrust::device_vector<float> d_sin(sin.size());
  thrust::device_vector<int32_t> d_token_position(
      token_position.size());
  copy_to_device(d_q, q);
  copy_to_device(d_k, k);
  copy_to_device(d_norm_weight, norm_weight);
  copy_to_device(d_cos, cos);
  copy_to_device(d_sin, sin);
  copy_to_device(d_token_position, token_position);

  const float scale = 1.0f / std::sqrt(float(GEMMA4_GLOBAL_HEAD_DIM));
  CHECK_CUDA(gemma4_flash_attention_global_fwd_bf16_norm_rope(
      raw_ptr(d_out), raw_ptr(d_lse), raw_ptr(d_q_prepared), raw_ptr(d_k_prepared),
      raw_ptr(d_v_prepared), raw_ptr(d_q), raw_ptr(d_k), raw_ptr(d_norm_weight),
      raw_ptr(d_norm_weight), raw_ptr(d_cos), raw_ptr(d_sin), raw_ptr(d_token_position),
      batch_size, seq_len, seq_len, scale, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> expected(q_count);
  std::vector<__nv_bfloat16> q_prepared = copy_to_host(d_q_prepared);
  std::vector<__nv_bfloat16> k_prepared = copy_to_host(d_k_prepared);
  std::vector<__nv_bfloat16> v_prepared = copy_to_host(d_v_prepared);
  std::vector<float> expected_lse(GEMMA4_NUM_QUERY_HEADS * seq_len);
  reference_attention(expected, q_prepared, k_prepared, v_prepared, seq_len,
                      GEMMA4_NUM_QUERY_HEADS, GEMMA4_GLOBAL_KV_HEADS,
                      GEMMA4_GLOBAL_HEAD_DIM, 0, scale, &expected_lse);
  compare_device_bf16(d_out, expected, 0.03125f,
                      "global D512 prefill partial tile");
  compare_float(copy_to_host(d_lse), expected_lse, 0.03125f,
                "global D512 prefill LSE");
}

// Validate global paged decode with one live key and direct single-split output.
void run_global_flash_decode_case() {
  Gemma4KvCacheConfig config = single_page_cache_config(
      GEMMA4_GLOBAL_KV_HEADS, GEMMA4_GLOBAL_HEAD_DIM, 0);
  const int q_heads = GEMMA4_NUM_QUERY_HEADS;
  const int q_count = q_heads * config.head_dim;
  const int64_t cache_count = cache_elements(config);
  std::vector<__nv_bfloat16> cache_v(cache_count);
  const int64_t v_row = gemma4_kv_cache_offset(config, 0, 0, 0, 0, 0);
  for (int d = 0; d < config.head_dim; ++d) {
    cache_v[v_row + d] = make_value(51000 + d);
  }
  std::vector<int32_t> page_table = {0};
  std::vector<int32_t> seq_lengths = {1};

  thrust::device_vector<__nv_bfloat16> d_q(q_count);
  thrust::device_vector<__nv_bfloat16> d_out(q_count);
  thrust::device_vector<__nv_bfloat16> d_cache_k(cache_count);
  thrust::device_vector<__nv_bfloat16> d_cache_v(
      cache_v.size());
  thrust::device_vector<int32_t> d_page_table(
      page_table.size());
  thrust::device_vector<int32_t> d_seq_lengths(
      seq_lengths.size());
  copy_to_device(d_cache_v, cache_v);
  copy_to_device(d_page_table, page_table);
  copy_to_device(d_seq_lengths, seq_lengths);
  memset_device(d_q, 0);
  memset_device(d_cache_k, 0);
  memset_device(d_out, 0x5a);

  CHECK_CUDA(gemma4_flash_attention_decode_paged_bf16(
      raw_ptr(d_out), nullptr, nullptr, nullptr, raw_ptr(d_q), raw_ptr(d_cache_k),
      raw_ptr(d_cache_v), raw_ptr(d_page_table), raw_ptr(d_seq_lengths), config, 0, 1,
      1.0f, 1, 1, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> expected(q_count);
  for (int qh = 0; qh < q_heads; ++qh) {
    for (int d = 0; d < config.head_dim; ++d) {
      expected[qh * config.head_dim + d] = cache_v[v_row + d];
    }
  }
  compare_device_bf16(d_out, expected, 0.0f, "global flash decode");
}

// Check sliding folded decode ingress against a sparse CPU reference.
void run_sliding_norm_project_prepare_case() {
  constexpr int batch_size = 1;
  constexpr int cache_layer = 0;
  constexpr int head_dim = GEMMA4_SLIDING_HEAD_DIM;
  constexpr int q_cols = GEMMA4_SLIDING_Q_PROJ_SIZE;
  constexpr int kv_cols = GEMMA4_SLIDING_KV_PROJ_SIZE;
  constexpr int rotary_half = head_dim / 2;
  constexpr float q_weight = 0.5f;
  constexpr float k_weight = -0.25f;
  constexpr float v_weight = 0.75f;
  Gemma4KvCacheConfig config = single_page_cache_config(
      GEMMA4_SLIDING_KV_HEADS, head_dim, 1);

  std::vector<__nv_bfloat16> x(GEMMA4_HIDDEN_SIZE,
                               __float2bfloat16_rn(0.0f));
  std::vector<__nv_bfloat16> input_norm_weight(
      GEMMA4_HIDDEN_SIZE, __float2bfloat16_rn(1.0f));
  std::vector<__nv_bfloat16> head_norm_weight(
      head_dim, __float2bfloat16_rn(1.0f));
  std::vector<float> cos(rotary_half, 1.0f);
  std::vector<float> sin(rotary_half, 0.0f);
  std::vector<int32_t> page_table = {0};
  std::vector<int32_t> token_position = {0};
  x[0] = __float2bfloat16_rn(1.0f);
  input_norm_weight[0] = __float2bfloat16_rn(1.25f);

  __nv_bfloat16 normed_x0 = reference_sparse_hidden_rmsnorm_scalar(
      bf16_to_float(x[0]), bf16_to_float(input_norm_weight[0]));
  const size_t q_weight_count =
      static_cast<size_t>(q_cols) * GEMMA4_HIDDEN_SIZE;
  const size_t kv_weight_count =
      static_cast<size_t>(kv_cols) * GEMMA4_HIDDEN_SIZE;
  const int64_t cache_count = cache_elements(config);

  thrust::device_vector<__nv_bfloat16> d_x(x.size());
  thrust::device_vector<__nv_bfloat16> d_input_norm_weight(
      input_norm_weight.size());
  thrust::device_vector<__nv_bfloat16> d_head_norm(
      head_norm_weight.size());
  thrust::device_vector<float> d_cos(cos.size());
  thrust::device_vector<float> d_sin(sin.size());
  thrust::device_vector<int32_t> d_page_table(
      page_table.size());
  thrust::device_vector<int32_t> d_token_position(
      token_position.size());
  thrust::device_vector<__nv_bfloat16> d_w_q(q_weight_count);
  thrust::device_vector<__nv_bfloat16> d_w_k(kv_weight_count);
  thrust::device_vector<__nv_bfloat16> d_w_v(kv_weight_count);
  thrust::device_vector<__nv_bfloat16> d_q(q_cols);
  thrust::device_vector<__nv_bfloat16> d_cache_k(cache_count);
  thrust::device_vector<__nv_bfloat16> d_cache_v(cache_count);
  copy_to_device(d_x, x);
  copy_to_device(d_input_norm_weight, input_norm_weight);
  copy_to_device(d_head_norm, head_norm_weight);
  copy_to_device(d_cos, cos);
  copy_to_device(d_sin, sin);
  copy_to_device(d_page_table, page_table);
  copy_to_device(d_token_position, token_position);

  memset_device(d_w_q, 0);
  memset_device(d_w_k, 0);
  memset_device(d_w_v, 0);
  memset_device(d_q, 0);
  memset_device(d_cache_k, 0);
  memset_device(d_cache_v, 0);

  write_device_bf16(d_w_q, 0, q_weight);
  write_device_bf16(d_w_k, 0, k_weight);
  write_device_bf16(d_w_v, 0, v_weight);

  Gemma4AttentionProjectionWeights weights = {
      raw_ptr(d_w_q),
      raw_ptr(d_w_k),
      raw_ptr(d_w_v),
      0,
      0,
      0,
  };

  CHECK_CUDA(gemma4_flash_attention_decode_norm_project_prepare_paged_kv_bf16(
      raw_ptr(d_q), raw_ptr(d_cache_k), raw_ptr(d_cache_v), config,
      raw_ptr(d_page_table), raw_ptr(d_token_position), batch_size, cache_layer,
      raw_ptr(d_x), raw_ptr(d_input_norm_weight), weights, raw_ptr(d_head_norm),
      raw_ptr(d_head_norm), raw_ptr(d_cos), raw_ptr(d_sin), 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  float q_value = reference_project_scalar(normed_x0, q_weight);
  float k_value = reference_project_scalar(normed_x0, k_weight);
  float v_value = reference_project_scalar(normed_x0, v_weight);
  std::vector<__nv_bfloat16> expected_q(q_cols, __float2bfloat16_rn(0.0f));
  std::vector<__nv_bfloat16> expected_k(cache_count, __float2bfloat16_rn(0.0f));
  std::vector<__nv_bfloat16> expected_v(cache_count, __float2bfloat16_rn(0.0f));
  expected_q[0] = reference_head_rms_scalar(q_value, head_dim);
  expected_k[0] = reference_head_rms_scalar(k_value, head_dim);
  expected_v[0] = reference_head_rms_scalar(v_value, head_dim);

  compare_device_bf16(d_q, expected_q, 0.125f, "sliding norm-project Q");
  compare_device_bf16(d_cache_k, expected_k, 0.125f,
                      "sliding norm-project K");
  compare_device_bf16(d_cache_v, expected_v, 0.125f,
                      "sliding norm-project V");
}

// Check global prefill prepares K-derived V and runs full causal attention.
void run_global_prefill_norm_rope_case() {
  constexpr int batch_size = 1;
  constexpr int seq_len = 2;
  constexpr int rows = batch_size * seq_len;
  constexpr int q_count = rows * GEMMA4_GLOBAL_Q_PROJ_SIZE;
  constexpr int kv_count = rows * GEMMA4_GLOBAL_K_PROJ_SIZE;
  constexpr int rotary_half = GEMMA4_GLOBAL_HEAD_DIM / 8;
  constexpr int max_position = 4;

  std::vector<__nv_bfloat16> k(kv_count);
  for (int i = 0; i < kv_count; ++i) {
    k[i] = make_value(110000 + i);
  }

  std::vector<__nv_bfloat16> norm_weight(
      GEMMA4_GLOBAL_HEAD_DIM, __float2bfloat16_rn(1.0f));
  std::vector<int32_t> token_position = {2, 4};
  std::vector<float> cos((max_position + 1) * rotary_half, 1.0f);
  std::vector<float> sin(cos.size(), 0.0f);
  cos[token_position[0] * rotary_half] = 0.0f;
  sin[token_position[0] * rotary_half] = 1.0f;
  cos[token_position[1] * rotary_half] = 0.0f;
  sin[token_position[1] * rotary_half] = -1.0f;

  thrust::device_vector<__nv_bfloat16> d_q(q_count);
  thrust::device_vector<__nv_bfloat16> d_k(k.size());
  thrust::device_vector<__nv_bfloat16> d_q_prepared(q_count);
  thrust::device_vector<__nv_bfloat16> d_k_prepared(kv_count);
  thrust::device_vector<__nv_bfloat16> d_v_prepared(kv_count);
  thrust::device_vector<__nv_bfloat16> d_out(q_count);
  thrust::device_vector<__nv_bfloat16> d_norm_weight(
      norm_weight.size());
  thrust::device_vector<float> d_cos(cos.size());
  thrust::device_vector<float> d_sin(sin.size());
  thrust::device_vector<int32_t> d_token_position(
      token_position.size());
  copy_to_device(d_k, k);
  copy_to_device(d_norm_weight, norm_weight);
  copy_to_device(d_cos, cos);
  copy_to_device(d_sin, sin);
  copy_to_device(d_token_position, token_position);
  memset_device(d_q, 0);

  const float scale = 1.0f / std::sqrt(float(GEMMA4_GLOBAL_HEAD_DIM));
  CHECK_CUDA(gemma4_flash_attention_global_fwd_bf16_norm_rope(
      raw_ptr(d_out), nullptr, raw_ptr(d_q_prepared), raw_ptr(d_k_prepared),
      raw_ptr(d_v_prepared), raw_ptr(d_q), raw_ptr(d_k), raw_ptr(d_norm_weight),
      raw_ptr(d_norm_weight), raw_ptr(d_cos), raw_ptr(d_sin), raw_ptr(d_token_position),
      batch_size, seq_len, seq_len, scale, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> expected_v(kv_count);
  std::vector<float> row_inv_rms(rows);
  for (int row = 0; row < rows; ++row) {
    const int64_t row_base = int64_t(row) * GEMMA4_GLOBAL_HEAD_DIM;
    float sum_sq = 0.0f;
    for (int d = 0; d < GEMMA4_GLOBAL_HEAD_DIM; ++d) {
      float value = bf16_to_float(k[row_base + d]);
      sum_sq += value * value;
    }
    row_inv_rms[row] = reference_rms_scale(sum_sq, GEMMA4_GLOBAL_HEAD_DIM);
    for (int d = 0; d < GEMMA4_GLOBAL_HEAD_DIM; ++d) {
      float value = bf16_to_float(k[row_base + d]);
      expected_v[row_base + d] = __float2bfloat16_rn(value * row_inv_rms[row]);
    }
  }
  std::vector<__nv_bfloat16> v_prepared = copy_to_host(d_v_prepared);
  compare_bf16(v_prepared, expected_v, 0.00390625f,
               "global prefill K-derived V");

  // Check RoPE uses absolute token positions instead of local prefill rows.
  std::vector<__nv_bfloat16> k_prepared = copy_to_host(d_k_prepared);
  for (int row = 0; row < rows; ++row) {
    const int64_t row_base = int64_t(row) * GEMMA4_GLOBAL_HEAD_DIM;
    const int position = token_position[row];
    const float c = cos[position * rotary_half];
    const float s = sin[position * rotary_half];
    const float lo = bf16_to_float(k[row_base]) * row_inv_rms[row];
    const float hi =
        bf16_to_float(k[row_base + rotary_half]) * row_inv_rms[row];
    const float expected_lo =
        bf16_to_float(__float2bfloat16_rn(lo * c - hi * s));
    const float expected_hi =
        bf16_to_float(__float2bfloat16_rn(lo * s + hi * c));
    const float actual_lo = bf16_to_float(k_prepared[row_base]);
    const float actual_hi =
        bf16_to_float(k_prepared[row_base + rotary_half]);
    if (std::fabs(actual_lo - expected_lo) > 0.00390625f ||
        std::fabs(actual_hi - expected_hi) > 0.00390625f) {
      std::fprintf(stderr, "global prefill K RoPE used wrong position at row %d\n",
                   row);
      std::exit(1);
    }
  }

  std::vector<__nv_bfloat16> expected_out(q_count);
  auto kv_offset = [seq_len](int position, int dim) {
    return token_offset(0, position, 0, dim, seq_len, GEMMA4_GLOBAL_KV_HEADS,
                        GEMMA4_GLOBAL_HEAD_DIM);
  };
  auto q_offset = [seq_len](int position, int head, int dim) {
    return token_offset(0, position, head, dim, seq_len, GEMMA4_NUM_QUERY_HEADS,
                        GEMMA4_GLOBAL_HEAD_DIM);
  };
  for (int qh = 0; qh < GEMMA4_NUM_QUERY_HEADS; ++qh) {
    for (int d = 0; d < GEMMA4_GLOBAL_HEAD_DIM; ++d) {
      float v0 = bf16_to_float(v_prepared[kv_offset(0, d)]);
      float v1 = bf16_to_float(v_prepared[kv_offset(1, d)]);
      expected_out[q_offset(0, qh, d)] = __float2bfloat16_rn(v0);
      expected_out[q_offset(1, qh, d)] = __float2bfloat16_rn(0.5f * (v0 + v1));
    }
  }
  compare_device_bf16(d_out, expected_out, 0.03125f,
                      "global prefill attention");
}

}  // namespace

int main() {
  run_global_flash_decode_case();
  run_sliding_norm_project_prepare_case();
  run_global_prefill_norm_rope_case();
  run_sliding_prefill_reference_case(
      5, 5, "sliding prefill nonzero causal");
  run_sliding_prefill_reference_case(
      5, 2, "sliding prefill local window");
  run_sliding_prefill_reference_case(
      65, 65, "sliding prefill partial tile");
  run_global_prefill_reference_case();
  std::puts("flash attention tests passed");
  return 0;
}
