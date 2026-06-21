#include "gemma4_kv_cache.cuh"
#include "gemma4_flash_attention.cuh"
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

// Abort the test at the first CUDA error so the failing call site is explicit.
void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

// Convert BF16 test values back to float for comparisons.
float bf16_to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

// Generate deterministic nontrivial BF16 values without storing fixtures.
__nv_bfloat16 make_value(int seed) {
  int centered = ((seed * 37 + 17) % 257) - 128;
  return __float2bfloat16_rn(static_cast<float>(centered) / 64.0f);
}

// Return total BF16 slots in the paged K/V cache layout.
int64_t cache_elements(const Gemma4KvCacheConfig &config) {
  return int64_t(config.num_layers) * config.num_pages * config.page_size *
         config.num_heads * config.head_dim;
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

// Own one device allocation for short CUDA tests.
template <typename T>
struct DeviceBuffer {
  explicit DeviceBuffer(size_t count_) : count(count_) {
    CHECK_CUDA(cudaMalloc(&ptr, count * sizeof(T)));
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  ~DeviceBuffer() {
    if (ptr != nullptr) cudaFree(ptr);
  }

  T *get() const { return ptr; }

  size_t count = 0;
  T *ptr = nullptr;
};

// Copy one host vector into an equally sized device buffer.
template <typename T>
void copy_to_device(const DeviceBuffer<T> &dst, const std::vector<T> &src) {
  CHECK_CUDA(cudaMemcpy(dst.get(), src.data(), src.size() * sizeof(T),
                        cudaMemcpyHostToDevice));
}

// Copy one full device buffer back into a host vector.
template <typename T>
std::vector<T> copy_to_host(const DeviceBuffer<T> &src) {
  std::vector<T> dst(src.count);
  CHECK_CUDA(cudaMemcpy(dst.data(), src.get(), dst.size() * sizeof(T),
                        cudaMemcpyDeviceToHost));
  return dst;
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

// Materialize one CPU RMSNorm row for parity against fused decode ingress.
std::vector<__nv_bfloat16> reference_hidden_rmsnorm(
    const std::vector<__nv_bfloat16> &x,
    const std::vector<__nv_bfloat16> &weight) {
  float sum_sq = 0.0f;
  for (int i = 0; i < GEMMA4_HIDDEN_SIZE; ++i) {
    float value = bf16_to_float(x[i]);
    sum_sq += value * value;
  }

  float scale =
      1.0f / std::sqrt(sum_sq / float(GEMMA4_HIDDEN_SIZE) +
                       GEMMA4_RMS_NORM_EPS);
  std::vector<__nv_bfloat16> out(GEMMA4_HIDDEN_SIZE);
  for (int i = 0; i < GEMMA4_HIDDEN_SIZE; ++i) {
    float value = bf16_to_float(x[i]);
    float gamma = bf16_to_float(weight[i]);
    out[i] = __float2bfloat16_rn(value * scale * gamma);
  }
  return out;
}

// Reproduce the one-term BF16 projection rounding used by the CUDA path.
float reference_project_scalar(__nv_bfloat16 x, float weight) {
  __nv_bfloat16 bf16_weight = __float2bfloat16_rn(weight);
  float product = bf16_to_float(x) * bf16_to_float(bf16_weight);
  return bf16_to_float(__float2bfloat16_rn(product));
}

// Reproduce per-head RMSNorm for a sparse head with only dim 0 nonzero.
__nv_bfloat16 reference_head_rms_scalar(float value, int head_dim) {
  float scale =
      1.0f / std::sqrt((value * value) / float(head_dim) +
                       GEMMA4_RMS_NORM_EPS);
  return __float2bfloat16_rn(value * scale);
}

// Write one BF16 value into a large zeroed device matrix.
void write_device_bf16(const DeviceBuffer<__nv_bfloat16> &dst,
                       int64_t index,
                       float value) {
  __nv_bfloat16 bf16 = __float2bfloat16_rn(value);
  CHECK_CUDA(cudaMemcpy(dst.get() + index, &bf16, sizeof(bf16),
                        cudaMemcpyHostToDevice));
}

// Compute a small CPU reference for paged decode attention.
void reference_decode_attention(std::vector<__nv_bfloat16> &out,
                                const std::vector<__nv_bfloat16> &q,
                                const std::vector<__nv_bfloat16> &k_by_pos,
                                const std::vector<__nv_bfloat16> &v_by_pos,
                                const std::vector<int32_t> &seq_lengths,
                                const Gemma4KvCacheConfig &config,
                                int batch_size,
                                int q_heads,
                                int max_seq,
                                float scale) {
  int group_size = q_heads / config.num_heads;
  for (int b = 0; b < batch_size; ++b) {
    int first_key = config.window_size > 0
                        ? std::max(0, seq_lengths[b] - config.window_size)
                        : 0;
    for (int qh = 0; qh < q_heads; ++qh) {
      int kh = qh / group_size;
      std::vector<float> scores(seq_lengths[b] - first_key);
      float max_score = -INFINITY;
      for (int pos = first_key; pos < seq_lengths[b]; ++pos) {
        float dot = 0.0f;
        for (int d = 0; d < config.head_dim; ++d) {
          float qv =
              bf16_to_float(q[((int64_t)b * q_heads + qh) * config.head_dim + d]);
          float kv = bf16_to_float(k_by_pos[token_offset(
              b, pos, kh, d, max_seq, config.num_heads, config.head_dim)]);
          dot += qv * kv;
        }
        float score = dot * scale;
        scores[pos - first_key] = score;
        max_score = std::max(max_score, score);
      }

      float denom = 0.0f;
      for (float score : scores) {
        denom += std::exp(score - max_score);
      }
      for (int d = 0; d < config.head_dim; ++d) {
        float value = 0.0f;
        for (int pos = first_key; pos < seq_lengths[b]; ++pos) {
          float weight = std::exp(scores[pos - first_key] - max_score) / denom;
          float vv = bf16_to_float(v_by_pos[token_offset(
              b, pos, kh, d, max_seq, config.num_heads, config.head_dim)]);
          value += weight * vv;
        }
        out[((int64_t)b * q_heads + qh) * config.head_dim + d] =
            __float2bfloat16_rn(value);
      }
    }
  }
}

// Check CUTE cache layout math and stale sliding-page protection.
void run_layout_and_slot_guard_case() {
  Gemma4KvCacheConfig config = {2, 7, 4, 5, 3, 8, 0};
  auto cache_layout = gemma4_kv_cache_layout(config);
  int64_t got = cache_layout(1, 2, 3, 1, 5);
  int64_t expected = (((((int64_t)1 * 7 + 2) * 4 + 3) * 3 + 1) * 8 + 5);
  if (got != expected) {
    std::fprintf(stderr, "cache layout got=%lld expected=%lld\n",
                 static_cast<long long>(got),
                 static_cast<long long>(expected));
    std::exit(1);
  }

  Gemma4KvCacheConfig sliding = {1, 6, 4, 3, 2, 16, 8};
  std::vector<int32_t> page_table(sliding.max_pages_per_seq, -1);
  std::vector<int32_t> slot_logical_pages(sliding.max_pages_per_seq, -1);
  Gemma4KvPageAllocator allocator(sliding.num_pages);
  int first_page = gemma4_kv_cache_ensure_page(
      page_table, slot_logical_pages, allocator, sliding, 1, 0, 0);
  int wrapped_page = gemma4_kv_cache_ensure_page(
      page_table, slot_logical_pages, allocator, sliding, 1, 0, 12);
  int stale_page = gemma4_kv_cache_ensure_page(
      page_table, slot_logical_pages, allocator, sliding, 1, 0, 0);
  if (first_page < 0 || wrapped_page != first_page || stale_page >= 0) {
    std::fprintf(stderr, "sliding stale slot guard failed\n");
    std::exit(1);
  }
}

// Exercise invalid positive physical pages for the vector writer.
void run_invalid_page_write_case() {
  Gemma4KvCacheConfig config = {1, 1, 4, 2, 2, 16, 0};
  std::vector<int32_t> page_table(config.max_pages_per_seq, -1);
  std::vector<int32_t> token_batch = {0};
  std::vector<int32_t> token_position = {0};
  page_table[0] = config.num_pages;

  std::vector<__nv_bfloat16> k(config.num_heads * config.head_dim);
  std::vector<__nv_bfloat16> v(k.size());
  for (int i = 0; i < static_cast<int>(k.size()); ++i) {
    k[i] = make_value(43000 + i);
    v[i] = make_value(44000 + i);
  }

  DeviceBuffer<__nv_bfloat16> d_cache_k(cache_elements(config));
  DeviceBuffer<__nv_bfloat16> d_cache_v(cache_elements(config));
  DeviceBuffer<__nv_bfloat16> d_k(k.size());
  DeviceBuffer<__nv_bfloat16> d_v(v.size());
  DeviceBuffer<int32_t> d_page_table(page_table.size());
  DeviceBuffer<int32_t> d_token_batch(token_batch.size());
  DeviceBuffer<int32_t> d_token_position(token_position.size());
  CHECK_CUDA(cudaMemset(d_cache_k.get(), 0,
                        cache_elements(config) * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_cache_v.get(), 0,
                        cache_elements(config) * sizeof(__nv_bfloat16)));
  copy_to_device(d_k, k);
  copy_to_device(d_v, v);
  copy_to_device(d_page_table, page_table);
  copy_to_device(d_token_batch, token_batch);
  copy_to_device(d_token_position, token_position);

  CHECK_CUDA(gemma4_kv_cache_write_bf16(
      d_cache_k.get(), d_cache_v.get(), config, d_page_table.get(),
      d_token_batch.get(), d_token_position.get(), 1, 0, d_k.get(), d_v.get(), 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> zero(cache_elements(config),
                                  __float2bfloat16_rn(0.0f));
  compare_bf16(copy_to_host(d_cache_k), zero, 0.0f, "invalid page cache K");
  compare_bf16(copy_to_host(d_cache_v), zero, 0.0f, "invalid page cache V");
}

// Check invalid positive pages are ignored by generic paged decode too.
void run_invalid_page_decode_case() {
  Gemma4KvCacheConfig config = {1, 1, 4, 2, 2, 16, 0};
  int batch_size = 1;
  int q_heads = 4;
  int num_splits = 1;
  std::vector<int32_t> page_table(config.max_pages_per_seq, -1);
  std::vector<int32_t> seq_lengths = {1};
  std::vector<__nv_bfloat16> q(q_heads * config.head_dim);
  page_table[0] = config.num_pages;
  for (int i = 0; i < static_cast<int>(q.size()); ++i) {
    q[i] = make_value(45000 + i);
  }

  DeviceBuffer<__nv_bfloat16> d_cache_k(cache_elements(config));
  DeviceBuffer<__nv_bfloat16> d_cache_v(cache_elements(config));
  DeviceBuffer<__nv_bfloat16> d_q(q.size());
  DeviceBuffer<__nv_bfloat16> d_out(q.size());
  DeviceBuffer<float> d_partial_m(
      gemma4_paged_decode_partial_m_elements(batch_size, q_heads, num_splits));
  DeviceBuffer<float> d_partial_l(d_partial_m.count);
  DeviceBuffer<float> d_partial_acc(gemma4_paged_decode_partial_acc_elements(
      batch_size, q_heads, num_splits, config.head_dim));
  DeviceBuffer<int32_t> d_page_table(page_table.size());
  DeviceBuffer<int32_t> d_seq_lengths(seq_lengths.size());
  CHECK_CUDA(cudaMemset(d_cache_k.get(), 0,
                        cache_elements(config) * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_cache_v.get(), 0,
                        cache_elements(config) * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_out.get(), 0x5a, q.size() * sizeof(__nv_bfloat16)));
  copy_to_device(d_q, q);
  copy_to_device(d_page_table, page_table);
  copy_to_device(d_seq_lengths, seq_lengths);

  CHECK_CUDA(gemma4_paged_decode_attention_bf16(
      d_out.get(), d_partial_m.get(), d_partial_l.get(), d_partial_acc.get(),
      d_q.get(), d_cache_k.get(), d_cache_v.get(), d_page_table.get(),
      d_seq_lengths.get(), config, 0, batch_size, q_heads, 0.25f, 1,
      num_splits, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> zero(q.size(), __float2bfloat16_rn(0.0f));
  compare_bf16(copy_to_host(d_out), zero, 0.0f, "invalid page decode");
}

// Validate global cache write plus generic paged decode against a CPU reference.
void run_global_write_decode_case() {
  Gemma4KvCacheConfig config = {2, 16, 4, 8, 2, 16, 0};
  int batch_size = 2;
  int q_heads = 4;
  int max_seq = 10;
  int layer = 1;
  int num_splits = 4;
  std::vector<int32_t> seq_lengths = {7, 10};
  std::vector<int32_t> page_table(batch_size * config.max_pages_per_seq, -1);
  page_table[0] = 3;
  page_table[1] = 1;
  page_table[config.max_pages_per_seq + 0] = 6;
  page_table[config.max_pages_per_seq + 1] = 5;
  page_table[config.max_pages_per_seq + 2] = 8;

  int token_count = seq_lengths[0] + seq_lengths[1];
  std::vector<int32_t> token_batch(token_count);
  std::vector<int32_t> token_position(token_count);
  std::vector<__nv_bfloat16> flat_k(token_count * config.num_heads *
                                    config.head_dim);
  std::vector<__nv_bfloat16> flat_v(flat_k.size());
  std::vector<__nv_bfloat16> by_pos_k(batch_size * max_seq *
                                      config.num_heads * config.head_dim);
  std::vector<__nv_bfloat16> by_pos_v(by_pos_k.size());
  int token = 0;
  for (int b = 0; b < batch_size; ++b) {
    for (int pos = 0; pos < seq_lengths[b]; ++pos, ++token) {
      token_batch[token] = b;
      token_position[token] = pos;
      for (int h = 0; h < config.num_heads; ++h) {
        for (int d = 0; d < config.head_dim; ++d) {
          __nv_bfloat16 kv = make_value(1000 * b + 100 * pos + 17 * h + d);
          __nv_bfloat16 vv = make_value(3000 + 1000 * b + 100 * pos +
                                        17 * h + d);
          flat_k[(token * config.num_heads + h) * config.head_dim + d] = kv;
          flat_v[(token * config.num_heads + h) * config.head_dim + d] = vv;
          by_pos_k[token_offset(b, pos, h, d, max_seq, config.num_heads,
                                config.head_dim)] = kv;
          by_pos_v[token_offset(b, pos, h, d, max_seq, config.num_heads,
                                config.head_dim)] = vv;
        }
      }
    }
  }

  std::vector<__nv_bfloat16> q(batch_size * q_heads * config.head_dim);
  for (int i = 0; i < static_cast<int>(q.size()); ++i) {
    q[i] = make_value(5000 + i);
  }
  std::vector<__nv_bfloat16> expected(q.size());
  reference_decode_attention(expected, q, by_pos_k, by_pos_v, seq_lengths,
                             config, batch_size, q_heads, max_seq, 0.25f);

  DeviceBuffer<__nv_bfloat16> d_cache_k(cache_elements(config));
  DeviceBuffer<__nv_bfloat16> d_cache_v(cache_elements(config));
  DeviceBuffer<__nv_bfloat16> d_k(flat_k.size());
  DeviceBuffer<__nv_bfloat16> d_v(flat_v.size());
  DeviceBuffer<__nv_bfloat16> d_q(q.size());
  DeviceBuffer<__nv_bfloat16> d_out(q.size());
  DeviceBuffer<float> d_partial_m(
      gemma4_paged_decode_partial_m_elements(batch_size, q_heads, num_splits));
  DeviceBuffer<float> d_partial_l(d_partial_m.count);
  DeviceBuffer<float> d_partial_acc(gemma4_paged_decode_partial_acc_elements(
      batch_size, q_heads, num_splits, config.head_dim));
  DeviceBuffer<int32_t> d_page_table(page_table.size());
  DeviceBuffer<int32_t> d_token_batch(token_batch.size());
  DeviceBuffer<int32_t> d_token_position(token_position.size());
  DeviceBuffer<int32_t> d_seq_lengths(seq_lengths.size());
  CHECK_CUDA(cudaMemset(d_cache_k.get(), 0,
                        cache_elements(config) * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_cache_v.get(), 0,
                        cache_elements(config) * sizeof(__nv_bfloat16)));
  copy_to_device(d_k, flat_k);
  copy_to_device(d_v, flat_v);
  copy_to_device(d_q, q);
  copy_to_device(d_page_table, page_table);
  copy_to_device(d_token_batch, token_batch);
  copy_to_device(d_token_position, token_position);
  copy_to_device(d_seq_lengths, seq_lengths);

  CHECK_CUDA(gemma4_kv_cache_write_bf16(
      d_cache_k.get(), d_cache_v.get(), config, d_page_table.get(),
      d_token_batch.get(), d_token_position.get(), token_count, layer, d_k.get(),
      d_v.get(), 0));
  CHECK_CUDA(gemma4_paged_decode_attention_bf16(
      d_out.get(), d_partial_m.get(), d_partial_l.get(), d_partial_acc.get(),
      d_q.get(), d_cache_k.get(), d_cache_v.get(), d_page_table.get(),
      d_seq_lengths.get(), config, layer, batch_size, q_heads, 0.25f, 3,
      num_splits, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  compare_bf16(copy_to_host(d_out), expected, 0.015625f,
               "global paged attention");
}

// Validate sliding FlashAttention decode over varlen, wrap, and extra splits.
void run_sliding_flash_decode_case() {
  Gemma4KvCacheConfig config = {
      1,
      6,
      4,
      3,
      GEMMA4_SLIDING_KV_HEADS,
      GEMMA4_SLIDING_HEAD_DIM,
      8,
  };
  int batch_size = 2;
  int q_heads = GEMMA4_NUM_QUERY_HEADS;
  int max_seq = 10;
  int layer = 0;
  int split_size = 3;
  int num_splits = 4;
  float scale = 1.0f / std::sqrt(float(config.head_dim));
  std::vector<int32_t> target_seq_lengths = {10, 6};
  std::vector<int32_t> page_table(batch_size * config.max_pages_per_seq, -1);
  std::vector<int32_t> slot_logical_pages(page_table.size(), -1);
  std::vector<int32_t> seq_lengths(batch_size, 0);
  Gemma4KvPageAllocator allocator(config.num_pages);

  DeviceBuffer<__nv_bfloat16> d_cache_k(cache_elements(config));
  DeviceBuffer<__nv_bfloat16> d_cache_v(cache_elements(config));
  DeviceBuffer<__nv_bfloat16> d_one_k(config.num_heads * config.head_dim);
  DeviceBuffer<__nv_bfloat16> d_one_v(config.num_heads * config.head_dim);
  DeviceBuffer<int32_t> d_page_table(page_table.size());
  DeviceBuffer<int32_t> d_token_batch(1);
  DeviceBuffer<int32_t> d_token_position(1);
  CHECK_CUDA(cudaMemset(d_cache_k.get(), 0,
                        cache_elements(config) * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_cache_v.get(), 0,
                        cache_elements(config) * sizeof(__nv_bfloat16)));

  std::vector<__nv_bfloat16> by_pos_k(batch_size * max_seq *
                                      config.num_heads * config.head_dim);
  std::vector<__nv_bfloat16> by_pos_v(by_pos_k.size());
  std::vector<__nv_bfloat16> one_k(config.num_heads * config.head_dim);
  std::vector<__nv_bfloat16> one_v(one_k.size());
  for (int b = 0; b < batch_size; ++b) {
    for (int pos = 0; pos < target_seq_lengths[b]; ++pos) {
      int ensured_page = gemma4_kv_cache_ensure_page(
          page_table, slot_logical_pages, allocator, config, batch_size, b, pos);
      if (ensured_page < 0) {
        std::fprintf(stderr, "sliding page allocation failed\n");
        std::exit(1);
      }
      seq_lengths[b] = pos + 1;
      for (int h = 0; h < config.num_heads; ++h) {
        for (int d = 0; d < config.head_dim; ++d) {
          __nv_bfloat16 kv = make_value(17000 + 1000 * b + 101 * pos + 17 * h + d);
          __nv_bfloat16 vv = make_value(27000 + 1000 * b + 101 * pos + 17 * h + d);
          one_k[h * config.head_dim + d] = kv;
          one_v[h * config.head_dim + d] = vv;
          by_pos_k[token_offset(b, pos, h, d, max_seq, config.num_heads,
                                config.head_dim)] = kv;
          by_pos_v[token_offset(b, pos, h, d, max_seq, config.num_heads,
                                config.head_dim)] = vv;
        }
      }

      std::vector<int32_t> one_batch = {b};
      std::vector<int32_t> one_position = {pos};
      copy_to_device(d_page_table, page_table);
      copy_to_device(d_token_batch, one_batch);
      copy_to_device(d_token_position, one_position);
      copy_to_device(d_one_k, one_k);
      copy_to_device(d_one_v, one_v);
      CHECK_CUDA(gemma4_kv_cache_write_bf16(
          d_cache_k.get(), d_cache_v.get(), config, d_page_table.get(),
          d_token_batch.get(), d_token_position.get(), 1, layer, d_one_k.get(),
          d_one_v.get(), 0));
    }
  }

  std::vector<__nv_bfloat16> q(batch_size * q_heads * config.head_dim);
  for (int i = 0; i < static_cast<int>(q.size()); ++i) {
    q[i] = make_value(37000 + i);
  }
  std::vector<__nv_bfloat16> expected(q.size());
  reference_decode_attention(expected, q, by_pos_k, by_pos_v, seq_lengths,
                             config, batch_size, q_heads, max_seq, scale);

  DeviceBuffer<__nv_bfloat16> d_q(q.size());
  DeviceBuffer<__nv_bfloat16> d_out(q.size());
  DeviceBuffer<float> d_partial_m(
      gemma4_paged_decode_partial_m_elements(batch_size, q_heads, num_splits));
  DeviceBuffer<float> d_partial_l(d_partial_m.count);
  DeviceBuffer<float> d_partial_acc(gemma4_paged_decode_partial_acc_elements(
      batch_size, q_heads, num_splits, config.head_dim));
  DeviceBuffer<int32_t> d_seq_lengths(seq_lengths.size());
  copy_to_device(d_q, q);
  copy_to_device(d_page_table, page_table);
  copy_to_device(d_seq_lengths, seq_lengths);
  CHECK_CUDA(cudaMemset(d_partial_m.get(), 0x7f, d_partial_m.count * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_partial_l.get(), 0x7f, d_partial_l.count * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_partial_acc.get(), 0x7f,
                        d_partial_acc.count * sizeof(float)));

  CHECK_CUDA(gemma4_flash_attention_sliding_decode_paged_bf16(
      d_out.get(), d_partial_m.get(), d_partial_l.get(), d_partial_acc.get(),
      d_q.get(), d_cache_k.get(), d_cache_v.get(), d_page_table.get(),
      d_seq_lengths.get(), config, layer, batch_size, scale, split_size,
      num_splits, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  compare_bf16(copy_to_host(d_out), expected, 0.015625f,
               "sliding flash decode");
}

// Check the folded decode ingress against a sparse CPU reference.
void run_norm_project_prepare_case(bool global) {
  constexpr int batch_size = 1;
  constexpr int cache_layer = 0;
  const int head_dim =
      global ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_SLIDING_HEAD_DIM;
  const int kv_heads =
      global ? GEMMA4_GLOBAL_KV_HEADS : GEMMA4_SLIDING_KV_HEADS;
  const int q_cols = GEMMA4_NUM_QUERY_HEADS * head_dim;
  const int kv_cols = kv_heads * head_dim;
  const int rotary_half = global ? GEMMA4_GLOBAL_HEAD_DIM / 8 : head_dim / 2;
  Gemma4KvCacheConfig config = {
      1,
      1,
      1,
      1,
      kv_heads,
      head_dim,
      global ? 0 : 1,
  };

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

  std::vector<__nv_bfloat16> normed_x =
      reference_hidden_rmsnorm(x, input_norm_weight);
  const size_t q_weight_count =
      static_cast<size_t>(q_cols) * GEMMA4_HIDDEN_SIZE;
  const size_t kv_weight_count =
      static_cast<size_t>(kv_cols) * GEMMA4_HIDDEN_SIZE;
  const int64_t cache_count = cache_elements(config);

  DeviceBuffer<__nv_bfloat16> d_x(x.size());
  DeviceBuffer<__nv_bfloat16> d_input_norm_weight(input_norm_weight.size());
  DeviceBuffer<__nv_bfloat16> d_q_norm(head_norm_weight.size());
  DeviceBuffer<__nv_bfloat16> d_k_norm(head_norm_weight.size());
  DeviceBuffer<float> d_cos(cos.size());
  DeviceBuffer<float> d_sin(sin.size());
  DeviceBuffer<int32_t> d_page_table(page_table.size());
  DeviceBuffer<int32_t> d_token_position(token_position.size());
  DeviceBuffer<__nv_bfloat16> d_w_q(q_weight_count);
  DeviceBuffer<__nv_bfloat16> d_w_k(kv_weight_count);
  DeviceBuffer<__nv_bfloat16> d_w_v(global ? 1 : kv_weight_count);
  DeviceBuffer<__nv_bfloat16> d_q(q_cols);
  DeviceBuffer<__nv_bfloat16> d_cache_k(cache_count);
  DeviceBuffer<__nv_bfloat16> d_cache_v(cache_count);

  copy_to_device(d_x, x);
  copy_to_device(d_input_norm_weight, input_norm_weight);
  copy_to_device(d_q_norm, head_norm_weight);
  copy_to_device(d_k_norm, head_norm_weight);
  copy_to_device(d_cos, cos);
  copy_to_device(d_sin, sin);
  copy_to_device(d_page_table, page_table);
  copy_to_device(d_token_position, token_position);
  CHECK_CUDA(cudaMemset(d_w_q.get(), 0,
                        q_weight_count * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_w_k.get(), 0,
                        kv_weight_count * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_w_v.get(), 0,
                        d_w_v.count * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_q.get(), 0, q_cols * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_cache_k.get(), 0,
                        cache_count * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMemset(d_cache_v.get(), 0,
                        cache_count * sizeof(__nv_bfloat16)));

  write_device_bf16(d_w_q, 0, 0.5f);
  write_device_bf16(d_w_k, 0, -0.25f);
  if (!global) write_device_bf16(d_w_v, 0, 0.75f);

  const __nv_bfloat16 *d_v_weight = global ? nullptr : d_w_v.get();
  Gemma4AttentionProjectionWeights weights = {
      d_w_q.get(),
      d_w_k.get(),
      d_v_weight,
      0,
      0,
      0,
  };

  CHECK_CUDA(gemma4_flash_attention_decode_norm_project_prepare_paged_kv_bf16(
      d_q.get(), d_cache_k.get(), d_cache_v.get(), config,
      d_page_table.get(), d_token_position.get(), batch_size, cache_layer,
      d_x.get(), d_input_norm_weight.get(), weights, d_q_norm.get(),
      d_k_norm.get(), d_cos.get(), d_sin.get(), 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  float q_value = reference_project_scalar(normed_x[0], 0.5f);
  float k_value = reference_project_scalar(normed_x[0], -0.25f);
  float v_value = global ? k_value : reference_project_scalar(normed_x[0], 0.75f);
  std::vector<__nv_bfloat16> expected_q(q_cols, __float2bfloat16_rn(0.0f));
  std::vector<__nv_bfloat16> expected_k(cache_count, __float2bfloat16_rn(0.0f));
  std::vector<__nv_bfloat16> expected_v(cache_count, __float2bfloat16_rn(0.0f));
  expected_q[0] = reference_head_rms_scalar(q_value, head_dim);
  expected_k[0] = reference_head_rms_scalar(k_value, head_dim);
  expected_v[0] = reference_head_rms_scalar(v_value, head_dim);

  compare_bf16(copy_to_host(d_q), expected_q, 0.125f,
               global ? "global norm-project Q" : "sliding norm-project Q");
  compare_bf16(copy_to_host(d_cache_k), expected_k, 0.125f,
               global ? "global norm-project K" : "sliding norm-project K");
  compare_bf16(copy_to_host(d_cache_v), expected_v, 0.125f,
               global ? "global norm-project V" : "sliding norm-project V");
}

// Check global prefill prepares K-derived V and runs full causal attention.
void run_global_prefill_norm_rope_case() {
  constexpr int batch_size = 1;
  constexpr int seq_len = 2;
  constexpr int rows = batch_size * seq_len;
  constexpr int q_count = rows * GEMMA4_GLOBAL_Q_PROJ_SIZE;
  constexpr int kv_count = rows * GEMMA4_GLOBAL_K_PROJ_SIZE;
  constexpr int rotary_half = GEMMA4_GLOBAL_HEAD_DIM / 8;

  std::vector<__nv_bfloat16> q(q_count, __float2bfloat16_rn(0.0f));
  std::vector<__nv_bfloat16> k(kv_count);
  for (int i = 0; i < kv_count; ++i) {
    k[i] = make_value(110000 + i);
  }

  std::vector<__nv_bfloat16> norm_weight(
      GEMMA4_GLOBAL_HEAD_DIM, __float2bfloat16_rn(1.0f));
  std::vector<float> cos(seq_len * rotary_half, 1.0f);
  std::vector<float> sin(cos.size(), 0.0f);

  DeviceBuffer<__nv_bfloat16> d_q(q.size());
  DeviceBuffer<__nv_bfloat16> d_k(k.size());
  DeviceBuffer<__nv_bfloat16> d_q_prepared(q.size());
  DeviceBuffer<__nv_bfloat16> d_k_prepared(k.size());
  DeviceBuffer<__nv_bfloat16> d_v_prepared(k.size());
  DeviceBuffer<__nv_bfloat16> d_out(q.size());
  DeviceBuffer<__nv_bfloat16> d_norm_weight(norm_weight.size());
  DeviceBuffer<float> d_cos(cos.size());
  DeviceBuffer<float> d_sin(sin.size());

  copy_to_device(d_q, q);
  copy_to_device(d_k, k);
  copy_to_device(d_norm_weight, norm_weight);
  copy_to_device(d_cos, cos);
  copy_to_device(d_sin, sin);

  const float scale = 1.0f / std::sqrt(float(GEMMA4_GLOBAL_HEAD_DIM));
  CHECK_CUDA(gemma4_flash_attention_global_fwd_bf16_norm_rope(
      d_out.get(), nullptr, d_q_prepared.get(), d_k_prepared.get(),
      d_v_prepared.get(), d_q.get(), d_k.get(), d_norm_weight.get(),
      d_norm_weight.get(), d_cos.get(), d_sin.get(), nullptr, batch_size, seq_len,
      seq_len, scale, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> expected_v(kv_count);
  for (int row = 0; row < rows; ++row) {
    float sum_sq = 0.0f;
    for (int d = 0; d < GEMMA4_GLOBAL_HEAD_DIM; ++d) {
      float value = bf16_to_float(k[row * GEMMA4_GLOBAL_HEAD_DIM + d]);
      sum_sq += value * value;
    }
    float inv_rms = 1.0f / std::sqrt(sum_sq / GEMMA4_GLOBAL_HEAD_DIM +
                                     GEMMA4_RMS_NORM_EPS);
    for (int d = 0; d < GEMMA4_GLOBAL_HEAD_DIM; ++d) {
      float value = bf16_to_float(k[row * GEMMA4_GLOBAL_HEAD_DIM + d]);
      expected_v[row * GEMMA4_GLOBAL_HEAD_DIM + d] =
          __float2bfloat16_rn(value * inv_rms);
    }
  }
  compare_bf16(copy_to_host(d_v_prepared), expected_v, 0.00390625f,
               "global prefill K-derived V");

  std::vector<__nv_bfloat16> expected_out(q_count);
  std::vector<__nv_bfloat16> v_prepared = copy_to_host(d_v_prepared);
  for (int qh = 0; qh < GEMMA4_NUM_QUERY_HEADS; ++qh) {
    for (int d = 0; d < GEMMA4_GLOBAL_HEAD_DIM; ++d) {
      float v0 = bf16_to_float(v_prepared[token_offset(
          0, 0, 0, d, seq_len, GEMMA4_GLOBAL_KV_HEADS,
          GEMMA4_GLOBAL_HEAD_DIM)]);
      float v1 = bf16_to_float(v_prepared[token_offset(
          0, 1, 0, d, seq_len, GEMMA4_GLOBAL_KV_HEADS,
          GEMMA4_GLOBAL_HEAD_DIM)]);
      expected_out[token_offset(
          0, 0, qh, d, seq_len, GEMMA4_NUM_QUERY_HEADS,
          GEMMA4_GLOBAL_HEAD_DIM)] = __float2bfloat16_rn(v0);
      expected_out[token_offset(
          0, 1, qh, d, seq_len, GEMMA4_NUM_QUERY_HEADS,
          GEMMA4_GLOBAL_HEAD_DIM)] = __float2bfloat16_rn(0.5f * (v0 + v1));
    }
  }
  compare_bf16(copy_to_host(d_out), expected_out, 0.03125f,
               "global prefill attention");
}

// Validate argument guards for sliding decode and parked persistent decode.
void run_decode_invalid_args_case() {
  DeviceBuffer<__nv_bfloat16> d_bf16(1);
  DeviceBuffer<float> d_float(1);
  DeviceBuffer<int32_t> d_i32(1);
  Gemma4KvCacheConfig config = {
      1,
      1,
      64,
      16,
      GEMMA4_SLIDING_KV_HEADS,
      GEMMA4_SLIDING_HEAD_DIM,
      0,
  };

  cudaError_t status = gemma4_flash_attention_sliding_decode_paged_bf16(
      d_bf16.get(), d_float.get(), d_float.get(), d_float.get(),
      d_bf16.get(), d_bf16.get(), d_bf16.get(), d_i32.get(), d_i32.get(),
      config, 0, 1, 0.25f, 64, 16, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected invalid sliding window config\n");
    std::exit(1);
  }

  config.window_size = 1024;
  status = gemma4_flash_attention_sliding_decode_paged_bf16(
      d_bf16.get(), d_float.get(), d_float.get(), d_float.get(),
      d_bf16.get(), d_bf16.get(), d_bf16.get(), d_i32.get(), d_i32.get(),
      config, 0, 1, 0.25f, 64, 15, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected invalid underprovisioned splits\n");
    std::exit(1);
  }

  Gemma4KvCacheConfig global_config = {
      1,
      3,
      2,
      3,
      GEMMA4_GLOBAL_KV_HEADS,
      GEMMA4_GLOBAL_HEAD_DIM,
      0,
  };
  DeviceBuffer<__nv_bfloat16> d_global_q(
      GEMMA4_NUM_QUERY_HEADS * GEMMA4_GLOBAL_HEAD_DIM);
  status = gemma4_flash_attention_decode_paged_bf16(
      d_global_q.get(), d_float.get(), d_float.get(), d_float.get(),
      d_global_q.get(), d_bf16.get(), d_bf16.get(), d_i32.get(), d_i32.get(),
      global_config, 0, 1, 0.25f, 2, 2, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected invalid global split coverage\n");
    std::exit(1);
  }

  int32_t scratch_i32 = static_cast<int32_t>(
      gemma4_flash_attention_sliding_decode_persistent_scratch_i32(1, 16));
  status = gemma4_flash_attention_sliding_decode_paged_persistent_bf16(
      d_bf16.get(), d_float.get(), d_float.get(), d_float.get(), nullptr,
      scratch_i32, d_bf16.get(), d_bf16.get(), d_bf16.get(), d_i32.get(),
      d_i32.get(), config, 0, 1, 0.25f, 64, 16, 0, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected invalid persistent scratch pointer\n");
    std::exit(1);
  }
}

}  // namespace

int main() {
  run_layout_and_slot_guard_case();
  run_invalid_page_write_case();
  run_invalid_page_decode_case();
  run_global_write_decode_case();
  run_sliding_flash_decode_case();
  run_norm_project_prepare_case(false);
  run_norm_project_prepare_case(true);
  run_global_prefill_norm_rope_case();
  run_decode_invalid_args_case();
  std::puts("kv cache tests passed");
  return 0;
}
