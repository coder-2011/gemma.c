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
  run_decode_invalid_args_case();
  std::puts("kv cache tests passed");
  return 0;
}
