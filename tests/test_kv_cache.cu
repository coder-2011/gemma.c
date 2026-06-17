#include "gemma4_kv_cache.cuh"

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

float bf16_to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

__nv_bfloat16 make_value(int seed) {
  int centered = ((seed * 37 + 17) % 257) - 128;
  return __float2bfloat16_rn(static_cast<float>(centered) / 64.0f);
}

int64_t cache_elements(const Gemma4KvCacheConfig &config) {
  return int64_t(config.num_layers) * config.num_pages * config.page_size *
         config.num_heads * config.head_dim;
}

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
          float qv = bf16_to_float(
              q[((int64_t)b * q_heads + qh) * config.head_dim + d]);
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

void run_address_case() {
  Gemma4KvCacheConfig config = {2, 7, 4, 5, 3, 8, 0};
  int64_t got = gemma4_kv_cache_offset(config, 1, 2, 3, 1, 5);
  int64_t expected = (((((int64_t)1 * 7 + 2) * 4 + 3) * 3 + 1) * 8 + 5);
  if (got != expected) {
    std::fprintf(stderr, "address helper got=%lld expected=%lld\n",
                 static_cast<long long>(got),
                 static_cast<long long>(expected));
    std::exit(1);
  }
}

void run_global_write_and_attention_case() {
  Gemma4KvCacheConfig config = {2, 16, 4, 8, 2, 16, 0};
  int batch_size = 2;
  int q_heads = 4;
  int max_seq = 10;
  int layer = 1;
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

  __nv_bfloat16 *d_cache_k = nullptr;
  __nv_bfloat16 *d_cache_v = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  __nv_bfloat16 *d_v = nullptr;
  int32_t *d_page_table = nullptr;
  int32_t *d_token_batch = nullptr;
  int32_t *d_token_position = nullptr;
  CHECK_CUDA(cudaMalloc(&d_cache_k, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMalloc(&d_cache_v, cache_elements(config) * sizeof(*d_cache_v)));
  CHECK_CUDA(cudaMalloc(&d_k, flat_k.size() * sizeof(*d_k)));
  CHECK_CUDA(cudaMalloc(&d_v, flat_v.size() * sizeof(*d_v)));
  CHECK_CUDA(cudaMalloc(&d_page_table, page_table.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_token_batch, token_batch.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_token_position,
                        token_position.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMemset(d_cache_k, 0, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMemset(d_cache_v, 0, cache_elements(config) * sizeof(*d_cache_v)));
  CHECK_CUDA(cudaMemcpy(d_k, flat_k.data(), flat_k.size() * sizeof(*d_k),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_v, flat_v.data(), flat_v.size() * sizeof(*d_v),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_page_table, page_table.data(),
                        page_table.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_token_batch, token_batch.data(),
                        token_batch.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_token_position, token_position.data(),
                        token_position.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(gemma4_kv_cache_write_bf16(
      d_cache_k, d_cache_v, config, d_page_table, d_token_batch,
      d_token_position, token_count, layer, d_k, d_v, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> cache_k(cache_elements(config));
  CHECK_CUDA(cudaMemcpy(cache_k.data(), d_cache_k,
                        cache_k.size() * sizeof(cache_k[0]),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < token_count; ++i) {
    int b = token_batch[i];
    int pos = token_position[i];
    int page = page_table[b * config.max_pages_per_seq +
                          gemma4_kv_cache_page_slot(config, pos)];
    int page_offset = gemma4_kv_cache_page_offset(config, pos);
    for (int h = 0; h < config.num_heads; ++h) {
      for (int d = 0; d < config.head_dim; ++d) {
        int64_t got_offset =
            gemma4_kv_cache_offset(config, layer, page, page_offset, h, d);
        __nv_bfloat16 expected =
            by_pos_k[token_offset(b, pos, h, d, max_seq, config.num_heads,
                                  config.head_dim)];
        if (bf16_to_float(cache_k[got_offset]) != bf16_to_float(expected)) {
          std::fprintf(stderr, "cache write mismatch\n");
          std::exit(1);
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

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_out = nullptr;
  float *d_partial_m = nullptr;
  float *d_partial_l = nullptr;
  float *d_partial_acc = nullptr;
  int32_t *d_seq_lengths = nullptr;
  int num_splits = 4;
  CHECK_CUDA(cudaMalloc(&d_q, q.size() * sizeof(*d_q)));
  CHECK_CUDA(cudaMalloc(&d_out, q.size() * sizeof(*d_out)));
  CHECK_CUDA(cudaMalloc(&d_seq_lengths, seq_lengths.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_partial_m,
                        gemma4_paged_decode_partial_m_elements(
                            batch_size, q_heads, num_splits) *
                            sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_partial_l,
                        gemma4_paged_decode_partial_m_elements(
                            batch_size, q_heads, num_splits) *
                            sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_partial_acc,
                        gemma4_paged_decode_partial_acc_elements(
                            batch_size, q_heads, num_splits, config.head_dim) *
                            sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_q, q.data(), q.size() * sizeof(*d_q),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_seq_lengths, seq_lengths.data(),
                        seq_lengths.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(gemma4_paged_decode_attention_bf16(
      d_out, d_partial_m, d_partial_l, d_partial_acc, d_q, d_cache_k,
      d_cache_v, d_page_table, d_seq_lengths, config, layer, batch_size,
      q_heads, 0.25f, 3, num_splits, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> actual(q.size());
  CHECK_CUDA(cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(actual[0]),
                        cudaMemcpyDeviceToHost));
  compare_bf16(actual, expected, 0.015625f, "global paged attention");

  CHECK_CUDA(cudaFree(d_cache_k));
  CHECK_CUDA(cudaFree(d_cache_v));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_v));
  CHECK_CUDA(cudaFree(d_page_table));
  CHECK_CUDA(cudaFree(d_token_batch));
  CHECK_CUDA(cudaFree(d_token_position));
  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_seq_lengths));
  CHECK_CUDA(cudaFree(d_partial_m));
  CHECK_CUDA(cudaFree(d_partial_l));
  CHECK_CUDA(cudaFree(d_partial_acc));
}

void run_sliding_wrap_case() {
  Gemma4KvCacheConfig config = {1, 6, 4, 3, 2, 16, 8};
  int batch_size = 1;
  int q_heads = 4;
  int max_seq = 13;
  int layer = 0;
  std::vector<int32_t> page_table(config.max_pages_per_seq, -1);
  std::vector<int32_t> slot_logical_pages(config.max_pages_per_seq, -1);
  std::vector<int32_t> seq_lengths = {0};
  Gemma4KvPageAllocator allocator(config.num_pages);

  __nv_bfloat16 *d_cache_k = nullptr;
  __nv_bfloat16 *d_cache_v = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  __nv_bfloat16 *d_v = nullptr;
  int32_t *d_page_table = nullptr;
  int32_t *d_token_batch = nullptr;
  int32_t *d_token_position = nullptr;
  CHECK_CUDA(cudaMalloc(&d_cache_k, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMalloc(&d_cache_v, cache_elements(config) * sizeof(*d_cache_v)));
  CHECK_CUDA(cudaMalloc(&d_k, config.num_heads * config.head_dim * sizeof(*d_k)));
  CHECK_CUDA(cudaMalloc(&d_v, config.num_heads * config.head_dim * sizeof(*d_v)));
  CHECK_CUDA(cudaMalloc(&d_page_table, page_table.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_token_batch, sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_token_position, sizeof(int32_t)));
  CHECK_CUDA(cudaMemset(d_cache_k, 0, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMemset(d_cache_v, 0, cache_elements(config) * sizeof(*d_cache_v)));

  std::vector<__nv_bfloat16> by_pos_k(batch_size * max_seq *
                                      config.num_heads * config.head_dim);
  std::vector<__nv_bfloat16> by_pos_v(by_pos_k.size());
  int32_t h_batch = 0;
  std::vector<__nv_bfloat16> one_k(config.num_heads * config.head_dim);
  std::vector<__nv_bfloat16> one_v(one_k.size());
  for (int step = 0; step < max_seq; ++step) {
    int pos = gemma4_kv_cache_append_position(
        page_table, slot_logical_pages, seq_lengths, allocator, config,
        batch_size, 0);
    if (pos != step) {
      std::fprintf(stderr, "sliding allocator failed\n");
      std::exit(1);
    }
    for (int h = 0; h < config.num_heads; ++h) {
      for (int d = 0; d < config.head_dim; ++d) {
        __nv_bfloat16 kv = make_value(7000 + 100 * pos + 19 * h + d);
        __nv_bfloat16 vv = make_value(9000 + 100 * pos + 19 * h + d);
        one_k[h * config.head_dim + d] = kv;
        one_v[h * config.head_dim + d] = vv;
        by_pos_k[token_offset(0, pos, h, d, max_seq, config.num_heads,
                              config.head_dim)] = kv;
        by_pos_v[token_offset(0, pos, h, d, max_seq, config.num_heads,
                              config.head_dim)] = vv;
      }
    }

    int32_t h_position = pos;
    CHECK_CUDA(cudaMemcpy(d_page_table, page_table.data(),
                          page_table.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_token_batch, &h_batch, sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_token_position, &h_position, sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k, one_k.data(), one_k.size() * sizeof(one_k[0]),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v, one_v.data(), one_v.size() * sizeof(one_v[0]),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(gemma4_kv_cache_write_bf16(
        d_cache_k, d_cache_v, config, d_page_table, d_token_batch,
        d_token_position, 1, layer, d_k, d_v, 0));
  }

  std::vector<__nv_bfloat16> q(batch_size * q_heads * config.head_dim);
  for (int i = 0; i < static_cast<int>(q.size()); ++i) {
    q[i] = make_value(11000 + i);
  }
  std::vector<__nv_bfloat16> expected(q.size());
  reference_decode_attention(expected, q, by_pos_k, by_pos_v, seq_lengths,
                             config, batch_size, q_heads, max_seq, 0.25f);

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_out = nullptr;
  float *d_partial_m = nullptr;
  float *d_partial_l = nullptr;
  float *d_partial_acc = nullptr;
  int32_t *d_seq_lengths = nullptr;
  int num_splits = 3;
  CHECK_CUDA(cudaMalloc(&d_q, q.size() * sizeof(*d_q)));
  CHECK_CUDA(cudaMalloc(&d_out, q.size() * sizeof(*d_out)));
  CHECK_CUDA(cudaMalloc(&d_seq_lengths, sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_partial_m,
                        gemma4_paged_decode_partial_m_elements(
                            batch_size, q_heads, num_splits) *
                            sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_partial_l,
                        gemma4_paged_decode_partial_m_elements(
                            batch_size, q_heads, num_splits) *
                            sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_partial_acc,
                        gemma4_paged_decode_partial_acc_elements(
                            batch_size, q_heads, num_splits, config.head_dim) *
                            sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_q, q.data(), q.size() * sizeof(*d_q),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_seq_lengths, seq_lengths.data(), sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_page_table, page_table.data(),
                        page_table.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(gemma4_paged_decode_attention_bf16(
      d_out, d_partial_m, d_partial_l, d_partial_acc, d_q, d_cache_k,
      d_cache_v, d_page_table, d_seq_lengths, config, layer, batch_size,
      q_heads, 0.25f, 3, num_splits, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> actual(q.size());
  CHECK_CUDA(cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(actual[0]),
                        cudaMemcpyDeviceToHost));
  compare_bf16(actual, expected, 0.015625f, "sliding paged attention");

  CHECK_CUDA(cudaFree(d_cache_k));
  CHECK_CUDA(cudaFree(d_cache_v));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_v));
  CHECK_CUDA(cudaFree(d_page_table));
  CHECK_CUDA(cudaFree(d_token_batch));
  CHECK_CUDA(cudaFree(d_token_position));
  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_seq_lengths));
  CHECK_CUDA(cudaFree(d_partial_m));
  CHECK_CUDA(cudaFree(d_partial_l));
  CHECK_CUDA(cudaFree(d_partial_acc));
}

}  // namespace

int main() {
  run_address_case();
  run_global_write_and_attention_case();
  run_sliding_wrap_case();
  std::puts("kv cache tests passed");
  return 0;
}
