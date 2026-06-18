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

int64_t sliding_q_offset(int batch, int head, int dim) {
  return ((int64_t)batch * GEMMA4_NUM_QUERY_HEADS + head) *
             GEMMA4_SLIDING_HEAD_DIM +
         dim;
}

int64_t sliding_kv_offset(int batch, int head, int dim) {
  return ((int64_t)batch * GEMMA4_SLIDING_KV_HEADS + head) *
             GEMMA4_SLIDING_HEAD_DIM +
         dim;
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

void fill_sliding_rope_tables(std::vector<float> &cos,
                              std::vector<float> &sin,
                              int seq_len) {
  constexpr int kHeadDim = GEMMA4_SLIDING_HEAD_DIM;
  constexpr int kRotaryHalf = kHeadDim / 2;
  for (int row = 0; row < seq_len; ++row) {
    for (int i = 0; i < kRotaryHalf; ++i) {
      float freq = std::pow(GEMMA4_ROPE_THETA_SLIDING,
                            -float(2 * i) / float(kHeadDim));
      float angle = float(row) * freq;
      cos[size_t(row) * kRotaryHalf + i] = std::cos(angle);
      sin[size_t(row) * kRotaryHalf + i] = std::sin(angle);
    }
  }
}

float reference_sliding_rms_scale(const std::vector<__nv_bfloat16> &values,
                                  int64_t base) {
  float sum = 0.0f;
  for (int d = 0; d < GEMMA4_SLIDING_HEAD_DIM; ++d) {
    float value = bf16_to_float(values[base + d]);
    sum = std::fma(value, value, sum);
  }
  return 1.0f / std::sqrt(sum / float(GEMMA4_SLIDING_HEAD_DIM) +
                          GEMMA4_RMS_NORM_EPS);
}

void reference_sliding_weighted_rope_head(
    std::vector<__nv_bfloat16> &out,
    int64_t out_base,
    const std::vector<__nv_bfloat16> &in,
    int64_t in_base,
    const std::vector<__nv_bfloat16> &weight,
    const std::vector<float> &cos,
    const std::vector<float> &sin,
    int position) {
  constexpr int kHeadDim = GEMMA4_SLIDING_HEAD_DIM;
  constexpr int kRotaryHalf = kHeadDim / 2;
  float scale = reference_sliding_rms_scale(in, in_base);
  int64_t table_base = int64_t(position) * kRotaryHalf;
  for (int d = 0; d < kRotaryHalf; ++d) {
    float lo = bf16_to_float(in[in_base + d]) * scale * bf16_to_float(weight[d]);
    float hi = bf16_to_float(in[in_base + kRotaryHalf + d]) * scale *
               bf16_to_float(weight[kRotaryHalf + d]);
    float c = cos[table_base + d];
    float s = sin[table_base + d];
    out[out_base + d] = __float2bfloat16_rn(std::fma(-hi, s, lo * c));
    out[out_base + kRotaryHalf + d] =
        __float2bfloat16_rn(std::fma(lo, s, hi * c));
  }
}

void reference_sliding_scale_head(std::vector<__nv_bfloat16> &out,
                                  int64_t out_base,
                                  const std::vector<__nv_bfloat16> &in,
                                  int64_t in_base) {
  float scale = reference_sliding_rms_scale(in, in_base);
  for (int d = 0; d < GEMMA4_SLIDING_HEAD_DIM; ++d) {
    out[out_base + d] =
        __float2bfloat16_rn(bf16_to_float(in[in_base + d]) * scale);
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

void run_sliding_decode_prep_cache_case() {
  Gemma4KvCacheConfig config = {
      2,
      12,
      4,
      4,
      GEMMA4_SLIDING_KV_HEADS,
      GEMMA4_SLIDING_HEAD_DIM,
      8,
  };
  int batch_size = 2;
  int layer = 1;
  std::vector<int32_t> token_position = {5, 11};
  std::vector<int32_t> page_table(batch_size * config.max_pages_per_seq, -1);
  page_table[gemma4_kv_cache_page_slot(config, token_position[0])] = 3;
  page_table[config.max_pages_per_seq +
             gemma4_kv_cache_page_slot(config, token_position[1])] = 7;

  std::vector<__nv_bfloat16> q(batch_size * GEMMA4_NUM_QUERY_HEADS *
                               GEMMA4_SLIDING_HEAD_DIM);
  std::vector<__nv_bfloat16> k(batch_size * GEMMA4_SLIDING_KV_HEADS *
                               GEMMA4_SLIDING_HEAD_DIM);
  std::vector<__nv_bfloat16> v(k.size());
  std::vector<__nv_bfloat16> q_weight(GEMMA4_SLIDING_HEAD_DIM);
  std::vector<__nv_bfloat16> k_weight(GEMMA4_SLIDING_HEAD_DIM);
  std::vector<float> cos((token_position[1] + 1) *
                         (GEMMA4_SLIDING_HEAD_DIM / 2));
  std::vector<float> sin(cos.size());

  for (int i = 0; i < static_cast<int>(q.size()); ++i) q[i] = make_value(12000 + i);
  for (int i = 0; i < static_cast<int>(k.size()); ++i) {
    k[i] = make_value(22000 + i);
    v[i] = make_value(32000 + i);
  }
  for (int d = 0; d < GEMMA4_SLIDING_HEAD_DIM; ++d) {
    q_weight[d] = __float2bfloat16_rn(0.75f + 0.001f * float(d % 23));
    k_weight[d] = __float2bfloat16_rn(0.95f - 0.001f * float(d % 19));
  }
  fill_sliding_rope_tables(cos, sin, token_position[1] + 1);

  std::vector<__nv_bfloat16> expected_q(q.size());
  std::vector<__nv_bfloat16> expected_cache_k(cache_elements(config));
  std::vector<__nv_bfloat16> expected_cache_v(cache_elements(config));
  for (int b = 0; b < batch_size; ++b) {
    int pos = token_position[b];
    int slot = gemma4_kv_cache_page_slot(config, pos);
    int page = page_table[b * config.max_pages_per_seq + slot];
    int page_offset = gemma4_kv_cache_page_offset(config, pos);
    for (int h = 0; h < GEMMA4_NUM_QUERY_HEADS; ++h) {
      reference_sliding_weighted_rope_head(
          expected_q, sliding_q_offset(b, h, 0), q,
          sliding_q_offset(b, h, 0), q_weight, cos, sin, pos);
    }
    for (int h = 0; h < GEMMA4_SLIDING_KV_HEADS; ++h) {
      int64_t cache_base =
          gemma4_kv_cache_offset(config, layer, page, page_offset, h, 0);
      reference_sliding_weighted_rope_head(
          expected_cache_k, cache_base, k, sliding_kv_offset(b, h, 0),
          k_weight, cos, sin, pos);
      reference_sliding_scale_head(expected_cache_v, cache_base, v,
                                   sliding_kv_offset(b, h, 0));
    }
  }

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  __nv_bfloat16 *d_v = nullptr;
  __nv_bfloat16 *d_q_prepared = nullptr;
  __nv_bfloat16 *d_cache_k = nullptr;
  __nv_bfloat16 *d_cache_v = nullptr;
  __nv_bfloat16 *d_q_weight = nullptr;
  __nv_bfloat16 *d_k_weight = nullptr;
  int32_t *d_page_table = nullptr;
  int32_t *d_token_position = nullptr;
  float *d_cos = nullptr;
  float *d_sin = nullptr;
  CHECK_CUDA(cudaMalloc(&d_q, q.size() * sizeof(*d_q)));
  CHECK_CUDA(cudaMalloc(&d_k, k.size() * sizeof(*d_k)));
  CHECK_CUDA(cudaMalloc(&d_v, v.size() * sizeof(*d_v)));
  CHECK_CUDA(cudaMalloc(&d_q_prepared, q.size() * sizeof(*d_q_prepared)));
  CHECK_CUDA(cudaMalloc(&d_cache_k, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMalloc(&d_cache_v, cache_elements(config) * sizeof(*d_cache_v)));
  CHECK_CUDA(cudaMalloc(&d_q_weight, q_weight.size() * sizeof(*d_q_weight)));
  CHECK_CUDA(cudaMalloc(&d_k_weight, k_weight.size() * sizeof(*d_k_weight)));
  CHECK_CUDA(cudaMalloc(&d_page_table, page_table.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_token_position,
                        token_position.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_cos, cos.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sin, sin.size() * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_q_prepared, 0, q.size() * sizeof(*d_q_prepared)));
  CHECK_CUDA(cudaMemset(d_cache_k, 0, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMemset(d_cache_v, 0, cache_elements(config) * sizeof(*d_cache_v)));
  CHECK_CUDA(cudaMemcpy(d_q, q.data(), q.size() * sizeof(q[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k, k.data(), k.size() * sizeof(k[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_v, v.data(), v.size() * sizeof(v[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_q_weight, q_weight.data(),
                        q_weight.size() * sizeof(q_weight[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k_weight, k_weight.data(),
                        k_weight.size() * sizeof(k_weight[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_page_table, page_table.data(),
                        page_table.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_token_position, token_position.data(),
                        token_position.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_cos, cos.data(), cos.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_sin, sin.data(), sin.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(gemma4_flash_attention_sliding_decode_prepare_q_paged_kv_bf16(
      d_q_prepared, d_cache_k, d_cache_v, config, d_page_table,
      d_token_position, batch_size, layer, d_q, d_k, d_v, d_q_weight,
      d_k_weight, d_cos, d_sin, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> actual_q(q.size());
  std::vector<__nv_bfloat16> actual_cache_k(cache_elements(config));
  std::vector<__nv_bfloat16> actual_cache_v(cache_elements(config));
  CHECK_CUDA(cudaMemcpy(actual_q.data(), d_q_prepared,
                        actual_q.size() * sizeof(actual_q[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_cache_k.data(), d_cache_k,
                        actual_cache_k.size() * sizeof(actual_cache_k[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_cache_v.data(), d_cache_v,
                        actual_cache_v.size() * sizeof(actual_cache_v[0]),
                        cudaMemcpyDeviceToHost));
  compare_bf16(actual_q, expected_q, 0.03125f, "sliding decode prepared Q");
  compare_bf16(actual_cache_k, expected_cache_k, 0.03125f,
               "sliding decode cache K");
  compare_bf16(actual_cache_v, expected_cache_v, 0.015625f,
               "sliding decode cache V");

  CHECK_CUDA(cudaFree(d_sin));
  CHECK_CUDA(cudaFree(d_cos));
  CHECK_CUDA(cudaFree(d_token_position));
  CHECK_CUDA(cudaFree(d_page_table));
  CHECK_CUDA(cudaFree(d_k_weight));
  CHECK_CUDA(cudaFree(d_q_weight));
  CHECK_CUDA(cudaFree(d_cache_v));
  CHECK_CUDA(cudaFree(d_cache_k));
  CHECK_CUDA(cudaFree(d_q_prepared));
  CHECK_CUDA(cudaFree(d_v));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_q));
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

void run_sliding_flash_decode_attention_case(const char *label,
                                             int batch_size,
                                             int seq_len,
                                             int page_size,
                                             int max_pages_per_seq,
                                             int window_size,
                                             int split_size,
                                             int extra_num_splits = 0,
                                             bool stagger_seq_lengths = false) {
  // This case builds a paged sliding-cache fixture, runs the flash decode path,
  // then checks it against a CPU reference.
  Gemma4KvCacheConfig config = {
      1,
      batch_size * max_pages_per_seq,
      page_size,
      max_pages_per_seq,
      GEMMA4_SLIDING_KV_HEADS,
      GEMMA4_SLIDING_HEAD_DIM,
      window_size,
  };
  int layer = 0;
  int q_heads = GEMMA4_NUM_QUERY_HEADS;

  // Most tests use one sequence length for the whole batch. The staggered mode
  // gives batch rows different live split counts while keeping one max launch
  // shape, which exercises the graph-compatible overprovisioning contract.
  std::vector<int32_t> target_seq_lengths(batch_size, seq_len);
  if (stagger_seq_lengths) {
    const int step = std::max(1, split_size + 1);
    for (int b = 0; b < batch_size; ++b) {
      target_seq_lengths[b] = std::max(1, seq_len - b * step);
    }
  }

  // `num_splits` is the max scratch stride and launch z dimension. Individual
  // rows may have fewer actual splits; `extra_num_splits` deliberately adds
  // empty CTAs to prove they do not write or get reduced.
  int num_splits = 0;
  for (int batch_seq_len : target_seq_lengths) {
    const int first_key =
        window_size > 0 ? std::max(0, batch_seq_len - window_size) : 0;
    const int key_count = std::max(0, batch_seq_len - first_key);
    num_splits =
        std::max(num_splits, (key_count + split_size - 1) / split_size);
  }
  const int max_key_count = window_size > 0 ? window_size : seq_len;
  num_splits =
      std::max(num_splits, (max_key_count + split_size - 1) / split_size);
  num_splits = std::max(1, num_splits + extra_num_splits);
  float scale = 1.0f / std::sqrt(float(config.head_dim));

  std::vector<int32_t> page_table(batch_size * config.max_pages_per_seq, -1);
  std::vector<int32_t> slot_logical_pages(page_table.size(), -1);
  std::vector<int32_t> seq_lengths(batch_size, 0);
  Gemma4KvPageAllocator allocator(config.num_pages);

  __nv_bfloat16 *d_cache_k = nullptr;
  __nv_bfloat16 *d_cache_v = nullptr;
  __nv_bfloat16 *d_one_k = nullptr;
  __nv_bfloat16 *d_one_v = nullptr;
  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_direct_out = nullptr;
  int32_t *d_page_table = nullptr;
  int32_t *d_token_batch = nullptr;
  int32_t *d_token_position = nullptr;
  int32_t *d_seq_lengths = nullptr;
  float *d_partial_m = nullptr;
  float *d_partial_l = nullptr;
  float *d_partial_acc = nullptr;

  // Scratch sizes use the overprovisioned stride. The flash reducer must read
  // only live splits out of this layout, leaving extra slots irrelevant.
  const size_t partial_m_bytes =
      gemma4_paged_decode_partial_m_elements(batch_size, q_heads, num_splits) * sizeof(float);
  const size_t partial_acc_bytes =
      gemma4_paged_decode_partial_acc_elements(batch_size, q_heads, num_splits,
                                               config.head_dim) * sizeof(float);

  CHECK_CUDA(cudaMalloc(&d_cache_k, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMalloc(&d_cache_v, cache_elements(config) * sizeof(*d_cache_v)));
  CHECK_CUDA(cudaMalloc(&d_one_k, config.num_heads * config.head_dim * sizeof(*d_one_k)));
  CHECK_CUDA(cudaMalloc(&d_one_v, config.num_heads * config.head_dim * sizeof(*d_one_v)));
  CHECK_CUDA(cudaMalloc(&d_q, batch_size * q_heads * config.head_dim * sizeof(*d_q)));
  CHECK_CUDA(cudaMalloc(&d_direct_out,
                        batch_size * q_heads * config.head_dim * sizeof(*d_direct_out)));
  CHECK_CUDA(cudaMalloc(&d_page_table, page_table.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_token_batch, sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_token_position, sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_seq_lengths, seq_lengths.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_partial_m, partial_m_bytes));
  CHECK_CUDA(cudaMalloc(&d_partial_l, partial_m_bytes));
  CHECK_CUDA(cudaMalloc(&d_partial_acc, partial_acc_bytes));
  CHECK_CUDA(cudaMemset(d_cache_k, 0, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMemset(d_cache_v, 0, cache_elements(config) * sizeof(*d_cache_v)));

  std::vector<__nv_bfloat16> by_pos_k(batch_size * seq_len *
                                      config.num_heads * config.head_dim);
  std::vector<__nv_bfloat16> by_pos_v(by_pos_k.size());
  std::vector<__nv_bfloat16> one_k(config.num_heads * config.head_dim);
  std::vector<__nv_bfloat16> one_v(one_k.size());

  // Populate the cache through the production KV-cache writer one token at a
  // time. This exercises page allocation, page-table entries, and sliding wrap
  // behavior instead of constructing cache memory by hand.
  for (int b = 0; b < batch_size; ++b) {
    for (int pos = 0; pos < target_seq_lengths[b]; ++pos) {
      int ensured_page = gemma4_kv_cache_ensure_page(
          page_table, slot_logical_pages, allocator, config, batch_size, b, pos);
      if (ensured_page < 0) {
        std::fprintf(stderr, "%s page allocation failed\n", label);
        std::exit(1);
      }
      seq_lengths[b] = pos + 1;
      for (int h = 0; h < config.num_heads; ++h) {
        for (int d = 0; d < config.head_dim; ++d) {
          __nv_bfloat16 kv = make_value(17000 + 1000 * b + 101 * pos + 17 * h + d);
          __nv_bfloat16 vv = make_value(27000 + 1000 * b + 101 * pos + 17 * h + d);
          one_k[h * config.head_dim + d] = kv;
          one_v[h * config.head_dim + d] = vv;
          by_pos_k[token_offset(b, pos, h, d, seq_len, config.num_heads,
                                config.head_dim)] = kv;
          by_pos_v[token_offset(b, pos, h, d, seq_len, config.num_heads,
                                config.head_dim)] = vv;
        }
      }

      int32_t h_batch = b;
      int32_t h_position = pos;
      CHECK_CUDA(cudaMemcpy(d_page_table, page_table.data(),
                            page_table.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_token_batch, &h_batch, sizeof(int32_t),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_token_position, &h_position, sizeof(int32_t),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_one_k, one_k.data(),
                            one_k.size() * sizeof(one_k[0]),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_one_v, one_v.data(),
                            one_v.size() * sizeof(one_v[0]),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(gemma4_kv_cache_write_bf16(
          d_cache_k, d_cache_v, config, d_page_table, d_token_batch,
          d_token_position, 1, layer, d_one_k, d_one_v, 0));
    }
  }

  std::vector<__nv_bfloat16> q(batch_size * q_heads * config.head_dim);
  for (int i = 0; i < static_cast<int>(q.size()); ++i) {
    q[i] = make_value(37000 + i);
  }
  std::vector<__nv_bfloat16> expected(q.size());
  reference_decode_attention(expected, q, by_pos_k, by_pos_v, seq_lengths,
                             config, batch_size, q_heads, seq_len, scale);

  CHECK_CUDA(cudaMemcpy(d_q, q.data(), q.size() * sizeof(q[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_page_table, page_table.data(),
                        page_table.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_seq_lengths, seq_lengths.data(),
                        seq_lengths.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

  // Poison all partial scratch before the flash run. If the flash reducer reads
  // beyond actual_splits, these extra slots should explode the comparison.
  auto poison_partials = [&]() {
    CHECK_CUDA(cudaMemset(d_partial_m, 0x7f, partial_m_bytes));
    CHECK_CUDA(cudaMemset(d_partial_l, 0x7f, partial_m_bytes));
    CHECK_CUDA(cudaMemset(d_partial_acc, 0x7f, partial_acc_bytes));
  };
  poison_partials();
  CHECK_CUDA(gemma4_flash_attention_sliding_decode_paged_bf16(
      d_direct_out, d_partial_m, d_partial_l, d_partial_acc, d_q, d_cache_k,
      d_cache_v, d_page_table, d_seq_lengths, config, layer, batch_size,
      scale, split_size, num_splits, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> direct(q.size());
  CHECK_CUDA(cudaMemcpy(direct.data(), d_direct_out,
                        direct.size() * sizeof(direct[0]),
                        cudaMemcpyDeviceToHost));

  compare_bf16(direct, expected, 0.015625f, "sliding flash decode direct cpu");

  CHECK_CUDA(cudaFree(d_cache_k));
  CHECK_CUDA(cudaFree(d_cache_v));
  CHECK_CUDA(cudaFree(d_one_k));
  CHECK_CUDA(cudaFree(d_one_v));
  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_direct_out));
  CHECK_CUDA(cudaFree(d_page_table));
  CHECK_CUDA(cudaFree(d_token_batch));
  CHECK_CUDA(cudaFree(d_token_position));
  CHECK_CUDA(cudaFree(d_seq_lengths));
  CHECK_CUDA(cudaFree(d_partial_m));
  CHECK_CUDA(cudaFree(d_partial_l));
  CHECK_CUDA(cudaFree(d_partial_acc));
}

void run_sliding_flash_decode_invalid_args_case() {
  __nv_bfloat16 *d_bf16 = nullptr;
  float *d_float = nullptr;
  int32_t *d_i32 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_bf16, sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_float, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_i32, sizeof(int32_t)));

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
      d_bf16, d_float, d_float, d_float, d_bf16, d_bf16, d_bf16, d_i32,
      d_i32, config, 0, 1, 0.25f, 64, 16, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected invalid sliding window config\n");
    std::exit(1);
  }

  config.window_size = 1024;
  status = gemma4_flash_attention_sliding_decode_paged_bf16(
      d_bf16, d_float, d_float, d_float, d_bf16, d_bf16, d_bf16, d_i32,
      d_i32, config, 0, 1, 0.25f, 64, 15, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected invalid underprovisioned decode splits\n");
    std::exit(1);
  }

  CHECK_CUDA(cudaFree(d_i32));
  CHECK_CUDA(cudaFree(d_float));
  CHECK_CUDA(cudaFree(d_bf16));
}

}  // namespace

int main() {
  run_address_case();
  run_sliding_decode_prep_cache_case();
  run_global_write_and_attention_case();
  run_sliding_wrap_case();

  // Short and boundary cases cover partial final splits, exact page boundaries,
  // overprovisioned launch z dimensions, and batch rows with different lengths.
  run_sliding_flash_decode_attention_case(
      "sliding flash decode short", 1, 5, 8, 2, 8, 3);
  run_sliding_flash_decode_attention_case(
      "sliding flash decode shorter than split", 1, 5, 8, 2, 8, 8, 2);
  run_sliding_flash_decode_attention_case(
      "sliding flash decode exact splits", 1, 8, 4, 3, 8, 4);
  run_sliding_flash_decode_attention_case(
      "sliding flash decode overprovisioned", 1, 5, 8, 2, 8, 3, 3);
  run_sliding_flash_decode_attention_case(
      "sliding flash decode boundary", 2, 10, 4, 3, 8, 3);
  run_sliding_flash_decode_attention_case(
      "sliding flash decode varlen overprovisioned", 2, 10, 4, 3, 8, 3, 3, true);
  run_sliding_flash_decode_attention_case(
      "sliding flash decode wrap", 1, 13, 4, 3, 8, 5);
  run_sliding_flash_decode_invalid_args_case();
  std::puts("kv cache tests passed");
  return 0;
}
