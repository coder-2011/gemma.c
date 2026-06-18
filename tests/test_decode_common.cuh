#pragma once

#include "gemma4_kv_cache.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace gemma4_test {

inline void check_cuda(cudaError_t status,
                       const char *expr,
                       const char *file,
                       int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

inline float bf16_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

inline __nv_bfloat16 make_value(int seed) {
  int centered = ((seed * 37 + 17) % 257) - 128;
  return __float2bfloat16_rn(static_cast<float>(centered) / 64.0f);
}

inline int64_t cache_elements(const Gemma4KvCacheConfig &config) {
  return int64_t(config.num_layers) * config.num_pages * config.page_size *
         config.num_heads * config.head_dim;
}

inline int64_t sliding_q_offset(int batch, int head, int dim) {
  return ((int64_t)batch * GEMMA4_NUM_QUERY_HEADS + head) *
             GEMMA4_SLIDING_HEAD_DIM +
         dim;
}

inline int64_t sliding_kv_offset(int batch, int head, int dim) {
  return ((int64_t)batch * GEMMA4_SLIDING_KV_HEADS + head) *
             GEMMA4_SLIDING_HEAD_DIM +
         dim;
}

inline int64_t global_q_offset(int batch, int head, int dim) {
  return ((int64_t)batch * GEMMA4_NUM_QUERY_HEADS + head) *
             GEMMA4_GLOBAL_HEAD_DIM +
         dim;
}

inline int64_t global_kv_offset(int batch, int head, int dim) {
  return ((int64_t)batch * GEMMA4_GLOBAL_KV_HEADS + head) *
             GEMMA4_GLOBAL_HEAD_DIM +
         dim;
}

inline int64_t token_offset(int batch,
                            int position,
                            int head,
                            int dim,
                            int max_seq,
                            int heads,
                            int head_dim) {
  return (((int64_t)batch * max_seq + position) * heads + head) * head_dim +
         dim;
}

inline void compare_bf16(const std::vector<__nv_bfloat16> &actual,
                         const std::vector<__nv_bfloat16> &expected,
                         float tolerance,
                         const char *label) {
  float max_abs = 0.0f;
  int max_index = 0;
  for (int i = 0; i < static_cast<int>(actual.size()); ++i) {
    float diff =
        std::fabs(bf16_to_float(actual[i]) - bf16_to_float(expected[i]));
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

inline void fill_sliding_rope_tables(std::vector<float> &cos,
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

inline void fill_global_rope_tables(std::vector<float> &cos,
                                    std::vector<float> &sin,
                                    int seq_len) {
  constexpr int kHeadDim = GEMMA4_GLOBAL_HEAD_DIM;
  constexpr int kRotaryDim = GEMMA4_GLOBAL_HEAD_DIM / 4;
  constexpr int kRotaryHalf = kRotaryDim / 2;
  for (int row = 0; row < seq_len; ++row) {
    for (int i = 0; i < kRotaryHalf; ++i) {
      float freq = std::pow(GEMMA4_ROPE_THETA_GLOBAL,
                            -float(2 * i) / float(kHeadDim));
      float angle = float(row) * freq;
      cos[size_t(row) * kRotaryHalf + i] = std::cos(angle);
      sin[size_t(row) * kRotaryHalf + i] = std::sin(angle);
    }
  }
}

inline float reference_sliding_rms_scale(
    const std::vector<__nv_bfloat16> &values,
    int64_t base) {
  float sum = 0.0f;
  for (int d = 0; d < GEMMA4_SLIDING_HEAD_DIM; ++d) {
    float value = bf16_to_float(values[base + d]);
    sum = std::fma(value, value, sum);
  }
  return 1.0f / std::sqrt(sum / float(GEMMA4_SLIDING_HEAD_DIM) +
                          GEMMA4_RMS_NORM_EPS);
}

inline void reference_sliding_weighted_rope_head(
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
    float lo = bf16_to_float(in[in_base + d]) * scale *
               bf16_to_float(weight[d]);
    float hi = bf16_to_float(in[in_base + kRotaryHalf + d]) * scale *
               bf16_to_float(weight[kRotaryHalf + d]);
    float c = cos[table_base + d];
    float s = sin[table_base + d];
    out[out_base + d] = __float2bfloat16_rn(std::fma(-hi, s, lo * c));
    out[out_base + kRotaryHalf + d] =
        __float2bfloat16_rn(std::fma(lo, s, hi * c));
  }
}

inline void reference_sliding_scale_head(
    std::vector<__nv_bfloat16> &out,
    int64_t out_base,
    const std::vector<__nv_bfloat16> &in,
    int64_t in_base) {
  float scale = reference_sliding_rms_scale(in, in_base);
  for (int d = 0; d < GEMMA4_SLIDING_HEAD_DIM; ++d) {
    out[out_base + d] =
        __float2bfloat16_rn(bf16_to_float(in[in_base + d]) * scale);
  }
}

inline float reference_global_rms_scale(
    const std::vector<__nv_bfloat16> &values,
    int64_t base) {
  float sum = 0.0f;
  for (int d = 0; d < GEMMA4_GLOBAL_HEAD_DIM; ++d) {
    float value = bf16_to_float(values[base + d]);
    sum = std::fma(value, value, sum);
  }
  return 1.0f / std::sqrt(sum / float(GEMMA4_GLOBAL_HEAD_DIM) +
                          GEMMA4_RMS_NORM_EPS);
}

inline void reference_global_weighted_rope_head(
    std::vector<__nv_bfloat16> &out,
    int64_t out_base,
    const std::vector<__nv_bfloat16> &in,
    int64_t in_base,
    const std::vector<__nv_bfloat16> &weight,
    const std::vector<float> &cos,
    const std::vector<float> &sin,
    int position) {
  constexpr int kHeadDim = GEMMA4_GLOBAL_HEAD_DIM;
  constexpr int kRotaryDim = GEMMA4_GLOBAL_HEAD_DIM / 4;
  constexpr int kRotaryHalf = kRotaryDim / 2;
  float scale = reference_global_rms_scale(in, in_base);
  int64_t table_base = int64_t(position) * kRotaryHalf;
  for (int d = 0; d < kRotaryHalf; ++d) {
    float lo = bf16_to_float(in[in_base + d]) * scale *
               bf16_to_float(weight[d]);
    float hi = bf16_to_float(in[in_base + kRotaryHalf + d]) * scale *
               bf16_to_float(weight[kRotaryHalf + d]);
    float c = cos[table_base + d];
    float s = sin[table_base + d];
    out[out_base + d] = __float2bfloat16_rn(std::fma(-hi, s, lo * c));
    out[out_base + kRotaryHalf + d] =
        __float2bfloat16_rn(std::fma(lo, s, hi * c));
  }
  for (int d = kRotaryDim; d < kHeadDim; ++d) {
    float value =
        bf16_to_float(in[in_base + d]) * scale * bf16_to_float(weight[d]);
    out[out_base + d] = __float2bfloat16_rn(value);
  }
}

inline void reference_global_scale_head(
    std::vector<__nv_bfloat16> &out,
    int64_t out_base,
    const std::vector<__nv_bfloat16> &in,
    int64_t in_base) {
  float scale = reference_global_rms_scale(in, in_base);
  for (int d = 0; d < GEMMA4_GLOBAL_HEAD_DIM; ++d) {
    out[out_base + d] =
        __float2bfloat16_rn(bf16_to_float(in[in_base + d]) * scale);
  }
}

inline void reference_decode_attention(
    std::vector<__nv_bfloat16> &out,
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

}  // namespace gemma4_test

#ifndef CHECK_CUDA
#define CHECK_CUDA(expr) \
  gemma4_test::check_cuda((expr), #expr, __FILE__, __LINE__)
#endif
