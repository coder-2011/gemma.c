#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "gemma4.h"
#include "gemma4_bench_utils.cuh"
#include "gemma4_flash_attention.cuh"

namespace {

constexpr int kQHeads = GEMMA4_NUM_QUERY_HEADS;
constexpr int kKvHeads = GEMMA4_SLIDING_KV_HEADS;
constexpr int kHeadDim = GEMMA4_SLIDING_HEAD_DIM;
constexpr int kGqaRatio = kQHeads / kKvHeads;
constexpr int kRotaryHalf = kHeadDim / 2;

int parse_arg(char **argv, int argc, int index, int default_value) {
  return index < argc ? std::atoi(argv[index]) : default_value;
}

size_t q_elements(int batch_size, int seq_len) {
  return size_t(batch_size) * seq_len * kQHeads * kHeadDim;
}

size_t kv_elements(int batch_size, int seq_len) {
  return size_t(batch_size) * seq_len * kKvHeads * kHeadDim;
}

size_t rope_table_elements(int seq_len) {
  return size_t(seq_len) * kRotaryHalf;
}

size_t q_index(int batch, int row, int head, int dim, int seq_len) {
  return (((size_t(batch) * seq_len + row) * kQHeads + head) * kHeadDim) + dim;
}

size_t kv_index(int batch, int row, int head, int dim, int seq_len) {
  return (((size_t(batch) * seq_len + row) * kKvHeads + head) * kHeadDim) + dim;
}

void fill_random_bf16(std::vector<__nv_bfloat16> &values, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.35f);
  for (auto &value : values) {
    value = __float2bfloat16(dist(rng));
  }
}

void fill_norm_weight_bf16(std::vector<__nv_bfloat16> &values, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(1.0f, 0.05f);
  for (auto &value : values) {
    value = __float2bfloat16(dist(rng));
  }
}

void fill_rope_tables(std::vector<float> &cos, std::vector<float> &sin,
                      int seq_len) {
  for (int row = 0; row < seq_len; ++row) {
    for (int i = 0; i < kRotaryHalf; ++i) {
      const double freq =
          std::pow(double(GEMMA4_ROPE_THETA_SLIDING),
                   -double(2 * i) / double(kHeadDim));
      const double angle = double(row) * freq;
      cos[size_t(row) * kRotaryHalf + i] =
          static_cast<float>(std::cos(angle));
      sin[size_t(row) * kRotaryHalf + i] =
          static_cast<float>(std::sin(angle));
    }
  }
}

float rms_scale(const std::vector<__nv_bfloat16> &values, size_t base) {
  double sum = 0.0;
  for (int d = 0; d < kHeadDim; ++d) {
    const float x = __bfloat162float(values[base + d]);
    sum += double(x) * double(x);
  }
  return 1.0f / std::sqrt(float(sum / double(kHeadDim)) + GEMMA4_RMS_NORM_EPS);
}

void prepare_weighted_rope_head(std::vector<__nv_bfloat16> &out,
                                const std::vector<__nv_bfloat16> &in,
                                const std::vector<__nv_bfloat16> &weight,
                                const std::vector<float> &cos,
                                const std::vector<float> &sin,
                                size_t base,
                                int row) {
  std::array<float, kHeadDim> values{};
  const float scale = rms_scale(in, base);
  for (int d = 0; d < kHeadDim; ++d) {
    values[d] = __bfloat162float(in[base + d]) * scale *
                __bfloat162float(weight[d]);
  }

  const size_t table_base = size_t(row) * kRotaryHalf;
  for (int d = 0; d < kRotaryHalf; ++d) {
    const float lo = values[d];
    const float hi = values[kRotaryHalf + d];
    const float c = cos[table_base + d];
    const float s = sin[table_base + d];
    out[base + d] = __float2bfloat16_rn(std::fma(-hi, s, lo * c));
    out[base + kRotaryHalf + d] =
        __float2bfloat16_rn(std::fma(lo, s, hi * c));
  }
}

void prepare_scale_head(std::vector<__nv_bfloat16> &out,
                        const std::vector<__nv_bfloat16> &in,
                        size_t base) {
  const float scale = rms_scale(in, base);
  for (int d = 0; d < kHeadDim; ++d) {
    out[base + d] =
        __float2bfloat16_rn(__bfloat162float(in[base + d]) * scale);
  }
}

void prepare_qkv_norm_rope_host(
    std::vector<__nv_bfloat16> &q_out,
    std::vector<__nv_bfloat16> &k_out,
    std::vector<__nv_bfloat16> &v_out,
    const std::vector<__nv_bfloat16> &q,
    const std::vector<__nv_bfloat16> &k,
    const std::vector<__nv_bfloat16> &v,
    const std::vector<__nv_bfloat16> &q_norm_weight,
    const std::vector<__nv_bfloat16> &k_norm_weight,
    const std::vector<float> &cos,
    const std::vector<float> &sin,
    int batch_size,
    int seq_len) {
  for (int b = 0; b < batch_size; ++b) {
    for (int row = 0; row < seq_len; ++row) {
      for (int qh = 0; qh < kQHeads; ++qh) {
        prepare_weighted_rope_head(q_out, q, q_norm_weight, cos, sin,
                                   q_index(b, row, qh, 0, seq_len), row);
      }
      for (int kvh = 0; kvh < kKvHeads; ++kvh) {
        const size_t base = kv_index(b, row, kvh, 0, seq_len);
        prepare_weighted_rope_head(k_out, k, k_norm_weight, cos, sin,
                                   base, row);
        prepare_scale_head(v_out, v, base);
      }
    }
  }
}

std::vector<__nv_bfloat16> reference_sliding_attention(
    const std::vector<__nv_bfloat16> &q,
    const std::vector<__nv_bfloat16> &k,
    const std::vector<__nv_bfloat16> &v,
    int batch_size,
    int seq_len,
    int window_left,
    float softmax_scale) {
  std::vector<__nv_bfloat16> out(q_elements(batch_size, seq_len));
  std::vector<float> scores(seq_len);

  for (int b = 0; b < batch_size; ++b) {
    for (int row = 0; row < seq_len; ++row) {
      const int col_begin = std::max(0, row - window_left);
      const int col_end = row + 1;
      for (int qh = 0; qh < kQHeads; ++qh) {
        const int kvh = qh / kGqaRatio;
        float max_score = -INFINITY;
        for (int col = col_begin; col < col_end; ++col) {
          float score = 0.0f;
          for (int d = 0; d < kHeadDim; ++d) {
            const float qv = __bfloat162float(q[q_index(b, row, qh, d, seq_len)]);
            const float kv = __bfloat162float(k[kv_index(b, col, kvh, d, seq_len)]);
            score = fmaf(qv, kv, score);
          }
          score *= softmax_scale;
          scores[col] = score;
          max_score = std::max(max_score, score);
        }

        float denom = 0.0f;
        for (int col = col_begin; col < col_end; ++col) {
          const float weight = std::exp(scores[col] - max_score);
          scores[col] = weight;
          denom += weight;
        }
        const float inv_denom = 1.0f / denom;

        for (int d = 0; d < kHeadDim; ++d) {
          float acc = 0.0f;
          for (int col = col_begin; col < col_end; ++col) {
            const float vv = __bfloat162float(v[kv_index(b, col, kvh, d, seq_len)]);
            acc = fmaf(scores[col] * inv_denom, vv, acc);
          }
          out[q_index(b, row, qh, d, seq_len)] = __float2bfloat16(acc);
        }
      }
    }
  }
  return out;
}

DiffStats diff_stats_host(const std::vector<__nv_bfloat16> &lhs,
                          const std::vector<__nv_bfloat16> &rhs) {
  DiffStats stats;
  double sum_abs = 0.0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const float a = __bfloat162float(lhs[i]);
    const float b = __bfloat162float(rhs[i]);
    const float abs_diff = std::abs(a - b);
    const float denom = std::max(std::max(std::abs(a), std::abs(b)), 1.0f);
    stats.max_abs = std::max(stats.max_abs, abs_diff);
    stats.max_rel = std::max(stats.max_rel, abs_diff / denom);
    sum_abs += abs_diff;
  }
  stats.mean_abs = lhs.empty() ? 0.0f : float(sum_abs / double(lhs.size()));
  return stats;
}

void run_correctness(int batch_size, int seq_len, int window_left,
                     cudaStream_t stream) {
  const float scale = 1.0f / std::sqrt(float(kHeadDim));
  std::vector<__nv_bfloat16> h_q(q_elements(batch_size, seq_len));
  std::vector<__nv_bfloat16> h_k(kv_elements(batch_size, seq_len));
  std::vector<__nv_bfloat16> h_v(kv_elements(batch_size, seq_len));
  std::vector<__nv_bfloat16> h_q_prepared(h_q.size());
  std::vector<__nv_bfloat16> h_k_prepared(h_k.size());
  std::vector<__nv_bfloat16> h_v_prepared(h_v.size());
  std::vector<__nv_bfloat16> h_q_norm_weight(kHeadDim);
  std::vector<__nv_bfloat16> h_k_norm_weight(kHeadDim);
  std::vector<float> h_cos(rope_table_elements(seq_len));
  std::vector<float> h_sin(rope_table_elements(seq_len));
  fill_random_bf16(h_q, 0x4a17u);
  fill_random_bf16(h_k, 0x5b23u);
  fill_random_bf16(h_v, 0x6c31u);
  fill_norm_weight_bf16(h_q_norm_weight, 0x7d41u);
  fill_norm_weight_bf16(h_k_norm_weight, 0x8e59u);
  fill_rope_tables(h_cos, h_sin, seq_len);
  prepare_qkv_norm_rope_host(h_q_prepared, h_k_prepared, h_v_prepared,
                             h_q, h_k, h_v, h_q_norm_weight,
                             h_k_norm_weight, h_cos, h_sin,
                             batch_size, seq_len);

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  __nv_bfloat16 *d_v = nullptr;
  __nv_bfloat16 *d_q_prepared = nullptr;
  __nv_bfloat16 *d_k_prepared = nullptr;
  __nv_bfloat16 *d_v_prepared = nullptr;
  __nv_bfloat16 *d_q_norm_weight = nullptr;
  __nv_bfloat16 *d_k_norm_weight = nullptr;
  __nv_bfloat16 *d_out = nullptr;
  float *d_cos = nullptr;
  float *d_sin = nullptr;
  float *d_lse = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_v, h_v.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_q_prepared, h_q.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_k_prepared, h_k.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_v_prepared, h_v.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_q_norm_weight,
                        h_q_norm_weight.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_k_norm_weight,
                        h_k_norm_weight.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_out, h_q.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_cos, h_cos.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sin, h_sin.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse, size_t(batch_size) * kQHeads * seq_len * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_q_norm_weight, h_q_norm_weight.data(),
                        h_q_norm_weight.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_norm_weight, h_k_norm_weight.data(),
                        h_k_norm_weight.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), h_cos.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), h_sin.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
      d_out, d_lse, d_q_prepared, d_k_prepared, d_v_prepared,
      d_q, d_k, d_v, d_q_norm_weight, d_k_norm_weight, d_cos, d_sin,
      batch_size, seq_len, seq_len, window_left, scale, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> h_out(h_q.size());
  std::vector<__nv_bfloat16> h_q_gpu(h_q.size());
  std::vector<__nv_bfloat16> h_k_gpu(h_k.size());
  std::vector<__nv_bfloat16> h_v_gpu(h_v.size());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_q_gpu.data(), d_q_prepared,
                        h_q_gpu.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_k_gpu.data(), d_k_prepared,
                        h_k_gpu.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_v_gpu.data(), d_v_prepared,
                        h_v_gpu.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  std::vector<__nv_bfloat16> h_ref =
      reference_sliding_attention(h_q_prepared, h_k_prepared, h_v_prepared,
                                  batch_size, seq_len, window_left, scale);
  const DiffStats q_diff = diff_stats_host(h_q_gpu, h_q_prepared);
  const DiffStats k_diff = diff_stats_host(h_k_gpu, h_k_prepared);
  const DiffStats v_diff = diff_stats_host(h_v_gpu, h_v_prepared);
  const DiffStats diff = diff_stats_host(h_out, h_ref);
  std::cout << "correctness seq=" << seq_len
            << " max_abs=" << diff.max_abs
            << " mean_abs=" << diff.mean_abs
            << " max_rel=" << diff.max_rel << "\n";
  std::cout << "prep_correctness q_max_abs=" << q_diff.max_abs
            << " k_max_abs=" << k_diff.max_abs
            << " v_max_abs=" << v_diff.max_abs << "\n";

  CUDA_CHECK(cudaFree(d_lse));
  CUDA_CHECK(cudaFree(d_sin));
  CUDA_CHECK(cudaFree(d_cos));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_k_norm_weight));
  CUDA_CHECK(cudaFree(d_q_norm_weight));
  CUDA_CHECK(cudaFree(d_v_prepared));
  CUDA_CHECK(cudaFree(d_k_prepared));
  CUDA_CHECK(cudaFree(d_q_prepared));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_q));
}

double sliding_attention_flops(int batch_size, int seq_len, int window_left) {
  double key_count = 0.0;
  for (int row = 0; row < seq_len; ++row) {
    key_count += double(row - std::max(0, row - window_left) + 1);
  }
  return double(batch_size) * kQHeads * key_count * double(4 * kHeadDim);
}

void run_benchmark(int batch_size, int seq_len, int window_left, int warmup,
                   int iters, int trials, cudaStream_t stream) {
  const float scale = 1.0f / std::sqrt(float(kHeadDim));
  std::vector<__nv_bfloat16> h_q(q_elements(batch_size, seq_len));
  std::vector<__nv_bfloat16> h_k(kv_elements(batch_size, seq_len));
  std::vector<__nv_bfloat16> h_v(kv_elements(batch_size, seq_len));
  std::vector<__nv_bfloat16> h_q_prepared(h_q.size());
  std::vector<__nv_bfloat16> h_k_prepared(h_k.size());
  std::vector<__nv_bfloat16> h_v_prepared(h_v.size());
  std::vector<__nv_bfloat16> h_q_norm_weight(kHeadDim);
  std::vector<__nv_bfloat16> h_k_norm_weight(kHeadDim);
  std::vector<float> h_cos(rope_table_elements(seq_len));
  std::vector<float> h_sin(rope_table_elements(seq_len));
  fill_random_bf16(h_q, 0x1234u);
  fill_random_bf16(h_k, 0x2345u);
  fill_random_bf16(h_v, 0x3456u);
  fill_norm_weight_bf16(h_q_norm_weight, 0x4567u);
  fill_norm_weight_bf16(h_k_norm_weight, 0x5678u);
  fill_rope_tables(h_cos, h_sin, seq_len);
  prepare_qkv_norm_rope_host(h_q_prepared, h_k_prepared, h_v_prepared,
                             h_q, h_k, h_v, h_q_norm_weight,
                             h_k_norm_weight, h_cos, h_sin,
                             batch_size, seq_len);

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  __nv_bfloat16 *d_v = nullptr;
  __nv_bfloat16 *d_q_prepared = nullptr;
  __nv_bfloat16 *d_k_prepared = nullptr;
  __nv_bfloat16 *d_v_prepared = nullptr;
  __nv_bfloat16 *d_q_norm_weight = nullptr;
  __nv_bfloat16 *d_k_norm_weight = nullptr;
  __nv_bfloat16 *d_out = nullptr;
  float *d_cos = nullptr;
  float *d_sin = nullptr;
  float *d_lse = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_v, h_v.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_q_prepared, h_q.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_k_prepared, h_k.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_v_prepared, h_v.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_q_norm_weight,
                        h_q_norm_weight.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_k_norm_weight,
                        h_k_norm_weight.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_out, h_q.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_cos, h_cos.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_sin, h_sin.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse, size_t(batch_size) * kQHeads * seq_len * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_q_prepared, h_q_prepared.data(),
                        h_q_prepared.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_prepared, h_k_prepared.data(),
                        h_k_prepared.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_prepared, h_v_prepared.data(),
                        h_v_prepared.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_q_norm_weight, h_q_norm_weight.data(),
                        h_q_norm_weight.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_norm_weight, h_k_norm_weight.data(),
                        h_k_norm_weight.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), h_cos.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), h_sin.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  auto launch_prepared_fa = [&]() {
    CUDA_CHECK(gemma4_flash_attention_sliding_fwd_bf16(
        d_out, d_lse, d_q_prepared, d_k_prepared, d_v_prepared,
        batch_size, seq_len, seq_len, window_left, scale, stream));
  };
  auto launch_norm_rope_fa = [&]() {
    CUDA_CHECK(gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
        d_out, d_lse, d_q_prepared, d_k_prepared, d_v_prepared,
        d_q, d_k, d_v, d_q_norm_weight, d_k_norm_weight, d_cos, d_sin,
        batch_size, seq_len, seq_len, window_left, scale, stream));
  };
  const TimingStats prepared_timing =
      time_ms(launch_prepared_fa, stream, warmup, iters, trials);
  const TimingStats norm_rope_timing =
      time_ms(launch_norm_rope_fa, stream, warmup, iters, trials);
  const double tflops = sliding_attention_flops(batch_size, seq_len, window_left) /
                        (double(prepared_timing.best_ms) * 1.0e-3) / 1.0e12;
  const float overhead_ms =
      norm_rope_timing.best_ms - prepared_timing.best_ms;
  const float overhead_pct =
      prepared_timing.best_ms > 0.0f
          ? (100.0f * overhead_ms / prepared_timing.best_ms)
          : 0.0f;

  std::cout << "benchmark batch=" << batch_size
            << " seq=" << seq_len
            << " window_left=" << window_left
            << " warmup=" << warmup
            << " iters=" << iters
            << " trials=" << trials << "\n";
  std::cout << "prepared_fa best_ms=" << prepared_timing.best_ms
            << " avg_ms=" << prepared_timing.avg_ms
            << " approx_tflops_attention_only=" << tflops << "\n";
  std::cout << "norm_rope_plus_fa best_ms=" << norm_rope_timing.best_ms
            << " avg_ms=" << norm_rope_timing.avg_ms
            << " overhead_best_ms=" << overhead_ms
            << " overhead_best_pct=" << overhead_pct << "\n";
  std::cout << "kernel threads_per_block="
            << gemma4_flash_attention_sliding_threads_per_block()
            << " dynamic_smem_bytes="
            << gemma4_flash_attention_sliding_smem_bytes() << "\n";

  CUDA_CHECK(cudaFree(d_lse));
  CUDA_CHECK(cudaFree(d_sin));
  CUDA_CHECK(cudaFree(d_cos));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_k_norm_weight));
  CUDA_CHECK(cudaFree(d_q_norm_weight));
  CUDA_CHECK(cudaFree(d_v_prepared));
  CUDA_CHECK(cudaFree(d_k_prepared));
  CUDA_CHECK(cudaFree(d_q_prepared));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_q));
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const int seq_len = parse_arg(argv, argc, 1, 1024);
    const int iters = parse_arg(argv, argc, 2, 100);
    const int warmup = parse_arg(argv, argc, 3, 20);
    const int trials = parse_arg(argv, argc, 4, 3);
    const int batch_size = parse_arg(argv, argc, 5, 1);
    const int check_seq = parse_arg(argv, argc, 6, 64);
    const int window_left = GEMMA4_SLIDING_WINDOW;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (check_seq > 0) {
      run_correctness(batch_size, check_seq, window_left, stream);
    }
    run_benchmark(batch_size, seq_len, window_left, warmup, iters, trials, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception &err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  }
}
