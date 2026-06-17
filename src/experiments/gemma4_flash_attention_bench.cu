#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
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

struct SampleStats {
  float median_ms = 0.0f;
  float mean_ms = 0.0f;
  float trimmed_mean_ms = 0.0f;
  float min_ms = 0.0f;
  float max_ms = 0.0f;
  float p95_ms = 0.0f;
  float p99_ms = 0.0f;
  float stddev_ms = 0.0f;
  float iqr_ms = 0.0f;
  std::vector<float> samples_ms;
};

int parse_arg(char **argv, int argc, int index, int default_value) {
  return index < argc ? std::atoi(argv[index]) : default_value;
}

std::string parse_string_arg(char **argv,
                             int argc,
                             int index,
                             const char *default_value) {
  return index < argc ? std::string(argv[index]) : std::string(default_value);
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

__global__ void l2_flush_kernel(uint32_t *__restrict__ scratch,
                                int64_t words) {
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
       i += stride) {
    scratch[i] += 1u;
  }
}

void flush_l2(cudaStream_t stream,
              uint32_t *__restrict__ d_scratch,
              int64_t words) {
  if (d_scratch == nullptr || words <= 0) return;
  constexpr int kThreads = 256;
  const int64_t blocks64 = (words + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(std::min<int64_t>(4096, blocks64));
  l2_flush_kernel<<<blocks, kThreads, 0, stream>>>(d_scratch, words);
  CUDA_CHECK(cudaGetLastError());
}

float percentile(const std::vector<float> &sorted, float pct) {
  if (sorted.empty()) return 0.0f;
  if (sorted.size() == 1) return sorted.front();
  const float index = pct * 0.01f * float(sorted.size() - 1);
  const int lo = int(std::floor(index));
  const int hi = int(std::ceil(index));
  const float frac = index - float(lo);
  return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

float trimmed_mean(const std::vector<float> &sorted, float trim_fraction) {
  if (sorted.empty()) return 0.0f;
  const int trim = int(std::floor(sorted.size() * trim_fraction));
  const int begin = std::min<int>(trim, sorted.size() - 1);
  const int end = std::max<int>(begin + 1, sorted.size() - trim);
  const float sum =
      std::accumulate(sorted.begin() + begin, sorted.begin() + end, 0.0f);
  return sum / float(end - begin);
}

SampleStats summarize_samples(std::vector<float> values) {
  std::vector<float> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  SampleStats stats;
  stats.samples_ms = std::move(values);
  stats.median_ms = percentile(sorted, 50.0f);
  stats.mean_ms = std::accumulate(stats.samples_ms.begin(),
                                  stats.samples_ms.end(), 0.0f) /
                  float(stats.samples_ms.size());
  stats.trimmed_mean_ms = trimmed_mean(sorted, 0.1f);
  stats.min_ms = sorted.front();
  stats.max_ms = sorted.back();
  stats.p95_ms = percentile(sorted, 95.0f);
  stats.p99_ms = percentile(sorted, 99.0f);
  stats.iqr_ms = percentile(sorted, 75.0f) - percentile(sorted, 25.0f);
  float variance = 0.0f;
  for (float sample : stats.samples_ms) {
    const float diff = sample - stats.mean_ms;
    variance += diff * diff;
  }
  variance /= float(stats.samples_ms.size());
  stats.stddev_ms = std::sqrt(variance);
  return stats;
}

template <typename Fn>
SampleStats time_cuda_samples(Fn &&fn,
                              cudaStream_t stream,
                              int warmup,
                              int iters_per_sample,
                              int samples,
                              bool cold_cache,
                              uint32_t *__restrict__ d_l2_scratch,
                              int64_t l2_flush_words) {
  for (int i = 0; i < warmup; ++i) {
    if (cold_cache) flush_l2(stream, d_l2_scratch, l2_flush_words);
    fn();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> values(samples);
  for (int sample = 0; sample < samples; ++sample) {
    if (cold_cache) {
      float total_ms = 0.0f;
      for (int i = 0; i < iters_per_sample; ++i) {
        flush_l2(stream, d_l2_scratch, l2_flush_words);
        CUDA_CHECK(cudaEventRecord(start, stream));
        fn();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float iter_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms += iter_ms;
      }
      values[sample] = total_ms / float(iters_per_sample);
    } else {
      CUDA_CHECK(cudaEventRecord(start, stream));
      for (int i = 0; i < iters_per_sample; ++i) {
        fn();
      }
      CUDA_CHECK(cudaEventRecord(stop, stream));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float total_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
      values[sample] = total_ms / float(iters_per_sample);
    }
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return summarize_samples(std::move(values));
}

void print_stats(const char *name, const SampleStats &stats) {
  std::cout << name << " median_ms=" << stats.median_ms
            << " mean_ms=" << stats.mean_ms
            << " trimmed_mean_ms=" << stats.trimmed_mean_ms
            << " min_ms=" << stats.min_ms
            << " max_ms=" << stats.max_ms
            << " p95_ms=" << stats.p95_ms
            << " p99_ms=" << stats.p99_ms
            << " stddev_ms=" << stats.stddev_ms
            << " iqr_ms=" << stats.iqr_ms
            << " samples_ms=[";
  for (size_t i = 0; i < stats.samples_ms.size(); ++i) {
    std::cout << (i == 0 ? "" : ",") << stats.samples_ms[i];
  }
  std::cout << "]\n";
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

  CUDA_CHECK(gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
      d_out, nullptr, d_q_prepared, d_k_prepared, d_v_prepared,
      d_q, d_k, d_v, d_q_norm_weight, d_k_norm_weight, d_cos, d_sin,
      batch_size, seq_len, seq_len, window_left, scale, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__nv_bfloat16> h_out_no_lse(h_q.size());
  CUDA_CHECK(cudaMemcpy(h_out_no_lse.data(), d_out,
                        h_out_no_lse.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  const DiffStats no_lse_diff = diff_stats_host(h_out_no_lse, h_ref);

  std::cout << "correctness seq=" << seq_len
            << " max_abs=" << diff.max_abs
            << " mean_abs=" << diff.mean_abs
            << " max_rel=" << diff.max_rel << "\n";
  std::cout << "no_lse_correctness max_abs=" << no_lse_diff.max_abs
            << " mean_abs=" << no_lse_diff.mean_abs
            << " max_rel=" << no_lse_diff.max_rel << "\n";
  std::cout << "norm_rope_prep_correctness q_max_abs=" << q_diff.max_abs
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
                   int iters, int samples, const std::string &cache_mode,
                   int64_t flush_bytes, cudaStream_t stream) {
  if (cache_mode != "warm" && cache_mode != "cold") {
    throw std::runtime_error("cache mode must be warm or cold");
  }
  const bool cold_cache = cache_mode == "cold";
  const float scale = 1.0f / std::sqrt(float(kHeadDim));
  std::vector<__nv_bfloat16> h_q(q_elements(batch_size, seq_len));
  std::vector<__nv_bfloat16> h_k(kv_elements(batch_size, seq_len));
  std::vector<__nv_bfloat16> h_v(kv_elements(batch_size, seq_len));
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
  uint32_t *d_l2_scratch = nullptr;
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
  int64_t l2_flush_words = 0;
  if (cold_cache) {
    flush_bytes = flush_bytes > 0 ? flush_bytes : int64_t(64) * 1024 * 1024;
    l2_flush_words = flush_bytes / int64_t(sizeof(uint32_t));
    CUDA_CHECK(cudaMalloc(&d_l2_scratch, size_t(l2_flush_words) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_l2_scratch, 0, size_t(l2_flush_words) * sizeof(uint32_t)));
  }
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

  auto launch_norm_rope_fa = [&]() {
    CUDA_CHECK(gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
        d_out, nullptr, d_q_prepared, d_k_prepared, d_v_prepared,
        d_q, d_k, d_v, d_q_norm_weight, d_k_norm_weight, d_cos, d_sin,
        batch_size, seq_len, seq_len, window_left, scale, stream));
  };
  const SampleStats norm_rope_timing =
      time_cuda_samples(launch_norm_rope_fa, stream, warmup, iters, samples,
                        cold_cache, d_l2_scratch, l2_flush_words);
  const double tflops = sliding_attention_flops(batch_size, seq_len, window_left) /
                        (double(norm_rope_timing.median_ms) * 1.0e-3) / 1.0e12;

  std::cout << "benchmark batch=" << batch_size
            << " seq=" << seq_len
            << " window_left=" << window_left
            << " warmup=" << warmup
            << " iters=" << iters
            << " samples=" << samples
            << " cache=" << cache_mode
            << " flush_bytes=" << (cold_cache ? flush_bytes : 0)
            << " timing=cuda_events_same_stream"
            << " return_lse=false"
            << " launch_overhead=included\n";
  print_stats("norm_rope_plus_fa", norm_rope_timing);
  std::cout << "approx_attention_tflops_in_total_path_median=" << tflops << "\n";
  std::cout << "kernel threads_per_block="
            << gemma4_flash_attention_sliding_threads_per_block()
            << " dynamic_smem_bytes="
            << gemma4_flash_attention_sliding_smem_bytes() << "\n";

  if (d_l2_scratch != nullptr) CUDA_CHECK(cudaFree(d_l2_scratch));
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
    const int samples = parse_arg(argv, argc, 4, 3);
    const int batch_size = parse_arg(argv, argc, 5, 1);
    const int check_seq = parse_arg(argv, argc, 6, 64);
    const std::string cache_mode = parse_string_arg(argv, argc, 7, "warm");
    const int flush_mib = parse_arg(argv, argc, 8, 64);
    const int window_left = GEMMA4_SLIDING_WINDOW;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (check_seq > 0) {
      run_correctness(batch_size, check_seq, window_left, stream);
    }
    run_benchmark(batch_size, seq_len, window_left, warmup, iters, samples,
                  cache_mode, int64_t(flush_mib) * 1024 * 1024, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception &err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  }
}
