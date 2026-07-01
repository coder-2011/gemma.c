#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/version.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
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

int parse_arg(char **argv, int argc, int index, int default_value) {
  return index < argc ? std::atoi(argv[index]) : default_value;
}

std::string parse_string_arg(char **argv,
                             int argc,
                             int index,
                             const char *default_value) {
  return index < argc ? std::string(argv[index]) : std::string(default_value);
}

constexpr size_t q_elements(int batch_size, int seq_len) {
  return size_t(batch_size) * seq_len * kQHeads * kHeadDim;
}

constexpr size_t kv_elements(int batch_size, int seq_len) {
  return size_t(batch_size) * seq_len * kKvHeads * kHeadDim;
}

constexpr size_t rope_table_elements(int seq_len) {
  return size_t(seq_len) * kRotaryHalf;
}

constexpr size_t q_index(int batch, int row, int head, int dim, int seq_len) {
  return (((size_t(batch) * seq_len + row) * kQHeads + head) * kHeadDim) + dim;
}

constexpr size_t kv_index(int batch, int row, int head, int dim, int seq_len) {
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

template <typename Fn>
TimingStats time_cuda_samples(Fn &&fn,
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
  return summarize_timing_samples(std::move(values));
}

void print_stats(const char *name, const TimingStats &stats) {
  gemma4_bench_print_timing_stats(name, stats);
}

// Runs the LibTorch SDPA baseline for raw sliding prefill tensors.
at::Tensor run_torch_prefill_sdpa(const at::Tensor &q,
                                  const at::Tensor &k,
                                  const at::Tensor &v,
                                  double scale) {
  at::Tensor q_t = q.permute({0, 2, 1, 3});
  at::Tensor k_t = k.permute({0, 2, 1, 3});
  at::Tensor v_t = v.permute({0, 2, 1, 3});
  at::Tensor out_t = at::scaled_dot_product_attention(
      q_t, k_t, v_t, std::nullopt, 0.0, true,
      std::optional<double>(scale), true);
  return out_t.permute({0, 2, 1, 3}).contiguous();
}

// Applies Gemma Q/K RMSNorm in LibTorch for the decode-prep baseline row.
at::Tensor torch_rmsnorm(const at::Tensor &x,
                         const at::Tensor &weight,
                         bool use_weight) {
  at::Tensor y = x.to(at::kFloat);
  at::Tensor mean_sq = y.square().mean(std::vector<int64_t>{-1}, true);
  y = y * at::rsqrt(mean_sq + GEMMA4_RMS_NORM_EPS);
  if (use_weight) {
    y = y * weight.to(at::kFloat).view({1, 1, kHeadDim});
  }
  return y;
}

// Applies sliding RoPE using token-position-selected cos/sin rows.
at::Tensor torch_apply_rope(const at::Tensor &x,
                            const at::Tensor &cos,
                            const at::Tensor &sin,
                            const at::Tensor &token_position) {
  at::Tensor rows = token_position.to(at::kLong);
  at::Tensor c = cos.index_select(0, rows).unsqueeze(1);
  at::Tensor s = sin.index_select(0, rows).unsqueeze(1);
  at::Tensor lo = x.slice(-1, 0, kRotaryHalf);
  at::Tensor hi = x.slice(-1, kRotaryHalf, kHeadDim);
  return at::cat({lo * c - hi * s, lo * s + hi * c}, -1).to(at::kBFloat16);
}

// Runs the LibTorch decode Q/K/V norm, RoPE, and paged-cache write baseline.
void run_torch_decode_prep(at::Tensor &q_prepared,
                           at::Tensor &cache_k,
                           at::Tensor &cache_v,
                           const at::Tensor &q,
                           const at::Tensor &k,
                           const at::Tensor &v,
                           const at::Tensor &q_norm_weight,
                           const at::Tensor &k_norm_weight,
                           const at::Tensor &cos,
                           const at::Tensor &sin,
                           const at::Tensor &token_position,
                           int batch_size,
                           int pages_per_seq,
                           int slot,
                           int page_offset) {
  at::Tensor q_out =
      torch_apply_rope(torch_rmsnorm(q, q_norm_weight, true), cos, sin,
                       token_position);
  at::Tensor k_out =
      torch_apply_rope(torch_rmsnorm(k, k_norm_weight, true), cos, sin,
                       token_position);
  at::Tensor v_out = torch_rmsnorm(v, k_norm_weight, false).to(at::kBFloat16);
  q_prepared.copy_(q_out);

  const auto long_options =
      at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
  at::Tensor batch_index = at::arange(batch_size, long_options);
  at::Tensor physical_page = batch_index * pages_per_seq + slot;
  at::Tensor offset = at::full({batch_size}, page_offset, long_options);
  using at::indexing::Slice;
  cache_k.index_put_({0, physical_page, offset, Slice(), Slice()}, k_out);
  cache_v.index_put_({0, physical_page, offset, Slice(), Slice()}, v_out);
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
    int window_size,
    float softmax_scale) {
  std::vector<__nv_bfloat16> out(q_elements(batch_size, seq_len));
  std::vector<float> scores(seq_len);

  for (int b = 0; b < batch_size; ++b) {
    for (int row = 0; row < seq_len; ++row) {
      const int col_begin = std::max(0, row - window_size + 1);
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

void run_correctness(int batch_size, int seq_len, int window_size,
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

  thrust::device_vector<__nv_bfloat16> d_q(h_q.size());
  thrust::device_vector<__nv_bfloat16> d_k(h_k.size());
  thrust::device_vector<__nv_bfloat16> d_v(h_v.size());
  thrust::device_vector<__nv_bfloat16> d_q_prepared(h_q.size());
  thrust::device_vector<__nv_bfloat16> d_k_prepared(h_k.size());
  thrust::device_vector<__nv_bfloat16> d_v_prepared(h_v.size());
  thrust::device_vector<__nv_bfloat16> d_q_norm_weight(h_q_norm_weight.size());
  thrust::device_vector<__nv_bfloat16> d_k_norm_weight(h_k_norm_weight.size());
  thrust::device_vector<__nv_bfloat16> d_out(h_q.size());
  thrust::device_vector<float> d_cos(h_cos.size());
  thrust::device_vector<float> d_sin(h_sin.size());
  thrust::device_vector<float> d_lse(size_t(batch_size) * kQHeads * seq_len);
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_q), h_q.data(), h_q.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_k), h_k.data(), h_k.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_v), h_v.data(), h_v.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_q_norm_weight), h_q_norm_weight.data(),
                        h_q_norm_weight.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_k_norm_weight), h_k_norm_weight.data(),
                        h_k_norm_weight.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_cos), h_cos.data(), h_cos.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_sin), h_sin.data(), h_sin.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
      raw_ptr(d_out), raw_ptr(d_lse), raw_ptr(d_q_prepared), raw_ptr(d_k_prepared), raw_ptr(d_v_prepared),
      raw_ptr(d_q), raw_ptr(d_k), raw_ptr(d_v), raw_ptr(d_q_norm_weight), raw_ptr(d_k_norm_weight), raw_ptr(d_cos), raw_ptr(d_sin),
      nullptr, batch_size, seq_len, seq_len, window_size, scale, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> h_out(h_q.size());
  std::vector<__nv_bfloat16> h_q_gpu(h_q.size());
  std::vector<__nv_bfloat16> h_k_gpu(h_k.size());
  std::vector<__nv_bfloat16> h_v_gpu(h_v.size());
  CUDA_CHECK(cudaMemcpy(h_out.data(), raw_ptr(d_out), h_out.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_q_gpu.data(), raw_ptr(d_q_prepared),
                        h_q_gpu.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_k_gpu.data(), raw_ptr(d_k_prepared),
                        h_k_gpu.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_v_gpu.data(), raw_ptr(d_v_prepared),
                        h_v_gpu.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  const std::vector<__nv_bfloat16> h_ref =
      reference_sliding_attention(h_q_prepared, h_k_prepared, h_v_prepared,
                                  batch_size, seq_len, window_size, scale);
  const DiffStats q_diff = diff_stats_host(h_q_gpu, h_q_prepared);
  const DiffStats k_diff = diff_stats_host(h_k_gpu, h_k_prepared);
  const DiffStats v_diff = diff_stats_host(h_v_gpu, h_v_prepared);
  const DiffStats diff = diff_stats_host(h_out, h_ref);

  CUDA_CHECK(gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
      raw_ptr(d_out), nullptr, raw_ptr(d_q_prepared), raw_ptr(d_k_prepared), raw_ptr(d_v_prepared),
      raw_ptr(d_q), raw_ptr(d_k), raw_ptr(d_v), raw_ptr(d_q_norm_weight), raw_ptr(d_k_norm_weight), raw_ptr(d_cos), raw_ptr(d_sin),
      nullptr, batch_size, seq_len, seq_len, window_size, scale, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<__nv_bfloat16> h_out_no_lse(h_q.size());
  CUDA_CHECK(cudaMemcpy(h_out_no_lse.data(), raw_ptr(d_out),
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
}

double sliding_attention_flops(int batch_size, int seq_len, int window_size) {
  double key_count = 0.0;
  for (int row = 0; row < seq_len; ++row) {
    key_count += double(row - std::max(0, row - window_size + 1) + 1);
  }
  return double(batch_size) * kQHeads * key_count * double(4 * kHeadDim);
}

void run_benchmark(int batch_size, int seq_len, int window_size, int warmup,
                   int iters, int samples, const std::string &cache_mode,
                   int64_t flush_bytes, cudaStream_t stream) {
  if (cache_mode != "warm" && cache_mode != "cold") {
    throw std::runtime_error("cache mode must be warm or cold");
  }
  if (batch_size <= 0 || seq_len <= 0 || window_size <= 0 || warmup < 0 ||
      iters <= 0 || samples <= 0) {
    throw std::runtime_error("benchmark dimensions/counts must be positive");
  }
  const bool cold_cache = cache_mode == "cold";
  const float scale = 1.0f / std::sqrt(float(kHeadDim));
  std::vector<__nv_bfloat16> h_q(q_elements(batch_size, seq_len));
  std::vector<__nv_bfloat16> h_k(kv_elements(batch_size, seq_len));
  std::vector<__nv_bfloat16> h_v(kv_elements(batch_size, seq_len));
  std::vector<__nv_bfloat16> h_decode_q(q_elements(batch_size, 1));
  std::vector<__nv_bfloat16> h_decode_k(kv_elements(batch_size, 1));
  std::vector<__nv_bfloat16> h_decode_v(kv_elements(batch_size, 1));
  std::vector<__nv_bfloat16> h_q_norm_weight(kHeadDim);
  std::vector<__nv_bfloat16> h_k_norm_weight(kHeadDim);
  std::vector<float> h_cos(rope_table_elements(seq_len));
  std::vector<float> h_sin(rope_table_elements(seq_len));
  fill_random_bf16(h_q, 0x1234u);
  fill_random_bf16(h_k, 0x2345u);
  fill_random_bf16(h_v, 0x3456u);
  fill_random_bf16(h_decode_q, 0x1357u);
  fill_random_bf16(h_decode_k, 0x2468u);
  fill_random_bf16(h_decode_v, 0x3579u);
  fill_norm_weight_bf16(h_q_norm_weight, 0x4567u);
  fill_norm_weight_bf16(h_k_norm_weight, 0x5678u);
  fill_rope_tables(h_cos, h_sin, seq_len);

  const int decode_page_size = 64;
  const int decode_pages_per_seq =
      std::max(1, (seq_len + decode_page_size - 1) / decode_page_size);
  const Gemma4KvCacheConfig decode_cache_config = {
      1,
      batch_size * decode_pages_per_seq,
      decode_page_size,
      decode_pages_per_seq,
      batch_size,
      kKvHeads,
      kHeadDim,
      GEMMA4_SLIDING_WINDOW,
  };
  std::vector<int32_t> h_decode_token_position(batch_size, seq_len - 1);
  std::vector<int32_t> h_decode_page_table(
      batch_size * decode_pages_per_seq, -1);
  const int decode_slot = ((seq_len - 1) / decode_page_size) % decode_pages_per_seq;
  const int decode_page_offset = (seq_len - 1) % decode_page_size;
  // Decode attention scans the whole live window, so every logical page needs
  // a valid mapping rather than only the final cache-write page.
  for (int b = 0; b < batch_size; ++b) {
    for (int slot = 0; slot < decode_pages_per_seq; ++slot) {
      h_decode_page_table[b * decode_pages_per_seq + slot] =
          b * decode_pages_per_seq + slot;
    }
  }
  const size_t decode_cache_elems =
      size_t(decode_cache_config.num_layers) * decode_cache_config.num_pages *
      decode_cache_config.page_size * decode_cache_config.num_heads *
      decode_cache_config.head_dim;
  std::vector<__nv_bfloat16> h_decode_cache_k(decode_cache_elems);
  std::vector<__nv_bfloat16> h_decode_cache_v(decode_cache_elems);
  std::vector<int32_t> h_decode_seq_lengths(batch_size, seq_len);
  fill_random_bf16(h_decode_cache_k, 0x468au);
  fill_random_bf16(h_decode_cache_v, 0x579bu);
  const int decode_split_size = GEMMA4_SLIDING_DECODE_SPLIT_SIZE;
  // The decode launcher validates capacity against the full sliding window;
  // rows shorter than this cheaply exit overprovisioned split CTAs.
  const int decode_num_splits =
      (GEMMA4_SLIDING_WINDOW + decode_split_size - 1) / decode_split_size;
  const size_t decode_out_elems = q_elements(batch_size, 1);
  const size_t decode_partial_rows = size_t(batch_size) * kQHeads *
                                     decode_num_splits;
  int64_t l2_flush_words = 0;
  if (cold_cache) {
    flush_bytes = flush_bytes > 0 ? flush_bytes : int64_t(64) * 1024 * 1024;
    l2_flush_words = flush_bytes / int64_t(sizeof(uint32_t));
  }
  thrust::device_vector<__nv_bfloat16> d_q(h_q.size());
  thrust::device_vector<__nv_bfloat16> d_k(h_k.size());
  thrust::device_vector<__nv_bfloat16> d_v(h_v.size());
  thrust::device_vector<__nv_bfloat16> d_decode_q(h_decode_q.size());
  thrust::device_vector<__nv_bfloat16> d_decode_k(h_decode_k.size());
  thrust::device_vector<__nv_bfloat16> d_decode_v(h_decode_v.size());
  thrust::device_vector<__nv_bfloat16> d_q_prepared(h_q.size());
  thrust::device_vector<__nv_bfloat16> d_k_prepared(h_k.size());
  thrust::device_vector<__nv_bfloat16> d_v_prepared(h_v.size());
  thrust::device_vector<__nv_bfloat16> d_decode_q_prepared(h_decode_q.size());
  thrust::device_vector<__nv_bfloat16> d_decode_cache_k(decode_cache_elems);
  thrust::device_vector<__nv_bfloat16> d_decode_cache_v(decode_cache_elems);
  thrust::device_vector<__nv_bfloat16> d_decode_out(decode_out_elems);
  thrust::device_vector<float> d_decode_partial_m(decode_partial_rows);
  thrust::device_vector<float> d_decode_partial_l(decode_partial_rows);
  thrust::device_vector<float> d_decode_partial_acc(
      decode_partial_rows * kHeadDim);
  thrust::device_vector<__nv_bfloat16> d_q_norm_weight(h_q_norm_weight.size());
  thrust::device_vector<__nv_bfloat16> d_k_norm_weight(h_k_norm_weight.size());
  thrust::device_vector<__nv_bfloat16> d_out(h_q.size());
  thrust::device_vector<int32_t> d_decode_page_table(h_decode_page_table.size());
  thrust::device_vector<int32_t> d_decode_token_position(
      h_decode_token_position.size());
  thrust::device_vector<int32_t> d_decode_seq_lengths(
      h_decode_seq_lengths.size());
  thrust::device_vector<float> d_cos(h_cos.size());
  thrust::device_vector<float> d_sin(h_sin.size());
  thrust::device_vector<uint32_t> d_l2_scratch(static_cast<size_t>(l2_flush_words));
  if (cold_cache) {
    CUDA_CHECK(cudaMemset(raw_ptr(d_l2_scratch), 0,
                          d_l2_scratch.size() * sizeof(uint32_t)));
  }
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_q), h_q.data(), h_q.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_k), h_k.data(), h_k.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_v), h_v.data(), h_v.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_decode_q), h_decode_q.data(),
                        h_decode_q.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_decode_k), h_decode_k.data(),
                        h_decode_k.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_decode_v), h_decode_v.data(),
                        h_decode_v.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_decode_cache_k), h_decode_cache_k.data(),
                        h_decode_cache_k.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_decode_cache_v), h_decode_cache_v.data(),
                        h_decode_cache_v.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_q_norm_weight), h_q_norm_weight.data(),
                        h_q_norm_weight.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_k_norm_weight), h_k_norm_weight.data(),
                        h_k_norm_weight.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_cos), h_cos.data(), h_cos.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_sin), h_sin.data(), h_sin.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_decode_page_table), h_decode_page_table.data(),
                        h_decode_page_table.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_decode_token_position), h_decode_token_position.data(),
                        h_decode_token_position.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(raw_ptr(d_decode_seq_lengths),
                        h_decode_seq_lengths.data(),
                        h_decode_seq_lengths.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

  if (seq_len <= window_size) {
    c10::cuda::CUDAGuard torch_device_guard(0);
    const c10::cuda::CUDAStream torch_stream =
        c10::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard torch_stream_guard(torch_stream);
    at::manual_seed(1235u);
    const auto torch_options =
        at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    at::Tensor torch_q =
        at::randn({batch_size, seq_len, kQHeads, kHeadDim}, torch_options);
    at::Tensor torch_k =
        at::randn({batch_size, seq_len, kKvHeads, kHeadDim}, torch_options);
    at::Tensor torch_v =
        at::randn({batch_size, seq_len, kKvHeads, kHeadDim}, torch_options);
    at::Tensor torch_out;
    auto launch_torch_prefill = [&]() {
      torch_out = run_torch_prefill_sdpa(torch_q, torch_k, torch_v, scale);
    };
    launch_torch_prefill();
    CUDA_CHECK(cudaStreamSynchronize(torch_stream.stream()));
    std::cout << "benchmark_libtorch torch_version=" << TORCH_VERSION
              << "\n";
    print_stats("prefill_torch_sdpa_graphless",
                time_cuda_samples(launch_torch_prefill, torch_stream.stream(),
                                  warmup, iters, samples, cold_cache,
                                  raw_ptr(d_l2_scratch), l2_flush_words));
  } else {
    std::cout << "prefill_torch_sdpa_graphless skipped=seq_exceeds_window\n";
  }

  {
    c10::cuda::CUDAGuard torch_device_guard(0);
    const c10::cuda::CUDAStream torch_stream =
        c10::cuda::getStreamFromPool(false, 0);
    c10::cuda::CUDAStreamGuard torch_stream_guard(torch_stream);
    at::manual_seed(1234u);
    const auto torch_options =
        at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    const auto int_options =
        at::TensorOptions().device(at::kCUDA).dtype(at::kInt);
    at::Tensor torch_decode_q =
        at::randn({batch_size, kQHeads, kHeadDim}, torch_options);
    at::Tensor torch_decode_k =
        at::randn({batch_size, kKvHeads, kHeadDim}, torch_options);
    at::Tensor torch_decode_v =
        at::randn({batch_size, kKvHeads, kHeadDim}, torch_options);
    at::Tensor torch_q_weight =
        at::randn({kHeadDim}, torch_options) * 0.05 + 0.95;
    at::Tensor torch_k_weight =
        at::randn({kHeadDim}, torch_options) * 0.05 + 0.95;
    at::Tensor torch_cos =
        at::from_blob(h_cos.data(), {seq_len, kRotaryHalf},
                      at::TensorOptions().dtype(at::kFloat))
            .clone()
            .to(at::kCUDA);
    at::Tensor torch_sin =
        at::from_blob(h_sin.data(), {seq_len, kRotaryHalf},
                      at::TensorOptions().dtype(at::kFloat))
            .clone()
            .to(at::kCUDA);
    at::Tensor torch_token_position =
        at::full({batch_size}, seq_len - 1, int_options);
    at::Tensor torch_q_prepared =
        at::empty({batch_size, kQHeads, kHeadDim}, torch_options);
    at::Tensor torch_cache_k =
        at::zeros({1, batch_size * decode_pages_per_seq, decode_page_size,
                   kKvHeads, kHeadDim},
                  torch_options);
    at::Tensor torch_cache_v = at::zeros_like(torch_cache_k);
    auto launch_torch_decode_prep = [&]() {
      run_torch_decode_prep(torch_q_prepared, torch_cache_k, torch_cache_v,
                            torch_decode_q, torch_decode_k, torch_decode_v,
                            torch_q_weight, torch_k_weight, torch_cos,
                            torch_sin, torch_token_position, batch_size,
                            decode_pages_per_seq, decode_slot,
                            decode_page_offset);
    };
    launch_torch_decode_prep();
    CUDA_CHECK(cudaStreamSynchronize(torch_stream.stream()));
    print_stats("torch_decode_norm_rope_paged_kv_write",
                time_cuda_samples(launch_torch_decode_prep,
                                  torch_stream.stream(), warmup, iters,
                                  samples, cold_cache, raw_ptr(d_l2_scratch),
                                  l2_flush_words));
  }

  auto launch_norm_rope_fa = [&]() {
    CUDA_CHECK(gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
      raw_ptr(d_out), nullptr, raw_ptr(d_q_prepared), raw_ptr(d_k_prepared), raw_ptr(d_v_prepared),
      raw_ptr(d_q), raw_ptr(d_k), raw_ptr(d_v), raw_ptr(d_q_norm_weight), raw_ptr(d_k_norm_weight), raw_ptr(d_cos), raw_ptr(d_sin),
      nullptr, batch_size, seq_len, seq_len, window_size, scale, stream));
  };
  const TimingStats norm_rope_timing =
      time_cuda_samples(launch_norm_rope_fa, stream, warmup, iters, samples,
                        cold_cache, raw_ptr(d_l2_scratch), l2_flush_words);
  const double tflops = sliding_attention_flops(batch_size, seq_len, window_size) /
                        (double(norm_rope_timing.median_ms) * 1.0e-3) / 1.0e12;

  auto launch_decode_prep_cache = [&]() {
    CUDA_CHECK(gemma4_flash_attention_decode_prepare_q_paged_kv_bf16(
        raw_ptr(d_decode_q_prepared), raw_ptr(d_decode_cache_k),
        raw_ptr(d_decode_cache_v), decode_cache_config,
        raw_ptr(d_decode_page_table), raw_ptr(d_decode_token_position),
        batch_size, 0, raw_ptr(d_decode_q), raw_ptr(d_decode_k),
        raw_ptr(d_decode_v), raw_ptr(d_q_norm_weight),
        raw_ptr(d_k_norm_weight), raw_ptr(d_cos), raw_ptr(d_sin), stream));
  };
  const TimingStats decode_prep_cache_timing =
      time_cuda_samples(launch_decode_prep_cache, stream, warmup, iters,
                        samples, cold_cache, raw_ptr(d_l2_scratch), l2_flush_words);
  auto launch_decode_attention = [&]() {
    CUDA_CHECK(gemma4_flash_attention_decode_paged_bf16(
        raw_ptr(d_decode_out), raw_ptr(d_decode_partial_m),
        raw_ptr(d_decode_partial_l), raw_ptr(d_decode_partial_acc),
        raw_ptr(d_decode_q_prepared), raw_ptr(d_decode_cache_k),
        raw_ptr(d_decode_cache_v), raw_ptr(d_decode_page_table),
        raw_ptr(d_decode_seq_lengths), decode_cache_config, 0, batch_size,
        scale, decode_split_size, decode_num_splits, stream));
  };
  const TimingStats decode_attention_timing =
      time_cuda_samples(launch_decode_attention, stream, warmup, iters,
                        samples, cold_cache, raw_ptr(d_l2_scratch),
                        l2_flush_words);

  std::cout << "benchmark_contract name=flash_attention_bench"
            << " measurement=sliding_prefill_decode_prepare_decode_attention"
            << " timing=cuda_events_same_stream"
            << " cache_mode=" << cache_mode
            << " l2_flush_bytes=" << (cold_cache ? flush_bytes : 0)
            << " launch_overhead=queued_launches_only"
            << " host_wall_time=excluded"
            << " aggregation=raw_samples"
            << " correctness=cpp_reference"
            << " warmup=" << warmup
            << " iters_per_sample=" << iters
            << " samples=" << samples
            << "\n";
  std::cout << "benchmark batch=" << batch_size
            << " seq=" << seq_len
            << " window_size=" << window_size
            << " warmup=" << warmup
            << " iters=" << iters
            << " samples=" << samples
            << " cache=" << cache_mode
            << " flush_bytes=" << (cold_cache ? flush_bytes : 0)
            << " timing=cuda_events_same_stream"
            << " launch_overhead=included\n";
  print_stats("norm_rope_plus_fa", norm_rope_timing);
  print_stats("decode_norm_rope_paged_kv_write", decode_prep_cache_timing);
  print_stats("decode_paged_attention", decode_attention_timing);
  std::cout << "approx_attention_tflops_in_total_path_median=" << tflops << "\n";

}

}  // namespace

int main(int argc, char **argv) {
  try {
    const int seq_len = parse_arg(argv, argc, 1, 1024);
    const int iters = parse_arg(argv, argc, 2, 100);
    const int warmup = parse_arg(argv, argc, 3, 20);
    const int samples = parse_arg(argv, argc, 4, 15);
    const int batch_size = parse_arg(argv, argc, 5, 1);
    const int check_seq = parse_arg(argv, argc, 6, 64);
    const std::string cache_mode = parse_string_arg(argv, argc, 7, "warm");
    const int flush_mib = parse_arg(argv, argc, 8, 64);
    constexpr int window_size = GEMMA4_SLIDING_WINDOW;

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    gemma4_bench_print_common_metadata("flash_attention_bench");
    if (check_seq > 0) {
      run_correctness(batch_size, check_seq, window_size, stream);
    }
    run_benchmark(batch_size, seq_len, window_size, warmup, iters, samples,
                  cache_mode, int64_t(flush_mib) * 1024 * 1024, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception &err) {
    std::cerr << "error: " << err.what() << "\n";
    return 1;
  }
}
