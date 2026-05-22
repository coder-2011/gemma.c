#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "gemma4.h"
#include "gemma4_bench_utils.cuh"

extern "C" cudaError_t gemma4_flash_attention_sliding_fwd_bf16(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_softmax_lse,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    int batch_size,
    int seqlen_q,
    int seqlen_k,
    int window_left,
    float softmax_scale,
    cudaStream_t stream);

extern "C" size_t gemma4_flash_attention_sliding_smem_bytes();
extern "C" int gemma4_flash_attention_sliding_threads_per_block();

namespace {

constexpr int kQHeads = GEMMA4_NUM_QUERY_HEADS;
constexpr int kKvHeads = GEMMA4_SLIDING_KV_HEADS;
constexpr int kHeadDim = GEMMA4_SLIDING_HEAD_DIM;
constexpr int kGqaRatio = kQHeads / kKvHeads;

int parse_arg(char **argv, int argc, int index, int default_value) {
  return index < argc ? std::atoi(argv[index]) : default_value;
}

size_t q_elements(int batch_size, int seq_len) {
  return size_t(batch_size) * seq_len * kQHeads * kHeadDim;
}

size_t kv_elements(int batch_size, int seq_len) {
  return size_t(batch_size) * seq_len * kKvHeads * kHeadDim;
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
  fill_random_bf16(h_q, 0x4a17u);
  fill_random_bf16(h_k, 0x5b23u);
  fill_random_bf16(h_v, 0x6c31u);

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  __nv_bfloat16 *d_v = nullptr;
  __nv_bfloat16 *d_out = nullptr;
  float *d_lse = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_v, h_v.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_out, h_q.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_lse, size_t(batch_size) * kQHeads * seq_len * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(gemma4_flash_attention_sliding_fwd_bf16(
      d_out, d_lse, d_q, d_k, d_v, batch_size, seq_len, seq_len,
      window_left, scale, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> h_out(h_q.size());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  std::vector<__nv_bfloat16> h_ref =
      reference_sliding_attention(h_q, h_k, h_v, batch_size, seq_len,
                                  window_left, scale);
  const DiffStats diff = diff_stats_host(h_out, h_ref);
  std::cout << "correctness seq=" << seq_len
            << " max_abs=" << diff.max_abs
            << " mean_abs=" << diff.mean_abs
            << " max_rel=" << diff.max_rel << "\n";

  CUDA_CHECK(cudaFree(d_lse));
  CUDA_CHECK(cudaFree(d_out));
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
  fill_random_bf16(h_q, 0x1234u);
  fill_random_bf16(h_k, 0x2345u);
  fill_random_bf16(h_v, 0x3456u);

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  __nv_bfloat16 *d_v = nullptr;
  __nv_bfloat16 *d_out = nullptr;
  float *d_lse = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_v, h_v.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_out, h_q.size() * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_lse, size_t(batch_size) * kQHeads * seq_len * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));

  auto launch = [&]() {
    CUDA_CHECK(gemma4_flash_attention_sliding_fwd_bf16(
        d_out, d_lse, d_q, d_k, d_v, batch_size, seq_len, seq_len,
        window_left, scale, stream));
  };
  const TimingStats timing = time_ms(launch, stream, warmup, iters, trials);
  const double tflops = sliding_attention_flops(batch_size, seq_len, window_left) /
                        (double(timing.best_ms) * 1.0e-3) / 1.0e12;

  std::cout << "benchmark batch=" << batch_size
            << " seq=" << seq_len
            << " window_left=" << window_left
            << " warmup=" << warmup
            << " iters=" << iters
            << " trials=" << trials << "\n";
  std::cout << "custom_fa2_derived best_ms=" << timing.best_ms
            << " avg_ms=" << timing.avg_ms
            << " approx_tflops=" << tflops << "\n";
  std::cout << "kernel threads_per_block="
            << gemma4_flash_attention_sliding_threads_per_block()
            << " dynamic_smem_bytes="
            << gemma4_flash_attention_sliding_smem_bytes() << "\n";

  CUDA_CHECK(cudaFree(d_lse));
  CUDA_CHECK(cudaFree(d_out));
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
