#ifndef GEMMA4_BENCH_UTILS_CUH
#define GEMMA4_BENCH_UTILS_CUH

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

inline void gemma4_cuda_check(cudaError_t status, const char *expr,
                              const char *file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(file) + ":" +
                             std::to_string(line) +
                             ": CUDA error for " + expr + ": " +
                             cudaGetErrorString(status));
  }
}

#define GEMMA4_CUDA_CHECK(expr)                                                \
  gemma4_cuda_check((expr), #expr, __FILE__, __LINE__)

inline void gemma4_cublas_check(cublasStatus_t status, const char *expr,
                                const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(file) + ":" +
                             std::to_string(line) +
                             ": cuBLAS error for " + expr + ": " +
                             std::to_string(status));
  }
}

#define GEMMA4_CUBLAS_CHECK(expr)                                              \
  gemma4_cublas_check((expr), #expr, __FILE__, __LINE__)

struct Gemma4TimingStats {
  float best_ms = 0.0f;
  float avg_ms = 0.0f;
};

struct Gemma4DiffStats {
  float max_abs = 0.0f;
  float mean_abs = 0.0f;
  float max_rel = 0.0f;
};

template <typename Fn>
float gemma4_time_ms_once(Fn &&fn, cudaStream_t stream, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  GEMMA4_CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  GEMMA4_CUDA_CHECK(cudaEventCreate(&start));
  GEMMA4_CUDA_CHECK(cudaEventCreate(&stop));
  GEMMA4_CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  GEMMA4_CUDA_CHECK(cudaEventRecord(stop, stream));
  GEMMA4_CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  GEMMA4_CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  GEMMA4_CUDA_CHECK(cudaEventDestroy(start));
  GEMMA4_CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms / static_cast<float>(iters);
}

template <typename Fn>
Gemma4TimingStats gemma4_time_ms(Fn &&fn, cudaStream_t stream, int warmup,
                                 int iters, int trials) {
  Gemma4TimingStats stats;
  stats.best_ms = 1.0e30f;
  for (int i = 0; i < trials; ++i) {
    const float ms = gemma4_time_ms_once(fn, stream, warmup, iters);
    stats.best_ms = std::min(stats.best_ms, ms);
    stats.avg_ms += ms;
  }
  stats.avg_ms /= static_cast<float>(trials);
  return stats;
}

inline Gemma4DiffStats gemma4_diff_stats_bf16(const __nv_bfloat16 *lhs,
                                              const __nv_bfloat16 *rhs,
                                              int count) {
  std::vector<__nv_bfloat16> h_lhs(count);
  std::vector<__nv_bfloat16> h_rhs(count);
  GEMMA4_CUDA_CHECK(cudaMemcpy(h_lhs.data(), lhs,
                               count * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost));
  GEMMA4_CUDA_CHECK(cudaMemcpy(h_rhs.data(), rhs,
                               count * sizeof(__nv_bfloat16),
                               cudaMemcpyDeviceToHost));

  Gemma4DiffStats stats;
  double sum_abs = 0.0;
  for (int i = 0; i < count; ++i) {
    const float a = __bfloat162float(h_lhs[i]);
    const float b = __bfloat162float(h_rhs[i]);
    const float abs_diff = std::abs(a - b);
    const float denom = std::max(std::max(std::abs(a), std::abs(b)), 1.0f);
    stats.max_abs = std::max(stats.max_abs, abs_diff);
    stats.max_rel = std::max(stats.max_rel, abs_diff / denom);
    sum_abs += abs_diff;
  }
  stats.mean_abs = count > 0 ? static_cast<float>(sum_abs / double(count)) : 0.0f;
  return stats;
}

#endif
