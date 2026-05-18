#ifndef BENCH_UTILS_CUH
#define BENCH_UTILS_CUH

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

inline void cuda_check(cudaError_t status, const char *expr, const char *file,
                       int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(file) + ":" +
                             std::to_string(line) +
                             ": CUDA error for " + expr + ": " +
                             cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expr)                                                        \
  cuda_check((expr), #expr, __FILE__, __LINE__)

inline void cublas_check(cublasStatus_t status, const char *expr,
                         const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(file) + ":" +
                             std::to_string(line) +
                             ": cuBLAS error for " + expr + ": " +
                             std::to_string(status));
  }
}

#define CUBLAS_CHECK(expr)                                                      \
  cublas_check((expr), #expr, __FILE__, __LINE__)

struct TimingStats {
  float best_ms = 0.0f;
  float avg_ms = 0.0f;
};

struct DiffStats {
  float max_abs = 0.0f;
  float mean_abs = 0.0f;
  float max_rel = 0.0f;
};

template <typename Fn>
float time_ms_once(Fn &&fn, cudaStream_t stream, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms / static_cast<float>(iters);
}

template <typename Fn>
TimingStats time_ms(Fn &&fn, cudaStream_t stream, int warmup, int iters,
                    int trials) {
  TimingStats stats;
  stats.best_ms = 1.0e30f;
  for (int i = 0; i < trials; ++i) {
    const float ms = time_ms_once(fn, stream, warmup, iters);
    stats.best_ms = std::min(stats.best_ms, ms);
    stats.avg_ms += ms;
  }
  stats.avg_ms /= static_cast<float>(trials);
  return stats;
}

template <typename Fn>
float time_ms_graph_once(Fn &&fn, cudaStream_t stream, int warmup, int iters) {
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  bool capturing = false;

  try {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Capture and instantiate outside the measured region. The timed replay
    // replaces per-call CPU enqueue overhead with one graph launch amortized
    // over the captured iteration count.
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    capturing = true;
    for (int i = 0; i < iters; ++i) {
      fn();
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    capturing = false;

    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
    for (int i = 0; i < warmup; ++i) {
      CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / static_cast<float>(iters);
  } catch (...) {
    if (capturing) {
      cudaGraph_t failed_graph = nullptr;
      cudaStreamEndCapture(stream, &failed_graph);
      if (failed_graph != nullptr) {
        cudaGraphDestroy(failed_graph);
      }
    }
    if (graph_exec != nullptr) {
      cudaGraphExecDestroy(graph_exec);
    }
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
    }
    if (start != nullptr) {
      cudaEventDestroy(start);
    }
    if (stop != nullptr) {
      cudaEventDestroy(stop);
    }
    throw;
  }
}

template <typename Fn>
TimingStats time_ms_graph(Fn &&fn, cudaStream_t stream, int warmup, int iters,
                          int trials) {
  TimingStats stats;
  stats.best_ms = 1.0e30f;
  for (int i = 0; i < trials; ++i) {
    const float ms = time_ms_graph_once(fn, stream, warmup, iters);
    stats.best_ms = std::min(stats.best_ms, ms);
    stats.avg_ms += ms;
  }
  stats.avg_ms /= static_cast<float>(trials);
  return stats;
}

inline DiffStats diff_stats_bf16(const __nv_bfloat16 *lhs,
                                 const __nv_bfloat16 *rhs, int count) {
  std::vector<__nv_bfloat16> h_lhs(count);
  std::vector<__nv_bfloat16> h_rhs(count);
  CUDA_CHECK(cudaMemcpy(h_lhs.data(), lhs, count * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_rhs.data(), rhs, count * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));

  DiffStats stats;
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
