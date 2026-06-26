#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

inline void gemma4_bench_cuda_check(cudaError_t status, const char *expr,
                                    const char *file, int line) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             ": CUDA error for " + expr + ": " +
                             cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expr)                                                        \
  gemma4_bench_cuda_check((expr), #expr, __FILE__, __LINE__)

inline void gemma4_bench_cublas_check(cublasStatus_t status, const char *expr,
                                      const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             ": cuBLAS error for " + expr + ": " +
                             std::to_string(status));
  }
}

#define CUBLAS_CHECK(expr)                                                      \
  gemma4_bench_cublas_check((expr), #expr, __FILE__, __LINE__)

struct TimingStats {
  float best_ms = 0.0f;
  float avg_ms = 0.0f;
};

struct DiffStats {
  float max_abs = 0.0f;
  float mean_abs = 0.0f;
  float max_rel = 0.0f;
};

// Returns the raw CUDA pointer owned by a Thrust device vector.
template <typename T>
T *raw_ptr(thrust::device_vector<T> &v) {
  return thrust::raw_pointer_cast(v.data());
}

// Returns the raw const CUDA pointer owned by a Thrust device vector.
template <typename T>
const T *raw_ptr(const thrust::device_vector<T> &v) {
  return thrust::raw_pointer_cast(v.data());
}

// Mixes an integer index and seed into deterministic pseudo-random bits.
static __device__ inline uint32_t gemma4_bench_mix_u32_device(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

// Fills BF16 device memory with deterministic values in [-scale, scale].
static __global__ void gemma4_bench_fill_random_bf16_kernel(
    __nv_bfloat16 *ptr,
    size_t count,
    uint64_t seed,
    float scale) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }

  uint32_t x = uint32_t(i) ^ uint32_t(i >> 32) ^ uint32_t(seed) ^
               uint32_t(seed >> 32);
  x = gemma4_bench_mix_u32_device(x);
  const float u = float(x >> 8) * (1.0f / 16777216.0f);
  ptr[i] = __float2bfloat16_rn((u * 2.0f - 1.0f) * scale);
}

// Launches the deterministic BF16 filler outside benchmark timing windows.
inline void fill_random_bf16(__nv_bfloat16 *ptr,
                             size_t count,
                             uint64_t seed,
                             float scale,
                             cudaStream_t stream) {
  if (count == 0) {
    return;
  }

  constexpr int threads = 256;
  const int blocks = int((count + threads - 1) / threads);
  gemma4_bench_fill_random_bf16_kernel<<<blocks, threads, 0, stream>>>(
      ptr, count, seed, scale);
  CUDA_CHECK(cudaGetLastError());
}

// Streams a scratch buffer through L2 so cold-cache bench regions are explicit.
static __global__ void gemma4_bench_flush_cache_kernel(
    const uint32_t *__restrict__ in,
    uint32_t *__restrict__ out,
    size_t count) {
  uint32_t acc = 0;
  const size_t thread_index = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  const size_t stride = size_t(blockDim.x) * gridDim.x;
  for (size_t i = thread_index; i < count; i += stride) {
    acc ^= in[i] + uint32_t(i);
  }
  out[thread_index] = acc;
}

// Launches the fixed-shape cache flush used by cold-cache microbenchmarks.
inline void flush_cache(const uint32_t *in,
                        uint32_t *out,
                        size_t count,
                        cudaStream_t stream) {
  if (count == 0) {
    return;
  }

  constexpr int threads = 256;
  constexpr int blocks = 4096;
  gemma4_bench_flush_cache_kernel<<<blocks, threads, 0, stream>>>(
      in, out, count);
  CUDA_CHECK(cudaGetLastError());
}

// Converts bytes and elapsed milliseconds into effective GiB/s.
inline double gib_per_second(double bytes, float ms) {
  const double gib = bytes / (1024.0 * 1024.0 * 1024.0);
  return gib / (static_cast<double>(ms) / 1000.0);
}

// Builds a benchmark seed from an env override or host entropy.
inline uint64_t make_seed(const char *env_name) {
  if (const char *env = std::getenv(env_name)) {
    return std::strtoull(env, nullptr, 0);
  }

  // Combine random_device with the clock so repeated short runs get varied seeds.
  std::random_device rd;
  uint64_t seed = uint64_t(rd()) << 32;
  seed ^= uint64_t(rd());
  seed ^= uint64_t(std::chrono::high_resolution_clock::now()
                       .time_since_epoch()
                       .count());
  return seed;
}

// Flags timings in the range where event overhead and enqueue jitter can dominate.
inline void gemma4_bench_warn_if_tiny(const char *label, float ms) {
  if (ms > 0.0f && ms < 0.010f) {
    std::printf("benchmark_warning label=%s measured_ms=%.6f "
                "reason=sub_10us_timing_needs_repetition_or_ncu_check\n",
                label, ms);
  }
}

// Times repeated CUDA work with events on the same stream as the work.
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
  stats.best_ms = INFINITY;
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
  stats.best_ms = INFINITY;
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
