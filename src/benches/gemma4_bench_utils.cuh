#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

struct Gemma4BenchmarkContract {
  const char *benchmark = "unknown";
  const char *measurement = "kernel_only";
  const char *timing = "cuda_events_same_stream";
  const char *cache = "unspecified";
  const char *launch_overhead = "included";
  const char *aggregation = "best_and_average";
  const char *correctness = "checked_before_timing";
  const char *clock_policy = "unlocked_boost";
  const char *notes = "";
  int warmup = 0;
  int iters = 0;
  int samples = 0;
  int graph_inner_iters = 0;
};

template <typename T>
class DeviceBuffer {
 public:
  // Allocates device storage for a benchmark buffer, allowing zero-sized cases.
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ > 0) {
      CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
  }

  // Releases the device storage owned by this benchmark buffer.
  ~DeviceBuffer() {
    cudaFree(ptr_);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  // Returns the raw device pointer for APIs that do not use implicit conversion.
  T *get() { return ptr_; }

  // Returns the raw const device pointer for read-only APIs.
  const T *get() const { return ptr_; }

  // Returns the element count originally requested for this buffer.
  size_t count() const { return count_; }

  // Preserves older bench call sites that pass DeviceBuffer directly to kernels.
  operator T *() { return ptr_; }

  // Preserves older bench call sites that pass DeviceBuffer directly to readers.
  operator const T *() const { return ptr_; }

 private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

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
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    acc ^= in[i] + uint32_t(i);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
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

// Captures a one-line shell command result for best-effort benchmark metadata.
inline std::string gemma4_bench_capture_first_line(const char *command) {
  FILE *pipe = popen(command, "r");
  if (pipe == nullptr) {
    return "unavailable";
  }

  char buffer[1024] = {};
  std::string line = fgets(buffer, sizeof(buffer), pipe) != nullptr
                         ? std::string(buffer)
                         : "unavailable";
  pclose(pipe);
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
    line.pop_back();
  }
  return line.empty() ? "unavailable" : line;
}

// Prints CUDA, device, and nvidia-smi metadata needed to compare benchmark runs.
inline void gemma4_bench_print_environment(const char *benchmark_name) {
  int device = 0;
  cudaDeviceProp prop{};
  int driver = 0;
  int runtime = 0;
  char pci_bus_id[32] = {};

  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device));

  std::printf("benchmark_env name=%s cuda_device=%d gpu=\"%s\" pci_bus_id=%s "
              "compute_capability=sm_%d%d global_mem_bytes=%zu "
              "l2_cache_bytes=%d driver_cuda=%d runtime_cuda=%d\n",
              benchmark_name, device, prop.name, pci_bus_id, prop.major,
              prop.minor, static_cast<size_t>(prop.totalGlobalMem),
              prop.l2CacheSize, driver, runtime);

#if defined(GEMMA4_WEIGHT_LOAD_POLICY)
  constexpr int weight_load_policy = GEMMA4_WEIGHT_LOAD_POLICY;
#else
  constexpr int weight_load_policy = -1;
#endif

#if defined(__CUDACC_VER_MAJOR__)
  std::printf("benchmark_compile nvcc=%d.%d.%d arch_flags=compile_command "
              "weight_load_policy=%d\n",
              __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
              __CUDACC_VER_BUILD__, weight_load_policy);
#else
  std::printf("benchmark_compile nvcc=not_nvcc arch_flags=compile_command "
              "weight_load_policy=%d\n",
              weight_load_policy);
#endif

  const std::string smi = gemma4_bench_capture_first_line(
      "nvidia-smi --query-gpu=name,gpu_bus_id,driver_version,"
      "persistence_mode,ecc.mode.current,mig.mode.current,power.limit,"
      "clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu "
      "--format=csv,noheader,nounits 2>/dev/null");
  std::printf("benchmark_nvidia_smi fields=name,bus_id,driver,persistence,"
              "ecc,mig,power_limit_w,sm_clock_mhz,mem_clock_mhz,temp_c,"
              "power_draw_w,gpu_util_pct values=\"%s\"\n",
              smi.c_str());
}

// Prints the measurement contract beside timing rows so results remain auditable.
inline void gemma4_bench_print_contract(
    const Gemma4BenchmarkContract &contract) {
  std::printf("benchmark_contract name=%s measurement=%s timing=%s cache=%s "
              "launch_overhead=%s aggregation=%s correctness=%s "
              "clock_policy=%s warmup=%d iters=%d samples=%d "
              "graph_inner_iters=%d no_sleep_between_trials=true "
              "ncu_clock_control=use_--clock-control_none notes=\"%s\"\n",
              contract.benchmark, contract.measurement, contract.timing,
              contract.cache, contract.launch_overhead, contract.aggregation,
              contract.correctness, contract.clock_policy, contract.warmup,
              contract.iters, contract.samples, contract.graph_inner_iters,
              contract.notes);
}

// Flags timings in the range where event overhead and enqueue jitter can dominate.
inline void gemma4_bench_warn_if_tiny(const char *label, float ms) {
  if (ms > 0.0f && ms < 0.010f) {
    std::printf("benchmark_warning label=%s measured_ms=%.6f "
                "reason=sub_10us_timing_needs_repetition_or_ncu_check\n",
                label, ms);
  }
}

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
