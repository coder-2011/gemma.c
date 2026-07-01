#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
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
  float median_ms = 0.0f;
  float trimmed_mean_ms = 0.0f;
  float min_ms = 0.0f;
  float max_ms = 0.0f;
  float p95_ms = 0.0f;
  float p99_ms = 0.0f;
  float stddev_ms = 0.0f;
  float iqr_ms = 0.0f;
  std::vector<float> samples_ms;
};

struct DiffStats {
  float max_abs = 0.0f;
  float mean_abs = 0.0f;
  float max_rel = 0.0f;
};

struct Gemma4BenchToolAvailability {
  bool ncu = false;
  bool nsys = false;
};

inline bool gemma4_bench_command_available(const char *command) {
  std::string probe = "command -v ";
  probe += command;
  probe += " >/dev/null 2>&1";
  return std::system(probe.c_str()) == 0;
}

inline std::string gemma4_bench_read_command_line(const std::string &command) {
  std::array<char, 512> buffer{};
  FILE *pipe = popen(command.c_str(), "r");
  if (pipe == nullptr) {
    return "";
  }
  std::string output;
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  pclose(pipe);
  while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
    output.pop_back();
  }
  return output;
}

// Checks profiler availability exactly once per benchmark process.
inline const Gemma4BenchToolAvailability &gemma4_bench_tool_availability() {
  static const Gemma4BenchToolAvailability availability = {
      gemma4_bench_command_available("ncu"),
      gemma4_bench_command_available("nsys"),
  };
  return availability;
}

// Emits the CUDA device/build context needed to compare benchmark runs.
inline void gemma4_bench_print_cuda_env_once(const char *benchmark_name) {
  static bool printed = false;
  if (printed) {
    return;
  }
  printed = true;

  int device = 0;
  cudaDeviceProp prop{};
  int driver_version = 0;
  int runtime_version = 0;
  int memory_clock_khz = 0;
  int memory_bus_width_bits = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &memory_clock_khz, cudaDevAttrMemoryClockRate, device));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &memory_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device));

  std::printf(
      "benchmark_env_cuda benchmark=%s device=%d gpu=\"%s\" pci_bus_id=%04x:%02x:%02x "
      "sm=%d%d sms=%d global_mem_bytes=%zu l2_cache_bytes=%d "
      "memory_clock_khz=%d memory_bus_width_bits=%d driver_version=%d "
      "runtime_version=%d\n",
      benchmark_name, device, prop.name, prop.pciDomainID, prop.pciBusID,
      prop.pciDeviceID, prop.major, prop.minor, prop.multiProcessorCount,
      static_cast<size_t>(prop.totalGlobalMem), prop.l2CacheSize,
      memory_clock_khz, memory_bus_width_bits, driver_version,
      runtime_version);

#if defined(__CUDACC_VER_MAJOR__)
  std::printf(
      "benchmark_build_cuda benchmark=%s cudacc_version=%d.%d.%d target_sm=86\n",
      benchmark_name, __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__,
      __CUDACC_VER_BUILD__);
#endif

  const std::string nvidia_smi_query =
      "nvidia-smi --query-gpu=name,gpu_bus_id,driver_version,persistence_mode,"
      "ecc.mode.current,mig.mode.current,power.limit,clocks.sm,clocks.mem,"
      "temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits -i " +
      std::to_string(device) + " 2>/dev/null";
  const std::string nvidia_smi_values =
      gemma4_bench_read_command_line(nvidia_smi_query);
  if (!nvidia_smi_values.empty()) {
    std::printf(
        "benchmark_env_nvidia_smi benchmark=%s fields=\"name,gpu_bus_id,driver_version,"
        "persistence_mode,ecc_mode_current,mig_mode_current,power_limit_w,"
        "clocks_sm_mhz,clocks_mem_mhz,temperature_c,power_draw_w,"
        "utilization_gpu_pct\" values=\"%s\"\n",
        benchmark_name, nvidia_smi_values.c_str());
  } else {
    std::printf("benchmark_env_nvidia_smi benchmark=%s status=unavailable\n",
                benchmark_name);
  }
}

// Emits profiler and counter guidance without putting profiler overhead in timing windows.
inline void gemma4_bench_print_profiler_tools_once(const char *benchmark_name) {
  static bool printed = false;
  if (printed) {
    return;
  }
  printed = true;

  const Gemma4BenchToolAvailability &tools = gemma4_bench_tool_availability();
  std::printf(
      "benchmark_profiler_tools benchmark=%s ncu_available=%d nsys_available=%d "
      "ncu_metrics=\"gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,"
      "dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum,"
      "lts__t_bytes.sum,smsp__warps_active.avg.pct_of_peak_sustained_active\" "
      "ncu_note=\"collect_counters_outside_timing_loop; profiler_replay_may_change_cache_state\" "
      "nsys_trace=\"cuda,nvtx,osrt\"\n",
      benchmark_name, tools.ncu ? 1 : 0, tools.nsys ? 1 : 0);
}

inline void gemma4_bench_print_common_metadata(const char *benchmark_name) {
  gemma4_bench_print_cuda_env_once(benchmark_name);
  gemma4_bench_print_profiler_tools_once(benchmark_name);
}

inline float gemma4_bench_percentile_sorted(const std::vector<float> &sorted,
                                            float q) {
  if (sorted.empty()) {
    return 0.0f;
  }
  const float pos = q * static_cast<float>(sorted.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(pos));
  const size_t hi = std::min(lo + 1, sorted.size() - 1);
  const float frac = pos - static_cast<float>(lo);
  return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

inline TimingStats summarize_timing_samples(std::vector<float> values) {
  TimingStats stats;
  if (values.empty()) {
    return stats;
  }

  stats.samples_ms = std::move(values);
  std::vector<float> sorted = stats.samples_ms;
  std::sort(sorted.begin(), sorted.end());

  stats.min_ms = sorted.front();
  stats.max_ms = sorted.back();
  stats.best_ms = stats.min_ms;
  stats.median_ms = gemma4_bench_percentile_sorted(sorted, 0.50f);
  stats.p95_ms = gemma4_bench_percentile_sorted(sorted, 0.95f);
  stats.p99_ms = gemma4_bench_percentile_sorted(sorted, 0.99f);
  stats.iqr_ms = gemma4_bench_percentile_sorted(sorted, 0.75f) -
                 gemma4_bench_percentile_sorted(sorted, 0.25f);

  const float sum =
      std::accumulate(stats.samples_ms.begin(), stats.samples_ms.end(), 0.0f);
  stats.avg_ms = sum / static_cast<float>(stats.samples_ms.size());

  size_t trim = sorted.size() >= 10 ? sorted.size() / 10 : 0;
  const auto trim_begin = sorted.begin() + static_cast<std::ptrdiff_t>(trim);
  const auto trim_end = sorted.end() - static_cast<std::ptrdiff_t>(trim);
  const float trimmed_sum = std::accumulate(trim_begin, trim_end, 0.0f);
  stats.trimmed_mean_ms =
      trimmed_sum / static_cast<float>(std::distance(trim_begin, trim_end));

  if (stats.samples_ms.size() > 1) {
    float variance = 0.0f;
    for (float sample : stats.samples_ms) {
      const float delta = sample - stats.avg_ms;
      variance += delta * delta;
    }
    stats.stddev_ms =
        std::sqrt(variance / static_cast<float>(stats.samples_ms.size() - 1));
  }
  return stats;
}

inline void gemma4_bench_print_timing_stats(const char *name,
                                            const char *context,
                                            const TimingStats &stats) {
  std::printf(
      "benchmark_timing name=%s%s%s median_ms=%.6f mean_ms=%.6f "
      "trimmed_mean_ms=%.6f min_ms=%.6f "
      "max_ms=%.6f p95_ms=%.6f p99_ms=%.6f stddev_ms=%.6f iqr_ms=%.6f "
      "samples_ms=[",
      name, context == nullptr || context[0] == '\0' ? "" : " ",
      context == nullptr ? "" : context, stats.median_ms, stats.avg_ms,
      stats.trimmed_mean_ms, stats.min_ms, stats.max_ms, stats.p95_ms,
      stats.p99_ms, stats.stddev_ms, stats.iqr_ms);
  for (size_t i = 0; i < stats.samples_ms.size(); ++i) {
    std::printf("%s%.6f", i == 0 ? "" : ",", stats.samples_ms[i]);
  }
  std::printf("]\n");
}

inline void gemma4_bench_print_timing_stats(const char *name,
                                            const TimingStats &stats) {
  gemma4_bench_print_timing_stats(name, "", stats);
}

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
  std::vector<float> samples;
  samples.reserve(static_cast<size_t>(trials));
  for (int i = 0; i < trials; ++i) {
    samples.push_back(time_ms_once(fn, stream, warmup, iters));
  }
  return summarize_timing_samples(std::move(samples));
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
  std::vector<float> samples;
  samples.reserve(static_cast<size_t>(trials));
  for (int i = 0; i < trials; ++i) {
    samples.push_back(time_ms_graph_once(fn, stream, warmup, iters));
  }
  return summarize_timing_samples(std::move(samples));
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
