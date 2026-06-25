#include "gemma4_kv_cache.cuh"
#include "gemma4_flash_attention.cuh"
#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_bench_utils.cuh"

#include <ATen/ATen.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/version.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

struct BenchOptions {
  int seq_len = 4096;
  int page_size = 64;
  int split_size = GEMMA4_SLIDING_DECODE_SPLIT_SIZE;
  int global_split_size = 64;
  int warmup = 25;
  int iters = 100;
  int samples = 15;
  int extra_splits = 0;
  int torch_q_heads = GEMMA4_NUM_QUERY_HEADS;
  int torch_kv_heads = GEMMA4_GLOBAL_KV_HEADS;
  int torch_head_dim = GEMMA4_GLOBAL_HEAD_DIM;
  std::string cache_mode = "warm";
  int64_t flush_bytes = 0;
};

float bf16_to_float(__nv_bfloat16 value) { return __bfloat162float(value); }

__nv_bfloat16 make_value(int seed) {
  int centered = ((seed * 37 + 17) % 257) - 128;
  return __float2bfloat16_rn(static_cast<float>(centered) / 64.0f);
}

int64_t cache_elements(const Gemma4KvCacheConfig &config) {
  return int64_t(config.num_layers) * config.num_pages * config.page_size *
         config.num_heads * config.head_dim;
}

__global__ void l2_flush_kernel(uint32_t *__restrict__ scratch,
                                int64_t words) {
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < words;
       i += stride) {
    scratch[i] = scratch[i] + 1u;
  }
}

void flush_l2(cudaStream_t stream,
              uint32_t *__restrict__ d_scratch,
              int64_t words) {
  if (d_scratch == nullptr || words <= 0) return;
  constexpr int kThreads = 256;
  int blocks = static_cast<int>(std::min<int64_t>(4096, div_up(words, kThreads)));
  l2_flush_kernel<<<blocks, kThreads, 0, stream>>>(d_scratch, words);
  CUDA_CHECK(cudaGetLastError());
}

float percentile(const std::vector<float> &sorted, float pct) {
  if (sorted.empty()) return 0.0f;
  if (sorted.size() == 1) return sorted.front();
  float index = pct * 0.01f * static_cast<float>(sorted.size() - 1);
  int lo = static_cast<int>(std::floor(index));
  int hi = static_cast<int>(std::ceil(index));
  float frac = index - static_cast<float>(lo);
  return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

float trimmed_mean(const std::vector<float> &sorted, float trim_fraction) {
  if (sorted.empty()) return 0.0f;
  int trim = static_cast<int>(std::floor(sorted.size() * trim_fraction));
  int begin = std::min<int>(trim, sorted.size() - 1);
  int end = std::max<int>(begin + 1, sorted.size() - trim);
  float sum = std::accumulate(sorted.begin() + begin, sorted.begin() + end, 0.0f);
  return sum / static_cast<float>(end - begin);
}

SampleStats summarize_samples(std::vector<float> values) {
  std::vector<float> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  SampleStats stats;
  stats.samples_ms = std::move(values);
  stats.median_ms = percentile(sorted, 50.0f);
  stats.min_ms = sorted.front();
  stats.max_ms = sorted.back();
  stats.p95_ms = percentile(sorted, 95.0f);
  stats.p99_ms = percentile(sorted, 99.0f);
  stats.iqr_ms = percentile(sorted, 75.0f) - percentile(sorted, 25.0f);
  stats.mean_ms = std::accumulate(stats.samples_ms.begin(),
                                  stats.samples_ms.end(), 0.0f) /
                  static_cast<float>(stats.samples_ms.size());
  stats.trimmed_mean_ms = trimmed_mean(sorted, 0.1f);
  float variance = 0.0f;
  for (float sample : stats.samples_ms) {
    float diff = sample - stats.mean_ms;
    variance += diff * diff;
  }
  variance /= static_cast<float>(stats.samples_ms.size());
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
      values[sample] = total_ms / static_cast<float>(iters_per_sample);
    } else {
      CUDA_CHECK(cudaEventRecord(start, stream));
      for (int i = 0; i < iters_per_sample; ++i) {
        fn();
      }
      CUDA_CHECK(cudaEventRecord(stop, stream));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float total_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
      values[sample] = total_ms / static_cast<float>(iters_per_sample);
    }
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return summarize_samples(std::move(values));
}

void print_stats(const char *name, const SampleStats &stats) {
  std::printf(
      "%s median_ms=%.6f mean_ms=%.6f trimmed_mean_ms=%.6f min_ms=%.6f "
      "max_ms=%.6f p95_ms=%.6f p99_ms=%.6f stddev_ms=%.6f iqr_ms=%.6f "
      "samples_ms=[",
      name, stats.median_ms, stats.mean_ms, stats.trimmed_mean_ms,
      stats.min_ms, stats.max_ms, stats.p95_ms, stats.p99_ms,
      stats.stddev_ms, stats.iqr_ms);
  for (size_t i = 0; i < stats.samples_ms.size(); ++i) {
    std::printf("%s%.6f", i == 0 ? "" : ",", stats.samples_ms[i]);
  }
  std::printf("]\n");
}

void cpu_decode_reference(std::vector<__nv_bfloat16> &expected,
                          const std::vector<__nv_bfloat16> &q,
                          const std::vector<__nv_bfloat16> &k,
                          const std::vector<__nv_bfloat16> &v,
                          int key_start,
                          int key_count,
                          int q_heads,
                          int kv_heads,
                          int head_dim,
                          float scale) {
  int group_size = q_heads / kv_heads;
  for (int qh = 0; qh < q_heads; ++qh) {
    int kh = qh / group_size;
    std::vector<float> scores(key_count);
    float max_score = -INFINITY;
    for (int key = 0; key < key_count; ++key) {
      int pos = key_start + key;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += bf16_to_float(q[qh * head_dim + d]) *
               bf16_to_float(k[(pos * kv_heads + kh) * head_dim + d]);
      }
      scores[key] = dot * scale;
      max_score = std::max(max_score, scores[key]);
    }
    float denom = 0.0f;
    for (float score : scores) {
      denom += std::exp(score - max_score);
    }
    for (int d = 0; d < head_dim; ++d) {
      float value = 0.0f;
      for (int key = 0; key < key_count; ++key) {
        int pos = key_start + key;
        float weight = std::exp(scores[key] - max_score) / denom;
        value += weight *
                 bf16_to_float(v[(pos * kv_heads + kh) * head_dim + d]);
      }
      expected[qh * head_dim + d] = __float2bfloat16_rn(value);
    }
  }
}

// Calls the same eager SDPA op as the deleted Python benchmark, with GQA enabled.
at::Tensor run_torch_sdpa(const at::Tensor &q,
                          const at::Tensor &k,
                          const at::Tensor &v,
                          double scale) {
  return at::scaled_dot_product_attention(
      q, k, v, std::nullopt, 0.0, false, std::optional<double>(scale), true);
}

// Runs the former Python full-decode path: write final K/V rows, then SDPA.
at::Tensor run_torch_full_decode(at::Tensor &k,
                                 at::Tensor &v,
                                 const at::Tensor &new_k,
                                 const at::Tensor &new_v,
                                 const at::Tensor &q,
                                 int seq_len,
                                 double scale) {
  k.narrow(2, seq_len - 1, 1).copy_(new_k);
  v.narrow(2, seq_len - 1, 1).copy_(new_v);
  return run_torch_sdpa(q, k, v, scale);
}

void check_attention_correctness(const std::vector<__nv_bfloat16> &actual,
                                 const std::vector<__nv_bfloat16> &expected) {
  float max_abs = 0.0f;
  float mean_abs = 0.0f;
  for (int i = 0; i < static_cast<int>(actual.size()); ++i) {
    float diff = std::fabs(bf16_to_float(actual[i]) -
                           bf16_to_float(expected[i]));
    max_abs = std::max(max_abs, diff);
    mean_abs += diff;
  }
  mean_abs /= static_cast<float>(actual.size());
  std::printf("correctness max_abs=%.6f mean_abs=%.6f\n", max_abs, mean_abs);
  if (max_abs > 0.03125f) {
    throw std::runtime_error("paged decode attention correctness check failed");
  }
}

void print_environment(const cudaDeviceProp &prop,
                       int driver,
                       int runtime,
                       int clock_rate_khz,
                       int memory_clock_rate_khz) {
  std::printf(
      "gpu=%s sm=%d%d memory_gb=%.2f driver=%d runtime=%d l2_cache_bytes=%d "
      "sms=%d clock_rate_khz=%d memory_clock_rate_khz=%d\n",
      prop.name, prop.major, prop.minor,
      double(prop.totalGlobalMem) / 1.0e9, driver, runtime,
      prop.l2CacheSize, prop.multiProcessorCount, clock_rate_khz,
      memory_clock_rate_khz);
}

BenchOptions parse_args(int argc, char **argv) {
  BenchOptions options;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto need_value = [&](const char *name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (arg == "--cache") {
      options.cache_mode = need_value("--cache");
    } else if (arg == "--flush-bytes") {
      options.flush_bytes = std::stoll(need_value("--flush-bytes"));
    } else if (arg == "--flush-mib") {
      options.flush_bytes = std::stoll(need_value("--flush-mib")) * 1024ll *
                            1024ll;
    } else if (arg == "--seq-len") {
      options.seq_len = std::stoi(need_value("--seq-len"));
    } else if (arg == "--page-size") {
      options.page_size = std::stoi(need_value("--page-size"));
    } else if (arg == "--split-size") {
      options.split_size = std::stoi(need_value("--split-size"));
    } else if (arg == "--global-split-size") {
      options.global_split_size = std::stoi(need_value("--global-split-size"));
    } else if (arg == "--q-heads") {
      options.torch_q_heads = std::stoi(need_value("--q-heads"));
    } else if (arg == "--kv-heads") {
      options.torch_kv_heads = std::stoi(need_value("--kv-heads"));
    } else if (arg == "--head-dim") {
      options.torch_head_dim = std::stoi(need_value("--head-dim"));
    } else if (arg == "--warmup") {
      options.warmup = std::stoi(need_value("--warmup"));
    } else if (arg == "--iters") {
      options.iters = std::stoi(need_value("--iters"));
    } else if (arg == "--samples") {
      options.samples = std::stoi(need_value("--samples"));
    } else if (arg == "--extra-splits") {
      options.extra_splits = std::stoi(need_value("--extra-splits"));
    } else {
      positional.push_back(arg);
    }
  }

  if (positional.size() > 0) options.seq_len = std::stoi(positional[0]);
  if (positional.size() > 1) options.page_size = std::stoi(positional[1]);
  if (positional.size() > 2) options.split_size = std::stoi(positional[2]);
  if (positional.size() > 3) options.warmup = std::stoi(positional[3]);
  if (positional.size() > 4) options.iters = std::stoi(positional[4]);
  if (positional.size() > 5) options.samples = std::stoi(positional[5]);
  if (positional.size() > 6) {
    throw std::runtime_error("usage: gemma4_kv_cache_bench [seq page split "
                             "warmup iters samples] [--cache warm|cold] "
                             "[--flush-bytes N] [--flush-mib N] "
                             "[--seq-len N] [--q-heads N] [--kv-heads N] "
                             "[--head-dim N] [--global-split-size N] "
                             "[--extra-splits N]");
  }
  if (options.cache_mode != "warm" && options.cache_mode != "cold") {
    throw std::runtime_error("--cache must be warm or cold");
  }
  if (options.seq_len <= 0 || options.page_size <= 0 ||
      options.split_size <= 0 || options.global_split_size <= 0 ||
      options.warmup < 0 || options.iters <= 0 ||
      options.samples <= 0 || options.extra_splits < 0 ||
      options.torch_q_heads <= 0 || options.torch_kv_heads <= 0 ||
      options.torch_head_dim <= 0) {
    throw std::runtime_error("benchmark dimensions/counts must be positive");
  }
  if (options.torch_q_heads % options.torch_kv_heads != 0) {
    throw std::runtime_error("--q-heads must be divisible by --kv-heads");
  }
  return options;
}

}  // namespace

int main(int argc, char **argv) {
  BenchOptions options = parse_args(argc, argv);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int driver = 0;
  int runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  int clock_rate_khz = 0;
  int memory_clock_rate_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate,
                                    device));
  CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_rate_khz,
                                    cudaDevAttrMemoryClockRate, device));

  int seq_len = options.seq_len;
  int page_size = options.page_size;
  int split_size = options.split_size;
  int global_split_size = options.global_split_size;
  int warmup = options.warmup;
  int iters = options.iters;
  int samples = options.samples;
  bool cold_cache = options.cache_mode == "cold";
  const int torch_q_heads = options.torch_q_heads;
  const int torch_kv_heads = options.torch_kv_heads;
  const int torch_head_dim = options.torch_head_dim;

  // Cold-cache mode defines an explicit L2 flush before each measured sample.
  // Warm-cache mode leaves repeated accesses to measure steady-state reuse.
  int64_t flush_bytes =
      cold_cache ? options.flush_bytes : 0;
  if (cold_cache && flush_bytes == 0) {
    flush_bytes = std::max<int64_t>(64ll * 1024 * 1024,
                                    int64_t(prop.l2CacheSize) * 4);
  }

  int pages_for_seq = div_up(seq_len, page_size);
  int max_pages_per_seq =
      std::max(pages_for_seq, div_up(GEMMA4_SLIDING_WINDOW, page_size) + 1);
  Gemma4KvCacheConfig config =
      gemma4_kv_cache_make_config(false, max_pages_per_seq, page_size,
                                  max_pages_per_seq);
  int batch_size = 1;
  int layer = 0;
  int q_heads = GEMMA4_NUM_QUERY_HEADS;

  // Decode attends over the active sliding window, not necessarily the full
  // constructed sequence. `actual_splits` is the live work; `num_splits` is the
  // fixed scratch/layout stride used to test graph-compatible overprovisioning.
  int key_count = config.window_size > 0 ? std::min(seq_len, config.window_size)
                                         : seq_len;
  int actual_splits = div_up(key_count, split_size);
  int num_splits = actual_splits + options.extra_splits;
  float scale = 1.0f / std::sqrt(float(config.head_dim));

  Gemma4BenchmarkContract contract;
  contract.benchmark = "kv_cache_bench";
  contract.cache = options.cache_mode.c_str();
  contract.aggregation = "median_mean_trimmed_mean_p95_p99_stddev_iqr_raw_samples";
  contract.correctness = "flash_decode_vs_cpu_reference_checked";
  contract.notes = "cold mode flushes L2 before each measured invocation";
  contract.warmup = warmup;
  contract.iters = iters;
  contract.samples = samples;
  gemma4_bench_print_environment(contract.benchmark);
  gemma4_bench_print_contract(contract);
  print_environment(prop, driver, runtime, clock_rate_khz,
                    memory_clock_rate_khz);
  std::printf(
      "contract=typical_kernel_microbenchmark timing=CUDA_event_gpu_timeline "
      "cache_mode=%s l2_flush_bytes=%lld launch_overhead=queued_launches_only "
      "host_wall_time=excluded stability_scope=single_process "
      "min_effect_for_claim_pct=5 seq_len=%d page_size=%d split_size=%d "
      "key_count=%d actual_splits=%d splits=%d extra_splits=%d warmup=%d "
      "iters_per_sample=%d samples=%d\n",
      options.cache_mode.c_str(), static_cast<long long>(flush_bytes), seq_len,
      page_size, split_size, key_count, actual_splits, num_splits,
      options.extra_splits, warmup, iters, samples);
  std::printf(
      "build=nvcc flags=\"-std=c++17 -O3 -arch=sm_86 -Isrc\" "
      "dtype=bf16 layout=\"cache=[layers,pages,page_size,kv_heads,head_dim]\" "
      "batch_size=%d q_heads=%d kv_heads=%d head_dim=%d\n",
      batch_size, q_heads, config.num_heads, config.head_dim);

  // LibTorch rows preserve the deleted Python SDPA baseline's global-like shape.
  {
    c10::cuda::CUDAGuard torch_device_guard(
        static_cast<c10::DeviceIndex>(device));
    const c10::cuda::CUDAStream torch_stream =
        c10::cuda::getStreamFromPool(false, device);
    c10::cuda::CUDAStreamGuard torch_stream_guard(torch_stream);
    at::manual_seed(1234u);

    const auto torch_options =
        at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    at::Tensor torch_q =
        at::randn({1, torch_q_heads, 1, torch_head_dim}, torch_options);
    at::Tensor torch_k =
        at::randn({1, torch_kv_heads, seq_len, torch_head_dim}, torch_options);
    at::Tensor torch_v =
        at::randn({1, torch_kv_heads, seq_len, torch_head_dim}, torch_options);
    at::Tensor torch_new_k =
        at::randn({1, torch_kv_heads, 1, torch_head_dim}, torch_options);
    at::Tensor torch_new_v =
        at::randn({1, torch_kv_heads, 1, torch_head_dim}, torch_options);
    at::Tensor torch_out;
    DeviceBuffer<uint8_t> d_torch_l2_scratch_bytes(flush_bytes);
    uint32_t *d_torch_l2_scratch =
        reinterpret_cast<uint32_t *>(d_torch_l2_scratch_bytes.get());
    if (flush_bytes > 0) {
      CUDA_CHECK(cudaMemsetAsync(d_torch_l2_scratch, 0, flush_bytes,
                                 torch_stream.stream()));
    }

    const double torch_scale = 1.0 / std::sqrt(double(torch_head_dim));
    auto torch_attention_only = [&]() {
      torch_out = run_torch_sdpa(torch_q, torch_k, torch_v, torch_scale);
    };
    auto torch_full_decode = [&]() {
      torch_out = run_torch_full_decode(torch_k, torch_v, torch_new_k,
                                        torch_new_v, torch_q, seq_len,
                                        torch_scale);
    };

    torch_attention_only();
    CUDA_CHECK(cudaStreamSynchronize(torch_stream.stream()));
    std::printf(
        "benchmark_libtorch torch_version=%s dtype=bf16 "
        "layout=\"q=[1,Hq,1,D],k/v=[1,Hkv,S,D]\" seq_len=%d q_heads=%d "
        "kv_heads=%d head_dim=%d scale=%.9f\n",
        TORCH_VERSION, seq_len, torch_q_heads, torch_kv_heads, torch_head_dim,
        torch_scale);
    print_stats("torch_attention_only",
                time_cuda_samples(torch_attention_only, torch_stream.stream(),
                                  warmup, iters, samples, cold_cache,
                                  d_torch_l2_scratch,
                                  flush_bytes / sizeof(uint32_t)));
    print_stats("torch_full_decode_write_plus_attention",
                time_cuda_samples(torch_full_decode, torch_stream.stream(),
                                  warmup, iters, samples, cold_cache,
                                  d_torch_l2_scratch,
                                  flush_bytes / sizeof(uint32_t)));

    at::Tensor torch_sliding_q =
        at::randn({1, q_heads, 1, config.head_dim}, torch_options);
    at::Tensor torch_sliding_k =
        at::randn({1, config.num_heads, key_count, config.head_dim},
                  torch_options);
    at::Tensor torch_sliding_v =
        at::randn({1, config.num_heads, key_count, config.head_dim},
                  torch_options);
    auto torch_sliding_decode = [&]() {
      torch_out = run_torch_sdpa(torch_sliding_q, torch_sliding_k,
                                 torch_sliding_v, scale);
    };
    torch_sliding_decode();
    CUDA_CHECK(cudaStreamSynchronize(torch_stream.stream()));
    std::printf(
        "benchmark_libtorch_sliding seq_len=%d key_count=%d q_heads=%d "
        "kv_heads=%d head_dim=%d\n",
        seq_len, key_count, q_heads, config.num_heads, config.head_dim);
    print_stats("torch_sliding_decode_attention",
                time_cuda_samples(torch_sliding_decode, torch_stream.stream(),
                                  warmup, iters, samples, cold_cache,
                                  d_torch_l2_scratch,
                                  flush_bytes / sizeof(uint32_t)));
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<int32_t> page_table(max_pages_per_seq);
  std::iota(page_table.begin(), page_table.end(), 0);
  std::vector<int32_t> seq_lengths = {seq_len};
  std::vector<int32_t> token_batch(seq_len, 0);
  std::vector<int32_t> token_position(seq_len);
  std::iota(token_position.begin(), token_position.end(), 0);
  std::vector<int32_t> one_batch = {0};
  std::vector<int32_t> one_position = {seq_len - 1};

  // Host K/V represent the full prefill sequence. `one_k`/`one_v` below are
  // the final token reused by the steady-state decode-write microbenchmark.
  int kv_elems = seq_len * config.num_heads * config.head_dim;
  std::vector<__nv_bfloat16> h_k(kv_elems);
  std::vector<__nv_bfloat16> h_v(kv_elems);
  for (int i = 0; i < kv_elems; ++i) {
    h_k[i] = make_value(i);
    h_v[i] = make_value(1000003 + i);
  }
  std::vector<__nv_bfloat16> h_q(q_heads * config.head_dim);
  for (int i = 0; i < static_cast<int>(h_q.size()); ++i) {
    h_q[i] = make_value(2000003 + i);
  }

  const size_t work_scratch_i32_size =
      gemma4_flash_attention_sliding_decode_persistent_scratch_i32(
          batch_size, num_splits);
  const int32_t work_scratch_i32 =
      static_cast<int32_t>(work_scratch_i32_size);
  const size_t partial_m_elements =
      static_cast<size_t>(batch_size) * q_heads * num_splits;
  const size_t partial_acc_elements = partial_m_elements * config.head_dim;

  // Partial scratch is allocated for `num_splits`, including any extra empty
  // splits requested by --extra-splits.
  DeviceBuffer<__nv_bfloat16> d_cache_k(
      static_cast<size_t>(cache_elements(config)));
  DeviceBuffer<__nv_bfloat16> d_cache_v(
      static_cast<size_t>(cache_elements(config)));
  DeviceBuffer<__nv_bfloat16> d_k(h_k.size());
  DeviceBuffer<__nv_bfloat16> d_v(h_v.size());
  DeviceBuffer<__nv_bfloat16> d_one_k(config.num_heads * config.head_dim);
  DeviceBuffer<__nv_bfloat16> d_one_v(config.num_heads * config.head_dim);
  DeviceBuffer<__nv_bfloat16> d_q(h_q.size());
  DeviceBuffer<__nv_bfloat16> d_out(h_q.size());
  DeviceBuffer<int32_t> d_page_table(page_table.size());
  DeviceBuffer<int32_t> d_seq_lengths(seq_lengths.size());
  DeviceBuffer<int32_t> d_token_batch(token_batch.size());
  DeviceBuffer<int32_t> d_token_position(token_position.size());
  DeviceBuffer<int32_t> d_one_batch(one_batch.size());
  DeviceBuffer<int32_t> d_one_position(one_position.size());
  DeviceBuffer<int32_t> d_work_scratch(work_scratch_i32_size);
  DeviceBuffer<float> d_partial_m(partial_m_elements);
  DeviceBuffer<float> d_partial_l(partial_m_elements);
  DeviceBuffer<float> d_partial_acc(partial_acc_elements);
  DeviceBuffer<uint8_t> d_l2_scratch_bytes(flush_bytes);
  uint32_t *d_l2_scratch =
      reinterpret_cast<uint32_t *>(d_l2_scratch_bytes.get());
  if (flush_bytes > 0) {
    CUDA_CHECK(cudaMemsetAsync(d_l2_scratch, 0, flush_bytes, stream));
  }

  CUDA_CHECK(cudaMemcpyAsync(d_k, h_k.data(), h_k.size() * sizeof(h_k[0]),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_v, h_v.data(), h_v.size() * sizeof(h_v[0]),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_one_k,
                             h_k.data() + (seq_len - 1) * config.num_heads *
                                              config.head_dim,
                             config.num_heads * config.head_dim *
                                 sizeof(h_k[0]),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_one_v,
                             h_v.data() + (seq_len - 1) * config.num_heads *
                                              config.head_dim,
                             config.num_heads * config.head_dim *
                                 sizeof(h_v[0]),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_q, h_q.data(), h_q.size() * sizeof(h_q[0]),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_page_table, page_table.data(),
                             page_table.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_seq_lengths, seq_lengths.data(), sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_token_batch, token_batch.data(),
                             token_batch.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_token_position, token_position.data(),
                             token_position.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_one_batch, one_batch.data(), sizeof(int32_t),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_one_position, one_position.data(),
                             sizeof(int32_t), cudaMemcpyHostToDevice, stream));

  CUDA_CHECK(gemma4_kv_cache_write_bf16(
      d_cache_k, d_cache_v, config, d_page_table, d_token_batch,
      d_token_position, seq_len, layer, d_k, d_v, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<__nv_bfloat16> expected(h_q.size());
  std::vector<__nv_bfloat16> actual(h_q.size());
  const int key_start = seq_len - key_count;
  cpu_decode_reference(expected, h_q, h_k, h_v, key_start, key_count, q_heads,
                       config.num_heads, config.head_dim, scale);

  // Correctness check: flash paged decode against the CPU reference.
  CUDA_CHECK(gemma4_flash_attention_sliding_decode_paged_bf16(
      d_out, d_partial_m, d_partial_l, d_partial_acc, d_q, d_cache_k,
      d_cache_v, d_page_table, d_seq_lengths, config, layer, batch_size,
      scale, split_size, num_splits, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(actual[0]),
                        cudaMemcpyDeviceToHost));
  check_attention_correctness(actual, expected);

  bool persistent_supported = true;
  cudaError_t persistent_status =
      gemma4_flash_attention_sliding_decode_paged_persistent_bf16(
      d_out, d_partial_m, d_partial_l, d_partial_acc, d_work_scratch,
      work_scratch_i32, d_q, d_cache_k, d_cache_v, d_page_table,
      d_seq_lengths, config, layer, batch_size, scale, split_size,
      num_splits, 0, stream);
  if (persistent_status == cudaErrorNotSupported) {
    persistent_supported = false;
    CUDA_CHECK(cudaGetLastError());
  } else {
    CUDA_CHECK(persistent_status);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(actual.data(), d_out, actual.size() * sizeof(actual[0]),
                          cudaMemcpyDeviceToHost));
    check_attention_correctness(actual, expected);
  }

  // Each lambda enqueues exactly the work named by its label. The timing helper
  // wraps these in CUDA events on the same stream and optionally flushes L2.
  auto prefill_write = [&]() {
    CUDA_CHECK(gemma4_kv_cache_write_bf16(
        d_cache_k, d_cache_v, config, d_page_table, d_token_batch,
        d_token_position, seq_len, layer, d_k, d_v, stream));
  };
  auto decode_write = [&]() {
    CUDA_CHECK(gemma4_kv_cache_write_bf16(
        d_cache_k, d_cache_v, config, d_page_table, d_one_batch,
        d_one_position, 1, layer, d_one_k, d_one_v, stream));
  };
  auto flash_decode_attention = [&]() {
    CUDA_CHECK(gemma4_flash_attention_sliding_decode_paged_bf16(
        d_out, d_partial_m, d_partial_l, d_partial_acc, d_q, d_cache_k,
        d_cache_v, d_page_table, d_seq_lengths, config, layer, batch_size,
        scale, split_size, num_splits, stream));
  };
  auto flash_decode_attention_persistent = [&]() {
    CUDA_CHECK(gemma4_flash_attention_sliding_decode_paged_persistent_bf16(
        d_out, d_partial_m, d_partial_l, d_partial_acc, d_work_scratch,
        work_scratch_i32, d_q, d_cache_k, d_cache_v, d_page_table,
        d_seq_lengths, config, layer, batch_size, scale, split_size,
        num_splits, 0, stream));
  };
  auto flash_full_decode = [&]() {
    decode_write();
    flash_decode_attention();
  };

  print_stats("prefill_cache_write",
              time_cuda_samples(prefill_write, stream, warmup, iters, samples,
                                cold_cache, d_l2_scratch,
                                flush_bytes / sizeof(uint32_t)));
  print_stats("decode_cache_write",
              time_cuda_samples(decode_write, stream, warmup, iters, samples,
                                cold_cache, d_l2_scratch,
                                flush_bytes / sizeof(uint32_t)));
  print_stats("flash_decode_paged_attention_direct",
              time_cuda_samples(flash_decode_attention, stream, warmup, iters,
                                samples, cold_cache, d_l2_scratch,
                                flush_bytes / sizeof(uint32_t)));
  if (persistent_supported) {
    print_stats("flash_decode_paged_attention_persistent",
                time_cuda_samples(flash_decode_attention_persistent, stream,
                                  warmup, iters, samples, cold_cache,
                                  d_l2_scratch,
                                  flush_bytes / sizeof(uint32_t)));
  } else {
    std::printf("flash_decode_paged_attention_persistent skipped=not_supported\n");
  }
  print_stats("flash_full_decode_write_plus_attention",
              time_cuda_samples(flash_full_decode, stream, warmup, iters,
                                samples, cold_cache, d_l2_scratch,
                                flush_bytes / sizeof(uint32_t)));

  {
    const int global_pages = div_up(seq_len, page_size);
    Gemma4KvCacheConfig global_config =
        gemma4_kv_cache_make_config(true, global_pages, page_size, global_pages);
    const int global_capacity = global_pages * page_size;
    const int global_splits = div_up(global_capacity, global_split_size);
    const float global_scale = 1.0f / std::sqrt(float(global_config.head_dim));
    const int global_kv_elems =
        seq_len * global_config.num_heads * global_config.head_dim;
    std::vector<int32_t> global_page_table(global_pages);
    std::iota(global_page_table.begin(), global_page_table.end(), 0);
    std::vector<__nv_bfloat16> global_h_k(global_kv_elems);
    std::vector<__nv_bfloat16> global_h_v(global_kv_elems);
    std::vector<__nv_bfloat16> global_h_q(q_heads * global_config.head_dim);
    for (int i = 0; i < global_kv_elems; ++i) {
      global_h_k[i] = make_value(3000003 + i);
      global_h_v[i] = make_value(4000003 + i);
    }
    for (int i = 0; i < static_cast<int>(global_h_q.size()); ++i) {
      global_h_q[i] = make_value(5000003 + i);
    }

    const size_t global_partial_m_elements =
        static_cast<size_t>(batch_size) * q_heads * global_splits;
    const size_t global_partial_acc_elements =
        global_partial_m_elements * global_config.head_dim;

    DeviceBuffer<__nv_bfloat16> d_global_cache_k(
        static_cast<size_t>(cache_elements(global_config)));
    DeviceBuffer<__nv_bfloat16> d_global_cache_v(
        static_cast<size_t>(cache_elements(global_config)));
    DeviceBuffer<__nv_bfloat16> d_global_k(global_h_k.size());
    DeviceBuffer<__nv_bfloat16> d_global_v(global_h_v.size());
    DeviceBuffer<__nv_bfloat16> d_global_q(global_h_q.size());
    DeviceBuffer<__nv_bfloat16> d_global_out(global_h_q.size());
    DeviceBuffer<int32_t> d_global_page_table(global_page_table.size());
    DeviceBuffer<float> d_global_partial_m(global_partial_m_elements);
    DeviceBuffer<float> d_global_partial_l(global_partial_m_elements);
    DeviceBuffer<float> d_global_partial_acc(global_partial_acc_elements);

    CUDA_CHECK(cudaMemcpyAsync(d_global_k, global_h_k.data(),
                               global_h_k.size() * sizeof(global_h_k[0]),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_global_v, global_h_v.data(),
                               global_h_v.size() * sizeof(global_h_v[0]),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_global_q, global_h_q.data(),
                               global_h_q.size() * sizeof(global_h_q[0]),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_global_page_table, global_page_table.data(),
                               global_page_table.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(gemma4_kv_cache_write_bf16(
        d_global_cache_k, d_global_cache_v, global_config, d_global_page_table,
        d_token_batch, d_token_position, seq_len, layer, d_global_k,
        d_global_v, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__nv_bfloat16> global_expected(global_h_q.size());
    std::vector<__nv_bfloat16> global_actual(global_h_q.size());
    cpu_decode_reference(global_expected, global_h_q, global_h_k, global_h_v,
                         0, seq_len, q_heads, global_config.num_heads,
                         global_config.head_dim, global_scale);

    auto global_decode_attention = [&]() {
      CUDA_CHECK(gemma4_flash_attention_decode_paged_bf16(
          d_global_out, d_global_partial_m, d_global_partial_l,
          d_global_partial_acc, d_global_q, d_global_cache_k, d_global_cache_v,
          d_global_page_table, d_seq_lengths, global_config, layer, batch_size,
          global_scale, global_split_size, global_splits, stream));
    };
    global_decode_attention();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(global_actual.data(), d_global_out,
                          global_actual.size() * sizeof(global_actual[0]),
                          cudaMemcpyDeviceToHost));
    check_attention_correctness(global_actual, global_expected);

    std::printf(
        "global_decode_shape seq_len=%d page_size=%d split_size=%d splits=%d "
        "q_heads=%d kv_heads=%d head_dim=%d\n",
        seq_len, page_size, global_split_size, global_splits, q_heads,
        global_config.num_heads, global_config.head_dim);
    print_stats("global_decode_paged_attention_direct",
                time_cuda_samples(global_decode_attention, stream, warmup,
                                  iters, samples, cold_cache, d_l2_scratch,
                                  flush_bytes / sizeof(uint32_t)));
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
