#include "gemma4_kv_cache.cuh"
#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_bench_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct BenchOptions {
  int seq_len = 4096;
  int page_size = 64;
  int warmup = 25;
  int iters = 100;
  int samples = 15;
  std::string cache_mode = "warm";
  int64_t flush_bytes = 0;
};

__nv_bfloat16 make_value(int seed) {
  const int centered = ((seed * 37 + 17) % 257) - 128;
  return __float2bfloat16_rn(static_cast<float>(centered) / 64.0f);
}

constexpr int64_t cache_elements(const Gemma4KvCacheConfig &config) {
  return int64_t(config.num_layers) * config.num_pages * config.page_size *
         config.num_heads * config.head_dim;
}

__global__ void l2_flush_kernel(uint32_t *__restrict__ scratch,
                                int64_t words) {
  const int64_t stride = int64_t(blockDim.x) * gridDim.x;
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

  return summarize_timing_samples(std::move(values));
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
    } else if (arg == "--warmup") {
      options.warmup = std::stoi(need_value("--warmup"));
    } else if (arg == "--iters") {
      options.iters = std::stoi(need_value("--iters"));
    } else if (arg == "--samples") {
      options.samples = std::stoi(need_value("--samples"));
    } else {
      positional.push_back(arg);
    }
  }

  if (positional.size() > 0) options.seq_len = std::stoi(positional[0]);
  if (positional.size() > 1) options.page_size = std::stoi(positional[1]);
  if (positional.size() > 2) options.warmup = std::stoi(positional[2]);
  if (positional.size() > 3) options.iters = std::stoi(positional[3]);
  if (positional.size() > 4) options.samples = std::stoi(positional[4]);
  if (positional.size() > 5) {
    throw std::runtime_error("usage: gemma4_kv_cache_bench [seq page "
                             "warmup iters samples] [--cache warm|cold] "
                             "[--flush-bytes N] [--flush-mib N] "
                             "[--seq-len N] [--page-size N]");
  }
  if (options.cache_mode != "warm" && options.cache_mode != "cold") {
    throw std::runtime_error("--cache must be warm or cold");
  }
  if (options.seq_len <= 0 || options.page_size <= 0 ||
      options.warmup < 0 || options.iters <= 0 || options.samples <= 0) {
    throw std::runtime_error("benchmark dimensions/counts must be positive");
  }
  return options;
}

}  // namespace

int main(int argc, char **argv) {
  const BenchOptions options = parse_args(argc, argv);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  gemma4_bench_print_common_metadata("kv_cache_bench");

  const int seq_len = options.seq_len;
  const int page_size = options.page_size;
  const int warmup = options.warmup;
  const int iters = options.iters;
  const int samples = options.samples;
  const bool cold_cache = options.cache_mode == "cold";

  // Cold-cache mode defines an explicit L2 flush before each measured sample.
  // Warm-cache mode leaves repeated accesses to measure steady-state reuse.
  int64_t flush_bytes =
      cold_cache ? options.flush_bytes : 0;
  if (cold_cache && flush_bytes == 0) {
    int l2_cache_bytes = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize,
                                      device));
    flush_bytes = std::max<int64_t>(64ll * 1024 * 1024,
                                    int64_t(l2_cache_bytes) * 4);
  }

  const int pages_for_seq = div_up(seq_len, page_size);
  const int max_pages_per_seq =
      std::max(pages_for_seq, div_up(GEMMA4_SLIDING_WINDOW, page_size) + 1);
  const Gemma4KvCacheConfig config =
      gemma4_kv_cache_make_config(false, max_pages_per_seq, page_size,
                                  max_pages_per_seq);
  const int batch_size = 1;
  const int layer = 0;

  std::printf(
      "benchmark_contract name=kv_cache_bench contract=typical_kernel_microbenchmark timing=CUDA_event_gpu_timeline "
      "cache_mode=%s l2_flush_bytes=%lld launch_overhead=queued_launches_only "
      "host_wall_time=excluded stability_scope=single_process "
      "min_effect_for_claim_pct=5 seq_len=%d page_size=%d warmup=%d "
      "iters_per_sample=%d samples=%d\n",
      options.cache_mode.c_str(), static_cast<long long>(flush_bytes), seq_len,
      page_size, warmup, iters, samples);
  std::printf(
      "build=nvcc flags=\"-std=c++17 -O3 -arch=sm_86 -Isrc\" "
      "dtype=bf16 layout=\"cache=[layers,pages,page_size,kv_heads,head_dim]\" "
      "batch_size=%d kv_heads=%d head_dim=%d\n",
      batch_size, config.num_heads, config.head_dim);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::vector<int32_t> page_table(max_pages_per_seq);
  std::iota(page_table.begin(), page_table.end(), 0);
  const std::vector<int32_t> token_batch(seq_len, 0);
  std::vector<int32_t> token_position(seq_len);
  std::iota(token_position.begin(), token_position.end(), 0);
  const std::vector<int32_t> one_batch = {0};
  const std::vector<int32_t> one_position = {seq_len - 1};

  // Host K/V represent the full prefill sequence. `one_k`/`one_v` below are
  // the final token reused by the steady-state decode-write microbenchmark.
  const int kv_elems = seq_len * config.num_heads * config.head_dim;
  std::vector<__nv_bfloat16> h_k(kv_elems);
  std::vector<__nv_bfloat16> h_v(kv_elems);
  for (int i = 0; i < kv_elems; ++i) {
    h_k[i] = make_value(i);
    h_v[i] = make_value(1000003 + i);
  }

  thrust::device_vector<__nv_bfloat16> d_cache_k(
      static_cast<size_t>(cache_elements(config)));
  thrust::device_vector<__nv_bfloat16> d_cache_v(
      static_cast<size_t>(cache_elements(config)));
  thrust::device_vector<__nv_bfloat16> d_k(h_k.size());
  thrust::device_vector<__nv_bfloat16> d_v(h_v.size());
  thrust::device_vector<__nv_bfloat16> d_one_k(config.num_heads * config.head_dim);
  thrust::device_vector<__nv_bfloat16> d_one_v(config.num_heads * config.head_dim);
  thrust::device_vector<int32_t> d_page_table(page_table.size());
  thrust::device_vector<int32_t> d_token_batch(token_batch.size());
  thrust::device_vector<int32_t> d_token_position(token_position.size());
  thrust::device_vector<int32_t> d_one_batch(one_batch.size());
  thrust::device_vector<int32_t> d_one_position(one_position.size());
  thrust::device_vector<uint8_t> d_l2_scratch_bytes(flush_bytes);
  uint32_t *d_l2_scratch =
      reinterpret_cast<uint32_t *>(raw_ptr(d_l2_scratch_bytes));
  if (flush_bytes > 0) {
    CUDA_CHECK(cudaMemsetAsync(d_l2_scratch, 0, flush_bytes));
  }

  CUDA_CHECK(cudaMemcpyAsync(raw_ptr(d_k), h_k.data(), h_k.size() * sizeof(h_k[0]),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(raw_ptr(d_v), h_v.data(), h_v.size() * sizeof(h_v[0]),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(raw_ptr(d_one_k),
                             h_k.data() + (seq_len - 1) * config.num_heads *
                                              config.head_dim,
                             config.num_heads * config.head_dim *
                                 sizeof(h_k[0]),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(raw_ptr(d_one_v),
                             h_v.data() + (seq_len - 1) * config.num_heads *
                                              config.head_dim,
                             config.num_heads * config.head_dim *
                                 sizeof(h_v[0]),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(raw_ptr(d_page_table), page_table.data(),
                             page_table.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(raw_ptr(d_token_batch), token_batch.data(),
                             token_batch.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(raw_ptr(d_token_position), token_position.data(),
                             token_position.size() * sizeof(int32_t),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(raw_ptr(d_one_batch), one_batch.data(), sizeof(int32_t),
                             cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(raw_ptr(d_one_position), one_position.data(),
                             sizeof(int32_t), cudaMemcpyHostToDevice));

  CUDA_CHECK(gemma4_kv_cache_write_bf16(
      raw_ptr(d_cache_k), raw_ptr(d_cache_v), config, raw_ptr(d_page_table), raw_ptr(d_token_batch),
      raw_ptr(d_token_position), seq_len, layer, raw_ptr(d_k), raw_ptr(d_v),
      stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Each lambda enqueues exactly the work named by its label. The timing helper
  // wraps these in CUDA events on the same stream and optionally flushes L2.
  auto prefill_write = [&]() {
    CUDA_CHECK(gemma4_kv_cache_write_bf16(
        raw_ptr(d_cache_k), raw_ptr(d_cache_v), config, raw_ptr(d_page_table), raw_ptr(d_token_batch),
        raw_ptr(d_token_position), seq_len, layer, raw_ptr(d_k), raw_ptr(d_v),
        stream));
  };
  auto decode_write = [&]() {
    CUDA_CHECK(gemma4_kv_cache_write_bf16(
        raw_ptr(d_cache_k), raw_ptr(d_cache_v), config, raw_ptr(d_page_table), raw_ptr(d_one_batch),
        raw_ptr(d_one_position), 1, layer, raw_ptr(d_one_k), raw_ptr(d_one_v),
        stream));
  };

  gemma4_bench_print_timing_stats(
      "prefill_cache_write",
      time_cuda_samples(prefill_write, stream, warmup, iters, samples,
                        cold_cache, d_l2_scratch,
                        flush_bytes / sizeof(uint32_t)));
  gemma4_bench_print_timing_stats(
      "decode_cache_write",
      time_cuda_samples(decode_write, stream, warmup, iters, samples,
                        cold_cache, d_l2_scratch,
                        flush_bytes / sizeof(uint32_t)));

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
