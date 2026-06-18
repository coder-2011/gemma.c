#include "gemma4_embedding_gather.cuh"
#include "gemma4_bench_utils.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

namespace {

struct GatherTimingStats {
  float min_ms = 0.0f;
  float median_ms = 0.0f;
  float avg_ms = 0.0f;
  float max_ms = 0.0f;
  std::vector<float> samples_ms;
};

__global__ void l2_flush_kernel(uint32_t *__restrict__ scratch,
                                size_t words) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < words) {
    scratch[idx] = scratch[idx] * 1664525u + 1013904223u;
  }
}

std::vector<int> token_counts_up_to(int max_tokens) {
  std::vector<int> counts;
  for (int count : {1, 4, 16, 64, 256, 1024, 4096, 8192}) {
    if (count <= max_tokens) {
      counts.push_back(count);
    }
  }
  if (counts.empty() || counts.back() != max_tokens) {
    counts.push_back(max_tokens);
  }
  return counts;
}

void fill_token_ids(std::vector<int32_t> &token_ids, int count, int vocab_size) {
  uint32_t state = 0x12345678u;
  for (int i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    token_ids[i] = static_cast<int32_t>(state % static_cast<uint32_t>(vocab_size));
  }
}

void flush_l2(uint32_t *scratch, size_t words, cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks =
      static_cast<int>((words + static_cast<size_t>(threads) - 1) /
                       static_cast<size_t>(threads));
  l2_flush_kernel<<<blocks, threads, 0, stream>>>(scratch, words);
  CUDA_CHECK(cudaGetLastError());
}

template <typename Fn>
float time_gather_warm_once(Fn &&fn, cudaStream_t stream, int warmup, int iters) {
  return time_ms_once(fn, stream, warmup, iters);
}

template <typename Fn>
float time_gather_cold_once(Fn &&fn,
                            uint32_t *flush_scratch,
                            size_t flush_words,
                            cudaStream_t stream,
                            int warmup,
                            int iters) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) {
    flush_l2(flush_scratch, flush_words, stream);
    fn();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float total_ms = 0.0f;
  for (int i = 0; i < iters; ++i) {
    flush_l2(flush_scratch, flush_words, stream);
    CUDA_CHECK(cudaEventRecord(start, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    total_ms += elapsed_ms;
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms / static_cast<float>(iters);
}

template <typename Fn>
GatherTimingStats time_gather(Fn &&fn,
                              uint32_t *flush_scratch,
                              size_t flush_words,
                              cudaStream_t stream,
                              int warmup,
                              int iters,
                              int trials,
                              bool cold_cache) {
  GatherTimingStats stats;
  stats.samples_ms.reserve(trials);
  for (int i = 0; i < trials; ++i) {
    const float ms =
        cold_cache ? time_gather_cold_once(fn, flush_scratch, flush_words,
                                           stream, warmup, iters)
                   : time_gather_warm_once(fn, stream, warmup, iters);
    stats.samples_ms.push_back(ms);
  }

  std::vector<float> sorted = stats.samples_ms;
  std::sort(sorted.begin(), sorted.end());
  stats.min_ms = sorted.front();
  stats.max_ms = sorted.back();
  stats.median_ms = sorted[sorted.size() / 2];
  stats.avg_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0f) /
                 static_cast<float>(sorted.size());
  return stats;
}

std::string format_samples(const std::vector<float> &samples) {
  std::string result;
  char buf[32];
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i != 0) result += ";";
    std::snprintf(buf, sizeof(buf), "%.6f", samples[i]);
    result += buf;
  }
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 30;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 5;
  const int max_tokens = argc > 4 ? std::atoi(argv[4]) : 4096;
  const std::string cache_mode = argc > 5 ? argv[5] : "warm";
  const int flush_mib = argc > 6 ? std::atoi(argv[6]) : 128;
  const bool cold_cache = cache_mode == "cold";

  if (iters <= 0 || warmup < 0 || trials <= 0 || max_tokens <= 0 ||
      flush_mib <= 0 || (cache_mode != "warm" && cache_mode != "cold")) {
    std::fprintf(stderr,
                 "usage: %s [iters=200] [warmup=30] [trials=5] "
                 "[max_tokens=4096] [cache=warm|cold] [flush_mib=128]\n",
                 argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  __nv_bfloat16 *d_embeddings = nullptr;
  __nv_bfloat16 *d_out = nullptr;
  int32_t *d_token_ids = nullptr;
  uint32_t *d_flush = nullptr;

  const int hidden_size = GEMMA4_HIDDEN_SIZE;
  const int vocab_size = GEMMA4_VOCAB_SIZE;
  const size_t flush_words =
      static_cast<size_t>(flush_mib) * 1024 * 1024 / sizeof(uint32_t);

  const size_t embedding_elems = static_cast<size_t>(vocab_size) * hidden_size;
  const size_t embedding_bytes = embedding_elems * sizeof(__nv_bfloat16);
  const size_t out_elems = static_cast<size_t>(max_tokens) * hidden_size;
  const size_t out_bytes = out_elems * sizeof(__nv_bfloat16);

  CUDA_CHECK(cudaMalloc(&d_embeddings, embedding_bytes));
  CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
  CUDA_CHECK(cudaMalloc(&d_token_ids, static_cast<size_t>(max_tokens) * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_flush, flush_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemsetAsync(d_embeddings, 0, embedding_bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(d_out, 0, out_bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(d_flush, 1, flush_words * sizeof(uint32_t), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<int32_t> h_token_ids(max_tokens);

  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  std::printf("device=%s\n", prop.name);
  std::printf("shape=hidden%d,vocab%d,embedding_bytes=%zu,"
              "out_max_tokens=%d\n",
              hidden_size, vocab_size, embedding_bytes, max_tokens);
  std::printf("contract=embedding_gather_microbenchmark timing=CUDA_event_gpu_timeline "
              "cache_mode=%s l2_flush_bytes=%zu launch_overhead=queued_launches_only "
              "host_wall_time=excluded stability_scope=single_process "
              "min_effect_for_claim_pct=5 iters=%d warmup=%d trials=%d\n",
              cache_mode.c_str(),
              cold_cache ? flush_words * sizeof(uint32_t) : size_t{0},
              iters, warmup, trials);
  std::printf("tokens,min_ms,median_ms,avg_ms,max_ms,median_effective_gib_s,"
              "effective_mib,samples_ms\n");

  for (int token_count : token_counts_up_to(max_tokens)) {
    fill_token_ids(h_token_ids, token_count, vocab_size);
    CUDA_CHECK(cudaMemcpyAsync(d_token_ids, h_token_ids.data(),
                               static_cast<size_t>(token_count) *
                                   sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto run_gather = [&]() {
      CUDA_CHECK(gemma4_embedding_gather_bf16(
          d_out, d_token_ids, d_embeddings, token_count, stream));
    };

    run_gather();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    GatherTimingStats stats =
        time_gather(run_gather, d_flush, flush_words, stream, warmup, iters,
                    trials, cold_cache);
    const double moved_bytes = 2.0 * static_cast<double>(token_count) *
                               hidden_size * sizeof(__nv_bfloat16);
    const double moved_gib = moved_bytes / (1024.0 * 1024.0 * 1024.0);
    const double median_gib_s =
        moved_gib / (static_cast<double>(stats.median_ms) / 1000.0);
    const double moved_mib = moved_bytes / (1024.0 * 1024.0);
    const std::string samples = format_samples(stats.samples_ms);

    std::printf("%d,%.6f,%.6f,%.6f,%.6f,%.3f,%.3f,%s\n", token_count,
                stats.min_ms, stats.median_ms, stats.avg_ms, stats.max_ms,
                median_gib_s, moved_mib, samples.c_str());
  }

  CUDA_CHECK(cudaFree(d_embeddings));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_token_ids));
  CUDA_CHECK(cudaFree(d_flush));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
