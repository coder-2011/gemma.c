#include "gemma4_embedding_gather.cuh"
#include "gemma4_bench_utils.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

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

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 30;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 5;
  const int max_tokens = argc > 4 ? std::atoi(argv[4]) : 4096;

  if (iters <= 0 || warmup < 0 || trials <= 0 || max_tokens <= 0) {
    std::fprintf(stderr,
                 "usage: %s [iters=200] [warmup=30] [trials=5] "
                 "[max_tokens=4096]\n",
                 argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  gemma4_bench_print_common_metadata("embedding_gather_bench");

  constexpr int hidden_size = GEMMA4_HIDDEN_SIZE;
  constexpr int vocab_size = GEMMA4_VOCAB_SIZE;
  const uint64_t seed = make_seed("GEMMA4_EMBEDDING_GATHER_BENCH_SEED");

  const size_t embedding_elems = static_cast<size_t>(vocab_size) * hidden_size;
  const size_t embedding_bytes = embedding_elems * sizeof(__nv_bfloat16);
  const size_t out_elems = static_cast<size_t>(max_tokens) * hidden_size;
  const size_t out_bytes = out_elems * sizeof(__nv_bfloat16);

  thrust::device_vector<__nv_bfloat16> d_embeddings(embedding_elems);
  thrust::device_vector<__nv_bfloat16> d_out(out_elems);
  thrust::device_vector<int32_t> d_token_ids(max_tokens);
  fill_random_bf16(raw_ptr(d_embeddings), embedding_elems, seed, 1.0f, stream);
  CUDA_CHECK(cudaMemsetAsync(raw_ptr(d_out), 0, out_bytes));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<int32_t> h_token_ids(max_tokens);

  std::printf("benchmark_contract name=embedding_gather measurement=gather_kernel_only "
              "timing=cuda_events_same_stream cache=warm_repeated_embedding_table "
              "launch_overhead=excluded_from_gpu_elapsed_time aggregation=raw_trial_samples "
              "correctness=sampled_rows_vs_embedding_table warmup=%d iters=%d "
              "trials=%d dtype=bf16 seed=0x%llx\n",
              warmup, iters, trials, static_cast<unsigned long long>(seed));
  std::printf("shape=hidden%d,vocab%d,embedding_bytes=%zu,out_max_tokens=%d\n",
              hidden_size, vocab_size, embedding_bytes, max_tokens);
  std::printf("tokens,best_ms,avg_ms,best_effective_gib_s,avg_effective_gib_s,effective_mib\n");

  for (int token_count : token_counts_up_to(max_tokens)) {
    fill_token_ids(h_token_ids, token_count, vocab_size);
    CUDA_CHECK(cudaMemcpyAsync(raw_ptr(d_token_ids), h_token_ids.data(),
                               static_cast<size_t>(token_count) *
                                   sizeof(int32_t),
                               cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto run_gather = [&]() {
      CUDA_CHECK(gemma4_embedding_gather_bf16(
          raw_ptr(d_out), raw_ptr(d_token_ids), raw_ptr(d_embeddings),
          token_count, stream));
    };

    run_gather();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int32_t first_token = h_token_ids.front();
    const DiffStats first_row = diff_stats_bf16(
        raw_ptr(d_out), raw_ptr(d_embeddings) + size_t(first_token) * hidden_size,
        hidden_size);
    if (first_row.max_abs != 0.0f) {
      throw std::runtime_error("embedding gather first-row correctness failed");
    }

    const TimingStats stats = time_ms(run_gather, stream, warmup, iters, trials);
    const double moved_bytes = 2.0 * static_cast<double>(token_count) *
                               hidden_size * sizeof(__nv_bfloat16);
    const double moved_gib = moved_bytes / (1024.0 * 1024.0 * 1024.0);
    const double best_gib_s = moved_gib / (static_cast<double>(stats.best_ms) / 1000.0);
    const double avg_gib_s = moved_gib / (static_cast<double>(stats.avg_ms) / 1000.0);
    const double moved_mib = moved_bytes / (1024.0 * 1024.0);

    std::printf("%d,%.6f,%.6f,%.3f,%.3f,%.3f\n", token_count,
                stats.best_ms, stats.avg_ms, best_gib_s, avg_gib_s,
                moved_mib);
    char context[64];
    std::snprintf(context, sizeof(context), "tokens=%d", token_count);
    gemma4_bench_print_timing_stats("embedding_gather", context, stats);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
