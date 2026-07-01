#include "gemma4_rmsnorm.cuh"
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

std::vector<int> row_counts_up_to(int max_rows) {
  std::vector<int> counts;
  for (int rows : {4, 16, 64, 256, 1024, 4096, 8192}) {
    if (rows <= max_rows) {
      counts.push_back(rows);
    }
  }
  if (counts.empty() || counts.back() != max_rows) {
    counts.push_back(max_rows);
  }
  return counts;
}

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 20;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 5;
  const int max_rows = argc > 4 ? std::atoi(argv[4]) : 1024;

  if (iters <= 0 || warmup < 0 || trials <= 0 || max_rows <= 1) {
    std::fprintf(stderr,
                 "usage: %s [iters=200] [warmup=20] [trials=5] "
                 "[max_rows=1024]\n",
                 argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  gemma4_bench_print_common_metadata("rmsnorm_hidden_fused_bench");

  constexpr int width = GEMMA4_HIDDEN_SIZE;
  const size_t max_elems = static_cast<size_t>(max_rows) * width;
  const uint64_t seed = make_seed("GEMMA4_RMSNORM_BENCH_SEED");

  thrust::device_vector<__nv_bfloat16> d_inp1(max_elems);
  thrust::device_vector<__nv_bfloat16> d_inp2(max_elems);
  thrust::device_vector<__nv_bfloat16> d_weight(width);
  thrust::device_vector<__nv_bfloat16> d_residual(max_elems);
  thrust::device_vector<__nv_bfloat16> d_normed(max_elems);
  thrust::device_vector<__nv_bfloat16> d_ref_residual(max_elems);
  thrust::device_vector<__nv_bfloat16> d_ref_normed(max_elems);

  fill_random_bf16(raw_ptr(d_inp1), max_elems, seed ^ 0x1001u, 1.0f, stream);
  fill_random_bf16(raw_ptr(d_inp2), max_elems, seed ^ 0x2002u, 1.0f, stream);
  fill_random_bf16(raw_ptr(d_weight), width, seed ^ 0x3003u, 0.5f, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::printf("benchmark_contract name=rmsnorm_hidden_fused measurement=graph_replayed_fused_kernel "
              "timing=cuda_events_same_stream cache=warm_repeated_buffers "
              "launch_overhead=cuda_graph_replay aggregation=raw_trial_samples "
              "correctness=fused_vs_split_residual_then_rmsnorm warmup_graph_launches=%d "
              "iters=%d trials=%d dtype=bf16 seed=0x%llx\n",
              warmup, iters, trials, static_cast<unsigned long long>(seed));
  std::printf("shape=width%d,max_rows=%d\n", width, max_rows);
  std::printf("rows,fused_graph_best_ms,fused_graph_avg_ms,"
              "fused_graph_gib_s\n");

  for (int rows : row_counts_up_to(max_rows)) {
    const double bytes =
        double(rows) * width * sizeof(__nv_bfloat16) * 5.0;
    auto run_fused = [&]() {
      CUDA_CHECK(gemma4_residual_add_rmsnorm_bf16(
          raw_ptr(d_residual), raw_ptr(d_normed), raw_ptr(d_inp1), raw_ptr(d_inp2), raw_ptr(d_weight), rows, width,
          GEMMA4_RMS_NORM_EPS, stream));
    };
    auto run_split = [&]() {
      CUDA_CHECK(gemma4_residual_add_bf16(
          raw_ptr(d_ref_residual), raw_ptr(d_inp1), raw_ptr(d_inp2),
          rows * width, stream));
      CUDA_CHECK(gemma4_rmsnorm_bf16(
          raw_ptr(d_ref_normed), raw_ptr(d_ref_residual), raw_ptr(d_weight),
          rows, width, GEMMA4_RMS_NORM_EPS, stream));
    };

    run_fused();
    run_split();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const int elems = rows * width;
    const DiffStats residual_diff =
        diff_stats_bf16(raw_ptr(d_residual), raw_ptr(d_ref_residual), elems);
    const DiffStats normed_diff =
        diff_stats_bf16(raw_ptr(d_normed), raw_ptr(d_ref_normed), elems);
    if (residual_diff.max_abs > 0.0f || normed_diff.max_abs > 0.015625f) {
      throw std::runtime_error("fused residual+rmsnorm correctness failed");
    }

    const TimingStats graph_stats =
        time_ms_graph(run_fused, stream, warmup, iters, trials);
    std::printf("%d,%.6f,%.6f,%.3f\n", rows, graph_stats.best_ms,
                graph_stats.avg_ms,
                gib_per_second(bytes, graph_stats.best_ms));
    char context[64];
    std::snprintf(context, sizeof(context), "rows=%d", rows);
    gemma4_bench_print_timing_stats("rmsnorm_hidden_fused_graph", context,
                                    graph_stats);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
