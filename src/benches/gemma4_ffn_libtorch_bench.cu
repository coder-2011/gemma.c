#include "gemma4_bench_utils.cuh"
#include "gemma4_ffn.cuh"
#include "gemma4_rmsnorm.cuh"
#include "gemma4.h"

#include <ATen/ATen.h>
#include <ATen/ops/_fused_rms_norm.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/version.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <tuple>
#include <vector>

namespace {

// Returns a CUDA BF16 pointer for custom C ABI calls.
__nv_bfloat16 *bf16_ptr(at::Tensor &tensor) {
  return reinterpret_cast<__nv_bfloat16 *>(tensor.data_ptr());
}

// Returns a const CUDA BF16 pointer for setup-time host/device copies.
const __nv_bfloat16 *bf16_ptr(const at::Tensor &tensor) {
  return reinterpret_cast<const __nv_bfloat16 *>(tensor.data_ptr());
}

// Maps a natural hidden column into the decode FFN swizzled weight layout.
int decode_hidden_col(int col) {
  const int pack = col / GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS;
  const int lane = col % GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS;
  return gemma4_ffn_decode_device::hidden_pack_swizzle_index(pack) *
             GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS +
         lane;
}

// Packs random benchmark weights into the same decode layout as the checkpoint loader.
void copy_decode_layout_weights(
    at::Tensor &w_gate_up_decode,
    at::Tensor &w_down_decode,
    const at::Tensor &w_gate_up,
    const at::Tensor &w_down,
    cudaStream_t stream) {
  const size_t gate_up_count =
      static_cast<size_t>(GEMMA4_PACKED_FFN_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t down_count =
      static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE;
  std::vector<__nv_bfloat16> h_gate_up(gate_up_count);
  std::vector<__nv_bfloat16> h_gate_up_decode(gate_up_count);
  std::vector<__nv_bfloat16> h_down(down_count);
  std::vector<__nv_bfloat16> h_down_decode(down_count);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaMemcpy(h_gate_up.data(), bf16_ptr(w_gate_up),
                        gate_up_count * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_down.data(), bf16_ptr(w_down),
                        down_count * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));

  for (int row = 0; row < GEMMA4_INTERMEDIATE_SIZE; ++row) {
    for (int col = 0; col < GEMMA4_HIDDEN_SIZE; ++col) {
      const int dst_col = decode_hidden_col(col);
      h_gate_up_decode[(2 * row) * GEMMA4_HIDDEN_SIZE + dst_col] =
          h_gate_up[row * GEMMA4_HIDDEN_SIZE + col];
      h_gate_up_decode[(2 * row + 1) * GEMMA4_HIDDEN_SIZE + dst_col] =
          h_gate_up[(GEMMA4_INTERMEDIATE_SIZE + row) * GEMMA4_HIDDEN_SIZE + col];
      h_down_decode[row * GEMMA4_HIDDEN_SIZE + dst_col] =
          h_down[row * GEMMA4_HIDDEN_SIZE + col];
    }
  }

  CUDA_CHECK(cudaMemcpy(bf16_ptr(w_gate_up_decode), h_gate_up_decode.data(),
                        gate_up_count * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(bf16_ptr(w_down_decode), h_down_decode.data(),
                        down_count * sizeof(__nv_bfloat16),
                        cudaMemcpyHostToDevice));
}

// Prints one benchmark row in the existing CSV-ish FFN format.
void print_row(const char *path,
               const TimingStats &stats,
               float max_abs) {
  std::printf("decode,1,%d,%d,%s,%.6f,%.6f,%.6g\n",
              GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, path,
              stats.best_ms, stats.avg_ms, max_abs);
}

// Times graph replay when capture works, but keeps stream-loop timing usable.
template <typename Fn>
std::optional<TimingStats> try_time_ms_graph(
    const char *path,
    Fn &&fn,
    cudaStream_t stream,
    int warmup,
    int iters,
    int trials) {
  try {
    return time_ms_graph(fn, stream, warmup, iters, trials);
  } catch (const std::exception &error) {
    std::printf("benchmark_graph_unavailable path=%s reason=\"%s\"\n",
                path, error.what());
    return std::nullopt;
  }
}

// Prints an optional graph-replay timing row using the normal timing schema.
void print_graph_stats(const char *path,
                       const std::optional<TimingStats> &stats) {
  if (!stats.has_value()) {
    return;
  }
  gemma4_bench_print_timing_stats(
      "ffn_libtorch_bench_graph", path, *stats);
}

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 100;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 20;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 3;
  if (iters <= 0 || warmup < 0 || trials <= 0) {
    std::fprintf(stderr, "usage: %s [iters=100] [warmup=20] [trials=3]\n",
                 argv[0]);
    return 1;
  }

  const uint64_t seed = make_seed("GEMMA4_FFN_LIBTORCH_BENCH_SEED");
  const c10::cuda::CUDAStream torch_stream =
      c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard torch_guard(torch_stream);
  const cudaStream_t stream = torch_stream.stream();
  const auto bf16_options =
      at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);

  gemma4_bench_print_common_metadata("ffn_libtorch_bench");

  at::manual_seed(static_cast<int64_t>(seed ^ 0x1001u));
  at::Tensor x =
      at::empty({1, GEMMA4_HIDDEN_SIZE}, bf16_options).uniform_(-0.05, 0.05);
  at::manual_seed(static_cast<int64_t>(seed ^ 0x2002u));
  at::Tensor w_gate_up =
      at::empty({GEMMA4_PACKED_FFN_SIZE, GEMMA4_HIDDEN_SIZE}, bf16_options)
          .uniform_(-0.01, 0.01);
  at::manual_seed(static_cast<int64_t>(seed ^ 0x3003u));
  at::Tensor w_down =
      at::empty({GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE}, bf16_options)
          .uniform_(-0.01, 0.01);
  at::manual_seed(static_cast<int64_t>(seed ^ 0x4004u));
  at::Tensor residual =
      at::empty({1, GEMMA4_HIDDEN_SIZE}, bf16_options).uniform_(-0.05, 0.05);
  at::manual_seed(static_cast<int64_t>(seed ^ 0x5005u));
  at::Tensor rms_weight =
      at::empty({GEMMA4_HIDDEN_SIZE}, bf16_options).uniform_(0.95, 1.05);

  at::Tensor gate_up = at::empty({1, GEMMA4_PACKED_FFN_SIZE}, bf16_options);
  at::Tensor gate = gate_up.slice(1, 0, GEMMA4_INTERMEDIATE_SIZE);
  at::Tensor up =
      gate_up.slice(1, GEMMA4_INTERMEDIATE_SIZE, GEMMA4_PACKED_FFN_SIZE);
  at::Tensor torch_act = at::empty({1, GEMMA4_INTERMEDIATE_SIZE}, bf16_options);
  at::Tensor torch_out = at::empty({1, GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor torch_normed = at::empty({1, GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor torch_residual = at::empty({1, GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor dual_mlp = at::empty({1, GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor dual_normed = at::empty({1, GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor dual_residual = at::empty({1, GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor torch_rstd;
  at::Tensor w_gate_up_swizzled = at::empty_like(w_gate_up);
  at::Tensor w_down_swizzled = at::empty_like(w_down);
  thrust::device_vector<__nv_bfloat16> d_dual_scratch(
      gemma4_ffn_prefill_scratch_elements(1));
  Gemma4FfnPrefillScratch dual_scratch =
      gemma4_ffn_prefill_scratch_from_buffer(raw_ptr(d_dual_scratch), 1);

  copy_decode_layout_weights(
      w_gate_up_swizzled, w_down_swizzled, w_gate_up, w_down, stream);

  const std::optional<at::Tensor> rms_weight_opt(rms_weight);
  auto run_libtorch_full = [&]() {
    at::mm_out(gate_up, x, w_gate_up.t());
    at::gelu_out(torch_act, gate, "tanh");
    torch_act.mul_(up);
    at::mm_out(torch_out, torch_act, w_down);
    std::tie(torch_normed, torch_rstd) = at::_fused_rms_norm(
        torch_out, {GEMMA4_HIDDEN_SIZE}, rms_weight_opt,
        GEMMA4_RMS_NORM_EPS);
    at::add_out(torch_residual, residual, torch_normed);
  };
  auto run_dualgemm_chain = [&]() {
    CUDA_CHECK(gemma4_ffn_prefill_mlp_bf16(
        bf16_ptr(dual_mlp), bf16_ptr(x), bf16_ptr(w_gate_up_swizzled),
        bf16_ptr(w_down_swizzled), dual_scratch, 1, stream));
    CUDA_CHECK(gemma4_rmsnorm_bf16(
        bf16_ptr(dual_normed), bf16_ptr(dual_mlp), bf16_ptr(rms_weight),
        1, GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_NORM_EPS, stream));
    CUDA_CHECK(gemma4_residual_add_bf16(
        bf16_ptr(dual_residual), bf16_ptr(residual), bf16_ptr(dual_normed),
        GEMMA4_HIDDEN_SIZE, stream));
  };

  run_libtorch_full();
  run_dualgemm_chain();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  at::Tensor torch_normed_row = torch_normed.select(0, 0);
  at::Tensor torch_residual_row = torch_residual.select(0, 0);
  at::Tensor dual_normed_row = dual_normed.select(0, 0);
  at::Tensor dual_residual_row = dual_residual.select(0, 0);
  const DiffStats dual_normed_diff =
      diff_stats_bf16(bf16_ptr(dual_normed_row), bf16_ptr(torch_normed_row),
                      GEMMA4_HIDDEN_SIZE);
  const DiffStats dual_residual_diff =
      diff_stats_bf16(bf16_ptr(dual_residual_row), bf16_ptr(torch_residual_row),
                      GEMMA4_HIDDEN_SIZE);
  const float dual_max_abs = dual_normed_diff.max_abs > dual_residual_diff.max_abs
                                 ? dual_normed_diff.max_abs
                                 : dual_residual_diff.max_abs;
  constexpr float kTolerance = 0.125f;
  if (dual_max_abs > kTolerance) {
    std::fprintf(stderr,
                 "FFN correctness failed dualgemm_max_abs=%.6g "
                 "tolerance=%.6g\n",
                 dual_max_abs, kTolerance);
    return 1;
  }

  const TimingStats torch_stats =
      time_ms(run_libtorch_full, stream, warmup, iters, trials);
  const TimingStats dual_stats =
      time_ms(run_dualgemm_chain, stream, warmup, iters, trials);
  const std::optional<TimingStats> dual_graph_stats =
      try_time_ms_graph("dualgemm_chain_decode_layout", run_dualgemm_chain,
                        stream, warmup, iters, trials);

  int device = 0;
  int memory_clock_khz = 0;
  int memory_bus_width_bits = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &memory_clock_khz, cudaDevAttrMemoryClockRate, device));
  CUDA_CHECK(cudaDeviceGetAttribute(
      &memory_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device));
  const double peak_dram_bytes_per_s =
      double(memory_clock_khz) * 1000.0 * 2.0 *
      (double(memory_bus_width_bits) / 8.0);
  const double ffn_weight_bytes =
      double(GEMMA4_PACKED_FFN_SIZE + GEMMA4_INTERMEDIATE_SIZE) *
      double(GEMMA4_HIDDEN_SIZE) * double(sizeof(__nv_bfloat16));
  const double ffn_weight_stream_floor_ms =
      ffn_weight_bytes / peak_dram_bytes_per_s * 1000.0;

  std::printf("benchmark_contract name=ffn_libtorch_bench "
              "timing=cuda_events_same_stream cache=warm_repeated_buffers "
              "launch_overhead=events_exclude_host_enqueue "
              "aggregation=raw_trial_samples "
              "correctness=dualgemm_vs_libtorch "
              "warmup=%d iters=%d trials=%d rows=1 dtype=bf16\n",
              warmup, iters, trials);
  std::printf("benchmark_env torch_version=%s seed=0x%llx\n", TORCH_VERSION,
              static_cast<unsigned long long>(seed));
  std::printf("benchmark_roofline ffn_weight_mib=%.3f "
              "ffn_weight_stream_floor_ms=%.6f\n",
              ffn_weight_bytes / (1024.0 * 1024.0),
              ffn_weight_stream_floor_ms);
  std::printf("benchmark_correctness path=dualgemm_chain_normed_vs_libtorch "
              "max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n",
              dual_normed_diff.max_abs, dual_normed_diff.mean_abs,
              dual_normed_diff.max_rel);
  std::printf("benchmark_correctness path=dualgemm_chain_residual_vs_libtorch "
              "max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n",
              dual_residual_diff.max_abs, dual_residual_diff.mean_abs,
              dual_residual_diff.max_rel);
  gemma4_bench_print_timing_stats(
      "ffn_libtorch_bench", "path=libtorch_full_ffn", torch_stats);
  gemma4_bench_print_timing_stats(
      "ffn_libtorch_bench", "path=dualgemm_chain_decode_layout", dual_stats);
  print_graph_stats("path=dualgemm_chain_decode_layout", dual_graph_stats);
  std::printf("mode,rows,hidden,intermediate,path,best_ms,avg_ms,"
              "max_abs_vs_libtorch\n");
  print_row("libtorch_full_ffn", torch_stats, 0.0f);
  print_row("dualgemm_chain_decode_layout", dual_stats, dual_max_abs);
  return 0;
}
