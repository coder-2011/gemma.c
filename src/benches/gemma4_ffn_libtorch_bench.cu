#include "gemma4_bench_utils.cuh"
#include "gemma4_ffn.cuh"
#include "gemma4.h"

#include <ATen/ATen.h>
#include <ATen/ops/_fused_rms_norm.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/version.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <tuple>

namespace {

// Returns a CUDA BF16 pointer for custom C ABI calls.
__nv_bfloat16 *bf16_ptr(at::Tensor &tensor) {
  return reinterpret_cast<__nv_bfloat16 *>(tensor.data_ptr());
}

// Prints one benchmark row in the existing CSV-ish FFN format.
void print_row(const char *path,
               const TimingStats &stats,
               float max_abs) {
  std::printf("decode,1,%d,%d,%s,%.6f,%.6f,%.6g\n",
              GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, path,
              stats.best_ms, stats.avg_ms, max_abs);
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
  at::Tensor torch_rstd;
  at::Tensor w_gate_up_swizzled = at::empty_like(w_gate_up);
  at::Tensor w_down_swizzled = at::empty_like(w_down);
  at::Tensor custom_residual = at::empty({GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor custom_normed = at::empty({GEMMA4_HIDDEN_SIZE}, bf16_options);
  thrust::device_vector<unsigned char> d_scratch(sizeof(Gemma4FfnDecodeScratch));

  CUDA_CHECK(gemma4_ffn_decode_swizzle_weights_bf16(
      bf16_ptr(w_gate_up_swizzled), bf16_ptr(w_gate_up),
      bf16_ptr(w_down_swizzled), bf16_ptr(w_down), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

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
  auto run_custom_decode = [&]() {
    CUDA_CHECK(gemma4_ffn_decode_fused_bf16(
        bf16_ptr(custom_residual), bf16_ptr(custom_normed), bf16_ptr(x),
        bf16_ptr(residual), bf16_ptr(rms_weight), bf16_ptr(w_gate_up_swizzled),
        bf16_ptr(w_down_swizzled),
        reinterpret_cast<Gemma4FfnDecodeScratch *>(raw_ptr(d_scratch)),
        nullptr, GEMMA4_RMS_NORM_EPS, stream));
  };

  run_libtorch_full();
  run_custom_decode();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  at::Tensor torch_normed_row = torch_normed.select(0, 0);
  at::Tensor torch_residual_row = torch_residual.select(0, 0);
  const DiffStats normed_diff =
      diff_stats_bf16(bf16_ptr(custom_normed), bf16_ptr(torch_normed_row),
                      GEMMA4_HIDDEN_SIZE);
  const DiffStats residual_diff =
      diff_stats_bf16(bf16_ptr(custom_residual), bf16_ptr(torch_residual_row),
                      GEMMA4_HIDDEN_SIZE);
  const float custom_max_abs =
      std::max(normed_diff.max_abs, residual_diff.max_abs);
  constexpr float kTolerance = 0.125f;
  if (custom_max_abs > kTolerance) {
    std::fprintf(stderr,
                 "custom decode correctness failed max_abs=%.6g "
                 "tolerance=%.6g\n",
                 custom_max_abs, kTolerance);
    return 1;
  }

  const TimingStats torch_stats =
      time_ms(run_libtorch_full, stream, warmup, iters, trials);
  const TimingStats custom_stats =
      time_ms(run_custom_decode, stream, warmup, iters, trials);

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
              "aggregation=raw_trial_samples correctness=custom_vs_libtorch "
              "warmup=%d iters=%d trials=%d rows=1 dtype=bf16\n",
              warmup, iters, trials);
  std::printf("benchmark_env torch_version=%s seed=0x%llx\n", TORCH_VERSION,
              static_cast<unsigned long long>(seed));
  std::printf("benchmark_roofline ffn_weight_mib=%.3f "
              "ffn_weight_stream_floor_ms=%.6f\n",
              ffn_weight_bytes / (1024.0 * 1024.0),
              ffn_weight_stream_floor_ms);
  std::printf("benchmark_correctness path=custom_decode_normed_vs_libtorch "
              "max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n",
              normed_diff.max_abs, normed_diff.mean_abs, normed_diff.max_rel);
  std::printf("benchmark_correctness path=custom_decode_residual_vs_libtorch "
              "max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n",
              residual_diff.max_abs, residual_diff.mean_abs,
              residual_diff.max_rel);
  gemma4_bench_print_timing_stats(
      "ffn_libtorch_bench", "path=libtorch_full_ffn", torch_stats);
  gemma4_bench_print_timing_stats(
      "ffn_libtorch_bench", "path=custom_fused_decode", custom_stats);
  std::printf("mode,rows,hidden,intermediate,path,best_ms,avg_ms,"
              "max_abs_vs_libtorch\n");
  print_row("libtorch_full_ffn", torch_stats, 0.0f);
  print_row("custom_fused_decode", custom_stats, custom_max_abs);
  std::printf("custom_decode_vs_libtorch_full_speedup,decode,1,%d,%d,%.6f\n",
              GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE,
              torch_stats.best_ms / custom_stats.best_ms);
  if (custom_stats.best_ms < ffn_weight_stream_floor_ms) {
    std::printf("benchmark_warning label=custom_fused_decode "
                "measured_ms=%.6f floor_ms=%.6f "
                "reason=below_single_ffn_weight_stream_floor\n",
                custom_stats.best_ms, ffn_weight_stream_floor_ms);
  }
  return 0;
}
