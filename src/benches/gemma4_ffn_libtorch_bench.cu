#include "gemma4_bench_utils.cuh"
#include "gemma4_ffn.cuh"
#include "gemma4.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/version.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

// Returns a CUDA BF16 pointer for custom C ABI calls.
__nv_bfloat16 *bf16_ptr(at::Tensor &tensor) {
  return reinterpret_cast<__nv_bfloat16 *>(tensor.data_ptr());
}

// Times work with a device-wide sync so escaped-stream library work is included.
template <typename Fn>
TimingStats time_ms_device_sync(Fn &&fn, int warmup, int iters, int trials) {
  TimingStats stats;
  stats.best_ms = INFINITY;
  for (int trial = 0; trial < trials; ++trial) {
    for (int i = 0; i < warmup; ++i) {
      fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      fn();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> elapsed = stop - start;
    const float ms = static_cast<float>(elapsed.count()) /
                     static_cast<float>(iters);
    stats.best_ms = std::min(stats.best_ms, ms);
    stats.avg_ms += ms;
  }
  stats.avg_ms /= static_cast<float>(trials);
  return stats;
}

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 100;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 20;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 3;
  const int tokens = argc > 4 ? std::atoi(argv[4]) : 1;
  if (iters <= 0 || warmup < 0 || trials <= 0 || tokens <= 0) {
    std::fprintf(stderr, "usage: %s [iters=100] [warmup=20] [trials=3] [tokens=1]\n", argv[0]);
    return 1;
  }
  constexpr int kDecodeRows = 1;

  const uint64_t seed = make_seed("GEMMA4_FFN_LIBTORCH_BENCH_SEED");
  const c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard torch_guard(torch_stream);
  const cudaStream_t stream = torch_stream.stream();
  const auto bf16_options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);

  const size_t x_elems = static_cast<size_t>(tokens) * GEMMA4_HIDDEN_SIZE;

  at::manual_seed(static_cast<int64_t>(seed ^ 0x1001u));
  at::Tensor x = at::empty({tokens, GEMMA4_HIDDEN_SIZE}, bf16_options).uniform_(-0.05, 0.05);
  at::manual_seed(static_cast<int64_t>(seed ^ 0x2002u));
  at::Tensor w_gate_up = at::empty({GEMMA4_PACKED_FFN_SIZE, GEMMA4_HIDDEN_SIZE}, bf16_options).uniform_(-0.01, 0.01);
  at::manual_seed(static_cast<int64_t>(seed ^ 0x3003u));
  at::Tensor w_down = at::empty({GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE}, bf16_options).uniform_(-0.01, 0.01);
  at::manual_seed(static_cast<int64_t>(seed ^ 0x4004u));
  at::Tensor residual = at::empty({tokens, GEMMA4_HIDDEN_SIZE}, bf16_options).uniform_(-0.05, 0.05);
  at::manual_seed(static_cast<int64_t>(seed ^ 0x5005u));
  at::Tensor rms_weight = at::empty({GEMMA4_HIDDEN_SIZE}, bf16_options).uniform_(0.95, 1.05);

  at::Tensor w_gate_up_t = w_gate_up.t();
  at::Tensor gate_up = at::empty({tokens, GEMMA4_PACKED_FFN_SIZE}, bf16_options);
  at::Tensor gate = gate_up.slice(1, 0, GEMMA4_INTERMEDIATE_SIZE);
  at::Tensor up = gate_up.slice(1, GEMMA4_INTERMEDIATE_SIZE, GEMMA4_PACKED_FFN_SIZE);
  at::Tensor torch_act = at::empty({tokens, GEMMA4_INTERMEDIATE_SIZE}, bf16_options);
  at::Tensor torch_out = at::empty({tokens, GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor torch_decode_normed = at::empty({tokens, GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor torch_decode_residual = at::empty({tokens, GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor custom_prefill_out = at::empty({tokens, GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor w_gate_up_swizzled = at::empty_like(w_gate_up);
  at::Tensor w_down_swizzled = at::empty_like(w_down);
  at::Tensor custom_residual_out = at::empty({GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor custom_normed_out = at::empty({GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor prefill_scratch_buffer = at::empty(
      {static_cast<int64_t>(gemma4_ffn_prefill_scratch_elements(tokens))}, bf16_options);
  thrust::device_vector<unsigned char> d_custom_scratch(
      sizeof(Gemma4FfnDecodeScratch));

  CUDA_CHECK(gemma4_ffn_decode_swizzle_weights_bf16(
      bf16_ptr(w_gate_up_swizzled), bf16_ptr(w_gate_up),
      bf16_ptr(w_down_swizzled), bf16_ptr(w_down), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  constexpr size_t kMinFlushBytes = 256ull * 1024ull * 1024ull;
  constexpr size_t kFlushOutCount = size_t(4096) * 256;
  const size_t flush_bytes =
      std::max<size_t>(kMinFlushBytes,
                       static_cast<size_t>(prop.l2CacheSize) * 4ull);
  const size_t flush_count = flush_bytes / sizeof(uint32_t);
  thrust::device_vector<uint32_t> d_flush_in(flush_count);
  thrust::device_vector<uint32_t> d_flush_out(kFlushOutCount);
  CUDA_CHECK(cudaMemsetAsync(raw_ptr(d_flush_in), 0x5a, flush_bytes));
  CUDA_CHECK(cudaMemsetAsync(raw_ptr(d_flush_out), 0, kFlushOutCount * sizeof(uint32_t)));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  Gemma4FfnPrefillScratch prefill_scratch =
      gemma4_ffn_prefill_scratch_from_buffer(bf16_ptr(prefill_scratch_buffer), tokens);
  auto run_libtorch_geglu_only = [&]() {
    at::mm_out(gate_up, x, w_gate_up_t);
    at::gelu_out(torch_act, gate, "tanh");
    torch_act.mul_(up);
  };
  auto run_libtorch_down_only = [&]() { at::mm_out(torch_out, torch_act, w_down); };
  auto run_libtorch_full = [&]() {
    at::mm_out(gate_up, x, w_gate_up_t);
    at::gelu_out(torch_act, gate, "tanh");
    torch_act.mul_(up);
    at::mm_out(torch_out, torch_act, w_down);
  };
  auto run_libtorch_decode_reference = [&]() {
    run_libtorch_full();
    at::Tensor torch_out_f32 = torch_out.to(at::kFloat);
    at::Tensor rms_weight_f32 = rms_weight.to(at::kFloat);
    at::Tensor rms_scale = at::rsqrt(
        torch_out_f32.square().mean(1, true) + GEMMA4_RMS_NORM_EPS);
    at::Tensor normed_f32 = torch_out_f32 * rms_scale * rms_weight_f32;
    torch_decode_normed.copy_(normed_f32.to(at::kBFloat16));
    at::Tensor residual_f32 = residual.to(at::kFloat);
    torch_decode_residual.copy_(
        (residual_f32 + torch_decode_normed.to(at::kFloat)).to(at::kBFloat16));
  };
  auto run_custom_prefill_mlp = [&]() {
    CUDA_CHECK(gemma4_ffn_prefill_mlp_bf16(
        bf16_ptr(custom_prefill_out), bf16_ptr(x), bf16_ptr(w_gate_up_swizzled),
        bf16_ptr(w_down_swizzled), prefill_scratch, tokens, stream));
  };
  auto run_custom_decode = [&]() {
    CUDA_CHECK(gemma4_ffn_decode_fused_bf16(
        bf16_ptr(custom_residual_out), bf16_ptr(custom_normed_out), bf16_ptr(x),
        bf16_ptr(residual), bf16_ptr(rms_weight), bf16_ptr(w_gate_up_swizzled),
        bf16_ptr(w_down_swizzled),
        reinterpret_cast<Gemma4FfnDecodeScratch *>(raw_ptr(d_custom_scratch)),
        nullptr, GEMMA4_RMS_NORM_EPS, stream));
  };
  auto run_custom_clear = [&]() {
    CUDA_CHECK(cudaMemsetAsync(raw_ptr(d_custom_scratch), 0, sizeof(Gemma4FfnDecodeScratch)));
  };
  auto run_cold_custom_decode = [&]() {
    flush_cache(raw_ptr(d_flush_in), raw_ptr(d_flush_out), flush_count, stream);
    run_custom_decode();
  };
  auto run_cold_custom_clear = [&]() {
    flush_cache(raw_ptr(d_flush_in), raw_ptr(d_flush_out), flush_count, stream);
    run_custom_clear();
  };

  run_libtorch_decode_reference();
  run_custom_prefill_mlp();
  run_custom_decode();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const DiffStats custom_prefill_vs_libtorch =
      diff_stats_bf16(bf16_ptr(custom_prefill_out), bf16_ptr(torch_out), int(x_elems));
  at::Tensor torch_decode_normed_row = torch_decode_normed.select(0, 0);
  at::Tensor torch_decode_residual_row = torch_decode_residual.select(0, 0);
  const DiffStats custom_decode_normed_vs_libtorch =
      diff_stats_bf16(bf16_ptr(custom_normed_out),
                      bf16_ptr(torch_decode_normed_row), GEMMA4_HIDDEN_SIZE);
  const DiffStats custom_decode_residual_vs_libtorch =
      diff_stats_bf16(bf16_ptr(custom_residual_out),
                      bf16_ptr(torch_decode_residual_row), GEMMA4_HIDDEN_SIZE);
  const float custom_decode_max_abs = std::max(
      custom_decode_normed_vs_libtorch.max_abs,
      custom_decode_residual_vs_libtorch.max_abs);
  constexpr float kDecodeTolerance = 0.125f;
  if (custom_decode_max_abs > kDecodeTolerance) {
    std::fprintf(stderr,
                 "custom decode correctness failed max_abs=%.6g "
                 "tolerance=%.6g\n",
                 custom_decode_max_abs, kDecodeTolerance);
    return 1;
  }

  const TimingStats torch_geglu_stats = time_ms(run_libtorch_geglu_only, stream, warmup, iters, trials);
  const TimingStats torch_down_stats = time_ms(run_libtorch_down_only, stream, warmup, iters, trials);
  const TimingStats torch_full_stats = time_ms(run_libtorch_full, stream, warmup, iters, trials);
  const TimingStats custom_decode_wall_stats =
      time_ms_device_sync(run_custom_decode, warmup, iters, trials);
  const TimingStats custom_prefill_stats = time_ms(run_custom_prefill_mlp, stream, warmup, iters, trials);
  const TimingStats custom_clear_stats = time_ms(run_custom_clear, stream, warmup, iters, trials);
  const TimingStats cold_custom_decode_stats = time_ms(run_cold_custom_decode, stream, warmup, iters, trials);
  const TimingStats cold_custom_clear_stats = time_ms(run_cold_custom_clear, stream, warmup, iters, trials);
  const TimingStats cold_flush_stats = time_ms(
      [&]() {
        flush_cache(raw_ptr(d_flush_in), raw_ptr(d_flush_out), flush_count,
                    stream);
      },
      stream, warmup, iters, trials);

  std::printf("benchmark_contract name=ffn_libtorch_bench timing=cuda_events_same_stream_decode_uses_device_sync_wall cache=warm_repeated_buffers_and_cold_l2_flush launch_overhead=events_exclude_host_enqueue_decode_sync_includes_host_overhead warmup=%d iters=%d trials=%d prefill_rows=%d decode_rows=%d dtype=bf16\n", warmup, iters, trials, tokens, kDecodeRows);
  std::printf("benchmark_env gpu=\"%s\" l2_cache_bytes=%d torch_version=%s seed=0x%llx\n", prop.name, prop.l2CacheSize, TORCH_VERSION, static_cast<unsigned long long>(seed));
  std::printf("benchmark_correctness path=custom_prefill_mlp_vs_libtorch_full max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n", custom_prefill_vs_libtorch.max_abs, custom_prefill_vs_libtorch.mean_abs, custom_prefill_vs_libtorch.max_rel);
  std::printf("benchmark_correctness path=custom_decode_normed_vs_libtorch max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n", custom_decode_normed_vs_libtorch.max_abs, custom_decode_normed_vs_libtorch.mean_abs, custom_decode_normed_vs_libtorch.max_rel);
  std::printf("benchmark_correctness path=custom_decode_residual_vs_libtorch max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n", custom_decode_residual_vs_libtorch.max_abs, custom_decode_residual_vs_libtorch.mean_abs, custom_decode_residual_vs_libtorch.max_rel);
  std::printf("mode,rows,hidden,intermediate,path,best_ms,avg_ms,max_abs_vs_libtorch\n");
  std::printf("prefill,%d,%d,%d,libtorch_geglu,%.6f,%.6f,0\n", tokens, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, torch_geglu_stats.best_ms, torch_geglu_stats.avg_ms);
  std::printf("prefill,%d,%d,%d,libtorch_down,%.6f,%.6f,0\n", tokens, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, torch_down_stats.best_ms, torch_down_stats.avg_ms);
  std::printf("prefill,%d,%d,%d,libtorch_full_ffn,%.6f,%.6f,0\n", tokens, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, torch_full_stats.best_ms, torch_full_stats.avg_ms);
  std::printf("decode,%d,%d,%d,custom_fused_decode,%.6f,%.6f,%.6g\n", kDecodeRows, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, custom_decode_wall_stats.best_ms, custom_decode_wall_stats.avg_ms, custom_decode_max_abs);
  std::printf("prefill,%d,%d,%d,custom_prefill_mlp,%.6f,%.6f,%.6g\n", tokens, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, custom_prefill_stats.best_ms, custom_prefill_stats.avg_ms, custom_prefill_vs_libtorch.max_abs);
  std::printf("decode,%d,%d,%d,custom_scratch_clear,%.6f,%.6f,\n", kDecodeRows, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, custom_clear_stats.best_ms, custom_clear_stats.avg_ms);
  std::printf("decode,%d,%d,%d,custom_fused_decode_cold,%.6f,%.6f,\n", kDecodeRows, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, cold_custom_decode_stats.best_ms, cold_custom_decode_stats.avg_ms);
  std::printf("decode,%d,%d,%d,custom_scratch_clear_cold,%.6f,%.6f,\n", kDecodeRows, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, cold_custom_clear_stats.best_ms, cold_custom_clear_stats.avg_ms);
  std::printf("cache,0,%d,%d,cache_flush_only,%.6f,%.6f,\n", GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, cold_flush_stats.best_ms, cold_flush_stats.avg_ms);
  std::printf("derived_metric,mode,rows,hidden,intermediate,value\n");
  std::printf("custom_decode_wall_sync_best_ms,decode,%d,%d,%d,%.6f\n", kDecodeRows, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, custom_decode_wall_stats.best_ms);
  std::printf("cold_decode_minus_flush_ms,decode,%d,%d,%d,%.6f\n", kDecodeRows, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, cold_custom_decode_stats.best_ms - cold_flush_stats.best_ms);
  std::printf("cold_clear_minus_flush_ms,decode,%d,%d,%d,%.6f\n", kDecodeRows, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, cold_custom_clear_stats.best_ms - cold_flush_stats.best_ms);
  std::printf("cold_decode_minus_flush_clear_ms,decode,%d,%d,%d,%.6f\n", kDecodeRows, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, cold_custom_decode_stats.best_ms - cold_custom_clear_stats.best_ms);
  // Only report this ratio when both paths ran one row.
  if (tokens == kDecodeRows) {
    std::printf("custom_decode_vs_libtorch_full_speedup,decode,%d,%d,%d,%.6f\n", kDecodeRows, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE, torch_full_stats.best_ms / custom_decode_wall_stats.best_ms);
  }
  return 0;
}
