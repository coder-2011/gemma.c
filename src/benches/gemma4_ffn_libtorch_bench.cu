#include "gemma4_bench_utils.cuh"
#include "gemma4_ffn.cuh"
#include "gemma4.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/version.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

// Replays a captured LibTorch CUDA graph and reports per-original-iteration time.
template <typename Fn>
TimingStats time_libtorch_graph_ms(Fn &&fn,
                                   c10::cuda::CUDAStream stream,
                                   int warmup,
                                   int iters,
                                   int trials) {
  c10::cuda::CUDAStreamGuard guard(stream);
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

  at::cuda::CUDAGraph graph;
  graph.capture_begin();
  for (int i = 0; i < iters; ++i) fn();
  graph.capture_end();

  TimingStats stats =
      time_ms([&]() { graph.replay(); }, stream.stream(), warmup, 1, trials);
  stats.best_ms /= static_cast<float>(iters);
  stats.avg_ms /= static_cast<float>(iters);
  return stats;
}

// Returns a CUDA BF16 pointer for custom C ABI calls.
__nv_bfloat16 *bf16_ptr(at::Tensor &tensor) {
  return reinterpret_cast<__nv_bfloat16 *>(tensor.data_ptr());
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

  const uint64_t seed = make_seed("GEMMA4_FFN_LIBTORCH_BENCH_SEED");
  const c10::cuda::CUDAStream torch_stream = c10::cuda::getStreamFromPool(false, 0);
  c10::cuda::CUDAStreamGuard torch_guard(torch_stream);
  const cudaStream_t stream = torch_stream.stream();
  const auto bf16_options = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);

  const size_t x_elems = static_cast<size_t>(tokens) * GEMMA4_HIDDEN_SIZE;
  const size_t act_elems = static_cast<size_t>(tokens) * GEMMA4_INTERMEDIATE_SIZE;
  const size_t gate_up_weight_elems = static_cast<size_t>(GEMMA4_PACKED_FFN_SIZE) * GEMMA4_HIDDEN_SIZE;
  const size_t down_weight_elems = static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE;

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
  at::Tensor custom_prefill_out = at::empty({tokens, GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor w_gate_up_swizzled = at::empty_like(w_gate_up);
  at::Tensor w_down_swizzled = at::empty_like(w_down);
  at::Tensor custom_residual_out = at::empty({GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor custom_normed_out = at::empty({GEMMA4_HIDDEN_SIZE}, bf16_options);
  at::Tensor prefill_scratch_buffer = at::empty(
      {static_cast<int64_t>(gemma4_ffn_prefill_scratch_elements(tokens))}, bf16_options);
  DeviceBuffer<Gemma4FfnDecodeScratch> d_custom_scratch(1);

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
  DeviceBuffer<uint32_t> d_flush_in(flush_count);
  DeviceBuffer<uint32_t> d_flush_out(kFlushOutCount);
  CUDA_CHECK(cudaMemsetAsync(d_flush_in, 0x5a, flush_bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(d_flush_out, 0, kFlushOutCount * sizeof(uint32_t), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const double geglu_bytes =
      double(x_elems) * sizeof(__nv_bfloat16) +
      double(gate_up_weight_elems) * sizeof(__nv_bfloat16) +
      double(act_elems) * sizeof(__nv_bfloat16);
  const double down_bytes =
      double(act_elems) * sizeof(__nv_bfloat16) +
      double(down_weight_elems) * sizeof(__nv_bfloat16) +
      double(x_elems) * sizeof(__nv_bfloat16);
  const double full_bytes = geglu_bytes + down_bytes;
  const double custom_decode_bytes =
      full_bytes + double(x_elems + GEMMA4_HIDDEN_SIZE + 2 * x_elems) *
                       sizeof(__nv_bfloat16);
  const double custom_prefill_bytes =
      full_bytes + double(4 * x_elems) * sizeof(__nv_bfloat16);

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
  auto run_custom_prefill_mlp = [&]() {
    CUDA_CHECK(gemma4_ffn_prefill_mlp_bf16(
        bf16_ptr(custom_prefill_out), bf16_ptr(x), bf16_ptr(w_gate_up_swizzled),
        bf16_ptr(w_down_swizzled), prefill_scratch, tokens, stream));
  };
  auto run_custom_decode = [&]() {
    CUDA_CHECK(gemma4_ffn_decode_fused_bf16(
        bf16_ptr(custom_residual_out), bf16_ptr(custom_normed_out), bf16_ptr(x),
        bf16_ptr(residual), bf16_ptr(rms_weight), bf16_ptr(w_gate_up_swizzled),
        bf16_ptr(w_down_swizzled), d_custom_scratch.get(), GEMMA4_RMS_NORM_EPS, stream));
  };
  auto run_custom_clear = [&]() {
    CUDA_CHECK(cudaMemsetAsync(d_custom_scratch.get(), 0, sizeof(Gemma4FfnDecodeScratch), stream));
  };
  auto run_cold_custom_decode = [&]() {
    flush_cache(d_flush_in, d_flush_out, flush_count, stream);
    run_custom_decode();
  };
  auto run_cold_custom_clear = [&]() {
    flush_cache(d_flush_in, d_flush_out, flush_count, stream);
    run_custom_clear();
  };

  run_libtorch_full();
  run_custom_prefill_mlp();
  run_custom_decode();
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const DiffStats custom_prefill_vs_libtorch =
      diff_stats_bf16(bf16_ptr(custom_prefill_out), bf16_ptr(torch_out), int(x_elems));

  const TimingStats torch_geglu_stats = time_ms(run_libtorch_geglu_only, stream, warmup, iters, trials);
  const TimingStats torch_down_stats = time_ms(run_libtorch_down_only, stream, warmup, iters, trials);
  const TimingStats torch_full_stats = time_ms(run_libtorch_full, stream, warmup, iters, trials);
  const TimingStats custom_decode_stats = time_ms(run_custom_decode, stream, warmup, iters, trials);
  const TimingStats custom_prefill_stats = time_ms(run_custom_prefill_mlp, stream, warmup, iters, trials);
  const TimingStats custom_clear_stats = time_ms(run_custom_clear, stream, warmup, iters, trials);
  const TimingStats cold_custom_decode_stats = time_ms(run_cold_custom_decode, stream, warmup, iters, trials);
  const TimingStats cold_custom_clear_stats = time_ms(run_cold_custom_clear, stream, warmup, iters, trials);
  const TimingStats cold_flush_stats = time_ms([&]() { flush_cache(d_flush_in, d_flush_out, flush_count, stream); }, stream, warmup, iters, trials);
  const TimingStats torch_geglu_graph_stats = time_libtorch_graph_ms(run_libtorch_geglu_only, torch_stream, warmup, iters, trials);
  const TimingStats torch_down_graph_stats = time_libtorch_graph_ms(run_libtorch_down_only, torch_stream, warmup, iters, trials);
  const TimingStats torch_full_graph_stats = time_libtorch_graph_ms(run_libtorch_full, torch_stream, warmup, iters, trials);
  const TimingStats custom_decode_graph_stats = time_ms_graph(run_custom_decode, stream, warmup, iters, trials);
  const TimingStats custom_prefill_graph_stats = time_ms_graph(run_custom_prefill_mlp, stream, warmup, iters, trials);
  const TimingStats custom_clear_graph_stats = time_ms_graph(run_custom_clear, stream, warmup, iters, trials);

  std::printf("benchmark_contract name=ffn_libtorch_bench timing=cuda_events_same_stream cache=warm_repeated_buffers_and_cold_l2_flush launch_overhead=direct_rows_include_enqueue_graph_rows_exclude_capture warmup=%d iters=%d trials=%d tokens=%d dtype=bf16\n", warmup, iters, trials, tokens);
  std::printf("benchmark_env gpu=\"%s\" l2_cache_bytes=%d torch_version=%s seed=0x%llx\n", prop.name, prop.l2CacheSize, TORCH_VERSION, static_cast<unsigned long long>(seed));
  std::printf("benchmark_correctness path=custom_prefill_mlp_vs_libtorch_full max_abs=%.6g mean_abs=%.6g max_rel=%.6g\n", custom_prefill_vs_libtorch.max_abs, custom_prefill_vs_libtorch.mean_abs, custom_prefill_vs_libtorch.max_rel);
  std::printf("path,best_ms,avg_ms,graph_best_ms,graph_avg_ms,rough_gib_s,max_abs_vs_libtorch\n");
  std::printf("libtorch_geglu,%.6f,%.6f,%.6f,%.6f,%.3f,0\n", torch_geglu_stats.best_ms, torch_geglu_stats.avg_ms, torch_geglu_graph_stats.best_ms, torch_geglu_graph_stats.avg_ms, gib_per_second(geglu_bytes, torch_geglu_stats.best_ms));
  std::printf("libtorch_down,%.6f,%.6f,%.6f,%.6f,%.3f,0\n", torch_down_stats.best_ms, torch_down_stats.avg_ms, torch_down_graph_stats.best_ms, torch_down_graph_stats.avg_ms, gib_per_second(down_bytes, torch_down_stats.best_ms));
  std::printf("libtorch_full_ffn,%.6f,%.6f,%.6f,%.6f,%.3f,0\n", torch_full_stats.best_ms, torch_full_stats.avg_ms, torch_full_graph_stats.best_ms, torch_full_graph_stats.avg_ms, gib_per_second(full_bytes, torch_full_stats.best_ms));
  std::printf("custom_fused_decode,%.6f,%.6f,%.6f,%.6f,%.3f,\n", custom_decode_stats.best_ms, custom_decode_stats.avg_ms, custom_decode_graph_stats.best_ms, custom_decode_graph_stats.avg_ms, gib_per_second(custom_decode_bytes, custom_decode_stats.best_ms));
  std::printf("custom_prefill_mlp,%.6f,%.6f,%.6f,%.6f,%.3f,%.6g\n", custom_prefill_stats.best_ms, custom_prefill_stats.avg_ms, custom_prefill_graph_stats.best_ms, custom_prefill_graph_stats.avg_ms, gib_per_second(custom_prefill_bytes, custom_prefill_stats.best_ms), custom_prefill_vs_libtorch.max_abs);
  std::printf("custom_scratch_clear,%.6f,%.6f,%.6f,%.6f,0.000,\n", custom_clear_stats.best_ms, custom_clear_stats.avg_ms, custom_clear_graph_stats.best_ms, custom_clear_graph_stats.avg_ms);
  std::printf("custom_fused_decode_cold,%.6f,%.6f,-1.000000,-1.000000,%.3f,\n", cold_custom_decode_stats.best_ms, cold_custom_decode_stats.avg_ms, gib_per_second(custom_decode_bytes, cold_custom_decode_stats.best_ms));
  std::printf("custom_scratch_clear_cold,%.6f,%.6f,-1.000000,-1.000000,0.000,\n", cold_custom_clear_stats.best_ms, cold_custom_clear_stats.avg_ms);
  std::printf("cache_flush_only,%.6f,%.6f,-1.000000,-1.000000,0.000,\n", cold_flush_stats.best_ms, cold_flush_stats.avg_ms);
  std::printf("derived_metric,value\n");
  std::printf("libtorch_full_graph_best_ms,%.6f\n", torch_full_graph_stats.best_ms);
  std::printf("custom_prefill_graph_best_ms,%.6f\n", custom_prefill_graph_stats.best_ms);
  std::printf("custom_decode_graph_best_ms,%.6f\n", custom_decode_graph_stats.best_ms);
  std::printf("custom_scratch_clear_graph_best_ms,%.6f\n", custom_clear_graph_stats.best_ms);
  std::printf("custom_decode_minus_clear_graph_ms,%.6f\n", custom_decode_graph_stats.best_ms - custom_clear_graph_stats.best_ms);
  std::printf("cold_decode_minus_flush_ms,%.6f\n", cold_custom_decode_stats.best_ms - cold_flush_stats.best_ms);
  std::printf("cold_clear_minus_flush_ms,%.6f\n", cold_custom_clear_stats.best_ms - cold_flush_stats.best_ms);
  std::printf("cold_decode_minus_flush_clear_ms,%.6f\n", cold_custom_decode_stats.best_ms - cold_custom_clear_stats.best_ms);
  std::printf("custom_prefill_graph_vs_libtorch_full_graph_speedup,%.6f\n", torch_full_graph_stats.best_ms / custom_prefill_graph_stats.best_ms);
  std::printf("custom_decode_graph_vs_libtorch_full_graph_speedup,%.6f\n", torch_full_graph_stats.best_ms / custom_decode_graph_stats.best_ms);
  return 0;
}
