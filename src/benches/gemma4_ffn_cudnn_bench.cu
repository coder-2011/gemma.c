#include "gemma4_bench_utils.cuh"
#include "gemma4_ffn.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cudnn_frontend.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <unordered_map>
#include <vector>

namespace {

template <typename Fn>
TimingStats time_ms_cold(Fn &&fn,
                         cudaStream_t stream,
                         int warmup,
                         int iters,
                         int trials,
                         const uint32_t *flush_in,
                         uint32_t *flush_out,
                         size_t flush_count) {
  auto run_cold = [&]() {
    flush_cache(flush_in, flush_out, flush_count, stream);
    fn();
  };
  return time_ms(run_cold, stream, warmup, iters, trials);
}

namespace fe = cudnn_frontend;

struct CudnnHandle {
  cudnnHandle_t handle = nullptr;

  explicit CudnnHandle(cudaStream_t stream) {
    cudnnCreate(&handle);
    cudnnSetStream(handle, stream);
  }

  ~CudnnHandle() {
    cudnnDestroy(handle);
  }

  CudnnHandle(const CudnnHandle &) = delete;
  CudnnHandle &operator=(const CudnnHandle &) = delete;
};

struct CudnnGraphBase {
  CudnnHandle cudnn;
  fe::graph::Graph graph;
  void *workspace = nullptr;
  int64_t workspace_size = 0;

  explicit CudnnGraphBase(cudaStream_t stream) : cudnn(stream) {
    graph.set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
  }

  ~CudnnGraphBase() {
    cudaFree(workspace);
  }

  void build(const char *) {
    (void)graph.validate();
    (void)graph.build_operation_graph(cudnn.handle);
    (void)graph.create_execution_plans({fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK});
    (void)graph.check_support();
    (void)graph.build_plans();
    (void)graph.get_workspace_size(workspace_size);
    if (workspace_size > 0) {
      CUDA_CHECK(cudaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }
  }
};

std::shared_ptr<fe::graph::Tensor_attributes>
make_bf16_tensor(fe::graph::Graph &graph,
                 const char *name,
                 const std::vector<int64_t> &dim,
                 const std::vector<int64_t> &stride) {
  return graph.tensor(fe::graph::Tensor_attributes()
                          .set_name(name)
                          .set_data_type(fe::DataType_t::BFLOAT16)
                          .set_dim(dim)
                          .set_stride(stride));
}

struct CudnnGeGlu : public CudnnGraphBase {
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> w_gate;
  std::shared_ptr<fe::graph::Tensor_attributes> w_up;
  std::shared_ptr<fe::graph::Tensor_attributes> act;

  CudnnGeGlu(int tokens, cudaStream_t stream) : CudnnGraphBase(stream) {
    x = make_bf16_tensor(graph, "x", {1, tokens, GEMMA4_HIDDEN_SIZE},
                         {tokens * GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, 1});
    w_gate = make_bf16_tensor(graph, "w_gate", {1, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE},
                              {GEMMA4_HIDDEN_SIZE * GEMMA4_INTERMEDIATE_SIZE, GEMMA4_INTERMEDIATE_SIZE, 1});
    w_up = make_bf16_tensor(graph, "w_up", {1, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE},
                            {GEMMA4_HIDDEN_SIZE * GEMMA4_INTERMEDIATE_SIZE, GEMMA4_INTERMEDIATE_SIZE, 1});

    auto gate = graph.matmul(x, w_gate, fe::graph::Matmul_attributes().set_name("gate_matmul").set_compute_data_type(fe::DataType_t::FLOAT));
    gate->set_data_type(fe::DataType_t::BFLOAT16);
    auto up = graph.matmul(x, w_up, fe::graph::Matmul_attributes().set_name("up_matmul").set_compute_data_type(fe::DataType_t::FLOAT));
    up->set_data_type(fe::DataType_t::BFLOAT16);
    auto gelu = graph.pointwise(
        gate, fe::graph::Pointwise_attributes().set_name("gelu").set_mode(fe::PointwiseMode_t::GELU_APPROX_TANH_FWD).set_compute_data_type(fe::DataType_t::FLOAT));
    gelu->set_data_type(fe::DataType_t::BFLOAT16);
    act = graph.pointwise(gelu, up,
                          fe::graph::Pointwise_attributes().set_name("gate_mul").set_mode(fe::PointwiseMode_t::MUL).set_compute_data_type(fe::DataType_t::FLOAT));
    act->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);

    build("geglu");
  }

  void run(const __nv_bfloat16 *x_ptr,
           const __nv_bfloat16 *w_gate_ptr,
           const __nv_bfloat16 *w_up_ptr,
           __nv_bfloat16 *act_ptr) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
        variant_pack = {{x, const_cast<__nv_bfloat16 *>(x_ptr)},
                        {w_gate, const_cast<__nv_bfloat16 *>(w_gate_ptr)},
                        {w_up, const_cast<__nv_bfloat16 *>(w_up_ptr)},
                        {act, act_ptr}};
    (void)graph.execute(cudnn.handle, variant_pack, workspace);
  }
};

struct CudnnDown : public CudnnGraphBase {
  std::shared_ptr<fe::graph::Tensor_attributes> act;
  std::shared_ptr<fe::graph::Tensor_attributes> w_down;
  std::shared_ptr<fe::graph::Tensor_attributes> out;

  CudnnDown(int tokens, cudaStream_t stream) : CudnnGraphBase(stream) {
    act = make_bf16_tensor(graph, "act", {1, tokens, GEMMA4_INTERMEDIATE_SIZE},
                           {tokens * GEMMA4_INTERMEDIATE_SIZE, GEMMA4_INTERMEDIATE_SIZE, 1});
    w_down = make_bf16_tensor(graph, "w_down", {1, GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE},
                              {GEMMA4_INTERMEDIATE_SIZE * GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, 1});
    out = graph.matmul(act, w_down, fe::graph::Matmul_attributes().set_name("down_matmul").set_compute_data_type(fe::DataType_t::FLOAT));
    out->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);

    build("down");
  }

  void run(const __nv_bfloat16 *act_ptr,
           const __nv_bfloat16 *w_down_ptr,
           __nv_bfloat16 *out_ptr) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
        variant_pack = {{act, const_cast<__nv_bfloat16 *>(act_ptr)},
                        {w_down, const_cast<__nv_bfloat16 *>(w_down_ptr)},
                        {out, out_ptr}};
    (void)graph.execute(cudnn.handle, variant_pack, workspace);
  }
};

struct CudnnFullFfn : public CudnnGraphBase {
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> w_gate;
  std::shared_ptr<fe::graph::Tensor_attributes> w_up;
  std::shared_ptr<fe::graph::Tensor_attributes> w_down;
  std::shared_ptr<fe::graph::Tensor_attributes> out;

  CudnnFullFfn(int tokens, cudaStream_t stream) : CudnnGraphBase(stream) {
    x = make_bf16_tensor(graph, "x", {1, tokens, GEMMA4_HIDDEN_SIZE},
                         {tokens * GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, 1});
    w_gate = make_bf16_tensor(graph, "w_gate", {1, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE},
                              {GEMMA4_HIDDEN_SIZE * GEMMA4_INTERMEDIATE_SIZE, GEMMA4_INTERMEDIATE_SIZE, 1});
    w_up = make_bf16_tensor(graph, "w_up", {1, GEMMA4_HIDDEN_SIZE, GEMMA4_INTERMEDIATE_SIZE},
                            {GEMMA4_HIDDEN_SIZE * GEMMA4_INTERMEDIATE_SIZE, GEMMA4_INTERMEDIATE_SIZE, 1});
    w_down = make_bf16_tensor(graph, "w_down", {1, GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE},
                              {GEMMA4_INTERMEDIATE_SIZE * GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, 1});

    auto gate = graph.matmul(x, w_gate, fe::graph::Matmul_attributes().set_name("gate_matmul").set_compute_data_type(fe::DataType_t::FLOAT));
    gate->set_data_type(fe::DataType_t::BFLOAT16);
    auto up = graph.matmul(x, w_up, fe::graph::Matmul_attributes().set_name("up_matmul").set_compute_data_type(fe::DataType_t::FLOAT));
    up->set_data_type(fe::DataType_t::BFLOAT16);
    auto gelu = graph.pointwise(
        gate, fe::graph::Pointwise_attributes().set_name("gelu").set_mode(fe::PointwiseMode_t::GELU_APPROX_TANH_FWD).set_compute_data_type(fe::DataType_t::FLOAT));
    gelu->set_data_type(fe::DataType_t::BFLOAT16);
    auto act = graph.pointwise(gelu, up,
                               fe::graph::Pointwise_attributes().set_name("gate_mul").set_mode(fe::PointwiseMode_t::MUL).set_compute_data_type(fe::DataType_t::FLOAT));
    act->set_data_type(fe::DataType_t::BFLOAT16);
    out = graph.matmul(act, w_down, fe::graph::Matmul_attributes().set_name("down_matmul").set_compute_data_type(fe::DataType_t::FLOAT));
    out->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);

    build("full_ffn");
  }

  void run(const __nv_bfloat16 *x_ptr,
           const __nv_bfloat16 *w_gate_ptr,
           const __nv_bfloat16 *w_up_ptr,
           const __nv_bfloat16 *w_down_ptr,
           __nv_bfloat16 *out_ptr) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
        variant_pack = {{x, const_cast<__nv_bfloat16 *>(x_ptr)},
                        {w_gate, const_cast<__nv_bfloat16 *>(w_gate_ptr)},
                        {w_up, const_cast<__nv_bfloat16 *>(w_up_ptr)},
                        {w_down, const_cast<__nv_bfloat16 *>(w_down_ptr)},
                        {out, out_ptr}};
    (void)graph.execute(cudnn.handle, variant_pack, workspace);
  }
};

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 100;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 20;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 3;
  const int tokens = argc > 4 ? std::atoi(argv[4]) : 1;

  if (iters <= 0 || warmup < 0 || trials <= 0 || tokens <= 0) {
    std::fprintf(stderr,
                 "usage: %s [iters=100] [warmup=20] [trials=3] "
                 "[tokens=1]\n",
                 argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const uint64_t seed = make_seed("GEMMA4_FFN_CUDNN_BENCH_SEED");
  const size_t x_elems = static_cast<size_t>(tokens) * GEMMA4_HIDDEN_SIZE;
  const size_t act_elems = static_cast<size_t>(tokens) * GEMMA4_INTERMEDIATE_SIZE;
  const size_t gate_weight_elems = static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * GEMMA4_INTERMEDIATE_SIZE;
  const size_t up_weight_elems = static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * GEMMA4_INTERMEDIATE_SIZE;
  const size_t down_weight_elems = static_cast<size_t>(GEMMA4_INTERMEDIATE_SIZE) * GEMMA4_HIDDEN_SIZE;

  DeviceBuffer<__nv_bfloat16> d_x(x_elems);
  DeviceBuffer<__nv_bfloat16> d_w_gate(gate_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_up(up_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_gate_up_col_major_src(gate_weight_elems + up_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_gate_up_col_major(gate_weight_elems + up_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_down(down_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_down_swizzled(down_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_residual(x_elems);
  DeviceBuffer<__nv_bfloat16> d_rms_weight(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_act(act_elems);
  DeviceBuffer<__nv_bfloat16> d_out(x_elems);
  DeviceBuffer<__nv_bfloat16> d_prefill_out(x_elems);
  DeviceBuffer<__nv_bfloat16> d_prefill_scratch(
      gemma4_ffn_prefill_scratch_elements(tokens));
  DeviceBuffer<__nv_bfloat16> d_full_out(x_elems);
  DeviceBuffer<__nv_bfloat16> d_custom_residual_out(x_elems);
  DeviceBuffer<__nv_bfloat16> d_custom_normed_out(x_elems);
  DeviceBuffer<Gemma4FfnDecodeScratch> d_custom_scratch(1);

  fill_random_bf16(d_x, x_elems, seed ^ 0x1001u, 0.05f, stream);
  fill_random_bf16(d_w_gate, gate_weight_elems, seed ^ 0x2002u, 0.01f, stream);
  fill_random_bf16(d_w_up, up_weight_elems, seed ^ 0x3003u, 0.01f, stream);
  fill_random_bf16(d_w_gate_up_col_major_src,
                   gate_weight_elems + up_weight_elems, seed ^ 0x5005u,
                   0.01f, stream);
  fill_random_bf16(d_w_down, down_weight_elems, seed ^ 0x4004u, 0.01f, stream);
  fill_random_bf16(d_residual, x_elems, seed ^ 0x6006u, 0.05f, stream);
  fill_random_bf16(d_rms_weight, GEMMA4_HIDDEN_SIZE, seed ^ 0x7007u, 1.0f, stream);
  gemma4_ffn_decode_swizzle_weights_bf16(
      d_w_gate_up_col_major, d_w_gate_up_col_major_src, d_w_down_swizzled,
      d_w_down, stream);
  cudaStreamSynchronize(stream);
  int device = 0;
  cudaDeviceProp prop{};
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  const size_t flush_bytes =
      std::max<size_t>(256ull * 1024ull * 1024ull,
                       static_cast<size_t>(prop.l2CacheSize) * 4ull);
  const size_t flush_count = flush_bytes / sizeof(uint32_t);
  DeviceBuffer<uint32_t> d_flush_in(flush_count);
  DeviceBuffer<uint32_t> d_flush_out(size_t(4096) * 256);
  cudaMemsetAsync(d_flush_in, 0x5a, flush_bytes, stream);
  cudaMemsetAsync(d_flush_out, 0, size_t(4096) * 256 * sizeof(uint32_t), stream);
  cudaStreamSynchronize(stream);

  const double geglu_bytes =
      double(x_elems) * sizeof(__nv_bfloat16) * 2.0 +
      double(gate_weight_elems + up_weight_elems) * sizeof(__nv_bfloat16) +
      double(act_elems) * sizeof(__nv_bfloat16);
  const double down_bytes =
      double(act_elems) * sizeof(__nv_bfloat16) +
      double(down_weight_elems) * sizeof(__nv_bfloat16) +
      double(x_elems) * sizeof(__nv_bfloat16);
  const double split_bytes = geglu_bytes + down_bytes;
  const double custom_bytes = split_bytes + double(x_elems + GEMMA4_HIDDEN_SIZE + 2 * x_elems) * sizeof(__nv_bfloat16);

  Gemma4FfnPrefillScratch prefill_scratch = gemma4_ffn_prefill_scratch_from_buffer(d_prefill_scratch, tokens);
  auto run_custom_prefill_mlp = [&]() {
    gemma4_ffn_prefill_mlp_bf16(
        d_prefill_out, d_x, d_w_gate_up_col_major, d_w_down_swizzled,
        prefill_scratch, tokens, stream); };
  run_custom_prefill_mlp();
  cudaStreamSynchronize(stream);

  const TimingStats custom_prefill_mlp_stats =
      time_ms(run_custom_prefill_mlp, stream, warmup, iters, trials);
  const TimingStats custom_prefill_mlp_graph_stats = time_ms_graph(
      run_custom_prefill_mlp, stream, warmup, iters, trials);

  {
    CudnnGeGlu geglu(tokens, stream);
    CudnnDown down(tokens, stream);

    auto run_geglu = [&]() {
      geglu.run(d_x, d_w_gate, d_w_up, d_act);
    };
    auto run_down = [&]() {
      down.run(d_act, d_w_down, d_out);
    };
    auto run_split = [&]() {
      geglu.run(d_x, d_w_gate, d_w_up, d_act);
      down.run(d_act, d_w_down, d_out);
    };
    auto run_custom = [&]() {
      gemma4_ffn_decode_fused_bf16(
          d_custom_residual_out, d_custom_normed_out, d_x, d_residual,
          d_rms_weight, d_w_gate_up_col_major, d_w_down_swizzled,
          d_custom_scratch, GEMMA4_RMS_NORM_EPS, stream);
    };
    auto run_custom_clear = [&]() {
      cudaMemsetAsync(d_custom_scratch, 0, sizeof(Gemma4FfnDecodeScratch), stream);
    };

    run_split();
    run_custom();
    run_custom_prefill_mlp();
    cudaStreamSynchronize(stream);

    const TimingStats geglu_stats = time_ms(run_geglu, stream, warmup, iters, trials);
    const TimingStats down_stats = time_ms(run_down, stream, warmup, iters, trials);
    const TimingStats split_stats = time_ms(run_split, stream, warmup, iters, trials);
    const TimingStats custom_stats = time_ms(run_custom, stream, warmup, iters, trials);
    const TimingStats custom_clear_stats = time_ms(run_custom_clear, stream, warmup, iters, trials);
    const TimingStats cold_custom_stats = time_ms_cold(run_custom, stream, warmup, iters, trials, d_flush_in, d_flush_out, flush_count);
    const TimingStats cold_custom_clear_stats = time_ms_cold(run_custom_clear, stream, warmup, iters, trials, d_flush_in, d_flush_out, flush_count);
    const TimingStats cold_flush_stats = time_ms([&]() { flush_cache(d_flush_in, d_flush_out, flush_count, stream); }, stream, warmup, iters, trials);

    const TimingStats geglu_graph_stats =
        time_ms_graph(run_geglu, stream, warmup, iters, trials);
    const TimingStats down_graph_stats =
        time_ms_graph(run_down, stream, warmup, iters, trials);
    const TimingStats split_graph_stats =
        time_ms_graph(run_split, stream, warmup, iters, trials);
    const TimingStats custom_graph_stats =
        time_ms_graph(run_custom, stream, warmup, iters, trials);
    const TimingStats custom_clear_graph_stats =
        time_ms_graph(run_custom_clear, stream, warmup, iters, trials);

    float full_ms = -1.0f;
    float full_graph_ms = -1.0f;
    float full_max_abs = -1.0f;

    try {
      CudnnFullFfn full(tokens, stream);

      auto run_full = [&]() {
        full.run(d_x, d_w_gate, d_w_up, d_w_down, d_full_out);
      };
      run_full();
      run_split();
      cudaStreamSynchronize(stream);
      full_max_abs = diff_stats_bf16(d_full_out, d_out, int(x_elems)).max_abs;

      const TimingStats full_stats =
          time_ms(run_full, stream, warmup, iters, trials);
      full_ms = full_stats.best_ms;
      const TimingStats full_graph_stats =
          time_ms_graph(run_full, stream, warmup, iters, trials);
          full_graph_ms = full_graph_stats.best_ms;
    } catch (const std::exception &) {
    }

    std::printf("path,best_ms,avg_ms,graph_best_ms,graph_avg_ms,"
                "rough_gib_s,workspace_bytes,max_abs_vs_split\n");
    std::printf("geglu,%.6f,%.6f,%.6f,%.6f,%.3f,%lld,\n",
                geglu_stats.best_ms, geglu_stats.avg_ms,
                geglu_graph_stats.best_ms, geglu_graph_stats.avg_ms,
                gib_per_second(geglu_bytes, geglu_stats.best_ms),
                static_cast<long long>(geglu.workspace_size));
    std::printf("down,%.6f,%.6f,%.6f,%.6f,%.3f,%lld,\n",
                down_stats.best_ms, down_stats.avg_ms,
                down_graph_stats.best_ms, down_graph_stats.avg_ms,
                gib_per_second(down_bytes, down_stats.best_ms),
                static_cast<long long>(down.workspace_size));
    std::printf("geglu_plus_down,%.6f,%.6f,%.6f,%.6f,%.3f,%lld,\n",
                split_stats.best_ms, split_stats.avg_ms,
                split_graph_stats.best_ms, split_graph_stats.avg_ms,
                gib_per_second(split_bytes, split_stats.best_ms),
                static_cast<long long>(geglu.workspace_size +
                                       down.workspace_size));
    std::printf("custom_fused_decode,%.6f,%.6f,%.6f,%.6f,%.3f,0,\n",
                custom_stats.best_ms, custom_stats.avg_ms,
                custom_graph_stats.best_ms, custom_graph_stats.avg_ms,
                gib_per_second(custom_bytes, custom_stats.best_ms));
    std::printf("custom_prefill_mlp,%.6f,%.6f,%.6f,%.6f,%.3f,0,\n",
                custom_prefill_mlp_stats.best_ms,
                custom_prefill_mlp_stats.avg_ms,
                custom_prefill_mlp_graph_stats.best_ms,
                custom_prefill_mlp_graph_stats.avg_ms,
                gib_per_second(
                    split_bytes + double(4 * x_elems) * sizeof(__nv_bfloat16),
                    custom_prefill_mlp_stats.best_ms));
    std::printf("custom_scratch_clear,%.6f,%.6f,%.6f,%.6f,0.000,0,\n",
                custom_clear_stats.best_ms, custom_clear_stats.avg_ms,
                custom_clear_graph_stats.best_ms,
                custom_clear_graph_stats.avg_ms);
    std::printf("custom_fused_decode_cold,%.6f,%.6f,-1.000000,-1.000000,"
                "%.3f,0,\n",
                cold_custom_stats.best_ms, cold_custom_stats.avg_ms,
                gib_per_second(custom_bytes, cold_custom_stats.best_ms));
    std::printf("custom_scratch_clear_cold,%.6f,%.6f,-1.000000,-1.000000,"
                "0.000,0,\n",
                cold_custom_clear_stats.best_ms,
                cold_custom_clear_stats.avg_ms);
    std::printf("cache_flush_only,%.6f,%.6f,-1.000000,-1.000000,0.000,0,\n",
                cold_flush_stats.best_ms, cold_flush_stats.avg_ms);
    std::printf("full_ffn,%.6f,,%.6f,,%.3f,,%.6g\n",
                full_ms, full_graph_ms,
                full_ms > 0.0f ? gib_per_second(split_bytes, full_ms) : 0.0,
                full_max_abs);

    const float cudnn_device_ms = split_graph_stats.best_ms;
    const float custom_device_ms = custom_graph_stats.best_ms;
    const float custom_clear_device_ms = custom_clear_graph_stats.best_ms;
    const float custom_minus_clear_ms = custom_device_ms - custom_clear_device_ms;
    const float cold_custom_minus_flush_ms = cold_custom_stats.best_ms - cold_flush_stats.best_ms;
    const float cold_clear_minus_flush_ms = cold_custom_clear_stats.best_ms - cold_flush_stats.best_ms;
    const float cold_custom_minus_flush_clear_ms = cold_custom_minus_flush_ms - cold_clear_minus_flush_ms;
    const float custom_speedup = cudnn_device_ms / custom_device_ms;
    const float custom_minus_clear_speedup = cudnn_device_ms / custom_minus_clear_ms;
    const float cudnn_direct_overhead = split_stats.best_ms - cudnn_device_ms;
    const float custom_direct_overhead = custom_stats.best_ms - custom_device_ms;

    std::printf("overhead_factored_metric,value\n");
    std::printf("cudnn_split_device_ms,%.6f\n", cudnn_device_ms);
    std::printf("custom_device_ms,%.6f\n", custom_device_ms);
    std::printf("custom_scratch_clear_device_ms,%.6f\n", custom_clear_device_ms);
    std::printf("custom_minus_clear_device_ms,%.6f\n", custom_minus_clear_ms);
    std::printf("cold_custom_minus_flush_ms,%.6f\n", cold_custom_minus_flush_ms);
    std::printf("cold_clear_minus_flush_ms,%.6f\n", cold_clear_minus_flush_ms);
    std::printf("cold_custom_minus_flush_clear_ms,%.6f\n", cold_custom_minus_flush_clear_ms);
    std::printf("custom_vs_cudnn_split_speedup,%.6f\n", custom_speedup);
    std::printf("custom_minus_clear_vs_cudnn_split_speedup,%.6f\n", custom_minus_clear_speedup);
    std::printf("cudnn_direct_minus_device_ms,%.6f\n", cudnn_direct_overhead);
    std::printf("custom_direct_minus_device_ms,%.6f\n", custom_direct_overhead);
  }

  cudaStreamDestroy(stream);
  return 0;
}
