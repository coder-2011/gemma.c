#include "gemma4_bench_utils.cuh"
#include "gemma4_ffn_decode.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#if __has_include(<cudnn_frontend.h>)
#include <cudnn_frontend.h>
#define GEMMA4_HAS_CUDNN_FRONTEND 1
#else
#define GEMMA4_HAS_CUDNN_FRONTEND 0
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kHidden = GEMMA4_HIDDEN_SIZE;
constexpr int kIntermediate = GEMMA4_INTERMEDIATE_SIZE;

__device__ uint32_t mix_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__global__ void fill_random_bf16_kernel(__nv_bfloat16 *ptr,
                                        size_t count,
                                        uint64_t seed,
                                        float scale) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }

  uint32_t x = uint32_t(i) ^ uint32_t(i >> 32) ^ uint32_t(seed) ^
               uint32_t(seed >> 32);
  x = mix_u32(x);
  const float u = float(x >> 8) * (1.0f / 16777216.0f);
  ptr[i] = __float2bfloat16_rn((u * 2.0f - 1.0f) * scale);
}

void fill_random_bf16(__nv_bfloat16 *ptr,
                      size_t count,
                      uint64_t seed,
                      float scale,
                      cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = int((count + threads - 1) / threads);
  fill_random_bf16_kernel<<<blocks, threads, 0, stream>>>(
      ptr, count, seed, scale);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void flush_cache_kernel(const uint32_t *__restrict__ in,
                                   uint32_t *__restrict__ out,
                                   size_t count) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    acc ^= in[i] + uint32_t(i);
  }
  if ((blockIdx.x * blockDim.x + threadIdx.x) < gridDim.x * blockDim.x) {
    out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
  }
}

void flush_cache(const uint32_t *in,
                 uint32_t *out,
                 size_t count,
                 cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int blocks = 4096;
  flush_cache_kernel<<<blocks, threads, 0, stream>>>(in, out, count);
  CUDA_CHECK(cudaGetLastError());
}

uint64_t make_seed() {
  if (const char *env = std::getenv("GEMMA4_FFN_CUDNN_BENCH_SEED")) {
    return std::strtoull(env, nullptr, 0);
  }

  std::random_device rd;
  uint64_t seed = uint64_t(rd()) << 32;
  seed ^= uint64_t(rd());
  seed ^= uint64_t(std::chrono::high_resolution_clock::now()
                       .time_since_epoch()
                       .count());
  return seed;
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) {
    CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  operator T *() { return ptr_; }
  operator const T *() const { return ptr_; }

 private:
  T *ptr_ = nullptr;
};

double gib_per_second(double bytes, float ms) {
  const double gib = bytes / (1024.0 * 1024.0 * 1024.0);
  return gib / (static_cast<double>(ms) / 1000.0);
}

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

#if GEMMA4_HAS_CUDNN_FRONTEND

namespace fe = cudnn_frontend;

void check_cudnn(cudnnStatus_t status, const char *expr) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuDNN error for ") + expr + ": " +
                             cudnnGetErrorString(status));
  }
}

void check_fe(cudnn_frontend::error_t status, const char *expr) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string("cuDNN frontend error for ") + expr +
                             ": " + status.get_message());
  }
}

struct CudnnHandle {
  cudnnHandle_t handle = nullptr;

  explicit CudnnHandle(cudaStream_t stream) {
    check_cudnn(cudnnCreate(&handle), "cudnnCreate");
    check_cudnn(cudnnSetStream(handle, stream), "cudnnSetStream");
  }

  ~CudnnHandle() {
    if (handle != nullptr) {
      cudnnDestroy(handle);
    }
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
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
  }

  void build(const char *name) {
    check_fe(graph.validate(), (std::string(name) + ".validate").c_str());
    check_fe(graph.build_operation_graph(cudnn.handle),
             (std::string(name) + ".build_operation_graph").c_str());
    check_fe(graph.create_execution_plans(
                 {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
             (std::string(name) + ".create_execution_plans").c_str());
    check_fe(graph.check_support(),
             (std::string(name) + ".check_support").c_str());
    check_fe(graph.build_plans(),
             (std::string(name) + ".build_plans").c_str());
    check_fe(graph.get_workspace_size(workspace_size),
             (std::string(name) + ".get_workspace_size").c_str());
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

fe::graph::Matmul_attributes matmul_attrs(const char *name) {
  return fe::graph::Matmul_attributes()
      .set_name(name)
      .set_compute_data_type(fe::DataType_t::FLOAT);
}

fe::graph::Pointwise_attributes pointwise_attrs(const char *name,
                                                fe::PointwiseMode_t mode) {
  return fe::graph::Pointwise_attributes()
      .set_name(name)
      .set_mode(mode)
      .set_compute_data_type(fe::DataType_t::FLOAT);
}

struct CudnnGeGlu : public CudnnGraphBase {
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> w_gate;
  std::shared_ptr<fe::graph::Tensor_attributes> w_up;
  std::shared_ptr<fe::graph::Tensor_attributes> act;

  CudnnGeGlu(int tokens, cudaStream_t stream) : CudnnGraphBase(stream) {
    x = make_bf16_tensor(graph, "x", {1, tokens, kHidden},
                         {tokens * kHidden, kHidden, 1});
    w_gate = make_bf16_tensor(graph, "w_gate", {1, kHidden, kIntermediate},
                              {kHidden * kIntermediate, kIntermediate, 1});
    w_up = make_bf16_tensor(graph, "w_up", {1, kHidden, kIntermediate},
                            {kHidden * kIntermediate, kIntermediate, 1});

    auto gate = graph.matmul(x, w_gate, matmul_attrs("gate_matmul"));
    gate->set_data_type(fe::DataType_t::BFLOAT16);
    auto up = graph.matmul(x, w_up, matmul_attrs("up_matmul"));
    up->set_data_type(fe::DataType_t::BFLOAT16);
    auto gelu = graph.pointwise(
        up, pointwise_attrs("gelu", fe::PointwiseMode_t::GELU_APPROX_TANH_FWD));
    gelu->set_data_type(fe::DataType_t::BFLOAT16);
    act = graph.pointwise(gate, gelu,
                          pointwise_attrs("gate_mul", fe::PointwiseMode_t::MUL));
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
    check_fe(graph.execute(cudnn.handle, variant_pack, workspace),
             "geglu.execute");
  }
};

struct CudnnDown : public CudnnGraphBase {
  std::shared_ptr<fe::graph::Tensor_attributes> act;
  std::shared_ptr<fe::graph::Tensor_attributes> w_down;
  std::shared_ptr<fe::graph::Tensor_attributes> out;

  CudnnDown(int tokens, cudaStream_t stream) : CudnnGraphBase(stream) {
    act = make_bf16_tensor(graph, "act", {1, tokens, kIntermediate},
                           {tokens * kIntermediate, kIntermediate, 1});
    w_down = make_bf16_tensor(graph, "w_down", {1, kIntermediate, kHidden},
                              {kIntermediate * kHidden, kHidden, 1});
    out = graph.matmul(act, w_down, matmul_attrs("down_matmul"));
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
    check_fe(graph.execute(cudnn.handle, variant_pack, workspace),
             "down.execute");
  }
};

struct CudnnFullFfn : public CudnnGraphBase {
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> w_gate;
  std::shared_ptr<fe::graph::Tensor_attributes> w_up;
  std::shared_ptr<fe::graph::Tensor_attributes> w_down;
  std::shared_ptr<fe::graph::Tensor_attributes> out;

  CudnnFullFfn(int tokens, cudaStream_t stream) : CudnnGraphBase(stream) {
    x = make_bf16_tensor(graph, "x", {1, tokens, kHidden},
                         {tokens * kHidden, kHidden, 1});
    w_gate = make_bf16_tensor(graph, "w_gate", {1, kHidden, kIntermediate},
                              {kHidden * kIntermediate, kIntermediate, 1});
    w_up = make_bf16_tensor(graph, "w_up", {1, kHidden, kIntermediate},
                            {kHidden * kIntermediate, kIntermediate, 1});
    w_down = make_bf16_tensor(graph, "w_down", {1, kIntermediate, kHidden},
                              {kIntermediate * kHidden, kHidden, 1});

    auto gate = graph.matmul(x, w_gate, matmul_attrs("gate_matmul"));
    gate->set_data_type(fe::DataType_t::BFLOAT16);
    auto up = graph.matmul(x, w_up, matmul_attrs("up_matmul"));
    up->set_data_type(fe::DataType_t::BFLOAT16);
    auto gelu = graph.pointwise(
        up, pointwise_attrs("gelu", fe::PointwiseMode_t::GELU_APPROX_TANH_FWD));
    gelu->set_data_type(fe::DataType_t::BFLOAT16);
    auto act = graph.pointwise(gate, gelu,
                               pointwise_attrs("gate_mul", fe::PointwiseMode_t::MUL));
    act->set_data_type(fe::DataType_t::BFLOAT16);
    out = graph.matmul(act, w_down, matmul_attrs("down_matmul"));
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
    check_fe(graph.execute(cudnn.handle, variant_pack, workspace),
             "full_ffn.execute");
  }
};

#endif

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

  const uint64_t seed = make_seed();
  const size_t x_elems = static_cast<size_t>(tokens) * kHidden;
  const size_t act_elems = static_cast<size_t>(tokens) * kIntermediate;
  const size_t gate_weight_elems = static_cast<size_t>(kHidden) * kIntermediate;
  const size_t up_weight_elems = static_cast<size_t>(kHidden) * kIntermediate;
  const size_t down_weight_elems =
      static_cast<size_t>(kIntermediate) * kHidden;

  DeviceBuffer<__nv_bfloat16> d_x(x_elems);
  DeviceBuffer<__nv_bfloat16> d_w_gate(gate_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_up(up_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_gate_up_col_major_src(
      gate_weight_elems + up_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_gate_up_col_major(
      gate_weight_elems + up_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_down(down_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_w_down_swizzled(down_weight_elems);
  DeviceBuffer<__nv_bfloat16> d_residual(x_elems);
  DeviceBuffer<__nv_bfloat16> d_rms_weight(kHidden);
  DeviceBuffer<__nv_bfloat16> d_act(act_elems);
  DeviceBuffer<__nv_bfloat16> d_out(x_elems);
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
  fill_random_bf16(d_rms_weight, kHidden, seed ^ 0x7007u, 1.0f, stream);
  CUDA_CHECK(gemma4_ffn_decode_swizzle_weights_bf16(
      d_w_gate_up_col_major, d_w_gate_up_col_major_src, d_w_down_swizzled,
      d_w_down, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(gemma4_ffn_decode_configure_scratch_l2(d_custom_scratch, stream));

  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  const size_t flush_bytes =
      std::max<size_t>(256ull * 1024ull * 1024ull,
                       static_cast<size_t>(prop.l2CacheSize) * 4ull);
  const size_t flush_count = flush_bytes / sizeof(uint32_t);
  DeviceBuffer<uint32_t> d_flush_in(flush_count);
  DeviceBuffer<uint32_t> d_flush_out(size_t(4096) * 256);
  CUDA_CHECK(cudaMemsetAsync(d_flush_in, 0x5a, flush_bytes, stream));
  CUDA_CHECK(cudaMemsetAsync(d_flush_out, 0, size_t(4096) * 256 *
                                             sizeof(uint32_t), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const double geglu_bytes =
      double(x_elems) * sizeof(__nv_bfloat16) * 2.0 +
      double(gate_weight_elems + up_weight_elems) * sizeof(__nv_bfloat16) +
      double(act_elems) * sizeof(__nv_bfloat16);
  const double down_bytes =
      double(act_elems) * sizeof(__nv_bfloat16) +
      double(down_weight_elems) * sizeof(__nv_bfloat16) +
      double(x_elems) * sizeof(__nv_bfloat16);
  const double split_bytes = geglu_bytes + down_bytes;
  const double custom_bytes =
      split_bytes + double(x_elems + kHidden + 2 * x_elems) *
                        sizeof(__nv_bfloat16);

  std::printf("device=%s\n", prop.name);
  std::printf("shape=tokens%d,hidden%d,intermediate%d,seed=0x%llx\n",
              tokens, kHidden, kIntermediate,
              static_cast<unsigned long long>(seed));
  std::printf("iters=%d,warmup_iters=%d,trials=%d\n", iters, warmup, trials);
  std::printf("cache_flush_bytes=%zu\n", flush_bytes);
  std::printf("cudnn_frontend=%s\n",
#if GEMMA4_HAS_CUDNN_FRONTEND
              "compiled"
#else
              "not_compiled"
#endif
  );

#if !GEMMA4_HAS_CUDNN_FRONTEND
  std::fprintf(stderr, "cuDNN Frontend headers were not found at compile time\n");
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 1;
#else
  std::printf("cudnn_version=%zu\n", cudnnGetVersion());

  try {
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
      CUDA_CHECK(gemma4_ffn_decode_fused_bf16(
          d_custom_residual_out, d_custom_normed_out, d_x, d_residual,
          d_rms_weight, d_w_gate_up_col_major, d_w_down_swizzled,
          d_custom_scratch, GEMMA4_RMS_NORM_EPS, stream));
    };
    auto run_custom_clear = [&]() {
      CUDA_CHECK(cudaMemsetAsync(d_custom_scratch, 0,
                                 sizeof(Gemma4FfnDecodeScratch), stream));
    };

    run_split();
    run_custom();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const TimingStats geglu_stats =
        time_ms(run_geglu, stream, warmup, iters, trials);
    const TimingStats down_stats =
        time_ms(run_down, stream, warmup, iters, trials);
    const TimingStats split_stats =
        time_ms(run_split, stream, warmup, iters, trials);
    const TimingStats custom_stats =
        time_ms(run_custom, stream, warmup, iters, trials);
    const TimingStats custom_clear_stats =
        time_ms(run_custom_clear, stream, warmup, iters, trials);
    const TimingStats cold_custom_stats =
        time_ms_cold(run_custom, stream, warmup, iters, trials, d_flush_in,
                     d_flush_out, flush_count);
    const TimingStats cold_custom_clear_stats =
        time_ms_cold(run_custom_clear, stream, warmup, iters, trials,
                     d_flush_in, d_flush_out, flush_count);
    const TimingStats cold_flush_stats =
        time_ms([&]() { flush_cache(d_flush_in, d_flush_out, flush_count,
                                    stream); },
                stream, warmup, iters, trials);

    TimingStats geglu_graph_stats{-1.0f, -1.0f};
    TimingStats down_graph_stats{-1.0f, -1.0f};
    TimingStats split_graph_stats{-1.0f, -1.0f};
    TimingStats custom_graph_stats{-1.0f, -1.0f};
    TimingStats custom_clear_graph_stats{-1.0f, -1.0f};
    try {
      geglu_graph_stats = time_ms_graph(run_geglu, stream, warmup, iters,
                                        trials);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "geglu CUDA graph timing unavailable: %s\n",
                   e.what());
    }
    try {
      down_graph_stats = time_ms_graph(run_down, stream, warmup, iters,
                                       trials);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "down CUDA graph timing unavailable: %s\n",
                   e.what());
    }
    try {
      split_graph_stats = time_ms_graph(run_split, stream, warmup, iters,
                                        trials);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "split CUDA graph timing unavailable: %s\n",
                   e.what());
    }
    try {
      custom_graph_stats = time_ms_graph(run_custom, stream, warmup, iters,
                                         trials);
    } catch (const std::exception &e) {
      std::fprintf(stderr, "custom CUDA graph timing unavailable: %s\n",
                   e.what());
    }
    try {
      custom_clear_graph_stats =
          time_ms_graph(run_custom_clear, stream, warmup, iters, trials);
    } catch (const std::exception &e) {
      std::fprintf(stderr,
                   "custom scratch clear CUDA graph timing unavailable: %s\n",
                   e.what());
    }

    float full_ms = -1.0f;
    float full_graph_ms = -1.0f;
    float full_max_abs = -1.0f;
    int full_supported = 0;
    std::string full_status = "unsupported";

    try {
      CudnnFullFfn full(tokens, stream);
      full_supported = 1;
      full_status = "supported";

      auto run_full = [&]() {
        full.run(d_x, d_w_gate, d_w_up, d_w_down, d_full_out);
      };
      run_full();
      run_split();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      full_max_abs = diff_stats_bf16(d_full_out, d_out, int(x_elems)).max_abs;

      const TimingStats full_stats =
          time_ms(run_full, stream, warmup, iters, trials);
      full_ms = full_stats.best_ms;
      try {
        const TimingStats full_graph_stats =
            time_ms_graph(run_full, stream, warmup, iters, trials);
        full_graph_ms = full_graph_stats.best_ms;
      } catch (const std::exception &e) {
        full_status = std::string("supported_graph_timing_failed: ") +
                      e.what();
      }
    } catch (const std::exception &e) {
      full_status = e.what();
      if (full_status.size() > 240) {
        full_status.resize(240);
      }
    }

    std::printf("full_graph_supported=%d,status=\"%s\"\n", full_supported,
                full_status.c_str());
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
    const float custom_minus_clear_ms =
        custom_device_ms > 0.0f && custom_clear_device_ms > 0.0f
            ? std::max(custom_device_ms - custom_clear_device_ms, 0.0f)
            : -1.0f;
    const float cold_custom_minus_flush_ms =
        cold_custom_stats.best_ms > 0.0f && cold_flush_stats.best_ms > 0.0f
            ? std::max(cold_custom_stats.best_ms - cold_flush_stats.best_ms,
                       0.0f)
            : -1.0f;
    const float cold_clear_minus_flush_ms =
        cold_custom_clear_stats.best_ms > 0.0f &&
                cold_flush_stats.best_ms > 0.0f
            ? std::max(cold_custom_clear_stats.best_ms -
                           cold_flush_stats.best_ms,
                       0.0f)
            : -1.0f;
    const float cold_custom_minus_flush_clear_ms =
        cold_custom_minus_flush_ms > 0.0f && cold_clear_minus_flush_ms > 0.0f
            ? std::max(cold_custom_minus_flush_ms -
                           cold_clear_minus_flush_ms,
                       0.0f)
            : -1.0f;
    const float custom_speedup =
        cudnn_device_ms > 0.0f && custom_device_ms > 0.0f
            ? cudnn_device_ms / custom_device_ms
            : -1.0f;
    const float custom_minus_clear_speedup =
        cudnn_device_ms > 0.0f && custom_minus_clear_ms > 0.0f
            ? cudnn_device_ms / custom_minus_clear_ms
            : -1.0f;
    const float cudnn_direct_overhead =
        split_stats.best_ms > 0.0f && cudnn_device_ms > 0.0f
            ? std::max(split_stats.best_ms - cudnn_device_ms, 0.0f)
            : -1.0f;
    const float custom_direct_overhead =
        custom_stats.best_ms > 0.0f && custom_device_ms > 0.0f
            ? std::max(custom_stats.best_ms - custom_device_ms, 0.0f)
            : -1.0f;

    std::printf("overhead_factored_metric,value\n");
    std::printf("cudnn_split_device_ms,%.6f\n", cudnn_device_ms);
    std::printf("custom_device_ms,%.6f\n", custom_device_ms);
    std::printf("custom_scratch_clear_device_ms,%.6f\n",
                custom_clear_device_ms);
    std::printf("custom_minus_clear_device_ms,%.6f\n",
                custom_minus_clear_ms);
    std::printf("cold_custom_minus_flush_ms,%.6f\n",
                cold_custom_minus_flush_ms);
    std::printf("cold_clear_minus_flush_ms,%.6f\n",
                cold_clear_minus_flush_ms);
    std::printf("cold_custom_minus_flush_clear_ms,%.6f\n",
                cold_custom_minus_flush_clear_ms);
    std::printf("custom_vs_cudnn_split_speedup,%.6f\n", custom_speedup);
    std::printf("custom_minus_clear_vs_cudnn_split_speedup,%.6f\n",
                custom_minus_clear_speedup);
    std::printf("cudnn_direct_minus_device_ms,%.6f\n",
                cudnn_direct_overhead);
    std::printf("custom_direct_minus_device_ms,%.6f\n",
                custom_direct_overhead);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "benchmark failed: %s\n", e.what());
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 1;
  }
#endif

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
