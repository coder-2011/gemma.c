#include "gemma4_rmsnorm.cuh"
#include "gemma4_bench_utils.cuh"
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
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if GEMMA4_HAS_CUDNN_FRONTEND
#include <memory>
#include <unordered_map>
#endif

namespace {

__device__ uint32_t gemma4_rmsnorm_mix_u32_device(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__global__ void gemma4_rmsnorm_fill_random_bf16_kernel(__nv_bfloat16 *ptr,
                                                       size_t count,
                                                       uint64_t seed,
                                                       float scale) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }

  uint32_t x = uint32_t(i) ^ uint32_t(i >> 32) ^ uint32_t(seed) ^
               uint32_t(seed >> 32);
  x = gemma4_rmsnorm_mix_u32_device(x);
  float u = float(x >> 8) * (1.0f / 16777216.0f);
  ptr[i] = __float2bfloat16_rn((u * 2.0f - 1.0f) * scale);
}

uint64_t make_seed() {
  if (const char *env = std::getenv("GEMMA4_RMSNORM_BENCH_SEED")) {
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

void fill_random_bf16(__nv_bfloat16 *ptr,
                      size_t count,
                      uint64_t seed,
                      float scale,
                      cudaStream_t stream) {
  constexpr int threads = 256;
  int blocks = int((count + threads - 1) / threads);
  gemma4_rmsnorm_fill_random_bf16_kernel<<<blocks, threads, 0, stream>>>(
      ptr, count, seed, scale);
  CUDA_CHECK(cudaGetLastError());
}

std::vector<int> row_counts_up_to(int max_rows) {
  std::vector<int> counts;
  for (int rows : {1, 4, 16, 64, 256, 1024, 4096, 8192}) {
    if (rows <= max_rows) {
      counts.push_back(rows);
    }
  }
  if (counts.empty() || counts.back() != max_rows) {
    counts.push_back(max_rows);
  }
  return counts;
}

double gib_per_second(double bytes, float ms) {
  double gib = bytes / (1024.0 * 1024.0 * 1024.0);
  return gib / (static_cast<double>(ms) / 1000.0);
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

struct CudnnRmsnorm {
  cudnnHandle_t handle = nullptr;
  fe::graph::Graph graph;
  std::shared_ptr<fe::graph::Tensor_attributes> x;
  std::shared_ptr<fe::graph::Tensor_attributes> scale;
  std::shared_ptr<fe::graph::Tensor_attributes> y;
  std::shared_ptr<fe::graph::Tensor_attributes> inv_variance;
  void *workspace = nullptr;

  CudnnRmsnorm(int rows, int width, float eps, cudaStream_t stream) {
    check_cudnn(cudnnCreate(&handle), "cudnnCreate");
    check_cudnn(cudnnSetStream(handle, stream), "cudnnSetStream");

    graph.set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    x = graph.tensor(fe::graph::Tensor_attributes()
                         .set_name("x")
                         .set_data_type(fe::DataType_t::BFLOAT16)
                         .set_dim({rows, width, 1, 1})
                         .set_stride({width, 1, width, width}));
    scale = graph.tensor(fe::graph::Tensor_attributes()
                             .set_name("scale")
                             .set_data_type(fe::DataType_t::BFLOAT16)
                             .set_dim({1, width, 1, 1})
                             .set_stride({width, 1, width, width}));
    auto epsilon = graph.tensor(eps);
    auto options = fe::graph::Rmsnorm_attributes()
                       .set_forward_phase(fe::NormFwdPhase_t::TRAINING)
                       .set_epsilon(epsilon);
    auto outputs = graph.rmsnorm(x, scale, options);
    y = std::get<0>(outputs);
    inv_variance = std::get<1>(outputs);
    y->set_output(true).set_data_type(fe::DataType_t::BFLOAT16);
    inv_variance->set_output(true).set_data_type(fe::DataType_t::FLOAT);

    check_fe(graph.validate(), "validate");
    check_fe(graph.build_operation_graph(handle), "build_operation_graph");
    check_fe(graph.create_execution_plans(
                 {fe::HeurMode_t::A, fe::HeurMode_t::FALLBACK}),
             "create_execution_plans");
    check_fe(graph.check_support(), "check_support");
    check_fe(graph.build_plans(), "build_plans");

    int64_t workspace_size = 0;
    check_fe(graph.get_workspace_size(workspace_size), "get_workspace_size");
    if (workspace_size > 0) {
      CUDA_CHECK(cudaMalloc(&workspace, static_cast<size_t>(workspace_size)));
    }
  }

  ~CudnnRmsnorm() {
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
    if (handle != nullptr) {
      cudnnDestroy(handle);
    }
  }

  void run(const __nv_bfloat16 *inp,
           const __nv_bfloat16 *weight,
           __nv_bfloat16 *out,
           float *rstd) {
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *>
        variant_pack = {{x, const_cast<__nv_bfloat16 *>(inp)},
                        {scale, const_cast<__nv_bfloat16 *>(weight)},
                        {y, out},
                        {inv_variance, rstd}};
    check_fe(graph.execute(handle, variant_pack, workspace), "execute");
  }
};

#endif

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
  const int warmup = argc > 2 ? std::atoi(argv[2]) : 30;
  const int trials = argc > 3 ? std::atoi(argv[3]) : 5;
  const int max_rows = argc > 4 ? std::atoi(argv[4]) : 4096;

  if (iters <= 0 || warmup < 0 || trials <= 0 || max_rows <= 0) {
    std::fprintf(stderr,
                 "usage: %s [iters=200] [warmup=30] [trials=5] [max_rows=4096]\n",
                 argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const int width = GEMMA4_HIDDEN_SIZE;
  const size_t max_elems = static_cast<size_t>(max_rows) * width;
  const uint64_t seed = make_seed();

  __nv_bfloat16 *d_inp1 = nullptr;
  __nv_bfloat16 *d_inp2 = nullptr;
  __nv_bfloat16 *d_weight = nullptr;
  __nv_bfloat16 *d_rms_out = nullptr;
  __nv_bfloat16 *d_rms_cudnn_out = nullptr;
  __nv_bfloat16 *d_residual = nullptr;
  __nv_bfloat16 *d_split_residual = nullptr;
  __nv_bfloat16 *d_fused_normed = nullptr;
  __nv_bfloat16 *d_split_normed = nullptr;
  float *d_rstd = nullptr;
  float *d_cudnn_rstd = nullptr;
  float *d_fused_rstd = nullptr;
  float *d_split_rstd = nullptr;

  CUDA_CHECK(cudaMalloc(&d_inp1, max_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_inp2, max_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_weight, static_cast<size_t>(width) * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_rms_out, max_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_rms_cudnn_out, max_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_residual, max_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_split_residual, max_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_fused_normed, max_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_split_normed, max_elems * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&d_rstd, static_cast<size_t>(max_rows) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cudnn_rstd, static_cast<size_t>(max_rows) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fused_rstd, static_cast<size_t>(max_rows) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_split_rstd, static_cast<size_t>(max_rows) * sizeof(float)));

  fill_random_bf16(d_inp1, max_elems, seed ^ 0x1001u, 1.0f, stream);
  fill_random_bf16(d_inp2, max_elems, seed ^ 0x2002u, 1.0f, stream);
  fill_random_bf16(d_weight, static_cast<size_t>(width), seed ^ 0x3003u, 0.5f,
                   stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int device = 0;
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  std::printf("device=%s\n", prop.name);
  std::printf("shape=width%d,max_rows=%d,seed=0x%llx\n", width, max_rows,
              static_cast<unsigned long long>(seed));
  std::printf("iters=%d,warmup_iters=%d,trials=%d\n", iters, warmup, trials);
  std::printf("cudnn_frontend=%s\n",
#if GEMMA4_HAS_CUDNN_FRONTEND
              "compiled"
#else
              "not_compiled"
#endif
  );
  std::printf("rows,rms_ms,rms_gib_s,cudnn_ms,cudnn_gib_s,cudnn_max_abs,cudnn_rstd_max_abs,residual_ms,fused_ms,split_ms,fused_vs_split\n");

  for (int rows : row_counts_up_to(max_rows)) {
    const int count = rows * width;
    const double rms_bytes =
        double(rows) * width * sizeof(__nv_bfloat16) * 3.0 +
        double(rows) * sizeof(float);
    const double residual_bytes =
        double(rows) * width * sizeof(__nv_bfloat16) * 3.0;
    const double fused_bytes =
        double(rows) * width * sizeof(__nv_bfloat16) * 5.0 +
        double(rows) * sizeof(float);
    const double split_bytes = residual_bytes + rms_bytes;

    auto run_rms = [&]() {
      CUDA_CHECK(gemma4_rmsnorm_bf16(
          d_rms_out, d_rstd, d_inp1, d_weight, rows, width,
          GEMMA4_RMS_NORM_EPS, stream));
    };
    auto run_residual = [&]() {
      CUDA_CHECK(gemma4_residual_add_bf16(
          d_residual, d_inp1, d_inp2, count, stream));
    };
    auto run_fused = [&]() {
      CUDA_CHECK(gemma4_residual_add_rmsnorm_bf16(
          d_split_residual, d_fused_normed, d_fused_rstd, d_inp1, d_inp2,
          d_weight, rows, width, GEMMA4_RMS_NORM_EPS, stream));
    };
    auto run_split = [&]() {
      CUDA_CHECK(gemma4_residual_add_bf16(
          d_split_residual, d_inp1, d_inp2, count, stream));
      CUDA_CHECK(gemma4_rmsnorm_bf16(
          d_split_normed, d_split_rstd, d_split_residual, d_weight, rows, width,
          GEMMA4_RMS_NORM_EPS, stream));
    };

    run_rms();
    run_residual();
    run_fused();
    run_split();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    TimingStats rms_stats =
        time_ms(run_rms, stream, warmup, iters, trials);
    TimingStats residual_stats =
        time_ms(run_residual, stream, warmup, iters, trials);
    TimingStats fused_stats =
        time_ms(run_fused, stream, warmup, iters, trials);
    TimingStats split_stats =
        time_ms(run_split, stream, warmup, iters, trials);

    float cudnn_ms = -1.0f;
    double cudnn_gib_s = 0.0;
    float cudnn_max_abs = -1.0f;
    float cudnn_rstd_max_abs = -1.0f;

#if GEMMA4_HAS_CUDNN_FRONTEND
    try {
      CudnnRmsnorm cudnn(rows, width, GEMMA4_RMS_NORM_EPS, stream);
      auto run_cudnn = [&]() {
        cudnn.run(d_inp1, d_weight, d_rms_cudnn_out, d_cudnn_rstd);
      };
      run_cudnn();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      TimingStats cudnn_stats =
          time_ms(run_cudnn, stream, warmup, iters, trials);
      cudnn_ms = cudnn_stats.best_ms;
      cudnn_gib_s = gib_per_second(rms_bytes, cudnn_ms);

      DiffStats out_diff =
          diff_stats_bf16(d_rms_out, d_rms_cudnn_out, count);
      std::vector<float> h_custom_rstd(rows);
      std::vector<float> h_cudnn_rstd(rows);
      CUDA_CHECK(cudaMemcpy(h_custom_rstd.data(), d_rstd,
                                   static_cast<size_t>(rows) * sizeof(float),
                                   cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_cudnn_rstd.data(), d_cudnn_rstd,
                                   static_cast<size_t>(rows) * sizeof(float),
                                   cudaMemcpyDeviceToHost));
      float max_rstd = 0.0f;
      for (int i = 0; i < rows; ++i) {
        max_rstd =
            std::max(max_rstd, std::abs(h_custom_rstd[i] - h_cudnn_rstd[i]));
      }
      cudnn_max_abs = out_diff.max_abs;
      cudnn_rstd_max_abs = max_rstd;
    } catch (const std::exception &e) {
      std::fprintf(stderr, "cuDNN RMSNorm unavailable for rows=%d: %s\n", rows,
                   e.what());
    }
#endif

    std::printf("%d,%.6f,%.3f,%.6f,%.3f,%.6g,%.6g,%.6f,%.6f,%.6f,%.3f\n",
                rows, rms_stats.best_ms,
                gib_per_second(rms_bytes, rms_stats.best_ms), cudnn_ms,
                cudnn_gib_s, cudnn_max_abs, cudnn_rstd_max_abs,
                residual_stats.best_ms, fused_stats.best_ms,
                split_stats.best_ms, split_stats.best_ms / fused_stats.best_ms);
  }

  CUDA_CHECK(cudaFree(d_inp1));
  CUDA_CHECK(cudaFree(d_inp2));
  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_rms_out));
  CUDA_CHECK(cudaFree(d_rms_cudnn_out));
  CUDA_CHECK(cudaFree(d_residual));
  CUDA_CHECK(cudaFree(d_split_residual));
  CUDA_CHECK(cudaFree(d_fused_normed));
  CUDA_CHECK(cudaFree(d_split_normed));
  CUDA_CHECK(cudaFree(d_rstd));
  CUDA_CHECK(cudaFree(d_cudnn_rstd));
  CUDA_CHECK(cudaFree(d_fused_rstd));
  CUDA_CHECK(cudaFree(d_split_rstd));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
