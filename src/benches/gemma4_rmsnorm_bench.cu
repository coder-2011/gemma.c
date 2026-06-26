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
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#if GEMMA4_HAS_CUDNN_FRONTEND
#include <memory>
#include <unordered_map>
#endif

namespace {

__global__ void gemma4_rmsnorm_fill_constant_bf16_kernel(__nv_bfloat16 *ptr,
                                                         size_t count,
                                                         float value) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i < count) {
    ptr[i] = __float2bfloat16_rn(value);
  }
}

void fill_constant_bf16(__nv_bfloat16 *ptr,
                        size_t count,
                        float value,
                        cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = int((count + threads - 1) / threads);
  gemma4_rmsnorm_fill_constant_bf16_kernel<<<blocks, threads, 0, stream>>>(
      ptr, count, value);
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
    cudaFree(workspace);
    cudnnDestroy(handle);
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
  const int width = argc > 5 ? std::atoi(argv[5]) : GEMMA4_HIDDEN_SIZE;

  if (iters <= 0 || warmup < 0 || trials <= 0 || max_rows <= 0 ||
      width <= 0 || (width % 8) != 0) {
    std::fprintf(stderr,
                 "usage: %s [iters=200] [warmup=30] [trials=5] "
                 "[max_rows=4096] [width=%d]\n",
                 argv[0], GEMMA4_HIDDEN_SIZE);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  const size_t max_elems = static_cast<size_t>(max_rows) * width;
  const uint64_t seed = make_seed("GEMMA4_RMSNORM_BENCH_SEED");

  const size_t width_elems = static_cast<size_t>(width);
  const size_t max_row_elems = static_cast<size_t>(max_rows);

  DeviceBuffer<__nv_bfloat16> d_inp1(max_elems);
  DeviceBuffer<__nv_bfloat16> d_inp2(max_elems);
  DeviceBuffer<__nv_bfloat16> d_weight(width_elems);
  DeviceBuffer<__nv_bfloat16> d_ones_weight(width_elems);
  DeviceBuffer<__nv_bfloat16> d_rms_out(max_elems);
  DeviceBuffer<__nv_bfloat16> d_rms_cudnn_out(max_elems);
  DeviceBuffer<__nv_bfloat16> d_scale_free_out(max_elems);
  DeviceBuffer<__nv_bfloat16> d_scale_free_cudnn_out(max_elems);
  DeviceBuffer<__nv_bfloat16> d_residual(max_elems);
  DeviceBuffer<__nv_bfloat16> d_split_residual(max_elems);
  DeviceBuffer<__nv_bfloat16> d_fused_normed(max_elems);
  DeviceBuffer<__nv_bfloat16> d_split_normed(max_elems);
  DeviceBuffer<float> d_cudnn_rstd(max_row_elems);
  DeviceBuffer<float> d_scale_free_cudnn_rstd(max_row_elems);

  fill_random_bf16(d_inp1, max_elems, seed ^ 0x1001u, 1.0f, stream);
  fill_random_bf16(d_inp2, max_elems, seed ^ 0x2002u, 1.0f, stream);
  fill_random_bf16(d_weight, static_cast<size_t>(width), seed ^ 0x3003u, 0.5f, stream);
  fill_constant_bf16(d_ones_weight, static_cast<size_t>(width), 1.0f, stream);
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
  std::printf("rows,rms_ms,rms_gib_s,rms_graph_kernel_ms,"
              "rms_graph_kernel_gib_s,cudnn_ms,cudnn_gib_s,"
              "cudnn_graph_kernel_ms,cudnn_graph_kernel_gib_s,"
              "cudnn_max_abs,scale_free_ms,"
              "scale_free_gib_s,scale_free_graph_kernel_ms,"
              "scale_free_graph_kernel_gib_s,cudnn_one_scale_ms,"
              "cudnn_one_scale_gib_s,cudnn_one_scale_graph_kernel_ms,"
              "cudnn_one_scale_graph_kernel_gib_s,cudnn_one_scale_max_abs,"
              "scale_free_vs_cudnn_one_scale,residual_ms,fused_ms,"
              "split_ms,fused_vs_split,fused_graph_kernel_ms,"
              "split_graph_kernel_ms,fused_graph_vs_split_graph,"
              "cudnn_split_ms,cudnn_split_graph_ms,fused_vs_cudnn_split,"
              "cudnn_split_max_abs\n");

  for (int rows : row_counts_up_to(max_rows)) {
    const int count = rows * width;
    const double rms_bytes =
        double(rows) * width * sizeof(__nv_bfloat16) * 3.0;
    const double scale_free_bytes =
        double(rows) * width * sizeof(__nv_bfloat16) * 2.0;
    const double residual_bytes =
        double(rows) * width * sizeof(__nv_bfloat16) * 3.0;
    const double fused_bytes =
        double(rows) * width * sizeof(__nv_bfloat16) * 5.0;
    const double split_bytes = residual_bytes + rms_bytes;
    const bool has_fused = width == GEMMA4_HIDDEN_SIZE;

    auto run_rms = [&]() {
      CUDA_CHECK(gemma4_rmsnorm_bf16(d_rms_out, d_inp1, d_weight, rows, width,
                                     GEMMA4_RMS_NORM_EPS, stream));
    };
    auto run_residual = [&]() {
      CUDA_CHECK(gemma4_residual_add_bf16(d_residual, d_inp1, d_inp2, count, stream));
    };
    auto run_scale_free = [&]() {
      CUDA_CHECK(gemma4_rmsnorm_scale_free_bf16(
          d_scale_free_out, d_inp1, rows, width,
          GEMMA4_RMS_NORM_EPS, stream));
    };
    auto run_fused = [&]() {
      if (!has_fused) {
        return;
      }
      CUDA_CHECK(gemma4_residual_add_rmsnorm_bf16(
          d_split_residual, d_fused_normed, d_inp1, d_inp2, d_weight, rows,
          width, GEMMA4_RMS_NORM_EPS, stream));
    };
    auto run_split = [&]() {
      CUDA_CHECK(gemma4_residual_add_bf16(d_split_residual, d_inp1, d_inp2, count, stream));
      CUDA_CHECK(gemma4_rmsnorm_bf16(d_split_normed, d_split_residual,
                                     d_weight, rows, width,
                                     GEMMA4_RMS_NORM_EPS, stream));
    };

    run_rms();
    run_residual();
    run_scale_free();
    run_fused();
    run_split();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const TimingStats rms_stats =
        time_ms(run_rms, stream, warmup, iters, trials);
    const TimingStats residual_stats =
        time_ms(run_residual, stream, warmup, iters, trials);
    const TimingStats scale_free_stats =
        time_ms(run_scale_free, stream, warmup, iters, trials);
    TimingStats fused_stats{-1.0f, -1.0f};
    if (has_fused) {
      fused_stats = time_ms(run_fused, stream, warmup, iters, trials);
    }
    const TimingStats split_stats =
        time_ms(run_split, stream, warmup, iters, trials);

    float fused_graph_ms = -1.0f;
    float split_graph_ms = -1.0f;
    if (has_fused) {
      try {
        const TimingStats fused_graph_stats =
            time_ms_graph(run_fused, stream, warmup, iters, trials);
        fused_graph_ms = fused_graph_stats.best_ms;
      } catch (const std::exception &e) {
        std::fprintf(
            stderr,
            "fused residual+RMSNorm CUDA graph timing unavailable for "
            "rows=%d: %s\n",
            rows, e.what());
      }
    }
    try {
      const TimingStats split_graph_stats =
          time_ms_graph(run_split, stream, warmup, iters, trials);
      split_graph_ms = split_graph_stats.best_ms;
    } catch (const std::exception &e) {
      std::fprintf(stderr,
                   "split residual+RMSNorm CUDA graph timing unavailable for rows=%d: %s\n",
                   rows, e.what());
    }

    float rms_graph_ms = -1.0f;
    double rms_graph_gib_s = 0.0;
    try {
      const TimingStats rms_graph_stats =
          time_ms_graph(run_rms, stream, warmup, iters, trials);
      rms_graph_ms = rms_graph_stats.best_ms;
      rms_graph_gib_s = gib_per_second(rms_bytes, rms_graph_ms);
    } catch (const std::exception &e) {
      std::fprintf(stderr,
                   "custom RMSNorm CUDA graph timing unavailable for rows=%d: %s\n",
                   rows, e.what());
    }

    float scale_free_graph_ms = -1.0f;
    double scale_free_graph_gib_s = 0.0;
    try {
      const TimingStats scale_free_graph_stats =
          time_ms_graph(run_scale_free, stream, warmup, iters, trials);
      scale_free_graph_ms = scale_free_graph_stats.best_ms;
      scale_free_graph_gib_s =
          gib_per_second(scale_free_bytes, scale_free_graph_ms);
    } catch (const std::exception &e) {
      std::fprintf(stderr,
                   "custom scale-free RMSNorm CUDA graph timing unavailable "
                   "for rows=%d: %s\n",
                   rows, e.what());
    }

    float cudnn_ms = -1.0f;
    double cudnn_gib_s = 0.0;
    float cudnn_graph_ms = -1.0f;
    double cudnn_graph_gib_s = 0.0;
    float cudnn_max_abs = -1.0f;
    float cudnn_one_scale_ms = -1.0f;
    double cudnn_one_scale_gib_s = 0.0;
    float cudnn_one_scale_graph_ms = -1.0f;
    double cudnn_one_scale_graph_gib_s = 0.0;
    float cudnn_one_scale_max_abs = -1.0f;
    float cudnn_split_ms = -1.0f;
    float cudnn_split_graph_ms = -1.0f;
    float cudnn_split_max_abs = -1.0f;

#if GEMMA4_HAS_CUDNN_FRONTEND
    try {
      CudnnRmsnorm cudnn(rows, width, GEMMA4_RMS_NORM_EPS, stream);
      auto run_cudnn = [&]() {
        cudnn.run(d_inp1, d_weight, d_rms_cudnn_out, d_cudnn_rstd);
      };
      run_cudnn();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const TimingStats cudnn_stats =
          time_ms(run_cudnn, stream, warmup, iters, trials);
      cudnn_ms = cudnn_stats.best_ms;
      cudnn_gib_s = gib_per_second(rms_bytes, cudnn_ms);

      try {
        const TimingStats cudnn_graph_stats =
            time_ms_graph(run_cudnn, stream, warmup, iters, trials);
        cudnn_graph_ms = cudnn_graph_stats.best_ms;
        cudnn_graph_gib_s = gib_per_second(rms_bytes, cudnn_graph_ms);
      } catch (const std::exception &e) {
        std::fprintf(stderr,
                     "cuDNN RMSNorm CUDA graph timing unavailable for rows=%d: %s\n",
                     rows, e.what());
      }
      const DiffStats out_diff =
          diff_stats_bf16(d_rms_out, d_rms_cudnn_out, count);
      cudnn_max_abs = out_diff.max_abs;

      auto run_cudnn_one_scale = [&]() {
        cudnn.run(d_inp1, d_ones_weight, d_scale_free_cudnn_out,
                  d_scale_free_cudnn_rstd);
      };
      run_cudnn_one_scale();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const TimingStats cudnn_one_scale_stats =
          time_ms(run_cudnn_one_scale, stream, warmup, iters, trials);
      cudnn_one_scale_ms = cudnn_one_scale_stats.best_ms;
      cudnn_one_scale_gib_s = gib_per_second(rms_bytes, cudnn_one_scale_ms);

      try {
        const TimingStats cudnn_one_scale_graph_stats =
            time_ms_graph(run_cudnn_one_scale, stream, warmup, iters, trials);
        cudnn_one_scale_graph_ms = cudnn_one_scale_graph_stats.best_ms;
        cudnn_one_scale_graph_gib_s =
            gib_per_second(rms_bytes, cudnn_one_scale_graph_ms);
      } catch (const std::exception &e) {
        std::fprintf(stderr,
                     "cuDNN one-scale RMSNorm CUDA graph timing unavailable "
                     "for rows=%d: %s\n",
                     rows, e.what());
      }

      const DiffStats one_scale_out_diff = diff_stats_bf16(
          d_scale_free_out, d_scale_free_cudnn_out, count);
      cudnn_one_scale_max_abs = one_scale_out_diff.max_abs;

      auto run_cudnn_split = [&]() {
        CUDA_CHECK(gemma4_residual_add_bf16(d_residual, d_inp1, d_inp2, count, stream));
        cudnn.run(d_residual, d_weight, d_rms_cudnn_out, d_cudnn_rstd);
      };
      run_cudnn_split();
      CUDA_CHECK(cudaStreamSynchronize(stream));
      const TimingStats cudnn_split_stats =
          time_ms(run_cudnn_split, stream, warmup, iters, trials);
      cudnn_split_ms = cudnn_split_stats.best_ms;

      try {
        const TimingStats cudnn_split_graph_stats =
            time_ms_graph(run_cudnn_split, stream, warmup, iters, trials);
        cudnn_split_graph_ms = cudnn_split_graph_stats.best_ms;
      } catch (const std::exception &e) {
        std::fprintf(stderr,
                     "custom residual + cuDNN RMSNorm CUDA graph timing "
                     "unavailable for rows=%d: %s\n",
                     rows, e.what());
      }

      if (has_fused) {
        run_fused();
        run_cudnn_split();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const DiffStats split_out_diff =
            diff_stats_bf16(d_fused_normed, d_rms_cudnn_out, count);
        cudnn_split_max_abs = split_out_diff.max_abs;
      }
    } catch (const std::exception &e) {
      std::fprintf(stderr, "cuDNN RMSNorm unavailable for rows=%d: %s\n", rows,
                   e.what());
    }
#endif

    const float fused_graph_vs_split_graph =
        (fused_graph_ms > 0.0f && split_graph_ms > 0.0f)
            ? split_graph_ms / fused_graph_ms
            : -1.0f;
    const float fused_vs_cudnn_split =
        has_fused && cudnn_split_ms > 0.0f
            ? cudnn_split_ms / fused_stats.best_ms
            : -1.0f;
    const float fused_vs_split =
        has_fused ? split_stats.best_ms / fused_stats.best_ms : -1.0f;
    const float scale_free_vs_cudnn_one_scale =
        cudnn_one_scale_ms > 0.0f
            ? cudnn_one_scale_ms / scale_free_stats.best_ms
            : -1.0f;
    std::printf("%d,%.6f,%.3f,%.6f,%.3f,%.6f,%.3f,%.6f,%.3f,"
                "%.6g,%.6f,%.3f,%.6f,%.3f,%.6f,%.3f,"
                "%.6f,%.3f,%.6g,%.3f,%.6f,%.6f,%.6f,"
                "%.3f,%.6f,%.6f,%.3f,%.6f,%.6f,%.3f,%.6g\n",
                rows, rms_stats.best_ms,
                gib_per_second(rms_bytes, rms_stats.best_ms), rms_graph_ms,
                rms_graph_gib_s, cudnn_ms, cudnn_gib_s, cudnn_graph_ms,
                cudnn_graph_gib_s, cudnn_max_abs,
                scale_free_stats.best_ms,
                gib_per_second(scale_free_bytes, scale_free_stats.best_ms),
                scale_free_graph_ms, scale_free_graph_gib_s,
                cudnn_one_scale_ms, cudnn_one_scale_gib_s,
                cudnn_one_scale_graph_ms, cudnn_one_scale_graph_gib_s,
                cudnn_one_scale_max_abs, scale_free_vs_cudnn_one_scale,
                residual_stats.best_ms, fused_stats.best_ms,
                split_stats.best_ms, fused_vs_split,
                fused_graph_ms, split_graph_ms, fused_graph_vs_split_graph,
                cudnn_split_ms, cudnn_split_graph_ms, fused_vs_cudnn_split,
                cudnn_split_max_abs);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
