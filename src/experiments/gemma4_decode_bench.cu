#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int kHidden = 5376;
static constexpr int kPackedFfn = 43008;

extern "C" cudaError_t gemma4_ffn_gate_up_decode(const half *x,
                                                  const half *w_col_major,
                                                  half *y,
                                                  cudaStream_t stream);

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
                               cudaGetErrorString(status));                    \
    }                                                                          \
  } while (0)

#define CUDNN_CHECK(expr)                                                      \
  do {                                                                         \
    cudnnStatus_t status = (expr);                                             \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      throw std::runtime_error(std::string("cuDNN error: ") +                 \
                               cudnnGetErrorString(status));                   \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t status = (expr);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error("cuBLAS error: " + std::to_string(status));    \
    }                                                                          \
  } while (0)

struct CublasGemmDecode {
  cublasHandle_t handle = nullptr;

  explicit CublasGemmDecode(cudaStream_t stream) {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  }

  ~CublasGemmDecode() {
    if (handle != nullptr) {
      cublasDestroy(handle);
    }
  }

  void run(const half *x, const half *w_col_major, half *y) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, kPackedFfn, 1, kHidden, &alpha,
        w_col_major, CUDA_R_16F, kHidden, x, CUDA_R_16F, kHidden, &beta, y,
        CUDA_R_16F, kPackedFfn, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
};

struct CublasGemvDecode {
  cublasHandle_t handle = nullptr;

  explicit CublasGemvDecode(cudaStream_t stream) {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
  }

  ~CublasGemvDecode() {
    if (handle != nullptr) {
      cublasDestroy(handle);
    }
  }

  void run(const half *x, const half *w_col_major, half *y) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasHSHgemvStridedBatched(
        handle, CUBLAS_OP_T, kHidden, kPackedFfn, &alpha, w_col_major,
        kHidden, 0, x, 1, 0, &beta, y, 1, 0, 1));
  }
};

struct CudnnDecodeConv {
  cudnnHandle_t handle = nullptr;
  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnFilterDescriptor_t w_desc = nullptr;
  cudnnConvolutionDescriptor_t conv_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  void *workspace = nullptr;
  size_t workspace_bytes = 0;

  explicit CudnnDecodeConv(cudaStream_t stream) {
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnSetStream(handle, stream));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_HALF, 1, kHidden, 1, 1));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_HALF,
                                           CUDNN_TENSOR_NCHW, kPackedFfn,
                                           kHidden, 1, 1));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_HALF, 1, kPackedFfn, 1,
                                           1));

    int max_count = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &max_count));
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf(max_count);
    int returned = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, x_desc, w_desc, conv_desc, y_desc, max_count, &returned,
        perf.data()));

    bool found = false;
    for (int i = 0; i < returned; ++i) {
      if (perf[i].status == CUDNN_STATUS_SUCCESS) {
        algo = perf[i].algo;
        found = true;
        break;
      }
    }
    if (!found) {
      throw std::runtime_error("cuDNN did not return a valid convolution algo");
    }

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_bytes));
    if (workspace_bytes > 0) {
      CUDA_CHECK(cudaMalloc(&workspace, workspace_bytes));
    }

    std::fprintf(stderr, "cuDNN conv algo=%d workspace_bytes=%zu\n", int(algo),
                 workspace_bytes);
  }

  ~CudnnDecodeConv() {
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
    if (y_desc != nullptr) {
      cudnnDestroyTensorDescriptor(y_desc);
    }
    if (conv_desc != nullptr) {
      cudnnDestroyConvolutionDescriptor(conv_desc);
    }
    if (w_desc != nullptr) {
      cudnnDestroyFilterDescriptor(w_desc);
    }
    if (x_desc != nullptr) {
      cudnnDestroyTensorDescriptor(x_desc);
    }
    if (handle != nullptr) {
      cudnnDestroy(handle);
    }
  }

  void run(const half *x, const half *w, half *y) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha, x_desc, x, w_desc, w,
                                        conv_desc, algo, workspace,
                                        workspace_bytes, &beta, y_desc, y));
  }
};

struct TimingStats {
  float best_ms = 0.0f;
  float avg_ms = 0.0f;
};

template <typename Fn>
static float time_ms_once(Fn &&fn, cudaStream_t stream, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < iters; ++i) {
    fn();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return total_ms / iters;
}

template <typename Fn>
static TimingStats time_ms(Fn &&fn, cudaStream_t stream, int warmup, int iters,
                           int trials) {
  TimingStats stats;
  stats.best_ms = INFINITY;
  for (int i = 0; i < trials; ++i) {
    const float ms = time_ms_once(fn, stream, warmup, iters);
    stats.best_ms = std::min(stats.best_ms, ms);
    stats.avg_ms += ms;
  }
  stats.avg_ms /= trials;
  return stats;
}

static float max_abs_diff(const half *lhs, const half *rhs, int count) {
  std::vector<half> h_lhs(count);
  std::vector<half> h_rhs(count);
  CUDA_CHECK(cudaMemcpy(h_lhs.data(), lhs, count * sizeof(half),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_rhs.data(), rhs, count * sizeof(half),
                        cudaMemcpyDeviceToHost));

  float max_diff = 0.0f;
  for (int i = 0; i < count; ++i) {
    max_diff = std::max(max_diff, std::abs(__half2float(h_lhs[i]) -
                                           __half2float(h_rhs[i])));
  }
  return max_diff;
}

int main(int argc, char **argv) {
  const bool has_mode =
      argc > 1 && !std::isdigit(static_cast<unsigned char>(argv[1][0]));
  const std::string mode = has_mode ? argv[1] : "both";
  const int iter_arg = has_mode ? 2 : 1;
  const int warmup_arg = has_mode ? 3 : 2;
  const int trials_arg = has_mode ? 4 : 3;
  const int iters = argc > iter_arg ? std::atoi(argv[iter_arg]) : 100;
  const int warmup = argc > warmup_arg ? std::atoi(argv[warmup_arg]) : 20;
  const int trials = argc > trials_arg ? std::atoi(argv[trials_arg]) : 3;

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  half *x = nullptr;
  half *w = nullptr;
  half *custom_y = nullptr;
  half *cublas_y = nullptr;
  half *cublas_gemv_y = nullptr;
  half *cudnn_y = nullptr;
  CUDA_CHECK(cudaMalloc(&x, kHidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&w, size_t(kPackedFfn) * kHidden * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&custom_y, kPackedFfn * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&cublas_y, kPackedFfn * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&cublas_gemv_y, kPackedFfn * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&cudnn_y, kPackedFfn * sizeof(half)));

  std::vector<half> h_x(kHidden);
  std::vector<half> h_w(size_t(kPackedFfn) * kHidden);
  for (int i = 0; i < kHidden; ++i) {
    h_x[i] = __float2half((float((i % 17) - 8)) * 0.03125f);
  }
  for (size_t i = 0; i < h_w.size(); ++i) {
    h_w[i] = __float2half((float((i % 31) - 15)) * 0.015625f);
  }

  CUDA_CHECK(cudaMemcpyAsync(x, h_x.data(), h_x.size() * sizeof(half),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(w, h_w.data(), h_w.size() * sizeof(half),
                             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(custom_y, 0, kPackedFfn * sizeof(half), stream));
  CUDA_CHECK(cudaMemsetAsync(cublas_y, 0, kPackedFfn * sizeof(half), stream));
  CUDA_CHECK(
      cudaMemsetAsync(cublas_gemv_y, 0, kPackedFfn * sizeof(half), stream));
  CUDA_CHECK(cudaMemsetAsync(cudnn_y, 0, kPackedFfn * sizeof(half), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto run_custom = [&]() {
    CUDA_CHECK(gemma4_ffn_gate_up_decode(x, w, custom_y, stream));
  };

  run_custom();
  const TimingStats custom = time_ms(run_custom, stream, warmup, iters, trials);
  const double bytes = double(kPackedFfn) * double(kHidden) * sizeof(half);

  std::printf("shape=M1,N%d,K%d\n", kPackedFfn, kHidden);
  std::printf("mode=%s\n", mode.c_str());
  std::printf("iters=%d,warmup_iters=%d,trials=%d\n", iters, warmup, trials);
  std::printf("custom_best_ms=%.6f\n", custom.best_ms);
  std::printf("custom_avg_ms=%.6f\n", custom.avg_ms);
  std::printf("custom_best_weight_gbps=%.3f\n",
              bytes / (custom.best_ms * 1.0e6));

  if (mode == "both" || mode == "cublas") {
    CublasGemmDecode cublas(stream);
    auto run_cublas = [&]() { cublas.run(x, w, cublas_y); };

    run_cublas();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const float diff = max_abs_diff(custom_y, cublas_y, kPackedFfn);
    const TimingStats cublas_timing =
        time_ms(run_cublas, stream, warmup, iters, trials);

    std::printf("cublas_gemmex_best_ms=%.6f\n", cublas_timing.best_ms);
    std::printf("cublas_gemmex_avg_ms=%.6f\n", cublas_timing.avg_ms);
    std::printf("cublas_gemmex_best_weight_gbps=%.3f\n",
                bytes / (cublas_timing.best_ms * 1.0e6));
    std::printf("custom_vs_cublas_gemmex_speedup=%.6f\n",
                cublas_timing.best_ms / custom.best_ms);
    std::printf("cublas_gemmex_max_abs_diff=%.6g\n", diff);
  }

  if (mode == "both" || mode == "gemv") {
    CublasGemvDecode cublas_gemv(stream);
    auto run_cublas_gemv = [&]() { cublas_gemv.run(x, w, cublas_gemv_y); };

    run_cublas_gemv();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const float diff = max_abs_diff(custom_y, cublas_gemv_y, kPackedFfn);
    const TimingStats cublas_gemv_timing =
        time_ms(run_cublas_gemv, stream, warmup, iters, trials);

    std::printf("cublas_hshgemv_strided_batched_best_ms=%.6f\n",
                cublas_gemv_timing.best_ms);
    std::printf("cublas_hshgemv_strided_batched_avg_ms=%.6f\n",
                cublas_gemv_timing.avg_ms);
    std::printf("cublas_hshgemv_strided_batched_best_weight_gbps=%.3f\n",
                bytes / (cublas_gemv_timing.best_ms * 1.0e6));
    std::printf("custom_vs_cublas_hshgemv_strided_batched_speedup=%.6f\n",
                cublas_gemv_timing.best_ms / custom.best_ms);
    std::printf("cublas_hshgemv_strided_batched_max_abs_diff=%.6g\n", diff);
  }

  if (mode == "both" || mode == "cudnn") {
    CudnnDecodeConv cudnn(stream);
    auto run_cudnn = [&]() { cudnn.run(x, w, cudnn_y); };

    run_cudnn();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const float diff = max_abs_diff(custom_y, cudnn_y, kPackedFfn);
    const TimingStats cudnn_timing =
        time_ms(run_cudnn, stream, warmup, iters, trials);

    std::printf("cudnn_conv_best_ms=%.6f\n", cudnn_timing.best_ms);
    std::printf("cudnn_conv_avg_ms=%.6f\n", cudnn_timing.avg_ms);
    std::printf("cudnn_best_weight_gbps=%.3f\n",
                bytes / (cudnn_timing.best_ms * 1.0e6));
    std::printf("custom_vs_cudnn_speedup=%.6f\n",
                cudnn_timing.best_ms / custom.best_ms);
    std::printf("max_abs_diff=%.6g\n", diff);
    std::printf("cudnn_workspace_bytes=%zu\n", cudnn.workspace_bytes);
    std::printf("cudnn_conv_algo=%d\n", int(cudnn.algo));
  }

  cudaFree(x);
  cudaFree(w);
  cudaFree(custom_y);
  cudaFree(cublas_y);
  cudaFree(cublas_gemv_y);
  cudaFree(cudnn_y);
  cudaStreamDestroy(stream);
  return 0;
}
