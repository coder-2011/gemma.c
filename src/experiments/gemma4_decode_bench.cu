#include "gemma4_matmul_kernels.cuh"

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

using DecodeLaunch = cudaError_t (*)(const half *, const half *, half *,
                                     cudaStream_t);

struct DecodeOp {
  const char *name;
  int k;
  int n;
  int layers_per_token;
  DecodeLaunch launch;
};

static const DecodeOp kDecodeOps[] = {
    {"ffn_gate_up", 5376, 43008, 60, gemma4_ffn_gate_up_decode},
    {"ffn_down", 21504, 5376, 60, gemma4_ffn_down_decode},
    {"sliding_qkv", 5376, 16384, 50, gemma4_sliding_qkv_decode},
    {"sliding_o", 8192, 5376, 50, gemma4_sliding_o_decode},
    {"global_q", 5376, 16384, 10, gemma4_global_q_decode},
    {"global_k", 5376, 2048, 10, gemma4_global_k_decode},
    {"global_o", 16384, 5376, 10, gemma4_global_o_decode},
    {"final_logits", 5376, 262144, 1, gemma4_final_logits_decode},
};

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
                               cudaGetErrorString(status));                    \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t status = (expr);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      throw std::runtime_error("cuBLAS error: " + std::to_string(status));    \
    }                                                                          \
  } while (0)

__global__ void fill_half_kernel(half *ptr, size_t count, int seed) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }
  int value = int((i * 17 + size_t(seed) * 13) % 31) - 15;
  ptr[i] = __float2half(float(value) * 0.015625f);
}

static void fill_half(half *ptr, size_t count, int seed, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = int((count + threads - 1) / threads);
  fill_half_kernel<<<blocks, threads, 0, stream>>>(ptr, count, seed);
  CUDA_CHECK(cudaGetLastError());
}

struct CublasDecode {
  cublasHandle_t handle = nullptr;

  explicit CublasDecode(cudaStream_t stream) {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  }

  ~CublasDecode() {
    if (handle != nullptr) {
      cublasDestroy(handle);
    }
  }

  void gemv(const half *x, const half *w_col_major, half *y, int k, int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasHSHgemvStridedBatched(
        handle, CUBLAS_OP_T, k, n, &alpha, w_col_major, k, 0, x, 1, 0, &beta,
        y, 1, 0, 1));
  }

  void gemm_m1(const half *x, const half *w_col_major, half *y, int k, int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k,
                              &alpha, w_col_major, CUDA_R_16F, k, x,
                              CUDA_R_16F, k, &beta, y, CUDA_R_16F, n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
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

static bool should_run_op(const std::string &selected, const DecodeOp &op) {
  return selected == "all" || selected == op.name;
}

static void run_op(const DecodeOp &op, int iters, int warmup, int trials,
                   cudaStream_t stream, CublasDecode &cublas) {
  half *x = nullptr;
  half *w = nullptr;
  half *custom_y = nullptr;
  half *gemv_y = nullptr;
  half *gemm_y = nullptr;
  const size_t x_count = size_t(op.k);
  const size_t w_count = size_t(op.n) * size_t(op.k);
  const size_t y_count = size_t(op.n);

  CUDA_CHECK(cudaMalloc(&x, x_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&w, w_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&custom_y, y_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&gemv_y, y_count * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&gemm_y, y_count * sizeof(half)));

  fill_half(x, x_count, 1, stream);
  fill_half(w, w_count, 2, stream);
  CUDA_CHECK(cudaMemsetAsync(custom_y, 0, y_count * sizeof(half), stream));
  CUDA_CHECK(cudaMemsetAsync(gemv_y, 0, y_count * sizeof(half), stream));
  CUDA_CHECK(cudaMemsetAsync(gemm_y, 0, y_count * sizeof(half), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto run_custom = [&]() { CUDA_CHECK(op.launch(x, w, custom_y, stream)); };
  auto run_gemv = [&]() { cublas.gemv(x, w, gemv_y, op.k, op.n); };
  auto run_gemm = [&]() { cublas.gemm_m1(x, w, gemm_y, op.k, op.n); };

  run_custom();
  run_gemv();
  run_gemm();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const float gemv_diff = max_abs_diff(custom_y, gemv_y, op.n);
  const float gemm_diff = max_abs_diff(custom_y, gemm_y, op.n);

  const TimingStats custom = time_ms(run_custom, stream, warmup, iters, trials);
  const TimingStats gemv = time_ms(run_gemv, stream, warmup, iters, trials);
  const TimingStats gemm = time_ms(run_gemm, stream, warmup, iters, trials);
  const double bytes = double(op.n) * double(op.k) * sizeof(half);
  const double per_token_gb = bytes * double(op.layers_per_token) / 1.0e9;

  std::printf("op=%s,K=%d,N=%d,layers_per_token=%d,weight_gb_per_token=%.3f\n",
              op.name, op.k, op.n, op.layers_per_token, per_token_gb);
  std::printf("custom_best_ms=%.6f,custom_avg_ms=%.6f,custom_best_weight_gbps=%.3f\n",
              custom.best_ms, custom.avg_ms, bytes / (custom.best_ms * 1.0e6));
  std::printf("cublas_gemv_best_ms=%.6f,cublas_gemv_avg_ms=%.6f,cublas_gemv_best_weight_gbps=%.3f\n",
              gemv.best_ms, gemv.avg_ms, bytes / (gemv.best_ms * 1.0e6));
  std::printf("cublas_gemm_m1_best_ms=%.6f,cublas_gemm_m1_avg_ms=%.6f,cublas_gemm_m1_best_weight_gbps=%.3f\n",
              gemm.best_ms, gemm.avg_ms, bytes / (gemm.best_ms * 1.0e6));
  std::printf("custom_vs_cublas_gemv_speedup=%.6f\n",
              gemv.best_ms / custom.best_ms);
  std::printf("custom_vs_cublas_gemm_m1_speedup=%.6f\n",
              gemm.best_ms / custom.best_ms);
  std::printf("cublas_gemv_max_abs_diff=%.6g\n", gemv_diff);
  std::printf("cublas_gemm_m1_max_abs_diff=%.6g\n\n", gemm_diff);

  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(w));
  CUDA_CHECK(cudaFree(custom_y));
  CUDA_CHECK(cudaFree(gemv_y));
  CUDA_CHECK(cudaFree(gemm_y));
}

int main(int argc, char **argv) {
  const bool has_op =
      argc > 1 && !std::isdigit(static_cast<unsigned char>(argv[1][0]));
  const std::string selected = has_op ? argv[1] : "all";
  const int iter_arg = has_op ? 2 : 1;
  const int warmup_arg = has_op ? 3 : 2;
  const int trials_arg = has_op ? 4 : 3;
  const int iters = argc > iter_arg ? std::atoi(argv[iter_arg]) : 20;
  const int warmup = argc > warmup_arg ? std::atoi(argv[warmup_arg]) : 5;
  const int trials = argc > trials_arg ? std::atoi(argv[trials_arg]) : 2;

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CublasDecode cublas(stream);

  std::printf("selected=%s,iters=%d,warmup_iters=%d,trials=%d\n\n",
              selected.c_str(), iters, warmup, trials);

  bool ran_any = false;
  for (const DecodeOp &op : kDecodeOps) {
    if (!should_run_op(selected, op)) {
      continue;
    }
    ran_any = true;
    run_op(op, iters, warmup, trials, stream, cublas);
  }

  if (!ran_any) {
    std::fprintf(stderr, "unknown op '%s'\n", selected.c_str());
    cudaStreamDestroy(stream);
    return 2;
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
