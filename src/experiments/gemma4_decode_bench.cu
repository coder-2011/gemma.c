#include "gemma4_matmul_kernels.cuh"
#include "gemma4_bench_utils.cuh"
#include "gemma4.h"

#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

struct DecodeOp {
  const char *name;
  Gemma4Projection projection;
  int k;
  int n;
  int layers_per_token;
};

static const DecodeOp kDecodeOps[] = {
    {"ffn_gate_up", GEMMA4_PROJECTION_FFN_GATE_UP, GEMMA4_HIDDEN_SIZE,
     GEMMA4_PACKED_FFN_SIZE, GEMMA4_NUM_LAYERS},
    {"ffn_down", GEMMA4_PROJECTION_FFN_DOWN, GEMMA4_INTERMEDIATE_SIZE,
     GEMMA4_HIDDEN_SIZE, GEMMA4_NUM_LAYERS},
    {"sliding_qkv", GEMMA4_PROJECTION_SLIDING_QKV, GEMMA4_HIDDEN_SIZE,
     GEMMA4_SLIDING_QKV_SIZE, GEMMA4_SLIDING_LAYER_COUNT},
    {"sliding_o", GEMMA4_PROJECTION_SLIDING_O,
     GEMMA4_SLIDING_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE,
     GEMMA4_SLIDING_LAYER_COUNT},
    {"global_q", GEMMA4_PROJECTION_GLOBAL_Q, GEMMA4_HIDDEN_SIZE,
     GEMMA4_GLOBAL_Q_PROJ_SIZE, GEMMA4_GLOBAL_LAYER_COUNT},
    {"global_k", GEMMA4_PROJECTION_GLOBAL_K, GEMMA4_HIDDEN_SIZE,
     GEMMA4_GLOBAL_K_PROJ_SIZE, GEMMA4_GLOBAL_LAYER_COUNT},
    {"global_o", GEMMA4_PROJECTION_GLOBAL_O,
     GEMMA4_GLOBAL_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE,
     GEMMA4_GLOBAL_LAYER_COUNT},
    {"final_logits", GEMMA4_PROJECTION_FINAL_LOGITS, GEMMA4_HIDDEN_SIZE,
     GEMMA4_VOCAB_SIZE, 1},
};

__device__ uint32_t gemma4_mix_u32_device(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__global__ void gemma4_fill_random_bf16_kernel(__nv_bfloat16 *ptr,
                                               size_t count, uint64_t seed,
                                               float scale) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }

  uint32_t x = uint32_t(i) ^ uint32_t(i >> 32) ^ uint32_t(seed) ^
               uint32_t(seed >> 32);
  x = gemma4_mix_u32_device(x);
  const float u = float(x >> 8) * (1.0f / 16777216.0f);
  ptr[i] = __float2bfloat16_rn((u * 2.0f - 1.0f) * scale);
}

static uint64_t make_seed() {
  if (const char *env = std::getenv("GEMMA4_DECODE_BENCH_SEED")) {
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

static void fill_random_bf16(__nv_bfloat16 *ptr, size_t count, uint64_t seed,
                             float scale, cudaStream_t stream) {
  const int threads = 256;
  const int blocks = int((count + threads - 1) / threads);
  gemma4_fill_random_bf16_kernel<<<blocks, threads, 0, stream>>>(ptr, count, seed, scale);
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

  void gemv(const __nv_bfloat16 *x, const __nv_bfloat16 *w_col_major,
            __nv_bfloat16 *y, int k, int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasTSTgemvStridedBatched(
        handle, CUBLAS_OP_T, k, n, &alpha, w_col_major, k, 0, x, 1, 0, &beta,
        y, 1, 0, 1));
  }

  void gemm_m1(const __nv_bfloat16 *x, const __nv_bfloat16 *w_col_major,
               __nv_bfloat16 *y, int k, int n) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k,
                              &alpha, w_col_major, CUDA_R_16BF, k, x,
                              CUDA_R_16BF, k, &beta, y, CUDA_R_16BF, n,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
};

inline void cudnn_check(cudnnStatus_t status, const char *expr,
                        const char *file, int line) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(file) + ":" +
                             std::to_string(line) +
                             ": cuDNN error for " + expr + ": " +
                             cudnnGetErrorString(status));
  }
}

#define CUDNN_CHECK(expr)                                                      \
  cudnn_check((expr), #expr, __FILE__, __LINE__)

struct CudnnDecodeConv {
  cudnnHandle_t handle = nullptr;
  cudnnTensorDescriptor_t x_desc = nullptr;
  cudnnFilterDescriptor_t w_desc = nullptr;
  cudnnConvolutionDescriptor_t conv_desc = nullptr;
  cudnnTensorDescriptor_t y_desc = nullptr;
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  void *workspace = nullptr;
  size_t workspace_bytes = 0;

  CudnnDecodeConv(cudaStream_t stream, int k, int n) {
    CUDNN_CHECK(cudnnCreate(&handle));
    CUDNN_CHECK(cudnnSetStream(handle, stream));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_BFLOAT16, 1, k, 1, 1));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_BFLOAT16,
                                           CUDNN_TENSOR_NCHW, n, k, 1, 1));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));
    CUDNN_CHECK(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_BFLOAT16, 1, n, 1, 1));

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

  void conv(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
            __nv_bfloat16 *y) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(handle, &alpha, x_desc, x, w_desc, w,
                                        conv_desc, algo, workspace,
                                        workspace_bytes, &beta, y_desc, y));
  }
};

static bool should_run_op(const std::string &selected, const DecodeOp &op) {
  return selected == "all" || selected == op.name;
}

static void run_op(const DecodeOp &op, int iters, int warmup, int trials,
                   cudaStream_t stream, CublasDecode &cublas,
                   uint64_t base_seed) {
  __nv_bfloat16 *x = nullptr;
  __nv_bfloat16 *w = nullptr;
  __nv_bfloat16 *custom_y = nullptr;
  __nv_bfloat16 *gemv_y = nullptr;
  __nv_bfloat16 *gemm_y = nullptr;
  __nv_bfloat16 *cudnn_y = nullptr;
  const size_t x_count = size_t(op.k);
  const size_t w_count = size_t(op.n) * size_t(op.k);
  const size_t y_count = size_t(op.n);

  CUDA_CHECK(cudaMalloc(&x, x_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&w, w_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&custom_y, y_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&gemv_y, y_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&gemm_y, y_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&cudnn_y, y_count * sizeof(__nv_bfloat16)));

  const uint64_t x_seed = base_seed ^ (uint64_t(op.k) << 32) ^ uint64_t(op.n);
  const uint64_t w_seed =
      (base_seed + 0x9e3779b97f4a7c15ull) ^ (uint64_t(op.n) << 32) ^
      uint64_t(op.k);
  constexpr float kInputScale = 1.0f;
  constexpr float kWeightScale = 0.5f;
  fill_random_bf16(x, x_count, x_seed, kInputScale, stream);
  fill_random_bf16(w, w_count, w_seed, kWeightScale, stream);
  CUDA_CHECK(cudaMemsetAsync(custom_y, 0, y_count * sizeof(__nv_bfloat16), stream));
  CUDA_CHECK(cudaMemsetAsync(gemv_y, 0, y_count * sizeof(__nv_bfloat16), stream));
  CUDA_CHECK(cudaMemsetAsync(gemm_y, 0, y_count * sizeof(__nv_bfloat16), stream));
  CUDA_CHECK(cudaMemsetAsync(cudnn_y, 0, y_count * sizeof(__nv_bfloat16), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto run_custom = [&]() {
    CUDA_CHECK(gemma4_projection_decode(op.projection, x, w, custom_y, stream));
  };
  auto run_gemv = [&]() { cublas.gemv(x, w, gemv_y, op.k, op.n); };
  auto run_gemm = [&]() { cublas.gemm_m1(x, w, gemm_y, op.k, op.n); };

  run_custom();
  run_gemv();
  run_gemm();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const DiffStats gemv_diff = diff_stats_bf16(custom_y, gemv_y, op.n);
  const DiffStats gemm_diff = diff_stats_bf16(custom_y, gemm_y, op.n);

  const TimingStats custom = time_ms(run_custom, stream, warmup, iters, trials);
  const TimingStats gemv = time_ms(run_gemv, stream, warmup, iters, trials);
  const TimingStats gemm = time_ms(run_gemm, stream, warmup, iters, trials);
  const double bytes = double(op.n) * double(op.k) * sizeof(__nv_bfloat16);
  const double per_token_gb = bytes * double(op.layers_per_token) / 1.0e9;

  float cudnn_best_ms = -1.0f;
  float cudnn_avg_ms = -1.0f;
  double cudnn_gbps = 0.0;
  float cudnn_speedup = -1.0f;
  float cudnn_max_abs = -1.0f;
  float cudnn_mean_abs = -1.0f;
  float cudnn_max_rel = -1.0f;
  int cudnn_algo = -1;
  size_t cudnn_workspace_bytes = 0;
  try {
    CudnnDecodeConv cudnn(stream, op.k, op.n);
    auto run_cudnn = [&]() { cudnn.conv(x, w, cudnn_y); };
    run_cudnn();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const DiffStats cudnn_diff = diff_stats_bf16(custom_y, cudnn_y, op.n);
    const TimingStats cudnn_stats =
        time_ms(run_cudnn, stream, warmup, iters, trials);
    cudnn_best_ms = cudnn_stats.best_ms;
    cudnn_avg_ms = cudnn_stats.avg_ms;
    cudnn_gbps = bytes / (cudnn_best_ms * 1.0e6);
    cudnn_speedup = cudnn_best_ms / custom.best_ms;
    cudnn_max_abs = cudnn_diff.max_abs;
    cudnn_mean_abs = cudnn_diff.mean_abs;
    cudnn_max_rel = cudnn_diff.max_rel;
    cudnn_algo = int(cudnn.algo);
    cudnn_workspace_bytes = cudnn.workspace_bytes;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "cuDNN 1x1 conv unavailable for op=%s: %s\n",
                 op.name, e.what());
  }

  std::printf("op=%s,K=%d,N=%d,layers_per_token=%d,weight_gb_per_token=%.3f\n",
              op.name, op.k, op.n, op.layers_per_token, per_token_gb);
  std::printf("dtype=bf16,input_seed=%llu,weight_seed=%llu,input_scale=%.3f,weight_scale=%.3f\n",
              static_cast<unsigned long long>(x_seed),
              static_cast<unsigned long long>(w_seed), kInputScale,
              kWeightScale);
  std::printf("custom_best_ms=%.6f,custom_avg_ms=%.6f,custom_best_weight_gbps=%.3f\n",
              custom.best_ms, custom.avg_ms, bytes / (custom.best_ms * 1.0e6));
  std::printf("cublas_bf16_gemv_best_ms=%.6f,cublas_bf16_gemv_avg_ms=%.6f,cublas_bf16_gemv_best_weight_gbps=%.3f\n",
              gemv.best_ms, gemv.avg_ms, bytes / (gemv.best_ms * 1.0e6));
  std::printf("cublas_bf16_gemm_m1_best_ms=%.6f,cublas_bf16_gemm_m1_avg_ms=%.6f,cublas_bf16_gemm_m1_best_weight_gbps=%.3f\n",
              gemm.best_ms, gemm.avg_ms, bytes / (gemm.best_ms * 1.0e6));
  std::printf("cudnn_bf16_conv1x1_best_ms=%.6f,cudnn_bf16_conv1x1_avg_ms=%.6f,cudnn_bf16_conv1x1_best_weight_gbps=%.3f\n",
              cudnn_best_ms, cudnn_avg_ms, cudnn_gbps);
  std::printf("custom_vs_cublas_gemv_speedup=%.6f\n",
              gemv.best_ms / custom.best_ms);
  std::printf("custom_vs_cublas_gemm_m1_speedup=%.6f\n",
              gemm.best_ms / custom.best_ms);
  std::printf("custom_vs_cudnn_conv1x1_speedup=%.6f\n", cudnn_speedup);
  std::printf("cublas_bf16_gemv_max_abs_diff=%.6g,cublas_bf16_gemv_mean_abs_diff=%.6g,cublas_bf16_gemv_max_rel_diff=%.6g\n",
              gemv_diff.max_abs, gemv_diff.mean_abs, gemv_diff.max_rel);
  std::printf("cublas_bf16_gemm_m1_max_abs_diff=%.6g,cublas_bf16_gemm_m1_mean_abs_diff=%.6g,cublas_bf16_gemm_m1_max_rel_diff=%.6g\n",
              gemm_diff.max_abs, gemm_diff.mean_abs, gemm_diff.max_rel);
  std::printf("cudnn_bf16_conv1x1_max_abs_diff=%.6g,cudnn_bf16_conv1x1_mean_abs_diff=%.6g,cudnn_bf16_conv1x1_max_rel_diff=%.6g\n",
              cudnn_max_abs, cudnn_mean_abs, cudnn_max_rel);
  std::printf("cudnn_bf16_conv1x1_algo=%d,cudnn_bf16_conv1x1_workspace_bytes=%zu\n\n",
              cudnn_algo, cudnn_workspace_bytes);

  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(w));
  CUDA_CHECK(cudaFree(custom_y));
  CUDA_CHECK(cudaFree(gemv_y));
  CUDA_CHECK(cudaFree(gemm_y));
  CUDA_CHECK(cudaFree(cudnn_y));
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
  const uint64_t seed = make_seed();

  std::printf("selected=%s,iters=%d,warmup_iters=%d,trials=%d,dtype=bf16,base_seed=%llu\n\n",
              selected.c_str(), iters, warmup, trials,
              static_cast<unsigned long long>(seed));

  bool ran_any = false;
  for (const DecodeOp &op : kDecodeOps) {
    if (!should_run_op(selected, op)) {
      continue;
    }
    ran_any = true;
    run_op(op, iters, warmup, trials, stream, cublas, seed);
  }

  if (!ran_any) {
    std::fprintf(stderr, "unknown op '%s'\n", selected.c_str());
    cudaStreamDestroy(stream);
    return 2;
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
