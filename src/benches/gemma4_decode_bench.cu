#include "gemma4_matmul_kernels.cuh"
#include "gemma4_bench_utils.cuh"
#include "gemma4.h"

#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct DecodeOp {
  const char *name;
  Gemma4Projection projection;
  int k;
  int n;
  int layers_per_token;
};

static constexpr DecodeOp kDecodeOps[] = {
    {"ffn_gate_up", GEMMA4_PROJECTION_FFN_GATE_UP, GEMMA4_HIDDEN_SIZE,
     GEMMA4_PACKED_FFN_SIZE, GEMMA4_NUM_LAYERS},
    {"ffn_down", GEMMA4_PROJECTION_FFN_DOWN, GEMMA4_INTERMEDIATE_SIZE,
     GEMMA4_HIDDEN_SIZE, GEMMA4_NUM_LAYERS},
    {"sliding_q", GEMMA4_PROJECTION_SLIDING_Q, GEMMA4_HIDDEN_SIZE,
     GEMMA4_SLIDING_Q_PROJ_SIZE, GEMMA4_SLIDING_LAYER_COUNT},
    {"sliding_kv", GEMMA4_PROJECTION_SLIDING_KV, GEMMA4_HIDDEN_SIZE,
     GEMMA4_SLIDING_KV_PROJ_SIZE, GEMMA4_SLIDING_LAYER_COUNT},
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

struct CublasDecode {
  cublasHandle_t handle = nullptr;

  explicit CublasDecode(cudaStream_t stream) {
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  }

  ~CublasDecode() {
    cublasDestroy(handle);
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
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroy(handle);
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
  const size_t x_count = size_t(op.k);
  const size_t w_count = size_t(op.n) * size_t(op.k);
  const size_t y_count = size_t(op.n);

  thrust::device_vector<__nv_bfloat16> x(x_count);
  thrust::device_vector<__nv_bfloat16> w(w_count);
  thrust::device_vector<__nv_bfloat16> custom_y(y_count);
  thrust::device_vector<__nv_bfloat16> swizzled_y(y_count);
  thrust::device_vector<__nv_bfloat16> gemv_y(y_count);
  thrust::device_vector<__nv_bfloat16> gemm_y(y_count);
  thrust::device_vector<__nv_bfloat16> cudnn_y(y_count);

  const uint64_t x_seed = base_seed ^ (uint64_t(op.k) << 32) ^ uint64_t(op.n);
  const uint64_t w_seed =
      (base_seed + 0x9e3779b97f4a7c15ull) ^ (uint64_t(op.n) << 32) ^
      uint64_t(op.k);
  constexpr float kInputScale = 1.0f;
  constexpr float kWeightScale = 0.5f;
  fill_random_bf16(raw_ptr(x), x_count, x_seed, kInputScale, stream);
  fill_random_bf16(raw_ptr(w), w_count, w_seed, kWeightScale, stream);
  CUDA_CHECK(cudaMemsetAsync(raw_ptr(custom_y), 0, y_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemsetAsync(raw_ptr(swizzled_y), 0, y_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemsetAsync(raw_ptr(gemv_y), 0, y_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemsetAsync(raw_ptr(gemm_y), 0, y_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMemsetAsync(raw_ptr(cudnn_y), 0, y_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto run_custom = [&]() {
    CUDA_CHECK(gemma4_projection_decode(
        op.projection, raw_ptr(x), raw_ptr(w), raw_ptr(custom_y), stream));
  };
  auto run_swizzled = [&]() {
    CUDA_CHECK(gemma4_projection_decode_swizzled(
        op.projection, GEMMA4_DECODE_SWIZZLE_INTERLEAVE_16, raw_ptr(x),
        raw_ptr(w), raw_ptr(swizzled_y), stream));
  };
  auto run_gemv = [&]() { cublas.gemv(raw_ptr(x), raw_ptr(w), raw_ptr(gemv_y), op.k, op.n); };
  auto run_gemm = [&]() { cublas.gemm_m1(raw_ptr(x), raw_ptr(w), raw_ptr(gemm_y), op.k, op.n); };

  run_custom();
  run_swizzled();
  run_gemv();
  run_gemm();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const DiffStats swizzled_diff = diff_stats_bf16(raw_ptr(custom_y), raw_ptr(swizzled_y), op.n);
  const DiffStats gemv_diff = diff_stats_bf16(raw_ptr(custom_y), raw_ptr(gemv_y), op.n);
  const DiffStats gemm_diff = diff_stats_bf16(raw_ptr(custom_y), raw_ptr(gemm_y), op.n);

  const TimingStats custom = time_ms(run_custom, stream, warmup, iters, trials);
  const TimingStats swizzled =
      time_ms(run_swizzled, stream, warmup, iters, trials);
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
    auto run_cudnn = [&]() { cudnn.conv(raw_ptr(x), raw_ptr(w), raw_ptr(cudnn_y)); };
    run_cudnn();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const DiffStats cudnn_diff = diff_stats_bf16(raw_ptr(custom_y), raw_ptr(cudnn_y), op.n);
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

  char context[96];
  std::snprintf(context, sizeof(context), "op=%s variant=custom", op.name);
  gemma4_bench_print_timing_stats("decode_projection", context, custom);
  std::snprintf(context, sizeof(context), "op=%s variant=custom_swizzle16",
                op.name);
  gemma4_bench_print_timing_stats("decode_projection", context, swizzled);
  std::snprintf(context, sizeof(context), "op=%s variant=cublas_gemv",
                op.name);
  gemma4_bench_print_timing_stats("decode_projection", context, gemv);
  std::snprintf(context, sizeof(context), "op=%s variant=cublas_gemm_m1",
                op.name);
  gemma4_bench_print_timing_stats("decode_projection", context, gemm);

  std::printf("op=%s,K=%d,N=%d,layers_per_token=%d,weight_gb_per_token=%.3f\n",
              op.name, op.k, op.n, op.layers_per_token, per_token_gb);
  std::printf("dtype=bf16,input_seed=%llu,weight_seed=%llu,input_scale=%.3f,weight_scale=%.3f\n",
              static_cast<unsigned long long>(x_seed),
              static_cast<unsigned long long>(w_seed), kInputScale,
              kWeightScale);
  std::printf("custom_best_ms=%.6f,custom_avg_ms=%.6f,custom_best_weight_gbps=%.3f\n",
              custom.best_ms, custom.avg_ms, bytes / (custom.best_ms * 1.0e6));
  std::printf("custom_swizzle16_best_ms=%.6f,"
              "custom_swizzle16_avg_ms=%.6f,"
              "custom_swizzle16_best_weight_gbps=%.3f,"
              "custom_swizzle16_vs_identity=%.6f\n",
              swizzled.best_ms, swizzled.avg_ms,
              bytes / (swizzled.best_ms * 1.0e6),
              custom.best_ms / swizzled.best_ms);
  std::printf("cublas_bf16_gemv_best_ms=%.6f,"
              "cublas_bf16_gemv_avg_ms=%.6f,"
              "cublas_bf16_gemv_best_weight_gbps=%.3f\n",
              gemv.best_ms, gemv.avg_ms, bytes / (gemv.best_ms * 1.0e6));
  std::printf("cublas_bf16_gemm_m1_best_ms=%.6f,"
              "cublas_bf16_gemm_m1_avg_ms=%.6f,"
              "cublas_bf16_gemm_m1_best_weight_gbps=%.3f\n",
              gemm.best_ms, gemm.avg_ms, bytes / (gemm.best_ms * 1.0e6));
  std::printf("cudnn_bf16_conv1x1_best_ms=%.6f,"
              "cudnn_bf16_conv1x1_avg_ms=%.6f,"
              "cudnn_bf16_conv1x1_best_weight_gbps=%.3f\n",
              cudnn_best_ms, cudnn_avg_ms, cudnn_gbps);
  std::printf("custom_vs_cublas_gemv_speedup=%.6f\n",
              gemv.best_ms / custom.best_ms);
  std::printf("custom_vs_cublas_gemm_m1_speedup=%.6f\n",
              gemm.best_ms / custom.best_ms);
  std::printf("custom_vs_cudnn_conv1x1_speedup=%.6f\n", cudnn_speedup);
  std::printf("cublas_bf16_gemv_max_abs_diff=%.6g,"
              "cublas_bf16_gemv_mean_abs_diff=%.6g,"
              "cublas_bf16_gemv_max_rel_diff=%.6g\n",
              gemv_diff.max_abs, gemv_diff.mean_abs, gemv_diff.max_rel);
  std::printf("custom_swizzle16_max_abs_diff=%.6g,"
              "custom_swizzle16_mean_abs_diff=%.6g,"
              "custom_swizzle16_max_rel_diff=%.6g\n",
              swizzled_diff.max_abs, swizzled_diff.mean_abs,
              swizzled_diff.max_rel);
  std::printf("cublas_bf16_gemm_m1_max_abs_diff=%.6g,"
              "cublas_bf16_gemm_m1_mean_abs_diff=%.6g,"
              "cublas_bf16_gemm_m1_max_rel_diff=%.6g\n",
              gemm_diff.max_abs, gemm_diff.mean_abs, gemm_diff.max_rel);
  std::printf("cudnn_bf16_conv1x1_max_abs_diff=%.6g,"
              "cudnn_bf16_conv1x1_mean_abs_diff=%.6g,"
              "cudnn_bf16_conv1x1_max_rel_diff=%.6g\n",
              cudnn_max_abs, cudnn_mean_abs, cudnn_max_rel);
  std::printf("benchmark_correctness op=%s reference=cublas_and_optional_cudnn "
              "custom_swizzle16_max_abs=%.6g cublas_gemv_max_abs=%.6g "
              "cublas_gemm_m1_max_abs=%.6g cudnn_conv1x1_max_abs=%.6g "
              "status=diff_reported\n",
              op.name, swizzled_diff.max_abs, gemv_diff.max_abs,
              gemm_diff.max_abs, cudnn_max_abs);
  std::printf("cudnn_bf16_conv1x1_algo=%d,cudnn_bf16_conv1x1_workspace_bytes=%zu\n\n",
              cudnn_algo, cudnn_workspace_bytes);
}

int main(int argc, char **argv) {
  const bool has_op =
      argc > 1 && !std::isdigit(static_cast<unsigned char>(argv[1][0]));
  const std::string selected = has_op ? argv[1] : "all";
  const int iter_arg = has_op ? 2 : 1;
  const int warmup_arg = has_op ? 3 : 2;
  const int trials_arg = has_op ? 4 : 3;
  const int iters = argc > iter_arg ? std::atoi(argv[iter_arg]) : 100;
  const int warmup = argc > warmup_arg ? std::atoi(argv[warmup_arg]) : 20;
  const int trials = argc > trials_arg ? std::atoi(argv[trials_arg]) : 5;
  if (iters <= 0 || warmup < 0 || trials <= 0) {
    std::fprintf(stderr,
                 "usage: %s [op=all] [iters=100] [warmup=20] [trials=5]\n",
                 argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  gemma4_bench_print_common_metadata("decode_bench");
  CublasDecode cublas(stream);
  const uint64_t seed = make_seed("GEMMA4_DECODE_BENCH_SEED");

  std::printf("benchmark_contract name=decode_bench measurement=single_token_projection_decode "
              "timing=cuda_events_same_stream cache=warm_repeated_buffers "
              "launch_overhead=excluded_from_gpu_elapsed_time aggregation=raw_trial_samples "
              "correctness=custom_vs_cublas_and_optional_cudnn_diff_reported "
              "warmup=%d iters=%d trials=%d dtype=bf16 base_seed=%llu\n",
              warmup, iters, trials, static_cast<unsigned long long>(seed));
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
