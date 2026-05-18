#include "gemma4_matmul_kernels.cuh"
#include "gemma4.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using DecodeLaunch = cudaError_t (*)(const __nv_bfloat16 *,
                                     const __nv_bfloat16 *, __nv_bfloat16 *,
                                     cudaStream_t);

struct DecodeOp {
  const char *name;
  int k;
  int n;
  int layers_per_token;
  DecodeLaunch launch;
};

static const DecodeOp kDecodeOps[] = {
    {"ffn_gate_up", GEMMA4_HIDDEN_SIZE, GEMMA4_PACKED_FFN_SIZE,
     GEMMA4_NUM_LAYERS, gemma4_ffn_gate_up_decode},
    {"ffn_down", GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE,
     GEMMA4_NUM_LAYERS, gemma4_ffn_down_decode},
    {"sliding_qkv", GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_QKV_SIZE,
     GEMMA4_SLIDING_LAYER_COUNT, gemma4_sliding_qkv_decode},
    {"sliding_o", GEMMA4_SLIDING_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE,
     GEMMA4_SLIDING_LAYER_COUNT, gemma4_sliding_o_decode},
    {"global_q", GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_Q_PROJ_SIZE,
     GEMMA4_GLOBAL_LAYER_COUNT, gemma4_global_q_decode},
    {"global_k", GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_K_PROJ_SIZE,
     GEMMA4_GLOBAL_LAYER_COUNT, gemma4_global_k_decode},
    {"global_o", GEMMA4_GLOBAL_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE,
     GEMMA4_GLOBAL_LAYER_COUNT, gemma4_global_o_decode},
    {"final_logits", GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE, 1,
     gemma4_final_logits_decode},
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

__device__ uint32_t mix_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__global__ void fill_random_bf16_kernel(__nv_bfloat16 *ptr, size_t count,
                                        uint64_t seed, float scale) {
  size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }

  uint32_t x = uint32_t(i) ^ uint32_t(i >> 32) ^ uint32_t(seed) ^
               uint32_t(seed >> 32);
  x = mix_u32(x);
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
  fill_random_bf16_kernel<<<blocks, threads, 0, stream>>>(ptr, count, seed,
                                                          scale);
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

struct TimingStats {
  float best_ms = 0.0f;
  float avg_ms = 0.0f;
};

struct DiffStats {
  float max_abs = 0.0f;
  float mean_abs = 0.0f;
  float max_rel = 0.0f;
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

static DiffStats diff_stats(const __nv_bfloat16 *lhs, const __nv_bfloat16 *rhs,
                            int count) {
  std::vector<__nv_bfloat16> h_lhs(count);
  std::vector<__nv_bfloat16> h_rhs(count);
  CUDA_CHECK(cudaMemcpy(h_lhs.data(), lhs, count * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_rhs.data(), rhs, count * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToHost));

  DiffStats stats;
  double sum_abs = 0.0;
  for (int i = 0; i < count; ++i) {
    const float a = __bfloat162float(h_lhs[i]);
    const float b = __bfloat162float(h_rhs[i]);
    const float abs_diff = std::abs(a - b);
    const float denom = std::max(std::max(std::abs(a), std::abs(b)), 1.0f);
    stats.max_abs = std::max(stats.max_abs, abs_diff);
    stats.max_rel = std::max(stats.max_rel, abs_diff / denom);
    sum_abs += abs_diff;
  }
  stats.mean_abs = count > 0 ? float(sum_abs / double(count)) : 0.0f;
  return stats;
}

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
  const size_t x_count = size_t(op.k);
  const size_t w_count = size_t(op.n) * size_t(op.k);
  const size_t y_count = size_t(op.n);

  CUDA_CHECK(cudaMalloc(&x, x_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&w, w_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&custom_y, y_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&gemv_y, y_count * sizeof(__nv_bfloat16)));
  CUDA_CHECK(cudaMalloc(&gemm_y, y_count * sizeof(__nv_bfloat16)));

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
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto run_custom = [&]() { CUDA_CHECK(op.launch(x, w, custom_y, stream)); };
  auto run_gemv = [&]() { cublas.gemv(x, w, gemv_y, op.k, op.n); };
  auto run_gemm = [&]() { cublas.gemm_m1(x, w, gemm_y, op.k, op.n); };

  run_custom();
  run_gemv();
  run_gemm();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const DiffStats gemv_diff = diff_stats(custom_y, gemv_y, op.n);
  const DiffStats gemm_diff = diff_stats(custom_y, gemm_y, op.n);

  const TimingStats custom = time_ms(run_custom, stream, warmup, iters, trials);
  const TimingStats gemv = time_ms(run_gemv, stream, warmup, iters, trials);
  const TimingStats gemm = time_ms(run_gemm, stream, warmup, iters, trials);
  const double bytes = double(op.n) * double(op.k) * sizeof(__nv_bfloat16);
  const double per_token_gb = bytes * double(op.layers_per_token) / 1.0e9;

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
  std::printf("custom_vs_cublas_gemv_speedup=%.6f\n",
              gemv.best_ms / custom.best_ms);
  std::printf("custom_vs_cublas_gemm_m1_speedup=%.6f\n",
              gemm.best_ms / custom.best_ms);
  std::printf("cublas_bf16_gemv_max_abs_diff=%.6g,cublas_bf16_gemv_mean_abs_diff=%.6g,cublas_bf16_gemv_max_rel_diff=%.6g\n",
              gemv_diff.max_abs, gemv_diff.mean_abs, gemv_diff.max_rel);
  std::printf("cublas_bf16_gemm_m1_max_abs_diff=%.6g,cublas_bf16_gemm_m1_mean_abs_diff=%.6g,cublas_bf16_gemm_m1_max_rel_diff=%.6g\n\n",
              gemm_diff.max_abs, gemm_diff.mean_abs, gemm_diff.max_rel);

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
