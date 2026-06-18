#include "gemma4_matmul_kernels.cuh"
#include "gemma4.h"

// Gemma 4 31B dense projection kernels.

#ifndef GEMMA4_WEIGHT_LOAD_POLICY
#define GEMMA4_WEIGHT_LOAD_POLICY 0
#endif

#include "gemma4_matmul_device.cuh"

namespace {

constexpr int kDefaultThreads = 512;
constexpr int kDefaultColsPerBlock = 4;
constexpr int kDefaultMinBlocksPerSm = 2;
constexpr int kInterleaveSwizzleBlocks = 16;
constexpr int kFfnDownThreads = 1024;
constexpr int kFfnDownColsPerBlock = 4;
constexpr int kFfnDownMinBlocksPerSm = 1;
constexpr int kGlobalOThreads = 512;
constexpr int kGlobalOColsPerBlock = 4;
constexpr int kGlobalOMinBlocksPerSm = 1;
constexpr int kFinalLogitsThreads = 1024;
constexpr int kFinalLogitsColsPerBlock = 4;
constexpr int kFinalLogitsMinBlocksPerSm = 1;

static_assert(kDefaultThreads > 0 && kDefaultThreads <= 1024 &&
                  (kDefaultThreads % WARP_SIZE) == 0,
              "decode thread count must be a valid warp-multiple block size");
static_assert(kFfnDownThreads > 0 && kFfnDownThreads <= 1024 &&
                  (kFfnDownThreads % WARP_SIZE) == 0,
              "FFN-down thread count must be a valid warp-multiple block size");
static_assert(kGlobalOThreads > 0 && kGlobalOThreads <= 1024 &&
                  (kGlobalOThreads % WARP_SIZE) == 0,
              "global-O thread count must be a valid warp-multiple block size");
static_assert(kFinalLogitsThreads > 0 && kFinalLogitsThreads <= 1024 &&
                  (kFinalLogitsThreads % WARP_SIZE) == 0,
              "final-logits thread count must be a valid warp-multiple block size");

struct Gemma4ProjectionShape {
  int k;
  int n;
};

template <int K,
          int N,
          int ColsPerBlock,
          int Threads,
          int MinBlocksPerSM,
          int SwizzleTileBlocks>
__global__ __launch_bounds__(Threads, MinBlocksPerSM) void
gemma4_decode_gemv_cols_kernel(const __nv_bfloat16 *__restrict__ x,
                               const __nv_bfloat16 *__restrict__ w_col_major,
                               __nv_bfloat16 *__restrict__ y) {
  static_assert((N % ColsPerBlock) == 0,
                "decode GEMV N must be divisible by columns per block");

  constexpr int warps = Threads / WARP_SIZE;
  __shared__ float warp_sums[ColsPerBlock][warps];

  float sums[ColsPerBlock] = {};
  gemma4_matmul_device::decode_gemv_cols_device<
      K, N, ColsPerBlock, Threads, SwizzleTileBlocks>(
      x, w_col_major, y, blockIdx.x, warp_sums, sums);
}

static cudaError_t check_decode_args(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    const __nv_bfloat16 *__restrict__ y) {
  if (!x || !w_col_major || !y || !is_aligned_16(x) ||
      !is_aligned_16(w_col_major) || !is_aligned_16(y)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

template <int K,
          int N,
          int ColsPerBlock,
          int Threads,
          int MinBlocksPerSM,
          int SwizzleTileBlocks>
cudaError_t launch_decode_gemv(const __nv_bfloat16 *__restrict__ x,
                               const __nv_bfloat16 *__restrict__ w_col_major,
                               __nv_bfloat16 *__restrict__ y,
                               cudaStream_t stream) {
  const cudaError_t arg_status = check_decode_args(x, w_col_major, y);
  if (arg_status != cudaSuccess) {
    return arg_status;
  }

  constexpr int blocks = N / ColsPerBlock;
  gemma4_decode_gemv_cols_kernel<K,
                                 N,
                                 ColsPerBlock,
                                 Threads,
                                 MinBlocksPerSM,
                                 SwizzleTileBlocks>
      <<<blocks, Threads, 0, stream>>>(x, w_col_major, y);
  return cudaGetLastError();
}

static cublasStatus_t prefill_gemm(
    cublasHandle_t handle,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int m, int k, int n,
    cudaStream_t stream) {
  if (m <= 0) {
    return CUBLAS_STATUS_SUCCESS;
  }

  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // Y^T[N, M] = W^T[N, K] * X^T[K, M].
  return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                      w_col_major, CUDA_R_16BF, k, x, CUDA_R_16BF, k, &beta,
                      y, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

static bool projection_shape(Gemma4Projection projection,
                             Gemma4ProjectionShape &shape) {
  switch (projection) {
  case GEMMA4_PROJECTION_FFN_GATE_UP:
    shape = {GEMMA4_HIDDEN_SIZE, GEMMA4_PACKED_FFN_SIZE};
    return true;
  case GEMMA4_PROJECTION_FFN_DOWN:
    shape = {GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE};
    return true;
  case GEMMA4_PROJECTION_SLIDING_QKV:
    shape = {GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_QKV_SIZE};
    return true;
  case GEMMA4_PROJECTION_SLIDING_O:
    shape = {GEMMA4_SLIDING_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE};
    return true;
  case GEMMA4_PROJECTION_GLOBAL_Q:
    shape = {GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_Q_PROJ_SIZE};
    return true;
  case GEMMA4_PROJECTION_GLOBAL_K:
    shape = {GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_K_PROJ_SIZE};
    return true;
  case GEMMA4_PROJECTION_GLOBAL_O:
    shape = {GEMMA4_GLOBAL_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE};
    return true;
  case GEMMA4_PROJECTION_FINAL_LOGITS:
    shape = {GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE};
    return true;
  }
  return false;
}

template <int SwizzleTileBlocks>
cudaError_t projection_decode_impl(
    Gemma4Projection projection,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream) {
  switch (projection) {
  case GEMMA4_PROJECTION_FFN_GATE_UP:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_PACKED_FFN_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_FFN_DOWN:
    return launch_decode_gemv<
        GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE,
        kFfnDownColsPerBlock, kFfnDownThreads,
        kFfnDownMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_SLIDING_QKV:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_QKV_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_SLIDING_O:
    return launch_decode_gemv<GEMMA4_SLIDING_ATTENTION_OUT_SIZE,
                              GEMMA4_HIDDEN_SIZE, kDefaultColsPerBlock,
                              kDefaultThreads, kDefaultMinBlocksPerSm,
                              SwizzleTileBlocks>(x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_GLOBAL_Q:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_Q_PROJ_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_GLOBAL_K:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_K_PROJ_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_GLOBAL_O:
    return launch_decode_gemv<
        GEMMA4_GLOBAL_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE,
        kGlobalOColsPerBlock, kGlobalOThreads,
        kGlobalOMinBlocksPerSm, SwizzleTileBlocks>(x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_FINAL_LOGITS:
    return launch_decode_gemv<
        GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE,
        kFinalLogitsColsPerBlock, kFinalLogitsThreads,
        kFinalLogitsMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace

cublasStatus_t gemma4_projection_prefill(
    Gemma4Projection projection, cublasHandle_t handle,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    int m, cudaStream_t stream) {
  Gemma4ProjectionShape shape = {};
  if (!projection_shape(projection, shape)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  return prefill_gemm(handle, x, w_col_major, y, m, shape.k, shape.n, stream);
}

cudaError_t gemma4_projection_decode(Gemma4Projection projection,
                                     const __nv_bfloat16 *__restrict__ x,
                                     const __nv_bfloat16 *__restrict__ w_col_major,
                                     __nv_bfloat16 *__restrict__ y,
                                     cudaStream_t stream) {
  return projection_decode_impl<1>(projection, x, w_col_major, y, stream);
}

cudaError_t gemma4_projection_decode_swizzled(
    Gemma4Projection projection,
    Gemma4DecodeSwizzle swizzle,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream) {
  switch (swizzle) {
  case GEMMA4_DECODE_SWIZZLE_IDENTITY:
    return projection_decode_impl<1>(projection, x, w_col_major, y, stream);
  case GEMMA4_DECODE_SWIZZLE_INTERLEAVE_16:
    return projection_decode_impl<kInterleaveSwizzleBlocks>(
        projection, x, w_col_major, y, stream);
  }
  return cudaErrorInvalidValue;
}
