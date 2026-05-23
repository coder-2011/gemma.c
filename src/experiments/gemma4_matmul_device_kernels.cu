#include "gemma4_matmul_device_kernels.cuh"

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_matmul_device.cuh"

// Copied decode-GEMV launch surface with the core kernel body converted into a
// device function. The global wrapper below is deliberately thin so it can be
// compared directly against src/gemma4_matmul_kernels.cu.

namespace {

constexpr int kDefaultThreads = 512;
constexpr int kDefaultColsPerBlock = 8;
constexpr int kDefaultMinBlocksPerSm = 2;
constexpr int kInterleaveSwizzleBlocks = 16;
constexpr int kFfnDownThreads = 1024;
constexpr int kFfnDownColsPerBlock = 8;
constexpr int kFfnDownMinBlocksPerSm = 1;
constexpr int kGlobalOThreads = 512;
constexpr int kGlobalOColsPerBlock = 8;
constexpr int kGlobalOMinBlocksPerSm = 1;
constexpr int kFinalLogitsThreads = 1024;
constexpr int kFinalLogitsColsPerBlock = 8;
constexpr int kFinalLogitsMinBlocksPerSm = 1;

static_assert((kDefaultThreads % WARP_SIZE) == 0,
              "decode thread count must be a whole number of warps");

template <int K,
          int N,
          int ColsPerBlock,
          int Threads,
          int MinBlocksPerSM,
          int SwizzleTileBlocks>
__global__ __launch_bounds__(Threads, MinBlocksPerSM) void
gemma4_decode_gemv_cols_device_wrapper_kernel(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y) {
  constexpr int warps = Threads / WARP_SIZE;
  __shared__ float warp_sums[ColsPerBlock][warps];

  gemma4_matmul_device::decode_gemv_cols_device<
      K, N, ColsPerBlock, Threads, SwizzleTileBlocks>(
      x, w_col_major, y, blockIdx.x, warp_sums);
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
cudaError_t launch_decode_gemv_device_wrapped(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream) {
  const cudaError_t arg_status = check_decode_args(x, w_col_major, y);
  if (arg_status != cudaSuccess) {
    return arg_status;
  }

  constexpr int blocks = N / ColsPerBlock;
  gemma4_decode_gemv_cols_device_wrapper_kernel<
      K, N, ColsPerBlock, Threads, MinBlocksPerSM, SwizzleTileBlocks>
      <<<blocks, Threads, 0, stream>>>(x, w_col_major, y);
  return cudaGetLastError();
}

template <int SwizzleTileBlocks>
cudaError_t projection_decode_device_wrapped_impl(
    Gemma4Projection projection,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream) {
  switch (projection) {
  case GEMMA4_PROJECTION_FFN_GATE_UP:
    return launch_decode_gemv_device_wrapped<
        GEMMA4_HIDDEN_SIZE, GEMMA4_PACKED_FFN_SIZE,
        kDefaultColsPerBlock, kDefaultThreads,
        kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_FFN_DOWN:
    return launch_decode_gemv_device_wrapped<
        GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE,
        kFfnDownColsPerBlock, kFfnDownThreads,
        kFfnDownMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_SLIDING_QKV:
    return launch_decode_gemv_device_wrapped<
        GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_QKV_SIZE,
        kDefaultColsPerBlock, kDefaultThreads,
        kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_SLIDING_O:
    return launch_decode_gemv_device_wrapped<
        GEMMA4_SLIDING_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE,
        kDefaultColsPerBlock, kDefaultThreads,
        kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_GLOBAL_Q:
    return launch_decode_gemv_device_wrapped<
        GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_Q_PROJ_SIZE,
        kDefaultColsPerBlock, kDefaultThreads,
        kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_GLOBAL_K:
    return launch_decode_gemv_device_wrapped<
        GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_K_PROJ_SIZE,
        kDefaultColsPerBlock, kDefaultThreads,
        kDefaultMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_GLOBAL_O:
    return launch_decode_gemv_device_wrapped<
        GEMMA4_GLOBAL_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE,
        kGlobalOColsPerBlock, kGlobalOThreads,
        kGlobalOMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_FINAL_LOGITS:
    return launch_decode_gemv_device_wrapped<
        GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE,
        kFinalLogitsColsPerBlock, kFinalLogitsThreads,
        kFinalLogitsMinBlocksPerSm, SwizzleTileBlocks>(
        x, w_col_major, y, stream);
  }
  return cudaErrorInvalidValue;
}

}  // namespace

cudaError_t gemma4_projection_decode_device_wrapped(
    Gemma4Projection projection,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream) {
  return projection_decode_device_wrapped_impl<1>(
      projection, x, w_col_major, y, stream);
}

cudaError_t gemma4_projection_decode_device_wrapped_swizzled(
    Gemma4Projection projection,
    Gemma4DecodeSwizzle swizzle,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major,
    __nv_bfloat16 *__restrict__ y,
    cudaStream_t stream) {
  switch (swizzle) {
  case GEMMA4_DECODE_SWIZZLE_IDENTITY:
    return projection_decode_device_wrapped_impl<1>(
        projection, x, w_col_major, y, stream);
  case GEMMA4_DECODE_SWIZZLE_INTERLEAVE_16:
    return projection_decode_device_wrapped_impl<kInterleaveSwizzleBlocks>(
        projection, x, w_col_major, y, stream);
  }
  return cudaErrorInvalidValue;
}
