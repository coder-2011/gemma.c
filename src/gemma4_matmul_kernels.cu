#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Gemma 4 31B dense projection kernels.
//
// Prefill stays on a host-side library GEMM. CUDA library frontends are not
// callable from __device__ code, so the custom device path is decode-only.
static constexpr int kGemma4Hidden = 5376;
static constexpr int kGemma4FfnIntermediate = 21504;
static constexpr int kGemma4PackedFfn = 43008;
static constexpr int kGemma4SlidingQkv = 16384;
static constexpr int kGemma4SlidingAttentionOut = 8192;
static constexpr int kGemma4GlobalQ = 16384;
static constexpr int kGemma4GlobalK = 2048;
static constexpr int kGemma4GlobalAttentionOut = 16384;
static constexpr int kGemma4Vocab = 262144;
static constexpr int kGemma4DecodeColsPerBlock = 4;

#ifndef GEMMA4_DECODE_THREADS
#define GEMMA4_DECODE_THREADS 256
#endif

static constexpr int kGemma4DecodeThreads = GEMMA4_DECODE_THREADS;
static constexpr int kGemma4DecodeWarps = (kGemma4DecodeThreads + 31) / 32;
static_assert((kGemma4DecodeThreads % 32) == 0,
              "decode thread count must be a whole number of warps");

template <int K>
__device__ __forceinline__ void gemma4_decode_dot4_device(
    const half *x, const half *w_col_major, int col0, int thread_idx,
    int thread_count, float &sum0, float &sum1, float &sum2, float &sum3) {
  static_assert((K % 2) == 0, "decode GEMV K must be even for half2 loads");

  sum0 = 0.0f;
  sum1 = 0.0f;
  sum2 = 0.0f;
  sum3 = 0.0f;

  const half2 *x2 = reinterpret_cast<const half2 *>(x);
  const half2 *w2 = reinterpret_cast<const half2 *>(w_col_major);
  constexpr int k_half2 = K / 2;
  const int w_col0 = (col0 + 0) * k_half2;
  const int w_col1 = (col0 + 1) * k_half2;
  const int w_col2 = (col0 + 2) * k_half2;
  const int w_col3 = (col0 + 3) * k_half2;

#pragma unroll 4
  for (int k2 = thread_idx; k2 < k_half2; k2 += thread_count) {
    const float2 xv = __half22float2(x2[k2]);
    const float2 wv0 = __half22float2(w2[w_col0 + k2]);
    const float2 wv1 = __half22float2(w2[w_col1 + k2]);
    const float2 wv2 = __half22float2(w2[w_col2 + k2]);
    const float2 wv3 = __half22float2(w2[w_col3 + k2]);

    sum0 = fmaf(xv.x, wv0.x, sum0);
    sum0 = fmaf(xv.y, wv0.y, sum0);
    sum1 = fmaf(xv.x, wv1.x, sum1);
    sum1 = fmaf(xv.y, wv1.y, sum1);
    sum2 = fmaf(xv.x, wv2.x, sum2);
    sum2 = fmaf(xv.y, wv2.y, sum2);
    sum3 = fmaf(xv.x, wv3.x, sum3);
    sum3 = fmaf(xv.y, wv3.y, sum3);
  }
}

template <int K, int N>
__global__ __launch_bounds__(kGemma4DecodeThreads, 2) void
gemma4_decode_gemv4_kernel(const half *x, const half *w_col_major, half *y) {
  static_assert((N % kGemma4DecodeColsPerBlock) == 0,
                "decode GEMV N must be divisible by columns per block");

  __shared__ float warp_sums[kGemma4DecodeWarps][kGemma4DecodeColsPerBlock];

  const int col0 = blockIdx.x * kGemma4DecodeColsPerBlock;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warp_count = (blockDim.x + 31) / 32;

  float sum0;
  float sum1;
  float sum2;
  float sum3;
  gemma4_decode_dot4_device<K>(x, w_col_major, col0, threadIdx.x, blockDim.x,
                               sum0, sum1, sum2, sum3);

  for (int offset = 16; offset > 0; offset >>= 1) {
    sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
    sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
    sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
    sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
  }

  if (lane == 0) {
    warp_sums[warp][0] = sum0;
    warp_sums[warp][1] = sum1;
    warp_sums[warp][2] = sum2;
    warp_sums[warp][3] = sum3;
  }
  __syncthreads();

  sum0 = threadIdx.x < warp_count ? warp_sums[lane][0] : 0.0f;
  sum1 = threadIdx.x < warp_count ? warp_sums[lane][1] : 0.0f;
  sum2 = threadIdx.x < warp_count ? warp_sums[lane][2] : 0.0f;
  sum3 = threadIdx.x < warp_count ? warp_sums[lane][3] : 0.0f;

  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
      sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
      sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
      sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
    }
  }

  if (threadIdx.x == 0) {
    y[col0 + 0] = __float2half_rn(sum0);
    y[col0 + 1] = __float2half_rn(sum1);
    y[col0 + 2] = __float2half_rn(sum2);
    y[col0 + 3] = __float2half_rn(sum3);
  }
}

template <int K, int N>
cudaError_t gemma4_decode_gemv4(const half *x, const half *w_col_major, half *y,
                                cudaStream_t stream) {
  constexpr int blocks = N / kGemma4DecodeColsPerBlock;
  gemma4_decode_gemv4_kernel<K, N><<<blocks, kGemma4DecodeThreads, 0, stream>>>(
      x, w_col_major, y);
  return cudaGetLastError();
}

extern "C" cublasStatus_t gemma4_ffn_gate_up_prefill(
    cublasHandle_t handle, const half *x, const half *w_col_major, half *y,
    int m, cudaStream_t stream) {
  if (m <= 0) {
    return CUBLAS_STATUS_SUCCESS;
  }

  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status;
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  // cuBLAS is column-major. This computes the row-major result
  // Y[M, N] = X[M, K] * W[K, N] as:
  // Y^T[N, M] = W^T[N, K] * X^T[K, M].
  return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, kGemma4PackedFfn, m,
                      kGemma4Hidden, &alpha, w_col_major, CUDA_R_16F,
                      kGemma4Hidden, x, CUDA_R_16F, kGemma4Hidden, &beta, y,
                      CUDA_R_16F, kGemma4PackedFfn, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

extern "C" cudaError_t gemma4_ffn_gate_up_decode(const half *x,
                                                  const half *w_col_major,
                                                  half *y,
                                                  cudaStream_t stream) {
  return gemma4_decode_gemv4<kGemma4Hidden, kGemma4PackedFfn>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_ffn_down_decode(const half *x,
                                               const half *w_col_major, half *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv4<kGemma4FfnIntermediate, kGemma4Hidden>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_sliding_qkv_decode(const half *x,
                                                  const half *w_col_major,
                                                  half *y,
                                                  cudaStream_t stream) {
  return gemma4_decode_gemv4<kGemma4Hidden, kGemma4SlidingQkv>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_sliding_o_decode(const half *x,
                                                const half *w_col_major,
                                                half *y,
                                                cudaStream_t stream) {
  return gemma4_decode_gemv4<kGemma4SlidingAttentionOut, kGemma4Hidden>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_global_q_decode(const half *x,
                                               const half *w_col_major, half *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv4<kGemma4Hidden, kGemma4GlobalQ>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_global_k_decode(const half *x,
                                               const half *w_col_major, half *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv4<kGemma4Hidden, kGemma4GlobalK>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_global_o_decode(const half *x,
                                               const half *w_col_major, half *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv4<kGemma4GlobalAttentionOut, kGemma4Hidden>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_final_logits_decode(const half *x,
                                                   const half *w_col_major,
                                                   half *y,
                                                   cudaStream_t stream) {
  return gemma4_decode_gemv4<kGemma4Hidden, kGemma4Vocab>(
      x, w_col_major, y, stream);
}
