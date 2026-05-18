#include "gemma4.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Gemma 4 31B dense projection kernels.
//
// Prefill stays on a host-side library GEMM. CUDA library frontends are not
// callable from __device__ code, so the custom device path is decode-only.
#ifndef GEMMA4_DECODE_THREADS
#define GEMMA4_DECODE_THREADS 256
#endif

#ifndef GEMMA4_DECODE_COLS_PER_BLOCK
#define GEMMA4_DECODE_COLS_PER_BLOCK 4
#endif

#ifndef GEMMA4_DECODE_MIN_BLOCKS_PER_SM
#define GEMMA4_DECODE_MIN_BLOCKS_PER_SM 2
#endif

static constexpr int kGemma4DecodeThreads = GEMMA4_DECODE_THREADS;
static constexpr int kGemma4DecodeColsPerBlock =
    GEMMA4_DECODE_COLS_PER_BLOCK;
static constexpr int kGemma4DecodeMinBlocksPerSm =
    GEMMA4_DECODE_MIN_BLOCKS_PER_SM;
static_assert((kGemma4DecodeThreads % 32) == 0,
              "decode thread count must be a whole number of warps");

template <int K>
__device__ __forceinline__ void gemma4_decode_dot4_device(
    const __nv_bfloat16 *x, const __nv_bfloat16 *w_col_major, int col0,
    int thread_idx, int thread_count, float &sum0, float &sum1, float &sum2,
    float &sum3) {
  static_assert((K % 2) == 0,
                "decode GEMV K must be even for __nv_bfloat162 loads");

  sum0 = 0.0f;
  sum1 = 0.0f;
  sum2 = 0.0f;
  sum3 = 0.0f;

  const __nv_bfloat162 *x2 = reinterpret_cast<const __nv_bfloat162 *>(x);
  const __nv_bfloat162 *w2 =
      reinterpret_cast<const __nv_bfloat162 *>(w_col_major);
  constexpr int k_half2 = K / 2;
  const int w_col0 = (col0 + 0) * k_half2;
  const int w_col1 = (col0 + 1) * k_half2;
  const int w_col2 = (col0 + 2) * k_half2;
  const int w_col3 = (col0 + 3) * k_half2;

#pragma unroll 4
  for (int k2 = thread_idx; k2 < k_half2; k2 += thread_count) {
    const float2 xv = __bfloat1622float2(x2[k2]);
    const float2 wv0 = __bfloat1622float2(w2[w_col0 + k2]);
    const float2 wv1 = __bfloat1622float2(w2[w_col1 + k2]);
    const float2 wv2 = __bfloat1622float2(w2[w_col2 + k2]);
    const float2 wv3 = __bfloat1622float2(w2[w_col3 + k2]);

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

template <int K, int N, int Threads, int MinBlocksPerSm>
__global__ __launch_bounds__(Threads, MinBlocksPerSm) void
gemma4_decode_gemv4_kernel_fixed4_tuned(const __nv_bfloat16 *x,
                                        const __nv_bfloat16 *w_col_major,
                                        __nv_bfloat16 *y) {
  static_assert((N % 4) == 0,
                "fixed decode GEMV N must be divisible by four columns");
  static_assert((Threads % 32) == 0,
                "fixed decode GEMV thread count must be whole warps");

  constexpr int warps = (Threads + 31) / 32;
  __shared__ float warp_sums[warps][4];

  const int col0 = blockIdx.x * 4;
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
    y[col0 + 0] = __float2bfloat16_rn(sum0);
    y[col0 + 1] = __float2bfloat16_rn(sum1);
    y[col0 + 2] = __float2bfloat16_rn(sum2);
    y[col0 + 3] = __float2bfloat16_rn(sum3);
  }
}

template <int K, int ColsPerBlock>
__device__ __forceinline__ void gemma4_decode_dot_device(
    const __nv_bfloat16 *x, const __nv_bfloat16 *w_col_major, int col0,
    int thread_idx, int thread_count, float (&sums)[ColsPerBlock]) {
  static_assert((K % 2) == 0,
                "decode GEMV K must be even for __nv_bfloat162 loads");

#pragma unroll
  for (int col = 0; col < ColsPerBlock; ++col) {
    sums[col] = 0.0f;
  }

  const __nv_bfloat162 *x2 = reinterpret_cast<const __nv_bfloat162 *>(x);
  const __nv_bfloat162 *w2 =
      reinterpret_cast<const __nv_bfloat162 *>(w_col_major);
  constexpr int k_half2 = K / 2;

#pragma unroll 4
  for (int k2 = thread_idx; k2 < k_half2; k2 += thread_count) {
    const float2 xv = __bfloat1622float2(x2[k2]);
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const float2 wv = __bfloat1622float2(w2[(col0 + col) * k_half2 + k2]);
      sums[col] = fmaf(xv.x, wv.x, sums[col]);
      sums[col] = fmaf(xv.y, wv.y, sums[col]);
    }
  }
}

template <int K, int N, int ColsPerBlock>
__global__ __launch_bounds__(kGemma4DecodeThreads,
                             kGemma4DecodeMinBlocksPerSm) void
gemma4_decode_gemv4_kernel(const __nv_bfloat16 *x,
                           const __nv_bfloat16 *w_col_major,
                           __nv_bfloat16 *y) {
  static_assert((N % ColsPerBlock) == 0,
                "decode GEMV N must be divisible by columns per block");

  constexpr int warps = (kGemma4DecodeThreads + 31) / 32;
  __shared__ float warp_sums[warps][ColsPerBlock];

  const int col0 = blockIdx.x * ColsPerBlock;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warp_count = (blockDim.x + 31) / 32;

  float sums[ColsPerBlock];
  gemma4_decode_dot_device<K, ColsPerBlock>(
      x, w_col_major, col0, threadIdx.x, blockDim.x, sums);

  for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      sums[col] += __shfl_down_sync(0xffffffffu, sums[col], offset);
    }
  }

  if (lane == 0) {
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      warp_sums[warp][col] = sums[col];
    }
  }
  __syncthreads();

#pragma unroll
  for (int col = 0; col < ColsPerBlock; ++col) {
    sums[col] = threadIdx.x < warp_count ? warp_sums[lane][col] : 0.0f;
  }

  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int col = 0; col < ColsPerBlock; ++col) {
        sums[col] += __shfl_down_sync(0xffffffffu, sums[col], offset);
      }
    }
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      y[col0 + col] = __float2bfloat16_rn(sums[col]);
    }
  }
}

template <int K, int N>
cudaError_t gemma4_decode_gemv4(const __nv_bfloat16 *x,
                                const __nv_bfloat16 *w_col_major,
                                __nv_bfloat16 *y, cudaStream_t stream) {
  constexpr int blocks = N / kGemma4DecodeColsPerBlock;
  if constexpr (kGemma4DecodeColsPerBlock == 4) {
    gemma4_decode_gemv4_kernel_fixed4_tuned<
        K, N, kGemma4DecodeThreads, kGemma4DecodeMinBlocksPerSm>
        <<<blocks, kGemma4DecodeThreads, 0, stream>>>(x, w_col_major, y);
  } else {
    gemma4_decode_gemv4_kernel<K, N, kGemma4DecodeColsPerBlock>
        <<<blocks, kGemma4DecodeThreads, 0, stream>>>(x, w_col_major, y);
  }
  return cudaGetLastError();
}

template <int K, int N, int Threads, int MinBlocksPerSm>
cudaError_t gemma4_decode_gemv4_fixed4_tuned(const __nv_bfloat16 *x,
                                             const __nv_bfloat16 *w_col_major,
                                             __nv_bfloat16 *y,
                                             cudaStream_t stream) {
  constexpr int blocks = N / 4;
  gemma4_decode_gemv4_kernel_fixed4_tuned<K, N, Threads, MinBlocksPerSm>
      <<<blocks, Threads, 0, stream>>>(x, w_col_major, y);
  return cudaGetLastError();
}

static cublasStatus_t gemma4_bf16_prefill_gemm(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m, int k, int n,
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

  // cuBLAS is column-major. This computes the row-major result
  // Y[M, N] = X[M, K] * W[K, N] as:
  // Y^T[N, M] = W^T[N, K] * X^T[K, M].
  return cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                      w_col_major, CUDA_R_16BF, k, x, CUDA_R_16BF, k, &beta,
                      y, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

extern "C" cublasStatus_t gemma4_ffn_gate_up_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_bf16_prefill_gemm(handle, x, w_col_major, y, m,
                                  GEMMA4_HIDDEN_SIZE, GEMMA4_PACKED_FFN_SIZE,
                                  stream);
}

extern "C" cublasStatus_t gemma4_ffn_down_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_bf16_prefill_gemm(handle, x, w_col_major, y, m,
                                  GEMMA4_INTERMEDIATE_SIZE,
                                  GEMMA4_HIDDEN_SIZE, stream);
}

extern "C" cublasStatus_t gemma4_sliding_qkv_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_bf16_prefill_gemm(handle, x, w_col_major, y, m,
                                  GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_QKV_SIZE,
                                  stream);
}

extern "C" cublasStatus_t gemma4_sliding_o_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_bf16_prefill_gemm(handle, x, w_col_major, y, m,
                                  GEMMA4_SLIDING_ATTENTION_OUT_SIZE,
                                  GEMMA4_HIDDEN_SIZE, stream);
}

extern "C" cublasStatus_t gemma4_global_q_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_bf16_prefill_gemm(handle, x, w_col_major, y, m,
                                  GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_Q_PROJ_SIZE,
                                  stream);
}

extern "C" cublasStatus_t gemma4_global_k_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_bf16_prefill_gemm(handle, x, w_col_major, y, m,
                                  GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_K_PROJ_SIZE,
                                  stream);
}

extern "C" cublasStatus_t gemma4_global_o_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_bf16_prefill_gemm(handle, x, w_col_major, y, m,
                                  GEMMA4_GLOBAL_ATTENTION_OUT_SIZE,
                                  GEMMA4_HIDDEN_SIZE, stream);
}

extern "C" cublasStatus_t gemma4_final_logits_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_bf16_prefill_gemm(handle, x, w_col_major, y, m,
                                  GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE,
                                  stream);
}

extern "C" cudaError_t gemma4_ffn_gate_up_decode(const __nv_bfloat16 *x,
                                                  const __nv_bfloat16 *w_col_major,
                                                  __nv_bfloat16 *y,
                                                  cudaStream_t stream) {
  return gemma4_decode_gemv4<GEMMA4_HIDDEN_SIZE, GEMMA4_PACKED_FFN_SIZE>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_ffn_down_decode(const __nv_bfloat16 *x,
                                               const __nv_bfloat16 *w_col_major,
                                               __nv_bfloat16 *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv4_fixed4_tuned<GEMMA4_INTERMEDIATE_SIZE,
                                          GEMMA4_HIDDEN_SIZE, 256, 4>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_sliding_qkv_decode(const __nv_bfloat16 *x,
                                                  const __nv_bfloat16 *w_col_major,
                                                  __nv_bfloat16 *y,
                                                  cudaStream_t stream) {
  return gemma4_decode_gemv4<GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_QKV_SIZE>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_sliding_o_decode(const __nv_bfloat16 *x,
                                                const __nv_bfloat16 *w_col_major,
                                                __nv_bfloat16 *y,
                                                cudaStream_t stream) {
  return gemma4_decode_gemv4<GEMMA4_SLIDING_ATTENTION_OUT_SIZE,
                             GEMMA4_HIDDEN_SIZE>(x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_global_q_decode(const __nv_bfloat16 *x,
                                               const __nv_bfloat16 *w_col_major,
                                               __nv_bfloat16 *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv4<GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_Q_PROJ_SIZE>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_global_k_decode(const __nv_bfloat16 *x,
                                               const __nv_bfloat16 *w_col_major,
                                               __nv_bfloat16 *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv4<GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_K_PROJ_SIZE>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_global_o_decode(const __nv_bfloat16 *x,
                                               const __nv_bfloat16 *w_col_major,
                                               __nv_bfloat16 *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv4_fixed4_tuned<GEMMA4_GLOBAL_ATTENTION_OUT_SIZE,
                                          GEMMA4_HIDDEN_SIZE, 512, 1>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_final_logits_decode(const __nv_bfloat16 *x,
                                                   const __nv_bfloat16 *w_col_major,
                                                   __nv_bfloat16 *y,
                                                   cudaStream_t stream) {
  return gemma4_decode_gemv4_fixed4_tuned<GEMMA4_HIDDEN_SIZE,
                                          GEMMA4_VOCAB_SIZE, 512, 2>(
      x, w_col_major, y, stream);
}
