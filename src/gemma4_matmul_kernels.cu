#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Gemma 4 31B dense projection kernels.
//
// Prefill stays on a host-side library GEMM. CUDA library frontends are not
// callable from __device__ code, so the custom device path is decode-only.
static constexpr int kGemma4DecodeThreads = 256;
static constexpr int kGemma4DecodeColsPerBlock = 4;
static constexpr int kGemma4DecodeMinBlocksPerSM = 2;
static_assert((kGemma4DecodeThreads % WARP_SIZE) == 0,
              "decode thread count must be a whole number of warps");

using Gemma4Bf16Pack = Packed128<__nv_bfloat16>;
static constexpr int kGemma4Bf16PerPack = Gemma4Bf16Pack::size;
static constexpr int kGemma4Bf16PairsPerPack = kGemma4Bf16PerPack / 2;

union Gemma4Bf16x2Bits {
  __nv_bfloat162 value;
  uint32_t bits;
};

__device__ __forceinline__ __nv_bfloat162
gemma4_bf16_pack_pair(const Gemma4Bf16Pack &pack, int pair) {
  const __nv_bfloat162 *pairs =
      reinterpret_cast<const __nv_bfloat162 *>(pack.payload);
  return pairs[pair];
}

__device__ __forceinline__ void gemma4_accumulate_bf16_pack(
    const Gemma4Bf16Pack &x_pack, const Gemma4Bf16Pack &w_pack, float &sum) {
#pragma unroll
  for (int pair = 0; pair < kGemma4Bf16PairsPerPack; ++pair) {
    const float2 xv = __bfloat1622float2(gemma4_bf16_pack_pair(x_pack, pair));
    const float2 wv = __bfloat1622float2(gemma4_bf16_pack_pair(w_pack, pair));
    sum = fmaf(xv.x, wv.x, sum);
    sum = fmaf(xv.y, wv.y, sum);
  }
}

__device__ __forceinline__ void gemma4_store_bf16x4(
    __nv_bfloat16 *__restrict__ dst, float sum0, float sum1, float sum2,
    float sum3) {
  Gemma4Bf16x2Bits out01;
  Gemma4Bf16x2Bits out23;
  out01.value = __floats2bfloat162_rn(sum0, sum1);
  out23.value = __floats2bfloat162_rn(sum2, sum3);
  *reinterpret_cast<uint2 *>(dst) = make_uint2(out01.bits, out23.bits);
}

template <int K, int Threads>
__device__ __forceinline__ void gemma4_decode_gemv_fixed4_dot_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major, int col0, int thread_idx,
    float &sum0, float &sum1, float &sum2, float &sum3) {
  static_assert((K % kGemma4Bf16PerPack) == 0,
                "decode GEMV K must be divisible by Packed128 bf16 width");
  static_assert((Threads % WARP_SIZE) == 0,
                "decode thread count must be a whole number of warps");

  sum0 = 0.0f;
  sum1 = 0.0f;
  sum2 = 0.0f;
  sum3 = 0.0f;

  constexpr int packs_per_col = K / kGemma4Bf16PerPack;
  const __nv_bfloat16 *w_col0 = w_col_major + col0 * K;
  const __nv_bfloat16 *w_col1 = w_col0 + K;
  const __nv_bfloat16 *w_col2 = w_col1 + K;
  const __nv_bfloat16 *w_col3 = w_col2 + K;

#pragma unroll
  for (int pack_idx = thread_idx; pack_idx < packs_per_col;
       pack_idx += Threads) {
    const int element_idx = pack_idx * kGemma4Bf16PerPack;
    const Gemma4Bf16Pack x_pack = load128(x + element_idx);
    const Gemma4Bf16Pack w_pack0 = load128cs(w_col0 + element_idx);
    const Gemma4Bf16Pack w_pack1 = load128cs(w_col1 + element_idx);
    const Gemma4Bf16Pack w_pack2 = load128cs(w_col2 + element_idx);
    const Gemma4Bf16Pack w_pack3 = load128cs(w_col3 + element_idx);

    gemma4_accumulate_bf16_pack(x_pack, w_pack0, sum0);
    gemma4_accumulate_bf16_pack(x_pack, w_pack1, sum1);
    gemma4_accumulate_bf16_pack(x_pack, w_pack2, sum2);
    gemma4_accumulate_bf16_pack(x_pack, w_pack3, sum3);
  }
}

template <int K, int N, int Threads, int MinBlocksPerSM>
__global__ __launch_bounds__(Threads, MinBlocksPerSM) void
gemma4_decode_gemv_fixed4_kernel(const __nv_bfloat16 *__restrict__ x,
                                 const __nv_bfloat16 *__restrict__ w_col_major,
                                 __nv_bfloat16 *__restrict__ y) {
  static_assert((N % 4) == 0,
                "fixed decode GEMV N must be divisible by four columns");
  static_assert((Threads % WARP_SIZE) == 0,
                "fixed decode GEMV thread count must be whole warps");

  constexpr int warps = div_up(Threads, WARP_SIZE);
  __shared__ float warp_sums[warps][4];

  const int col0 = blockIdx.x * 4;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;
  const int warp_count = div_up(blockDim.x, WARP_SIZE);

  float sum0;
  float sum1;
  float sum2;
  float sum3;
  gemma4_decode_gemv_fixed4_dot_device<K, Threads>(
      x, w_col_major, col0, threadIdx.x, sum0, sum1, sum2, sum3);

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
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
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      sum0 += __shfl_down_sync(0xffffffffu, sum0, offset);
      sum1 += __shfl_down_sync(0xffffffffu, sum1, offset);
      sum2 += __shfl_down_sync(0xffffffffu, sum2, offset);
      sum3 += __shfl_down_sync(0xffffffffu, sum3, offset);
    }
  }

  if (threadIdx.x == 0) {
    gemma4_store_bf16x4(y + col0, sum0, sum1, sum2, sum3);
  }
}

template <int K, int ColsPerBlock, int Threads>
__device__ __forceinline__ void gemma4_decode_gemv_cols_dot_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major, int col0, int thread_idx,
    float (&sums)[ColsPerBlock]) {
  static_assert((K % kGemma4Bf16PerPack) == 0,
                "decode GEMV K must be divisible by Packed128 bf16 width");
  static_assert((Threads % WARP_SIZE) == 0,
                "decode thread count must be a whole number of warps");

#pragma unroll
  for (int col = 0; col < ColsPerBlock; ++col) {
    sums[col] = 0.0f;
  }

  constexpr int packs_per_col = K / kGemma4Bf16PerPack;

#pragma unroll
  for (int pack_idx = thread_idx; pack_idx < packs_per_col;
       pack_idx += Threads) {
    const int element_idx = pack_idx * kGemma4Bf16PerPack;
    const Gemma4Bf16Pack x_pack = load128(x + element_idx);
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const Gemma4Bf16Pack w_pack =
          load128cs(w_col_major + (col0 + col) * K + element_idx);
      gemma4_accumulate_bf16_pack(x_pack, w_pack, sums[col]);
    }
  }
}

template <int K, int N, int ColsPerBlock>
__global__ __launch_bounds__(kGemma4DecodeThreads,
                             kGemma4DecodeMinBlocksPerSM) void
gemma4_decode_gemv_cols_kernel(const __nv_bfloat16 *__restrict__ x,
                               const __nv_bfloat16 *__restrict__ w_col_major,
                               __nv_bfloat16 *__restrict__ y) {
  static_assert((N % ColsPerBlock) == 0,
                "decode GEMV N must be divisible by columns per block");

  constexpr int warps =
      div_up(kGemma4DecodeThreads, WARP_SIZE);
  __shared__ float warp_sums[warps][ColsPerBlock];

  const int col0 = blockIdx.x * ColsPerBlock;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;
  const int warp_count = div_up(blockDim.x, WARP_SIZE);

  float sums[ColsPerBlock];
  gemma4_decode_gemv_cols_dot_device<K, ColsPerBlock, kGemma4DecodeThreads>(
      x, w_col_major, col0, threadIdx.x, sums);

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
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
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
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
cudaError_t gemma4_decode_gemv(const __nv_bfloat16 *x,
                               const __nv_bfloat16 *w_col_major,
                               __nv_bfloat16 *y, cudaStream_t stream) {
  constexpr int blocks = N / kGemma4DecodeColsPerBlock;
  if constexpr (kGemma4DecodeColsPerBlock == 4) {
    gemma4_decode_gemv_fixed4_kernel<
        K, N, kGemma4DecodeThreads, kGemma4DecodeMinBlocksPerSM>
        <<<blocks, kGemma4DecodeThreads, 0, stream>>>(x, w_col_major, y);
  } else {
    gemma4_decode_gemv_cols_kernel<K, N, kGemma4DecodeColsPerBlock>
        <<<blocks, kGemma4DecodeThreads, 0, stream>>>(x, w_col_major, y);
  }
  return cudaGetLastError();
}

template <int K, int N, int Threads, int MinBlocksPerSM>
cudaError_t gemma4_decode_gemv_fixed4(const __nv_bfloat16 *x,
                                      const __nv_bfloat16 *w_col_major,
                                      __nv_bfloat16 *y,
                                      cudaStream_t stream) {
  constexpr int blocks = N / 4;
  gemma4_decode_gemv_fixed4_kernel<K, N, Threads, MinBlocksPerSM>
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
  return gemma4_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_PACKED_FFN_SIZE>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_ffn_down_decode(const __nv_bfloat16 *x,
                                               const __nv_bfloat16 *w_col_major,
                                               __nv_bfloat16 *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv_fixed4<GEMMA4_INTERMEDIATE_SIZE,
                                   GEMMA4_HIDDEN_SIZE, 256, 4>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_sliding_qkv_decode(const __nv_bfloat16 *x,
                                                  const __nv_bfloat16 *w_col_major,
                                                  __nv_bfloat16 *y,
                                                  cudaStream_t stream) {
  return gemma4_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_QKV_SIZE>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_sliding_o_decode(const __nv_bfloat16 *x,
                                                const __nv_bfloat16 *w_col_major,
                                                __nv_bfloat16 *y,
                                                cudaStream_t stream) {
  return gemma4_decode_gemv<GEMMA4_SLIDING_ATTENTION_OUT_SIZE,
                            GEMMA4_HIDDEN_SIZE>(x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_global_q_decode(const __nv_bfloat16 *x,
                                               const __nv_bfloat16 *w_col_major,
                                               __nv_bfloat16 *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_Q_PROJ_SIZE>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_global_k_decode(const __nv_bfloat16 *x,
                                               const __nv_bfloat16 *w_col_major,
                                               __nv_bfloat16 *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_K_PROJ_SIZE>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_global_o_decode(const __nv_bfloat16 *x,
                                               const __nv_bfloat16 *w_col_major,
                                               __nv_bfloat16 *y,
                                               cudaStream_t stream) {
  return gemma4_decode_gemv_fixed4<GEMMA4_GLOBAL_ATTENTION_OUT_SIZE,
                                   GEMMA4_HIDDEN_SIZE, 512, 1>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_final_logits_decode(const __nv_bfloat16 *x,
                                                   const __nv_bfloat16 *w_col_major,
                                                   __nv_bfloat16 *y,
                                                   cudaStream_t stream) {
  return gemma4_decode_gemv_fixed4<GEMMA4_HIDDEN_SIZE,
                                   GEMMA4_VOCAB_SIZE, 512, 2>(
      x, w_col_major, y, stream);
}
