#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// Gemma 4 31B dense projection kernels.

static constexpr int kGemma4DecodeThreads = 512;
static constexpr int kGemma4DecodeColsPerBlock = 8;
static constexpr int kGemma4DecodeMinBlocksPerSM = 2;
static_assert((kGemma4DecodeThreads % WARP_SIZE) == 0, "decode thread count must be a whole number of warps");

using Gemma4Bf16Pack = Packed128<__nv_bfloat16>;

static constexpr int kGemma4Bf16PerPack = Gemma4Bf16Pack::size;
static constexpr int kGemma4Bf16PairsPerPack = kGemma4Bf16PerPack / 2;
static_assert((kGemma4Bf16PerPack % 2) == 0, "Packed128 bf16 width must contain whole bf16 pairs");
static_assert(alignof(Gemma4Bf16Pack) >= alignof(__nv_bfloat162), "Packed128 bf16 payload must be aligned for bf16x2 access");

template <int Pair = 0>
__device__ __forceinline__ void gemma4_accumulate_bf16_pairs(
    const __nv_bfloat162 *x_pairs, const __nv_bfloat162 *w_pairs, float &sum) {
  static_assert(Pair >= 0 && Pair <= kGemma4Bf16PairsPerPack, "bf16 pair index is out of range");
  if constexpr (Pair < kGemma4Bf16PairsPerPack) {
    const float2 xv = __bfloat1622float2(x_pairs[Pair]);
    const float2 wv = __bfloat1622float2(w_pairs[Pair]);
    sum = fmaf(xv.x, wv.x, sum);
    sum = fmaf(xv.y, wv.y, sum);
    gemma4_accumulate_bf16_pairs<Pair + 1>(x_pairs, w_pairs, sum);
  }
}

__device__ __forceinline__ void gemma4_accumulate_bf16_pack(
    const Gemma4Bf16Pack &x_pack, const Gemma4Bf16Pack &w_pack, float &sum) {
  const auto *x_pairs = reinterpret_cast<const __nv_bfloat162 *>(x_pack.payload);
  const auto *w_pairs = reinterpret_cast<const __nv_bfloat162 *>(w_pack.payload);
  gemma4_accumulate_bf16_pairs<>(x_pairs, w_pairs, sum);
}

template <int ColsPerBlock>
__device__ __forceinline__ void gemma4_store_bf16_cols(
    __nv_bfloat16 *__restrict__ dst, const float (&sums)[ColsPerBlock]) {
  if constexpr (ColsPerBlock == kGemma4Bf16PerPack) {
    Gemma4Bf16Pack out;
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      out[col] = __float2bfloat16_rn(sums[col]);
    }
    store128(dst, out);
  } else {
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      dst[col] = __float2bfloat16_rn(sums[col]);
    }
  }
}

template <int K, int ColsPerBlock, int Threads>
__device__ __forceinline__ void gemma4_decode_gemv_cols_dot_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major, int col0, int thread_idx,
    float (&sums)[ColsPerBlock]) {
  static_assert((K % kGemma4Bf16PerPack) == 0, "decode GEMV K must be divisible by Packed128 bf16 width");
  static_assert((Threads % WARP_SIZE) == 0, "decode thread count must be a whole number of warps");

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

template <int K, int N, int ColsPerBlock, int Threads, int MinBlocksPerSM>
__global__ __launch_bounds__(Threads, MinBlocksPerSM) void
gemma4_decode_gemv_cols_kernel(const __nv_bfloat16 *__restrict__ x,
                               const __nv_bfloat16 *__restrict__ w_col_major,
                               __nv_bfloat16 *__restrict__ y) {
  static_assert((N % ColsPerBlock) == 0, "decode GEMV N must be divisible by columns per block");

  constexpr int warps = div_up(Threads, WARP_SIZE);
  __shared__ float warp_sums[ColsPerBlock][warps];

  const int col0 = blockIdx.x * ColsPerBlock;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;
  const int warp_count = div_up(blockDim.x, WARP_SIZE);

  float sums[ColsPerBlock];
  gemma4_decode_gemv_cols_dot_device<K, ColsPerBlock, Threads>(
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
      warp_sums[col][warp] = sums[col];
    }
  }
  __syncthreads();

#pragma unroll
  for (int col = 0; col < ColsPerBlock; ++col) {
    sums[col] = threadIdx.x < warp_count ? warp_sums[col][lane] : 0.0f;
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
    gemma4_store_bf16_cols<ColsPerBlock>(y + col0, sums);
  }
}

template <int K, int N>
cudaError_t gemma4_decode_gemv(const __nv_bfloat16 *x,
                               const __nv_bfloat16 *w_col_major,
                               __nv_bfloat16 *y, cudaStream_t stream) {
  constexpr int blocks = N / kGemma4DecodeColsPerBlock;
  gemma4_decode_gemv_cols_kernel<K, N, kGemma4DecodeColsPerBlock,
                                 kGemma4DecodeThreads,
                                 kGemma4DecodeMinBlocksPerSM>
      <<<blocks, kGemma4DecodeThreads, 0, stream>>>(x, w_col_major, y);
  return cudaGetLastError();
}

template <int K, int N, int Threads, int MinBlocksPerSM>
cudaError_t gemma4_decode_gemv_fixed8(const __nv_bfloat16 *x,
                                      const __nv_bfloat16 *w_col_major,
                                      __nv_bfloat16 *y,
                                      cudaStream_t stream) {
  constexpr int blocks = N / 8;
  gemma4_decode_gemv_cols_kernel<K, N, 8, Threads, MinBlocksPerSM>
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
  return gemma4_decode_gemv_fixed8<GEMMA4_INTERMEDIATE_SIZE,
                                   GEMMA4_HIDDEN_SIZE, 1024, 1>(
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
  return gemma4_decode_gemv_fixed8<GEMMA4_GLOBAL_ATTENTION_OUT_SIZE,
                                   GEMMA4_HIDDEN_SIZE, 512, 1>(
      x, w_col_major, y, stream);
}

extern "C" cudaError_t gemma4_final_logits_decode(const __nv_bfloat16 *x,
                                                   const __nv_bfloat16 *w_col_major,
                                                   __nv_bfloat16 *y,
                                                   cudaStream_t stream) {
  return gemma4_decode_gemv_fixed8<GEMMA4_HIDDEN_SIZE,
                                   GEMMA4_VOCAB_SIZE, 1024, 1>(
      x, w_col_major, y, stream);
}
