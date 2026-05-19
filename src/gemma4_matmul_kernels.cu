#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>

// Gemma 4 31B dense projection kernels.

#ifndef GEMMA4_DECODE_THREADS
#define GEMMA4_DECODE_THREADS 512
#endif
#ifndef GEMMA4_DECODE_COLS_PER_BLOCK
#define GEMMA4_DECODE_COLS_PER_BLOCK 8
#endif
#ifndef GEMMA4_DECODE_MIN_BLOCKS_PER_SM
#define GEMMA4_DECODE_MIN_BLOCKS_PER_SM 2
#endif
#ifndef GEMMA4_DECODE_FFN_DOWN_THREADS
#define GEMMA4_DECODE_FFN_DOWN_THREADS 1024
#endif
#ifndef GEMMA4_DECODE_FFN_DOWN_COLS_PER_BLOCK
#define GEMMA4_DECODE_FFN_DOWN_COLS_PER_BLOCK 8
#endif
#ifndef GEMMA4_DECODE_FFN_DOWN_MIN_BLOCKS_PER_SM
#define GEMMA4_DECODE_FFN_DOWN_MIN_BLOCKS_PER_SM 1
#endif
#ifndef GEMMA4_DECODE_GLOBAL_O_THREADS
#define GEMMA4_DECODE_GLOBAL_O_THREADS 512
#endif
#ifndef GEMMA4_DECODE_GLOBAL_O_COLS_PER_BLOCK
#define GEMMA4_DECODE_GLOBAL_O_COLS_PER_BLOCK 8
#endif
#ifndef GEMMA4_DECODE_GLOBAL_O_MIN_BLOCKS_PER_SM
#define GEMMA4_DECODE_GLOBAL_O_MIN_BLOCKS_PER_SM 1
#endif
#ifndef GEMMA4_DECODE_FINAL_LOGITS_THREADS
#define GEMMA4_DECODE_FINAL_LOGITS_THREADS 1024
#endif
#ifndef GEMMA4_DECODE_FINAL_LOGITS_COLS_PER_BLOCK
#define GEMMA4_DECODE_FINAL_LOGITS_COLS_PER_BLOCK 8
#endif
#ifndef GEMMA4_DECODE_FINAL_LOGITS_MIN_BLOCKS_PER_SM
#define GEMMA4_DECODE_FINAL_LOGITS_MIN_BLOCKS_PER_SM 1
#endif

static constexpr int kGemma4DecodeThreads = GEMMA4_DECODE_THREADS;
static constexpr int kGemma4DecodeColsPerBlock =
    GEMMA4_DECODE_COLS_PER_BLOCK;
static constexpr int kGemma4DecodeMinBlocksPerSM =
    GEMMA4_DECODE_MIN_BLOCKS_PER_SM;
static_assert((kGemma4DecodeThreads % WARP_SIZE) == 0, "decode thread count must be a whole number of warps");

using Gemma4Bf16Pack = Packed128<__nv_bfloat16>;

static constexpr int kGemma4Bf16PerPack = Gemma4Bf16Pack::size;
static constexpr int kGemma4Bf16PairsPerPack = kGemma4Bf16PerPack / 2;
static_assert(sizeof(Gemma4Bf16Pack) == sizeof(int4) && alignof(Gemma4Bf16Pack) >= alignof(int4), "Packed128 bf16 must map to one aligned int4 load");
static_assert((kGemma4Bf16PerPack % 2) == 0, "Packed128 bf16 width must contain whole bf16 pairs");
static_assert(alignof(Gemma4Bf16Pack) >= alignof(__nv_bfloat162), "Packed128 bf16 payload must be aligned for bf16x2 access");

__device__ __forceinline__ int gemma4_bf16_pack_element_index(int pack_idx) {
  return pack_idx * kGemma4Bf16PerPack;
}

template <int K>
__device__ __forceinline__ int gemma4_weight_pack_offset(int col,
                                                        int element_idx) {
  return col * K + element_idx;
}

__device__ __forceinline__ Gemma4Bf16Pack
gemma4_load_activation_pack(const __nv_bfloat16 *__restrict__ x,
                            int element_idx) {
  return load128(x + element_idx);
}

template <int K>
__device__ __forceinline__ Gemma4Bf16Pack
gemma4_load_streaming_weight_pack(
    const __nv_bfloat16 *__restrict__ w_col_major, int col, int element_idx) {
  return load128cs(w_col_major + gemma4_weight_pack_offset<K>(col, element_idx));
}

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

#if defined(GEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER) ||                          \
    defined(GEMMA4_DECODE_SHARED_STAGE_LOAD128) ||                             \
    defined(GEMMA4_DECODE_SHARED_STAGE_LOAD128CS)
#define GEMMA4_DECODE_SHARED_STAGE_WEIGHT_PACKS 1
#endif

#if defined(GEMMA4_DECODE_SHARED_STAGE_WEIGHT_PACKS)
template <int K>
__device__ __forceinline__ void
gemma4_stage_weight_pack(Gemma4Bf16Pack *dst_shared,
                         const __nv_bfloat16 *__restrict__ w_col_major,
                         int col, int element_idx) {
  static_assert(sizeof(Gemma4Bf16Pack) == 16,
                "shared-stage path requires one 16-byte weight pack");
#if defined(GEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER)
  __pipeline_memcpy_async(
      dst_shared, w_col_major + gemma4_weight_pack_offset<K>(col, element_idx),
      sizeof(Gemma4Bf16Pack));
#elif defined(GEMMA4_DECODE_SHARED_STAGE_LOAD128CS)
  *dst_shared =
      gemma4_load_streaming_weight_pack<K>(w_col_major, col, element_idx);
#else
  *dst_shared =
      load128(w_col_major + gemma4_weight_pack_offset<K>(col, element_idx));
#endif
}

__device__ __forceinline__ void gemma4_commit_staged_weight_pack() {
#if defined(GEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER)
  __pipeline_commit();
#endif
}

__device__ __forceinline__ void gemma4_wait_for_staged_weight_pack() {
#if defined(GEMMA4_DECODE_CP_ASYNC_DOUBLE_BUFFER)
  __pipeline_wait_prior(0);
#endif
}

template <int K, int ColsPerBlock, int Threads>
__device__ __forceinline__ void
gemma4_decode_gemv_cols_dot_shared_staged_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major, int col0, int thread_idx,
    Gemma4Bf16Pack (&weight_stages)[2][Threads],
    float (&sums)[ColsPerBlock]) {
  static_assert((K % kGemma4Bf16PerPack) == 0,
                "decode GEMV K must be divisible by Packed128 bf16 width");
  static_assert((Threads % WARP_SIZE) == 0,
                "decode thread count must be a whole number of warps");
  static_assert(ColsPerBlock > 0, "decode GEMV must compute at least one column");

  constexpr int packs_per_col = K / kGemma4Bf16PerPack;

#pragma unroll
  for (int pack_idx = thread_idx; pack_idx < packs_per_col;
       pack_idx += Threads) {
    const int element_idx = gemma4_bf16_pack_element_index(pack_idx);
    const Gemma4Bf16Pack x_pack = gemma4_load_activation_pack(x, element_idx);
    int stage = 0;

    gemma4_stage_weight_pack<K>(&weight_stages[stage][thread_idx], w_col_major,
                                col0, element_idx);
    gemma4_commit_staged_weight_pack();
    gemma4_wait_for_staged_weight_pack();

#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const int next_col = col + 1;
      const int next_stage = stage ^ 1;
      if constexpr (ColsPerBlock > 1) {
        if (next_col < ColsPerBlock) {
          gemma4_stage_weight_pack<K>(&weight_stages[next_stage][thread_idx],
                                      w_col_major, col0 + next_col,
                                      element_idx);
          gemma4_commit_staged_weight_pack();
        }
      }

      const Gemma4Bf16Pack w_pack = weight_stages[stage][thread_idx];
      gemma4_accumulate_bf16_pack(x_pack, w_pack, sums[col]);

      if constexpr (ColsPerBlock > 1) {
        if (next_col < ColsPerBlock) {
          gemma4_wait_for_staged_weight_pack();
          stage = next_stage;
        }
      }
    }
  }
}
#endif

template <int K, int ColsPerBlock, int Threads>
__device__ __forceinline__ void gemma4_decode_gemv_cols_dot_device(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major, int col0, int thread_idx,
    float (&sums)[ColsPerBlock]) {
  static_assert((K % kGemma4Bf16PerPack) == 0, "decode GEMV K must be divisible by Packed128 bf16 width");
  static_assert((Threads % WARP_SIZE) == 0, "decode thread count must be a whole number of warps");

  constexpr int packs_per_col = K / kGemma4Bf16PerPack;

#pragma unroll
  for (int pack_idx = thread_idx; pack_idx < packs_per_col;
       pack_idx += Threads) {
    const int element_idx = gemma4_bf16_pack_element_index(pack_idx);
    const Gemma4Bf16Pack x_pack = gemma4_load_activation_pack(x, element_idx);
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const Gemma4Bf16Pack w_pack = gemma4_load_streaming_weight_pack<K>(
          w_col_major, col0 + col, element_idx);
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

  float sums[ColsPerBlock] = {};
#if defined(GEMMA4_DECODE_SHARED_STAGE_WEIGHT_PACKS)
  __shared__ Gemma4Bf16Pack weight_stages[2][Threads];
  gemma4_decode_gemv_cols_dot_shared_staged_device<K, ColsPerBlock, Threads>(
      x, w_col_major, col0, threadIdx.x, weight_stages, sums);
#else
  gemma4_decode_gemv_cols_dot_device<K, ColsPerBlock, Threads>(
      x, w_col_major, col0, threadIdx.x, sums);
#endif

  warp_reduce_sum_to_lane0(sums);

  if (lane == 0) {
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      warp_sums[col][warp] = sums[col];
    }
  }
  __syncthreads();

#pragma unroll
  for (int col = 0; col < ColsPerBlock; ++col) {
    sums[col] = threadIdx.x < warps ? warp_sums[col][lane] : 0.0f;
  }

  if (warp == 0) {
    warp_reduce_sum_to_lane0(sums);
  }

  if (threadIdx.x == 0) {
    gemma4_store_bf16_cols<ColsPerBlock>(y + col0, sums);
  }
}

static cudaError_t gemma4_check_decode_gemv_args(
    const __nv_bfloat16 *x, const __nv_bfloat16 *w_col_major,
    const __nv_bfloat16 *y) {
  if (!x || !w_col_major || !y || !is_aligned_16(x) ||
      !is_aligned_16(w_col_major) || !is_aligned_16(y)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

template <int K, int N>
cudaError_t gemma4_decode_gemv(const __nv_bfloat16 *x,
                               const __nv_bfloat16 *w_col_major,
                               __nv_bfloat16 *y, cudaStream_t stream) {
  const cudaError_t arg_status =
      gemma4_check_decode_gemv_args(x, w_col_major, y);
  if (arg_status != cudaSuccess) {
    return arg_status;
  }

  constexpr int blocks = N / kGemma4DecodeColsPerBlock;
  gemma4_decode_gemv_cols_kernel<K, N, kGemma4DecodeColsPerBlock,
                                 kGemma4DecodeThreads,
                                 kGemma4DecodeMinBlocksPerSM>
      <<<blocks, kGemma4DecodeThreads, 0, stream>>>(x, w_col_major, y);
  return cudaGetLastError();
}

template <int K, int N, int ColsPerBlock, int Threads, int MinBlocksPerSM>
cudaError_t gemma4_decode_gemv_fixed_cols(const __nv_bfloat16 *x,
                                          const __nv_bfloat16 *w_col_major,
                                          __nv_bfloat16 *y,
                                          cudaStream_t stream) {
  const cudaError_t arg_status =
      gemma4_check_decode_gemv_args(x, w_col_major, y);
  if (arg_status != cudaSuccess) {
    return arg_status;
  }

  constexpr int blocks = N / ColsPerBlock;
  gemma4_decode_gemv_cols_kernel<K, N, ColsPerBlock, Threads, MinBlocksPerSM>
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

template <int K, int N>
static cublasStatus_t gemma4_prefill_projection(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_bf16_prefill_gemm(handle, x, w_col_major, y, m, K, N, stream);
}

template <int K, int N>
static cudaError_t gemma4_decode_projection(
    const __nv_bfloat16 *x, const __nv_bfloat16 *w_col_major,
    __nv_bfloat16 *y, cudaStream_t stream) {
  return gemma4_decode_gemv<K, N>(x, w_col_major, y, stream);
}

template <int K, int N, int ColsPerBlock, int Threads, int MinBlocksPerSM>
static cudaError_t gemma4_decode_projection_fixed_cols(
    const __nv_bfloat16 *x, const __nv_bfloat16 *w_col_major,
    __nv_bfloat16 *y, cudaStream_t stream) {
  return gemma4_decode_gemv_fixed_cols<K, N, ColsPerBlock, Threads,
                                       MinBlocksPerSM>(
      x, w_col_major, y, stream);
}

extern "C" {

cublasStatus_t gemma4_ffn_gate_up_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_prefill_projection<GEMMA4_HIDDEN_SIZE,
                                   GEMMA4_PACKED_FFN_SIZE>(
      handle, x, w_col_major, y, m, stream);
}

cublasStatus_t gemma4_ffn_down_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_prefill_projection<GEMMA4_INTERMEDIATE_SIZE,
                                   GEMMA4_HIDDEN_SIZE>(
      handle, x, w_col_major, y, m, stream);
}

cublasStatus_t gemma4_sliding_qkv_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_prefill_projection<GEMMA4_HIDDEN_SIZE,
                                   GEMMA4_SLIDING_QKV_SIZE>(
      handle, x, w_col_major, y, m, stream);
}

cublasStatus_t gemma4_sliding_o_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_prefill_projection<GEMMA4_SLIDING_ATTENTION_OUT_SIZE,
                                   GEMMA4_HIDDEN_SIZE>(
      handle, x, w_col_major, y, m, stream);
}

cublasStatus_t gemma4_global_q_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_prefill_projection<GEMMA4_HIDDEN_SIZE,
                                   GEMMA4_GLOBAL_Q_PROJ_SIZE>(
      handle, x, w_col_major, y, m, stream);
}

cublasStatus_t gemma4_global_k_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_prefill_projection<GEMMA4_HIDDEN_SIZE,
                                   GEMMA4_GLOBAL_K_PROJ_SIZE>(
      handle, x, w_col_major, y, m, stream);
}

cublasStatus_t gemma4_global_o_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_prefill_projection<GEMMA4_GLOBAL_ATTENTION_OUT_SIZE,
                                   GEMMA4_HIDDEN_SIZE>(
      handle, x, w_col_major, y, m, stream);
}

cublasStatus_t gemma4_final_logits_prefill(
    cublasHandle_t handle, const __nv_bfloat16 *x,
    const __nv_bfloat16 *w_col_major, __nv_bfloat16 *y, int m,
    cudaStream_t stream) {
  return gemma4_prefill_projection<GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE>(
      handle, x, w_col_major, y, m, stream);
}

cudaError_t gemma4_ffn_gate_up_decode(const __nv_bfloat16 *x,
                                      const __nv_bfloat16 *w_col_major,
                                      __nv_bfloat16 *y,
                                      cudaStream_t stream) {
  return gemma4_decode_projection<GEMMA4_HIDDEN_SIZE,
                                  GEMMA4_PACKED_FFN_SIZE>(
      x, w_col_major, y, stream);
}

cudaError_t gemma4_ffn_down_decode(const __nv_bfloat16 *x,
                                   const __nv_bfloat16 *w_col_major,
                                   __nv_bfloat16 *y, cudaStream_t stream) {
  return gemma4_decode_projection_fixed_cols<
      GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE,
      GEMMA4_DECODE_FFN_DOWN_COLS_PER_BLOCK, GEMMA4_DECODE_FFN_DOWN_THREADS,
      GEMMA4_DECODE_FFN_DOWN_MIN_BLOCKS_PER_SM>(
      x, w_col_major, y, stream);
}

cudaError_t gemma4_sliding_qkv_decode(const __nv_bfloat16 *x,
                                      const __nv_bfloat16 *w_col_major,
                                      __nv_bfloat16 *y,
                                      cudaStream_t stream) {
  return gemma4_decode_projection<GEMMA4_HIDDEN_SIZE,
                                  GEMMA4_SLIDING_QKV_SIZE>(
      x, w_col_major, y, stream);
}

cudaError_t gemma4_sliding_o_decode(const __nv_bfloat16 *x,
                                    const __nv_bfloat16 *w_col_major,
                                    __nv_bfloat16 *y, cudaStream_t stream) {
  return gemma4_decode_projection<GEMMA4_SLIDING_ATTENTION_OUT_SIZE,
                                  GEMMA4_HIDDEN_SIZE>(
      x, w_col_major, y, stream);
}

cudaError_t gemma4_global_q_decode(const __nv_bfloat16 *x,
                                   const __nv_bfloat16 *w_col_major,
                                   __nv_bfloat16 *y, cudaStream_t stream) {
  return gemma4_decode_projection<GEMMA4_HIDDEN_SIZE,
                                  GEMMA4_GLOBAL_Q_PROJ_SIZE>(
      x, w_col_major, y, stream);
}

cudaError_t gemma4_global_k_decode(const __nv_bfloat16 *x,
                                   const __nv_bfloat16 *w_col_major,
                                   __nv_bfloat16 *y, cudaStream_t stream) {
  return gemma4_decode_projection<GEMMA4_HIDDEN_SIZE,
                                  GEMMA4_GLOBAL_K_PROJ_SIZE>(
      x, w_col_major, y, stream);
}

cudaError_t gemma4_global_o_decode(const __nv_bfloat16 *x,
                                   const __nv_bfloat16 *w_col_major,
                                   __nv_bfloat16 *y, cudaStream_t stream) {
  return gemma4_decode_projection_fixed_cols<
      GEMMA4_GLOBAL_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE,
      GEMMA4_DECODE_GLOBAL_O_COLS_PER_BLOCK, GEMMA4_DECODE_GLOBAL_O_THREADS,
      GEMMA4_DECODE_GLOBAL_O_MIN_BLOCKS_PER_SM>(
      x, w_col_major, y, stream);
}

cudaError_t gemma4_final_logits_decode(const __nv_bfloat16 *x,
                                       const __nv_bfloat16 *w_col_major,
                                       __nv_bfloat16 *y,
                                       cudaStream_t stream) {
  return gemma4_decode_projection_fixed_cols<
      GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE,
      GEMMA4_DECODE_FINAL_LOGITS_COLS_PER_BLOCK,
      GEMMA4_DECODE_FINAL_LOGITS_THREADS,
      GEMMA4_DECODE_FINAL_LOGITS_MIN_BLOCKS_PER_SM>(
      x, w_col_major, y, stream);
}

}
