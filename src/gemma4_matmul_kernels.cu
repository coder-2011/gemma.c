#include "gemma4_matmul_kernels.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

// Gemma 4 31B dense projection kernels.

namespace {

constexpr int kDefaultThreads = 512;
constexpr int kDefaultColsPerBlock = 8;
constexpr int kDefaultMinBlocksPerSm = 2;
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

struct Gemma4ProjectionShape {
  int k;
  int n;
};

__device__ __forceinline__ int pack_offset(int pack_idx) {
  return pack_idx * kBf16Packed128Elements;
}

template <int K>
__device__ __forceinline__ int weight_offset(int col, int element_idx) {
  return col * K + element_idx;
}

__device__ __forceinline__ Bf16Packed128
load_activation_pack(const __nv_bfloat16 *__restrict__ x, int element_idx) {
  return load128(x + element_idx);
}

template <int K>
__device__ __forceinline__ Bf16Packed128
load_weight_pack(
    const __nv_bfloat16 *__restrict__ w_col_major, int col, int element_idx) {
  return load128cs(w_col_major + weight_offset<K>(col, element_idx));
}

template <int ColsPerBlock>
__device__ __forceinline__ void store_cols(
    __nv_bfloat16 *__restrict__ dst, const float (&sums)[ColsPerBlock]) {
  if constexpr (ColsPerBlock == kBf16Packed128Elements) {
    Bf16Packed128 out;
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
__device__ __forceinline__ void dot_cols(
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_col_major, int col0, int thread_idx,
    float (&sums)[ColsPerBlock]) {
  static_assert((K % kBf16Packed128Elements) == 0,
                "decode GEMV K must be divisible by Packed128 bf16 width");
  static_assert((Threads % WARP_SIZE) == 0,
                "decode thread count must be a whole number of warps");

  constexpr int packs_per_col = K / kBf16Packed128Elements;

#pragma unroll
  for (int pack_idx = thread_idx; pack_idx < packs_per_col; pack_idx += Threads) {
    const int element_idx = pack_offset(pack_idx);
    const Bf16Packed128 x_pack = load_activation_pack(x, element_idx);
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const Bf16Packed128 w_pack = load_weight_pack<K>(w_col_major, col0 + col, element_idx);
      gemma4_bf16_pack_accumulate_dot(x_pack, w_pack, sums[col]);
    }
  }
}

template <int K, int N, int ColsPerBlock, int Threads, int MinBlocksPerSM>
__global__ __launch_bounds__(Threads, MinBlocksPerSM) void
gemma4_decode_gemv_cols_kernel(const __nv_bfloat16 *__restrict__ x,
                               const __nv_bfloat16 *__restrict__ w_col_major,
                               __nv_bfloat16 *__restrict__ y) {
  static_assert((N % ColsPerBlock) == 0,
                "decode GEMV N must be divisible by columns per block");

  static_assert((Threads % WARP_SIZE) == 0,
                "decode thread count must be a whole number of warps");
  constexpr int warps = Threads / WARP_SIZE;
  __shared__ float warp_sums[ColsPerBlock][warps];

  const int col0 = blockIdx.x * ColsPerBlock;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;

  float sums[ColsPerBlock] = {};
  dot_cols<K, ColsPerBlock, Threads>(x, w_col_major, col0, threadIdx.x, sums);

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
    store_cols<ColsPerBlock>(y + col0, sums);
  }
}

static cudaError_t check_decode_args(
    const __nv_bfloat16 *x, const __nv_bfloat16 *w_col_major,
    const __nv_bfloat16 *y) {
  if (!x || !w_col_major || !y || !is_aligned_16(x) || !is_aligned_16(w_col_major) || !is_aligned_16(y)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

template <int K, int N, int ColsPerBlock, int Threads, int MinBlocksPerSM>
cudaError_t launch_decode_gemv(const __nv_bfloat16 *x,
                               const __nv_bfloat16 *w_col_major,
                               __nv_bfloat16 *y,
                               cudaStream_t stream) {
  const cudaError_t arg_status = check_decode_args(x, w_col_major, y);
  if (arg_status != cudaSuccess) {
    return arg_status;
  }

  constexpr int blocks = N / ColsPerBlock;
  gemma4_decode_gemv_cols_kernel<K, N, ColsPerBlock, Threads, MinBlocksPerSM><<<blocks, Threads, 0, stream>>>(x, w_col_major, y);
  return cudaGetLastError();
}

static cublasStatus_t prefill_gemm(
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

}  // namespace

cublasStatus_t gemma4_projection_prefill(
    Gemma4Projection projection, cublasHandle_t handle,
    const __nv_bfloat16 *x, const __nv_bfloat16 *w_col_major,
    __nv_bfloat16 *y, int m, cudaStream_t stream) {
  Gemma4ProjectionShape shape = {};
  if (!projection_shape(projection, shape)) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  return prefill_gemm(handle, x, w_col_major, y, m, shape.k, shape.n, stream);
}

cudaError_t gemma4_projection_decode(Gemma4Projection projection,
                                     const __nv_bfloat16 *x,
                                     const __nv_bfloat16 *w_col_major,
                                     __nv_bfloat16 *y, cudaStream_t stream) {
  switch (projection) {
  case GEMMA4_PROJECTION_FFN_GATE_UP:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_PACKED_FFN_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm>(x, w_col_major, y,
                                                       stream);
  case GEMMA4_PROJECTION_FFN_DOWN:
    return launch_decode_gemv<
        GEMMA4_INTERMEDIATE_SIZE, GEMMA4_HIDDEN_SIZE,
        kFfnDownColsPerBlock, kFfnDownThreads,
        kFfnDownMinBlocksPerSm>(x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_SLIDING_QKV:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_SLIDING_QKV_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm>(x, w_col_major, y,
                                                       stream);
  case GEMMA4_PROJECTION_SLIDING_O:
    return launch_decode_gemv<GEMMA4_SLIDING_ATTENTION_OUT_SIZE,
                              GEMMA4_HIDDEN_SIZE, kDefaultColsPerBlock,
                              kDefaultThreads, kDefaultMinBlocksPerSm>(
        x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_GLOBAL_Q:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_Q_PROJ_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm>(x, w_col_major, y,
                                                       stream);
  case GEMMA4_PROJECTION_GLOBAL_K:
    return launch_decode_gemv<GEMMA4_HIDDEN_SIZE, GEMMA4_GLOBAL_K_PROJ_SIZE,
                              kDefaultColsPerBlock, kDefaultThreads,
                              kDefaultMinBlocksPerSm>(x, w_col_major, y,
                                                       stream);
  case GEMMA4_PROJECTION_GLOBAL_O:
    return launch_decode_gemv<
        GEMMA4_GLOBAL_ATTENTION_OUT_SIZE, GEMMA4_HIDDEN_SIZE,
        kGlobalOColsPerBlock, kGlobalOThreads,
        kGlobalOMinBlocksPerSm>(x, w_col_major, y, stream);
  case GEMMA4_PROJECTION_FINAL_LOGITS:
    return launch_decode_gemv<
        GEMMA4_HIDDEN_SIZE, GEMMA4_VOCAB_SIZE,
        kFinalLogitsColsPerBlock, kFinalLogitsThreads,
        kFinalLogitsMinBlocksPerSm>(x, w_col_major, y, stream);
  }
  return cudaErrorInvalidValue;
}
