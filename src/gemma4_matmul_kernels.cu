#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Gemma 4 31B dense FFN packed gate+up projection:
//   X[M, 5376] * W[5376, 43008] -> Y[M, 43008]
//
// Prefill stays on a host-side library GEMM. CUDA library frontends are not
// callable from __device__ code, so the custom device path is decode-only.
static constexpr int kGemma4Hidden = 5376;
static constexpr int kGemma4HiddenHalf2 = kGemma4Hidden / 2;
static constexpr int kGemma4PackedFfn = 43008;
static constexpr int kGemma4DecodeThreads = 256;
static constexpr int kGemma4DecodeColsPerBlock = 4;

__device__ __forceinline__ void gemma4_decode_dot4_device(
    const half *x, const half *w_col_major, int col0, float &sum0,
    float &sum1, float &sum2, float &sum3) {
  __shared__ float warp_sums[8][kGemma4DecodeColsPerBlock];

  sum0 = 0.0f;
  sum1 = 0.0f;
  sum2 = 0.0f;
  sum3 = 0.0f;

  const half2 *x2 = reinterpret_cast<const half2 *>(x);
  const half2 *w2 = reinterpret_cast<const half2 *>(w_col_major);
  const int w_col0 = (col0 + 0) * kGemma4HiddenHalf2;
  const int w_col1 = (col0 + 1) * kGemma4HiddenHalf2;
  const int w_col2 = (col0 + 2) * kGemma4HiddenHalf2;
  const int w_col3 = (col0 + 3) * kGemma4HiddenHalf2;

#pragma unroll 4
  for (int k2 = threadIdx.x; k2 < kGemma4HiddenHalf2; k2 += blockDim.x) {
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

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

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

  const int warp_count = (blockDim.x + 31) / 32;
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
}

__global__ void gemma4_ffn_gate_up_decode_kernel(const half *x,
                                                 const half *w_col_major,
                                                 half *y) {
  const int col0 = blockIdx.x * kGemma4DecodeColsPerBlock;

  float sum0;
  float sum1;
  float sum2;
  float sum3;
  gemma4_decode_dot4_device(x, w_col_major, col0, sum0, sum1, sum2, sum3);

  if (threadIdx.x == 0) {
    y[col0 + 0] = __float2half_rn(sum0);
    y[col0 + 1] = __float2half_rn(sum1);
    y[col0 + 2] = __float2half_rn(sum2);
    y[col0 + 3] = __float2half_rn(sum3);
  }
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
  constexpr int blocks =
      kGemma4PackedFfn / kGemma4DecodeColsPerBlock;
  gemma4_ffn_gate_up_decode_kernel<<<blocks, kGemma4DecodeThreads, 0, stream>>>(
      x, w_col_major, y);
  return cudaGetLastError();
}
