/*
 * Adapted from llm.c/llmc/layernorm.cuh.
 *
 * The structure intentionally mirrors llm.c's forward LayerNorm and fused
 * residual+LayerNorm kernels, but the math is Gemma RMSNorm:
 *
 *   y = x * rsqrt(mean(x * x) + eps) * weight
 *
 * Gemma uses RMSNorm without a bias term and without mean subtraction.
 */

#include "gemma4_rmsnorm.cuh"
#include "gemma4_cuda_utils.cuh"

#include <math.h>
#include <stdint.h>

namespace {

using RmsnormPack = Packed128<floatX>;
constexpr int kFloatXPerPack = RmsnormPack::size;

__global__ void gemma4_rmsnorm_bf16_warp_kernel(
    floatX *out,
    float *rstd,
    const floatX *inp,
    const floatX *weight,
    int rows,
    int width,
    float eps) {
  int lane = threadIdx.x % WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int warps_per_block = blockDim.x / WARP_SIZE;
  int row = blockIdx.x * warps_per_block + warp;
  if (row >= rows) {
    return;
  }

  const floatX *x = inp + static_cast<int64_t>(row) * width;
  float sum_sq = 0.0f;
  for (int c = lane; c < width; c += WARP_SIZE) {
    float value = static_cast<float>(x[c]);
    sum_sq += value * value;
  }
  sum_sq = warp_reduce_sum(sum_sq);
  float scale = rsqrtf(sum_sq / static_cast<float>(width) + eps);
  if (lane == 0 && rstd != nullptr) {
    rstd[row] = scale;
  }

  floatX *y = out + static_cast<int64_t>(row) * width;
  for (int c = lane; c < width; c += WARP_SIZE) {
    float value = static_cast<float>(x[c]);
    float gamma = static_cast<float>(weight[c]);
    y[c] = static_cast<floatX>(value * scale * gamma);
  }
}

__global__ void gemma4_rmsnorm_bf16_shared_kernel(
    floatX *out,
    float *rstd,
    const floatX *inp,
    const floatX *weight,
    int rows,
    int width,
    int packs_per_row,
    float eps) {
  extern __shared__ unsigned char shared_bytes[];
  auto *shared = reinterpret_cast<RmsnormPack *>(shared_bytes);
  RmsnormPack *s_weight = shared;
  RmsnormPack *s_in = s_weight + packs_per_row +
                      static_cast<int64_t>(threadIdx.y) * packs_per_row;

  int packed_thread = threadIdx.x + WARP_SIZE * threadIdx.y;
  int packed_stride = WARP_SIZE * blockDim.y;
  for (int pack = packed_thread; pack < packs_per_row; pack += packed_stride) {
    s_weight[pack] =
        load128(weight + static_cast<int64_t>(pack) * kFloatXPerPack);
  }
  __syncthreads();

  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) {
    return;
  }

  const floatX *x = inp + static_cast<int64_t>(row) * width;
  float sum_sq = 0.0f;
  for (int pack = threadIdx.x; pack < packs_per_row;
       pack += WARP_SIZE) {
    RmsnormPack values =
        load128cs(x + static_cast<int64_t>(pack) * kFloatXPerPack);
    s_in[pack] = values;
    for (int k = 0; k < kFloatXPerPack; ++k) {
      float value = static_cast<float>(values[k]);
      sum_sq += value * value;
    }
  }

  sum_sq = warp_reduce_sum(sum_sq);
  float scale = rsqrtf(sum_sq / static_cast<float>(width) + eps);
  if (threadIdx.x == 0 && rstd != nullptr) {
    rstd[row] = scale;
  }

  floatX *y = out + static_cast<int64_t>(row) * width;
  for (int pack = threadIdx.x; pack < packs_per_row;
       pack += WARP_SIZE) {
    RmsnormPack values = s_in[pack];
    RmsnormPack gamma = s_weight[pack];
    RmsnormPack result;
    for (int k = 0; k < kFloatXPerPack; ++k) {
      result[k] = static_cast<floatX>(
          static_cast<float>(values[k]) * scale * static_cast<float>(gamma[k]));
    }
    store128cs(y + static_cast<int64_t>(pack) * kFloatXPerPack,
                      result);
  }
}

__global__ void gemma4_residual_add_bf16_kernel(
    floatX *out,
    const floatX *inp1,
    const floatX *inp2,
    int packs) {
  int pack = blockIdx.x * blockDim.x + threadIdx.x;
  if (pack >= packs) {
    return;
  }

  RmsnormPack a =
      load128cs(inp1 + static_cast<int64_t>(pack) * kFloatXPerPack);
  RmsnormPack b =
      load128cs(inp2 + static_cast<int64_t>(pack) * kFloatXPerPack);
  RmsnormPack result;
  for (int k = 0; k < kFloatXPerPack; ++k) {
    result[k] =
        static_cast<floatX>(static_cast<float>(a[k]) + static_cast<float>(b[k]));
  }
  store128(out + static_cast<int64_t>(pack) * kFloatXPerPack, result);
}

__global__ void gemma4_residual_add_rmsnorm_bf16_warp_kernel(
    floatX *residual,
    floatX *normed,
    float *rstd,
    const floatX *inp1,
    const floatX *inp2,
    const floatX *weight,
    int rows,
    int width,
    float eps) {
  int lane = threadIdx.x % WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int warps_per_block = blockDim.x / WARP_SIZE;
  int row = blockIdx.x * warps_per_block + warp;
  if (row >= rows) {
    return;
  }

  const floatX *x1 = inp1 + static_cast<int64_t>(row) * width;
  const floatX *x2 = inp2 + static_cast<int64_t>(row) * width;
  floatX *res = residual + static_cast<int64_t>(row) * width;
  float sum_sq = 0.0f;
  for (int c = lane; c < width; c += WARP_SIZE) {
    float value = static_cast<float>(x1[c]) + static_cast<float>(x2[c]);
    floatX rounded = static_cast<floatX>(value);
    res[c] = rounded;
    float rounded_value = static_cast<float>(rounded);
    sum_sq += rounded_value * rounded_value;
  }
  sum_sq = warp_reduce_sum(sum_sq);
  float scale = rsqrtf(sum_sq / static_cast<float>(width) + eps);
  if (lane == 0 && rstd != nullptr) {
    rstd[row] = scale;
  }

  floatX *y = normed + static_cast<int64_t>(row) * width;
  for (int c = lane; c < width; c += WARP_SIZE) {
    float value = static_cast<float>(res[c]);
    float gamma = static_cast<float>(weight[c]);
    y[c] = static_cast<floatX>(value * scale * gamma);
  }
}

__global__ void gemma4_residual_add_rmsnorm_bf16_shared_kernel(
    floatX *residual,
    floatX *normed,
    float *rstd,
    const floatX *inp1,
    const floatX *inp2,
    const floatX *weight,
    int rows,
    int width,
    int packs_per_row,
    float eps) {
  extern __shared__ unsigned char shared_bytes[];
  auto *shared = reinterpret_cast<RmsnormPack *>(shared_bytes);
  RmsnormPack *s_weight = shared;
  RmsnormPack *s_res = s_weight + packs_per_row +
                       static_cast<int64_t>(threadIdx.y) * packs_per_row;

  int packed_thread = threadIdx.x + WARP_SIZE * threadIdx.y;
  int packed_stride = WARP_SIZE * blockDim.y;
  for (int pack = packed_thread; pack < packs_per_row; pack += packed_stride) {
    s_weight[pack] =
        load128(weight + static_cast<int64_t>(pack) * kFloatXPerPack);
  }
  __syncthreads();

  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) {
    return;
  }

  const floatX *x1 = inp1 + static_cast<int64_t>(row) * width;
  const floatX *x2 = inp2 + static_cast<int64_t>(row) * width;
  floatX *residual_row = residual + static_cast<int64_t>(row) * width;

  float sum_sq = 0.0f;
  for (int pack = threadIdx.x; pack < packs_per_row;
       pack += WARP_SIZE) {
    RmsnormPack a =
        load128cs(x1 + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack b =
        load128cs(x2 + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack result;
    for (int k = 0; k < kFloatXPerPack; ++k) {
      float value = static_cast<float>(a[k]) + static_cast<float>(b[k]);
      result[k] = static_cast<floatX>(value);
      sum_sq += static_cast<float>(result[k]) * static_cast<float>(result[k]);
    }
    s_res[pack] = result;
    store128cs(
        residual_row + static_cast<int64_t>(pack) * kFloatXPerPack, result);
  }

  sum_sq = warp_reduce_sum(sum_sq);
  float scale = rsqrtf(sum_sq / static_cast<float>(width) + eps);
  if (threadIdx.x == 0 && rstd != nullptr) {
    rstd[row] = scale;
  }

  floatX *y = normed + static_cast<int64_t>(row) * width;
  for (int pack = threadIdx.x; pack < packs_per_row;
       pack += WARP_SIZE) {
    RmsnormPack values = s_res[pack];
    RmsnormPack gamma = s_weight[pack];
    RmsnormPack result;
    for (int k = 0; k < kFloatXPerPack; ++k) {
      result[k] = static_cast<floatX>(
          static_cast<float>(values[k]) * scale * static_cast<float>(gamma[k]));
    }
    store128cs(y + static_cast<int64_t>(pack) * kFloatXPerPack,
                      result);
  }
}

bool gemma4_rmsnorm_args_valid(const floatX *out,
                               const floatX *inp,
                               const floatX *weight,
                               int rows,
                               int width) {
  if (rows < 0 || width <= 0) {
    return false;
  }
  if (rows == 0) {
    return true;
  }
  if (out == nullptr || inp == nullptr || weight == nullptr) {
    return false;
  }
  if ((width % kFloatXPerPack) != 0) {
    return false;
  }
  return is_aligned_16(out) && is_aligned_16(inp) &&
         is_aligned_16(weight);
}

template <typename Kernel>
cudaError_t gemma4_set_max_dynamic_smem(Kernel kernel, size_t bytes) {
  return cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                              static_cast<int>(bytes));
}

}  // namespace

cudaError_t gemma4_rmsnorm_bf16(floatX *out,
                                float *rstd,
                                const floatX *inp,
                                const floatX *weight,
                                int rows,
                                int width,
                                float eps,
                                cudaStream_t stream) {
  if (!gemma4_rmsnorm_args_valid(out, inp, weight, rows, width)) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }

  constexpr int block_size = 256;
  constexpr int block_y = block_size / WARP_SIZE;
  int packs_per_row = width / kFloatXPerPack;
  size_t smem = static_cast<size_t>(1 + block_y) * packs_per_row *
                sizeof(RmsnormPack);
  int grid = div_up(rows, block_y);

  cudaError_t attr =
      gemma4_set_max_dynamic_smem(gemma4_rmsnorm_bf16_shared_kernel, smem);
  if (attr == cudaSuccess) {
    gemma4_rmsnorm_bf16_shared_kernel<<<grid, dim3(WARP_SIZE, block_y),
                                        smem, stream>>>(
        out, rstd, inp, weight, rows, width, packs_per_row, eps);
  } else {
    gemma4_rmsnorm_bf16_warp_kernel<<<grid, block_size, 0, stream>>>(
        out, rstd, inp, weight, rows, width, eps);
  }
  return cudaGetLastError();
}

cudaError_t gemma4_residual_add_bf16(floatX *out,
                                     const floatX *inp1,
                                     const floatX *inp2,
                                     int count,
                                     cudaStream_t stream) {
  if (count < 0) {
    return cudaErrorInvalidValue;
  }
  if (count == 0) {
    return cudaSuccess;
  }
  if (out == nullptr || inp1 == nullptr || inp2 == nullptr) {
    return cudaErrorInvalidValue;
  }
  if ((count % kFloatXPerPack) != 0) {
    return cudaErrorInvalidValue;
  }
  if (!is_aligned_16(out) || !is_aligned_16(inp1) ||
      !is_aligned_16(inp2)) {
    return cudaErrorInvalidValue;
  }

  constexpr int block_size = 256;
  int packs = count / kFloatXPerPack;
  int grid = div_up(packs, block_size);
  gemma4_residual_add_bf16_kernel<<<grid, block_size, 0, stream>>>(
      out, inp1, inp2, packs);
  return cudaGetLastError();
}

cudaError_t gemma4_residual_add_rmsnorm_bf16(floatX *residual,
                                             floatX *normed,
                                             float *rstd,
                                             const floatX *inp1,
                                             const floatX *inp2,
                                             const floatX *weight,
                                             int rows,
                                             int width,
                                             float eps,
                                             cudaStream_t stream) {
  if (!gemma4_rmsnorm_args_valid(normed, inp1, weight, rows, width)) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }
  if (residual == nullptr || inp2 == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (!is_aligned_16(residual) || !is_aligned_16(inp2)) {
    return cudaErrorInvalidValue;
  }

  constexpr int block_size = 256;
  constexpr int block_y = block_size / WARP_SIZE;
  int packs_per_row = width / kFloatXPerPack;
  size_t smem = static_cast<size_t>(1 + block_y) * packs_per_row *
                sizeof(RmsnormPack);
  int grid = div_up(rows, block_y);

  cudaError_t attr = gemma4_set_max_dynamic_smem(
      gemma4_residual_add_rmsnorm_bf16_shared_kernel, smem);
  if (attr == cudaSuccess) {
    gemma4_residual_add_rmsnorm_bf16_shared_kernel<<<
        grid, dim3(WARP_SIZE, block_y), smem, stream>>>(
        residual, normed, rstd, inp1, inp2, weight, rows, width, packs_per_row,
        eps);
  } else {
    gemma4_residual_add_rmsnorm_bf16_warp_kernel<<<grid, block_size, 0,
                                                   stream>>>(
        residual, normed, rstd, inp1, inp2, weight, rows, width, eps);
  }
  return cudaGetLastError();
}
