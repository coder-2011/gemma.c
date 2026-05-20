// Gemma RMSNorm: y = x * rsqrt(mean(x * x) + eps) * weight.

#include "gemma4_rmsnorm.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

#include <math.h>
#include <stdint.h>

namespace {

using RmsnormPack = Bf16Packed128;
constexpr int kFloatXPerPack = RmsnormPack::size;
constexpr int kDecodeRmsnormThreads = 768;
constexpr int kDecodeFusedThreads = 1024;
constexpr int kDecodePacks = GEMMA4_HIDDEN_SIZE / kFloatXPerPack;

template <int Threads>
__device__ __forceinline__ float gemma4_block_reduce_sum(float value) {
  static_assert((Threads % WARP_SIZE) == 0,
                "block reduction requires whole warps");
  constexpr int warps = Threads / WARP_SIZE;
  __shared__ float warp_sums[warps];
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  return value;
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void
gemma4_rmsnorm_bf16_decode_kernel(floatX *out,
                                  float *rstd,
                                  const floatX *inp,
                                  const floatX *weight,
                                  float eps) {
  static_assert((GEMMA4_HIDDEN_SIZE % kFloatXPerPack) == 0,
                "decode RMSNorm width must be divisible by Packed128 width");
  __shared__ RmsnormPack s_in[kDecodePacks];
  __shared__ float s_scale;

  float sum_sq = 0.0f;
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values =
        load128cs(inp + static_cast<int64_t>(pack) * kFloatXPerPack);
    s_in[pack] = values;
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
  }

  float block_sum = gemma4_block_reduce_sum<Threads>(sum_sq);
  if (threadIdx.x == 0) {
    float scale =
        rsqrtf(block_sum / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
    s_scale = scale;
    if (rstd != nullptr) {
      rstd[0] = scale;
    }
  }
  __syncthreads();

  float scale = s_scale;
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values = s_in[pack];
    RmsnormPack gamma =
        load128(weight + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
    store128(out + static_cast<int64_t>(pack) * kFloatXPerPack, result);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void
gemma4_residual_add_rmsnorm_bf16_decode_kernel(floatX *residual,
                                               floatX *normed,
                                               float *rstd,
                                               const floatX *inp1,
                                               const floatX *inp2,
                                               const floatX *weight,
                                               float eps) {
  static_assert((GEMMA4_HIDDEN_SIZE % kFloatXPerPack) == 0,
                "decode RMSNorm width must be divisible by Packed128 width");
  __shared__ RmsnormPack s_res[kDecodePacks];
  __shared__ float s_scale;

  float sum_sq = 0.0f;
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack a =
        load128cs(inp1 + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack b =
        load128cs(inp2 + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_add(a, b);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
    s_res[pack] = result;
    store128(residual + static_cast<int64_t>(pack) * kFloatXPerPack, result);
  }

  float block_sum = gemma4_block_reduce_sum<Threads>(sum_sq);
  if (threadIdx.x == 0) {
    float scale =
        rsqrtf(block_sum / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
    s_scale = scale;
    if (rstd != nullptr) {
      rstd[0] = scale;
    }
  }
  __syncthreads();

  float scale = s_scale;
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values = s_res[pack];
    RmsnormPack gamma =
        load128(weight + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
    store128(normed + static_cast<int64_t>(pack) * kFloatXPerPack, result);
  }
}

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
  int packs_per_row = width / kFloatXPerPack;
  float sum_sq = 0.0f;
  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values =
        load128(x + static_cast<int64_t>(pack) * kFloatXPerPack);
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
  }
  sum_sq = warp_reduce_sum(sum_sq);
  float scale = rsqrtf(sum_sq / static_cast<float>(width) + eps);
  if (lane == 0 && rstd != nullptr) {
    rstd[row] = scale;
  }

  floatX *y = out + static_cast<int64_t>(row) * width;
  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values =
        load128cs(x + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack gamma =
        load128(weight + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
    store128cs(y + static_cast<int64_t>(pack) * kFloatXPerPack, result);
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
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
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
    RmsnormPack result = gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
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
  RmsnormPack result = gemma4_bf16_pack_add(a, b);
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
  int packs_per_row = width / kFloatXPerPack;
  float sum_sq = 0.0f;
  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack a =
        load128cs(x1 + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack b =
        load128cs(x2 + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_add(a, b);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
    store128(res + static_cast<int64_t>(pack) * kFloatXPerPack, result);
  }
  sum_sq = warp_reduce_sum(sum_sq);
  float scale = rsqrtf(sum_sq / static_cast<float>(width) + eps);
  if (lane == 0 && rstd != nullptr) {
    rstd[row] = scale;
  }

  floatX *y = normed + static_cast<int64_t>(row) * width;
  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values =
        load128(res + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack gamma =
        load128(weight + static_cast<int64_t>(pack) * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
    store128cs(y + static_cast<int64_t>(pack) * kFloatXPerPack, result);
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
    RmsnormPack result = gemma4_bf16_pack_add(a, b);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
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
    RmsnormPack result = gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
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

struct RmsnormLaunchShape {
  static constexpr int block_size = 256;
  static constexpr int block_y = block_size / WARP_SIZE;

  int packs_per_row;
  size_t smem;
  int grid;
};

RmsnormLaunchShape gemma4_rmsnorm_launch_shape(int rows, int width) {
  int packs_per_row = width / kFloatXPerPack;
  return {
      .packs_per_row = packs_per_row,
      .smem = static_cast<size_t>(1 + RmsnormLaunchShape::block_y) *
              packs_per_row * sizeof(RmsnormPack),
      .grid = div_up(rows, RmsnormLaunchShape::block_y),
  };
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
  if (rows == 1 && width == GEMMA4_HIDDEN_SIZE) {
    gemma4_rmsnorm_bf16_decode_kernel<kDecodeRmsnormThreads>
        <<<1, kDecodeRmsnormThreads, 0, stream>>>(
            out, rstd, inp, weight, eps);
    return cudaGetLastError();
  }

  RmsnormLaunchShape shape = gemma4_rmsnorm_launch_shape(rows, width);

  cudaError_t attr =
      gemma4_set_max_dynamic_smem(gemma4_rmsnorm_bf16_shared_kernel,
                                  shape.smem);
  if (attr == cudaSuccess) {
    gemma4_rmsnorm_bf16_shared_kernel<<<
        shape.grid, dim3(WARP_SIZE, RmsnormLaunchShape::block_y), shape.smem,
        stream>>>(out, rstd, inp, weight, rows, width, shape.packs_per_row,
                  eps);
  } else {
    gemma4_rmsnorm_bf16_warp_kernel<<<
        shape.grid, RmsnormLaunchShape::block_size, 0, stream>>>(
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
  if (rows == 1 && width == GEMMA4_HIDDEN_SIZE) {
    gemma4_residual_add_rmsnorm_bf16_decode_kernel<kDecodeFusedThreads>
        <<<1, kDecodeFusedThreads, 0, stream>>>(
            residual, normed, rstd, inp1, inp2, weight, eps);
    return cudaGetLastError();
  }

  RmsnormLaunchShape shape = gemma4_rmsnorm_launch_shape(rows, width);

  cudaError_t attr = gemma4_set_max_dynamic_smem(
      gemma4_residual_add_rmsnorm_bf16_shared_kernel, shape.smem);
  if (attr == cudaSuccess) {
    gemma4_residual_add_rmsnorm_bf16_shared_kernel<<<
        shape.grid, dim3(WARP_SIZE, RmsnormLaunchShape::block_y), shape.smem,
        stream>>>(residual, normed, rstd, inp1, inp2, weight, rows, width,
                  shape.packs_per_row, eps);
  } else {
    gemma4_residual_add_rmsnorm_bf16_warp_kernel<<<
        shape.grid, RmsnormLaunchShape::block_size, 0, stream>>>(
        residual, normed, rstd, inp1, inp2, weight, rows, width, eps);
  }
  return cudaGetLastError();
}
