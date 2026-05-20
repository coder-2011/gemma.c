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
constexpr int kRmsnormBlockSize = 256;
constexpr int kRmsnormRowsPerBlock = kRmsnormBlockSize / WARP_SIZE;
constexpr int kResidualAddThreads = 256;

// -----------------------------------------------------------------------------
// CUDA helpers

__device__ __forceinline__ float gemma4_rmsnorm_scale(float sum_sq,
                                                      int width,
                                                      float eps) {
  return rsqrtf(sum_sq / static_cast<float>(width) + eps);
}

__device__ __forceinline__ void gemma4_store_rstd(float *__restrict__ rstd,
                                                  int row,
                                                  float scale) {
  if (rstd != nullptr) {
    rstd[row] = scale;
  }
}

__device__ __forceinline__ RmsnormPack gemma4_bf16_pack_scale(
    const RmsnormPack &values,
    float scale) {
  const __nv_bfloat162 *value_pairs = gemma4_bf16_pack_pairs(values);
  RmsnormPack result;
  __nv_bfloat162 *out_pairs = gemma4_bf16_pack_pairs(result);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    float2 value = __bfloat1622float2(value_pairs[p]);
    out_pairs[p] = __floats2bfloat162_rn(value.x * scale, value.y * scale);
  }
  return result;
}

template <bool HasWeight>
__device__ __forceinline__ RmsnormPack gemma4_apply_rmsnorm_pack(
    const RmsnormPack &values,
    const floatX *__restrict__ weight,
    int pack,
    float scale) {
  if constexpr (HasWeight) {
    RmsnormPack gamma = load128g(weight + pack * kFloatXPerPack);
    return gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
  } else {
    return gemma4_bf16_pack_scale(values, scale);
  }
}

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

// -----------------------------------------------------------------------------
// CUDA kernels

template <bool HasWeight, int Threads>
__global__ __launch_bounds__(Threads, 1) void
gemma4_rmsnorm_bf16_decode_kernel(floatX *out,
                                  float *__restrict__ rstd,
                                  const floatX *inp,
                                  const floatX *__restrict__ weight,
                                  float eps) {
  static_assert((GEMMA4_HIDDEN_SIZE % kFloatXPerPack) == 0,
                "decode RMSNorm width must be divisible by Packed128 width");
  __shared__ RmsnormPack s_in[kDecodePacks];
  __shared__ float s_scale;

  float sum_sq = 0.0f;
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values = load128cs(inp + pack * kFloatXPerPack);
    s_in[pack] = values;
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
  }

  float block_sum = gemma4_block_reduce_sum<Threads>(sum_sq);
  if (threadIdx.x == 0) {
    float scale = gemma4_rmsnorm_scale(block_sum, GEMMA4_HIDDEN_SIZE, eps);
    s_scale = scale;
    gemma4_store_rstd(rstd, 0, scale);
  }
  __syncthreads();

  float scale = s_scale;
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values = s_in[pack];
    RmsnormPack result =
        gemma4_apply_rmsnorm_pack<HasWeight>(values, weight, pack, scale);
    store128(out + pack * kFloatXPerPack, result);
  }
}

template <bool HasWeight, int Threads>
__global__ __launch_bounds__(Threads, 1) void
gemma4_residual_add_rmsnorm_bf16_decode_kernel(floatX *residual,
                                               floatX *normed,
                                               float *__restrict__ rstd,
                                               const floatX *inp1,
                                               const floatX *inp2,
                                               const floatX *__restrict__ weight,
                                               float eps) {
  static_assert((GEMMA4_HIDDEN_SIZE % kFloatXPerPack) == 0,
                "decode RMSNorm width must be divisible by Packed128 width");
  __shared__ RmsnormPack s_res[kDecodePacks];
  __shared__ float s_scale;

  float sum_sq = 0.0f;
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack a = load128cs(inp1 + pack * kFloatXPerPack);
    RmsnormPack b = load128cs(inp2 + pack * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_add(a, b);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
    s_res[pack] = result;
    store128(residual + pack * kFloatXPerPack, result);
  }

  float block_sum = gemma4_block_reduce_sum<Threads>(sum_sq);
  if (threadIdx.x == 0) {
    float scale = gemma4_rmsnorm_scale(block_sum, GEMMA4_HIDDEN_SIZE, eps);
    s_scale = scale;
    gemma4_store_rstd(rstd, 0, scale);
  }
  __syncthreads();

  float scale = s_scale;
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values = s_res[pack];
    RmsnormPack result =
        gemma4_apply_rmsnorm_pack<HasWeight>(values, weight, pack, scale);
    store128(normed + pack * kFloatXPerPack, result);
  }
}

template <bool HasWeight>
__global__ __launch_bounds__(kRmsnormBlockSize) void
gemma4_rmsnorm_bf16_warp_kernel(
    floatX *out,
    float *__restrict__ rstd,
    const floatX *inp,
    const floatX *__restrict__ weight,
    int rows,
    int width,
    float eps) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;
  const int warps_per_block = blockDim.x / WARP_SIZE;
  const int row = blockIdx.x * warps_per_block + warp;
  if (row >= rows) {
    return;
  }

  inp += row * width;
  out += row * width;

  const int packs_per_row = width / kFloatXPerPack;
  float sum_sq = 0.0f;
  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = load128g(inp + pack * kFloatXPerPack);
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
  }
  sum_sq = warp_reduce_sum(sum_sq);
  float scale = gemma4_rmsnorm_scale(sum_sq, width, eps);
  if (lane == 0) {
    gemma4_store_rstd(rstd, row, scale);
  }

  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = load128cs(inp + pack * kFloatXPerPack);
    RmsnormPack result =
        gemma4_apply_rmsnorm_pack<HasWeight>(values, weight, pack, scale);
    store128cs(out + pack * kFloatXPerPack, result);
  }
}

template <bool HasWeight>
__global__ __launch_bounds__(kRmsnormBlockSize) void
gemma4_rmsnorm_bf16_shared_kernel(
    floatX *out,
    float *__restrict__ rstd,
    const floatX *inp,
    const floatX *__restrict__ weight,
    int rows,
    int width,
    int packs_per_row,
    float eps) {
  extern __shared__ unsigned char shared_bytes[];
  auto *shared = reinterpret_cast<RmsnormPack *>(shared_bytes);
  RmsnormPack *s_weight = shared;
  RmsnormPack *s_in = shared + (HasWeight ? packs_per_row : 0) +
                      threadIdx.y * packs_per_row;

  if constexpr (HasWeight) {
    const int packed_thread = threadIdx.x + WARP_SIZE * threadIdx.y;
    const int packed_stride = WARP_SIZE * blockDim.y;
    for (int pack = packed_thread; pack < packs_per_row;
         pack += packed_stride) {
      s_weight[pack] = load128g(weight + pack * kFloatXPerPack);
    }
    __syncthreads();
  }

  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) {
    return;
  }

  inp += row * width;
  out += row * width;

  float sum_sq = 0.0f;
  for (int pack = threadIdx.x; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = load128cs(inp + pack * kFloatXPerPack);
    s_in[pack] = values;
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
  }

  sum_sq = warp_reduce_sum(sum_sq);
  float scale = gemma4_rmsnorm_scale(sum_sq, width, eps);
  if (threadIdx.x == 0) {
    gemma4_store_rstd(rstd, row, scale);
  }

  for (int pack = threadIdx.x; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = s_in[pack];
    RmsnormPack result;
    if constexpr (HasWeight) {
      result = gemma4_bf16_pack_apply_rmsnorm(values, s_weight[pack], scale);
    } else {
      result = gemma4_bf16_pack_scale(values, scale);
    }
    store128cs(out + pack * kFloatXPerPack, result);
  }
}

__global__ __launch_bounds__(kResidualAddThreads) void
gemma4_residual_add_bf16_kernel(
    floatX *out,
    const floatX *inp1,
    const floatX *inp2,
    int packs) {
  int pack = blockIdx.x * blockDim.x + threadIdx.x;
  if (pack >= packs) {
    return;
  }

  RmsnormPack a = load128cs(inp1 + pack * kFloatXPerPack);
  RmsnormPack b = load128cs(inp2 + pack * kFloatXPerPack);
  RmsnormPack result = gemma4_bf16_pack_add(a, b);
  store128(out + pack * kFloatXPerPack, result);
}

template <bool HasWeight>
__global__ __launch_bounds__(kRmsnormBlockSize) void
gemma4_residual_add_rmsnorm_bf16_warp_kernel(
    floatX *residual,
    floatX *normed,
    float *__restrict__ rstd,
    const floatX *inp1,
    const floatX *inp2,
    const floatX *__restrict__ weight,
    int rows,
    int width,
    float eps) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;
  const int warps_per_block = blockDim.x / WARP_SIZE;
  const int row = blockIdx.x * warps_per_block + warp;
  if (row >= rows) {
    return;
  }

  inp1 += row * width;
  inp2 += row * width;
  residual += row * width;
  normed += row * width;

  const int packs_per_row = width / kFloatXPerPack;
  float sum_sq = 0.0f;
  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack a = load128cs(inp1 + pack * kFloatXPerPack);
    RmsnormPack b = load128cs(inp2 + pack * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_add(a, b);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
    store128(residual + pack * kFloatXPerPack, result);
  }
  sum_sq = warp_reduce_sum(sum_sq);
  float scale = gemma4_rmsnorm_scale(sum_sq, width, eps);
  if (lane == 0) {
    gemma4_store_rstd(rstd, row, scale);
  }

  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = load128(residual + pack * kFloatXPerPack);
    RmsnormPack result =
        gemma4_apply_rmsnorm_pack<HasWeight>(values, weight, pack, scale);
    store128cs(normed + pack * kFloatXPerPack, result);
  }
}

template <bool HasWeight>
__global__ __launch_bounds__(kRmsnormBlockSize) void
gemma4_residual_add_rmsnorm_bf16_shared_kernel(
    floatX *residual,
    floatX *normed,
    float *__restrict__ rstd,
    const floatX *inp1,
    const floatX *inp2,
    const floatX *__restrict__ weight,
    int rows,
    int width,
    int packs_per_row,
    float eps) {
  extern __shared__ unsigned char shared_bytes[];
  auto *shared = reinterpret_cast<RmsnormPack *>(shared_bytes);
  RmsnormPack *s_weight = shared;
  RmsnormPack *s_res = shared + (HasWeight ? packs_per_row : 0) +
                       threadIdx.y * packs_per_row;

  if constexpr (HasWeight) {
    const int packed_thread = threadIdx.x + WARP_SIZE * threadIdx.y;
    const int packed_stride = WARP_SIZE * blockDim.y;
    for (int pack = packed_thread; pack < packs_per_row;
         pack += packed_stride) {
      s_weight[pack] = load128g(weight + pack * kFloatXPerPack);
    }
    __syncthreads();
  }

  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) {
    return;
  }

  inp1 += row * width;
  inp2 += row * width;
  residual += row * width;
  normed += row * width;

  float sum_sq = 0.0f;
  for (int pack = threadIdx.x; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack a = load128cs(inp1 + pack * kFloatXPerPack);
    RmsnormPack b = load128cs(inp2 + pack * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_add(a, b);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
    s_res[pack] = result;
    store128cs(residual + pack * kFloatXPerPack, result);
  }

  sum_sq = warp_reduce_sum(sum_sq);
  float scale = gemma4_rmsnorm_scale(sum_sq, width, eps);
  if (threadIdx.x == 0) {
    gemma4_store_rstd(rstd, row, scale);
  }

  for (int pack = threadIdx.x; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = s_res[pack];
    RmsnormPack result;
    if constexpr (HasWeight) {
      result = gemma4_bf16_pack_apply_rmsnorm(values, s_weight[pack], scale);
    } else {
      result = gemma4_bf16_pack_scale(values, scale);
    }
    store128cs(normed + pack * kFloatXPerPack, result);
  }
}

// -----------------------------------------------------------------------------
// Host launch helpers

bool gemma4_rmsnorm_args_valid(const floatX *out,
                               const floatX *inp,
                               const floatX *__restrict__ weight,
                               int rows,
                               int width,
                               bool has_weight) {
  if (rows < 0 || width <= 0) {
    return false;
  }
  if (rows == 0) {
    return true;
  }
  if (out == nullptr || inp == nullptr || (has_weight && weight == nullptr)) {
    return false;
  }
  if ((width % kFloatXPerPack) != 0) {
    return false;
  }
  return is_aligned_16(out) && is_aligned_16(inp) &&
         (!has_weight || is_aligned_16(weight));
}

struct RmsnormLaunchShape {
  static constexpr int block_size = kRmsnormBlockSize;
  static constexpr int block_y = kRmsnormRowsPerBlock;

  int packs_per_row;
  size_t smem;
  int grid;
};

RmsnormLaunchShape gemma4_rmsnorm_launch_shape(int rows,
                                               int width,
                                               bool has_weight) {
  int packs_per_row = width / kFloatXPerPack;
  return {
      .packs_per_row = packs_per_row,
      .smem = static_cast<size_t>((has_weight ? 1 : 0) +
                                  RmsnormLaunchShape::block_y) *
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

template <bool HasWeight>
cudaError_t gemma4_rmsnorm_bf16_impl(floatX *out,
                                     float *__restrict__ rstd,
                                     const floatX *inp,
                                     const floatX *__restrict__ weight,
                                     int rows,
                                     int width,
                                     float eps,
                                     cudaStream_t stream) {
  if (!gemma4_rmsnorm_args_valid(out, inp, weight, rows, width, HasWeight)) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }
  if (rows == 1 && width == GEMMA4_HIDDEN_SIZE) {
    gemma4_rmsnorm_bf16_decode_kernel<HasWeight, kDecodeRmsnormThreads>
        <<<1, kDecodeRmsnormThreads, 0, stream>>>(out, rstd, inp, weight, eps);
    return cudaGetLastError();
  }

  RmsnormLaunchShape shape =
      gemma4_rmsnorm_launch_shape(rows, width, HasWeight);

  cudaError_t attr = gemma4_set_max_dynamic_smem(
      gemma4_rmsnorm_bf16_shared_kernel<HasWeight>, shape.smem);
  if (attr == cudaSuccess) {
    gemma4_rmsnorm_bf16_shared_kernel<HasWeight><<<
        shape.grid, dim3(WARP_SIZE, RmsnormLaunchShape::block_y), shape.smem,
        stream>>>(out, rstd, inp, weight, rows, width, shape.packs_per_row,
                  eps);
  } else {
    gemma4_rmsnorm_bf16_warp_kernel<HasWeight><<<
        shape.grid, RmsnormLaunchShape::block_size, 0, stream>>>(
        out, rstd, inp, weight, rows, width, eps);
  }
  return cudaGetLastError();
}

template <bool HasWeight>
cudaError_t gemma4_residual_add_rmsnorm_bf16_impl(
    floatX *residual,
    floatX *normed,
    float *__restrict__ rstd,
    const floatX *inp1,
    const floatX *inp2,
    const floatX *__restrict__ weight,
    int rows,
    int width,
    float eps,
    cudaStream_t stream) {
  if (!gemma4_rmsnorm_args_valid(normed, inp1, weight, rows, width,
                                 HasWeight)) {
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
    gemma4_residual_add_rmsnorm_bf16_decode_kernel<
        HasWeight, kDecodeFusedThreads><<<1, kDecodeFusedThreads, 0, stream>>>(
        residual, normed, rstd, inp1, inp2, weight, eps);
    return cudaGetLastError();
  }

  RmsnormLaunchShape shape =
      gemma4_rmsnorm_launch_shape(rows, width, HasWeight);

  cudaError_t attr = gemma4_set_max_dynamic_smem(
      gemma4_residual_add_rmsnorm_bf16_shared_kernel<HasWeight>, shape.smem);
  if (attr == cudaSuccess) {
    gemma4_residual_add_rmsnorm_bf16_shared_kernel<HasWeight><<<
        shape.grid, dim3(WARP_SIZE, RmsnormLaunchShape::block_y), shape.smem,
        stream>>>(residual, normed, rstd, inp1, inp2, weight, rows, width,
                  shape.packs_per_row, eps);
  } else {
    gemma4_residual_add_rmsnorm_bf16_warp_kernel<HasWeight><<<
        shape.grid, RmsnormLaunchShape::block_size, 0, stream>>>(
        residual, normed, rstd, inp1, inp2, weight, rows, width, eps);
  }
  return cudaGetLastError();
}

}  // namespace

cudaError_t gemma4_rmsnorm_bf16(floatX *out,
                                float *__restrict__ rstd,
                                const floatX *inp,
                                const floatX *__restrict__ weight,
                                int rows,
                                int width,
                                float eps,
                                cudaStream_t stream) {
  if (weight != nullptr) {
    return gemma4_rmsnorm_bf16_impl<true>(
        out, rstd, inp, weight, rows, width, eps, stream);
  }
  return gemma4_rmsnorm_bf16_impl<false>(
      out, rstd, inp, weight, rows, width, eps, stream);
}

cudaError_t gemma4_rmsnorm_scale_free_bf16(floatX *out,
                                           float *__restrict__ rstd,
                                           const floatX *inp,
                                           int rows,
                                           int width,
                                           float eps,
                                           cudaStream_t stream) {
  return gemma4_rmsnorm_bf16_impl<false>(
      out, rstd, inp, nullptr, rows, width, eps, stream);
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
  if (!is_aligned_16(out) || !is_aligned_16(inp1) || !is_aligned_16(inp2)) {
    return cudaErrorInvalidValue;
  }

  int packs = count / kFloatXPerPack;
  int grid = div_up(packs, kResidualAddThreads);
  gemma4_residual_add_bf16_kernel<<<grid, kResidualAddThreads, 0, stream>>>(
      out, inp1, inp2, packs);
  return cudaGetLastError();
}

cudaError_t gemma4_residual_add_rmsnorm_bf16(floatX *residual,
                                             floatX *normed,
                                             float *__restrict__ rstd,
                                             const floatX *inp1,
                                             const floatX *inp2,
                                             const floatX *__restrict__ weight,
                                             int rows,
                                             int width,
                                             float eps,
                                             cudaStream_t stream) {
  if (weight != nullptr) {
    return gemma4_residual_add_rmsnorm_bf16_impl<true>(
        residual, normed, rstd, inp1, inp2, weight, rows, width, eps, stream);
  }
  return gemma4_residual_add_rmsnorm_bf16_impl<false>(
      residual, normed, rstd, inp1, inp2, weight, rows, width, eps, stream);
}
