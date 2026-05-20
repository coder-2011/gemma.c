// Gemma RMSNorm: y = x * rsqrt(mean(x * x) + eps) * weight.

#include "gemma4_rmsnorm.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

#include <math.h>
#include <stdint.h>

namespace {

using RmsnormPack = Bf16Packed128;
constexpr int kFloatXPerPack = RmsnormPack::size;
constexpr int kBf16OnePairBits = 0x3f803f80;
constexpr int kDecodeRmsnormThreads = 704;
constexpr int kDecodeFusedThreads = 672;
constexpr int kDecodePacks = GEMMA4_HIDDEN_SIZE / kFloatXPerPack;
constexpr int kRmsnormBlockSize = 64;
constexpr int kRmsnormRowsPerBlock = kRmsnormBlockSize / WARP_SIZE;
constexpr int kResidualAddThreads = 256;


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

__device__ __forceinline__ RmsnormPack gemma4_bf16_pack_ones() {
  return RmsnormPack{make_int4(kBf16OnePairBits, kBf16OnePairBits,
                               kBf16OnePairBits, kBf16OnePairBits)};
}

__device__ __forceinline__ RmsnormPack gemma4_apply_weighted_rmsnorm_pack(
    const RmsnormPack &values,
    const floatX *__restrict__ weight,
    int pack,
    float scale) {
  RmsnormPack gamma = load128g(weight + pack * kFloatXPerPack);
  return gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
}

__device__ __forceinline__ RmsnormPack gemma4_apply_scale_free_rmsnorm_pack(
    const RmsnormPack &values,
    float scale) {
  return gemma4_bf16_pack_apply_rmsnorm(
      values, gemma4_bf16_pack_ones(), scale);
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

template <int Threads>
__device__ __forceinline__ float gemma4_decode_rmsnorm_scale(
    float sum_sq,
    float *__restrict__ rstd,
    float &s_scale,
    float eps) {
  float block_sum = gemma4_block_reduce_sum<Threads>(sum_sq);
  if (threadIdx.x == 0) {
    float scale = gemma4_rmsnorm_scale(block_sum, GEMMA4_HIDDEN_SIZE, eps);
    s_scale = scale;
    gemma4_store_rstd(rstd, 0, scale);
  }
  __syncthreads();
  return s_scale;
}

__device__ __forceinline__ float gemma4_warp_rmsnorm_scale(
    float sum_sq,
    float *__restrict__ rstd,
    int row,
    int width,
    int lane,
    float eps) {
  sum_sq = warp_reduce_sum(sum_sq);
  float scale = gemma4_rmsnorm_scale(sum_sq, width, eps);
  if (lane == 0) {
    gemma4_store_rstd(rstd, row, scale);
  }
  return scale;
}

__device__ __forceinline__ float gemma4_accumulate_input_squares(
    const floatX *__restrict__ inp,
    int first_pack,
    int pack_stride,
    int packs_per_row) {
  float sum_sq = 0.0f;
  for (int pack = first_pack; pack < packs_per_row; pack += pack_stride) {
    RmsnormPack values = load128g(inp + pack * kFloatXPerPack);
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
  }
  return sum_sq;
}

__device__ __forceinline__ float gemma4_cache_input_squares(
    RmsnormPack *__restrict__ cache,
    const floatX *__restrict__ inp,
    int first_pack,
    int pack_stride,
    int packs_per_row) {
  float sum_sq = 0.0f;
  for (int pack = first_pack; pack < packs_per_row; pack += pack_stride) {
    RmsnormPack values = load128cs(inp + pack * kFloatXPerPack);
    cache[pack] = values;
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
  }
  return sum_sq;
}

__device__ __forceinline__ RmsnormPack gemma4_load_residual_pack(
    const floatX *__restrict__ inp1,
    const floatX *__restrict__ inp2,
    int pack) {
  RmsnormPack a = load128cs(inp1 + pack * kFloatXPerPack);
  RmsnormPack b = load128cs(inp2 + pack * kFloatXPerPack);
  return gemma4_bf16_pack_add(a, b);
}

// -----------------------------------------------------------------------------
// CUDA kernels

template <int Threads>
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

  float sum_sq =
      gemma4_cache_input_squares(s_in, inp, threadIdx.x, Threads, kDecodePacks);
  float scale = gemma4_decode_rmsnorm_scale<Threads>(
      sum_sq, rstd, s_scale, eps);
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values = s_in[pack];
    RmsnormPack result =
        gemma4_apply_weighted_rmsnorm_pack(values, weight, pack, scale);
    store128(out + pack * kFloatXPerPack, result);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void
gemma4_rmsnorm_scale_free_bf16_decode_kernel(floatX *out,
                                             float *__restrict__ rstd,
                                             const floatX *inp,
                                             float eps) {
  static_assert((GEMMA4_HIDDEN_SIZE % kFloatXPerPack) == 0,
                "decode RMSNorm width must be divisible by Packed128 width");
  __shared__ RmsnormPack s_in[kDecodePacks];
  __shared__ float s_scale;

  float sum_sq =
      gemma4_cache_input_squares(s_in, inp, threadIdx.x, Threads, kDecodePacks);
  float scale = gemma4_decode_rmsnorm_scale<Threads>(
      sum_sq, rstd, s_scale, eps);
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values = s_in[pack];
    RmsnormPack result = gemma4_apply_scale_free_rmsnorm_pack(values, scale);
    store128(out + pack * kFloatXPerPack, result);
  }
}

template <int Threads>
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
    RmsnormPack result = gemma4_load_residual_pack(inp1, inp2, pack);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
    s_res[pack] = result;
    store128(residual + pack * kFloatXPerPack, result);
  }

  float scale = gemma4_decode_rmsnorm_scale<Threads>(
      sum_sq, rstd, s_scale, eps);
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values = s_res[pack];
    RmsnormPack result =
        gemma4_apply_weighted_rmsnorm_pack(values, weight, pack, scale);
    store128(normed + pack * kFloatXPerPack, result);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void
gemma4_residual_add_rmsnorm_scale_free_bf16_decode_kernel(
    floatX *residual,
    floatX *normed,
    float *__restrict__ rstd,
    const floatX *inp1,
    const floatX *inp2,
    float eps) {
  static_assert((GEMMA4_HIDDEN_SIZE % kFloatXPerPack) == 0,
                "decode RMSNorm width must be divisible by Packed128 width");
  __shared__ RmsnormPack s_res[kDecodePacks];
  __shared__ float s_scale;

  float sum_sq = 0.0f;
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack result = gemma4_load_residual_pack(inp1, inp2, pack);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
    s_res[pack] = result;
    store128(residual + pack * kFloatXPerPack, result);
  }

  float scale = gemma4_decode_rmsnorm_scale<Threads>(
      sum_sq, rstd, s_scale, eps);
  for (int pack = threadIdx.x; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values = s_res[pack];
    RmsnormPack result = gemma4_apply_scale_free_rmsnorm_pack(values, scale);
    store128(normed + pack * kFloatXPerPack, result);
  }
}

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
  float sum_sq =
      gemma4_accumulate_input_squares(inp, lane, WARP_SIZE, packs_per_row);
  float scale = gemma4_warp_rmsnorm_scale(
      sum_sq, rstd, row, width, lane, eps);

  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = load128cs(inp + pack * kFloatXPerPack);
    RmsnormPack result =
        gemma4_apply_weighted_rmsnorm_pack(values, weight, pack, scale);
    store128cs(out + pack * kFloatXPerPack, result);
  }
}

__global__ __launch_bounds__(kRmsnormBlockSize) void
gemma4_rmsnorm_scale_free_bf16_warp_kernel(
    floatX *out,
    float *__restrict__ rstd,
    const floatX *inp,
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
  float sum_sq =
      gemma4_accumulate_input_squares(inp, lane, WARP_SIZE, packs_per_row);
  float scale = gemma4_warp_rmsnorm_scale(
      sum_sq, rstd, row, width, lane, eps);

  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = load128cs(inp + pack * kFloatXPerPack);
    RmsnormPack result = gemma4_apply_scale_free_rmsnorm_pack(values, scale);
    store128cs(out + pack * kFloatXPerPack, result);
  }
}

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
  RmsnormPack *s_in = shared + packs_per_row + threadIdx.y * packs_per_row;

  const int packed_thread = threadIdx.x + WARP_SIZE * threadIdx.y;
  const int packed_stride = WARP_SIZE * blockDim.y;
  for (int pack = packed_thread; pack < packs_per_row;
       pack += packed_stride) {
    s_weight[pack] = load128g(weight + pack * kFloatXPerPack);
  }
  __syncthreads();

  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) {
    return;
  }

  inp += row * width;
  out += row * width;

  float sum_sq = gemma4_cache_input_squares(
      s_in, inp, threadIdx.x, WARP_SIZE, packs_per_row);
  float scale = gemma4_warp_rmsnorm_scale(
      sum_sq, rstd, row, width, threadIdx.x, eps);

  for (int pack = threadIdx.x; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = s_in[pack];
    RmsnormPack result =
        gemma4_bf16_pack_apply_rmsnorm(values, s_weight[pack], scale);
    store128cs(out + pack * kFloatXPerPack, result);
  }
}

__global__ __launch_bounds__(kRmsnormBlockSize) void
gemma4_rmsnorm_scale_free_bf16_shared_kernel(
    floatX *out,
    float *__restrict__ rstd,
    const floatX *inp,
    int rows,
    int width,
    int packs_per_row,
    float eps) {
  extern __shared__ unsigned char shared_bytes[];
  auto *s_in = reinterpret_cast<RmsnormPack *>(shared_bytes) +
               threadIdx.y * packs_per_row;

  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) {
    return;
  }

  inp += row * width;
  out += row * width;

  float sum_sq = gemma4_cache_input_squares(
      s_in, inp, threadIdx.x, WARP_SIZE, packs_per_row);
  float scale = gemma4_warp_rmsnorm_scale(
      sum_sq, rstd, row, width, threadIdx.x, eps);

  for (int pack = threadIdx.x; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = s_in[pack];
    RmsnormPack result = gemma4_apply_scale_free_rmsnorm_pack(values, scale);
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
    RmsnormPack result = gemma4_load_residual_pack(inp1, inp2, pack);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
    store128(residual + pack * kFloatXPerPack, result);
  }
  float scale = gemma4_warp_rmsnorm_scale(
      sum_sq, rstd, row, width, lane, eps);

  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = load128(residual + pack * kFloatXPerPack);
    RmsnormPack result =
        gemma4_apply_weighted_rmsnorm_pack(values, weight, pack, scale);
    store128cs(normed + pack * kFloatXPerPack, result);
  }
}

__global__ __launch_bounds__(kRmsnormBlockSize) void
gemma4_residual_add_rmsnorm_scale_free_bf16_warp_kernel(
    floatX *residual,
    floatX *normed,
    float *__restrict__ rstd,
    const floatX *inp1,
    const floatX *inp2,
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
    RmsnormPack result = gemma4_load_residual_pack(inp1, inp2, pack);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
    store128(residual + pack * kFloatXPerPack, result);
  }
  float scale = gemma4_warp_rmsnorm_scale(
      sum_sq, rstd, row, width, lane, eps);

  for (int pack = lane; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = load128(residual + pack * kFloatXPerPack);
    RmsnormPack result = gemma4_apply_scale_free_rmsnorm_pack(values, scale);
    store128cs(normed + pack * kFloatXPerPack, result);
  }
}

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
  RmsnormPack *s_res = shared + packs_per_row + threadIdx.y * packs_per_row;

  const int packed_thread = threadIdx.x + WARP_SIZE * threadIdx.y;
  const int packed_stride = WARP_SIZE * blockDim.y;
  for (int pack = packed_thread; pack < packs_per_row;
       pack += packed_stride) {
    s_weight[pack] = load128g(weight + pack * kFloatXPerPack);
  }
  __syncthreads();

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
    RmsnormPack result = gemma4_load_residual_pack(inp1, inp2, pack);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
    s_res[pack] = result;
    store128cs(residual + pack * kFloatXPerPack, result);
  }

  float scale = gemma4_warp_rmsnorm_scale(
      sum_sq, rstd, row, width, threadIdx.x, eps);

  for (int pack = threadIdx.x; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = s_res[pack];
    RmsnormPack result =
        gemma4_bf16_pack_apply_rmsnorm(values, s_weight[pack], scale);
    store128cs(normed + pack * kFloatXPerPack, result);
  }
}

__global__ __launch_bounds__(kRmsnormBlockSize) void
gemma4_residual_add_rmsnorm_scale_free_bf16_shared_kernel(
    floatX *residual,
    floatX *normed,
    float *__restrict__ rstd,
    const floatX *inp1,
    const floatX *inp2,
    int rows,
    int width,
    int packs_per_row,
    float eps) {
  extern __shared__ unsigned char shared_bytes[];
  auto *s_res = reinterpret_cast<RmsnormPack *>(shared_bytes) +
                threadIdx.y * packs_per_row;

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
    RmsnormPack result = gemma4_load_residual_pack(inp1, inp2, pack);
    gemma4_bf16_pack_accumulate_square(result, sum_sq);
    s_res[pack] = result;
    store128cs(residual + pack * kFloatXPerPack, result);
  }

  float scale = gemma4_warp_rmsnorm_scale(
      sum_sq, rstd, row, width, threadIdx.x, eps);

  for (int pack = threadIdx.x; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = s_res[pack];
    RmsnormPack result = gemma4_apply_scale_free_rmsnorm_pack(values, scale);
    store128cs(normed + pack * kFloatXPerPack, result);
  }
}

// -----------------------------------------------------------------------------
// Host launch helpers

bool gemma4_rmsnorm_args_valid(const floatX *out,
                               const floatX *inp,
                               int rows,
                               int width) {
  if (rows < 0 || width <= 0) {
    return false;
  }
  if (rows == 0) {
    return true;
  }
  return out != nullptr && inp != nullptr &&
         (width % kFloatXPerPack) == 0 &&
         is_aligned_16(out) && is_aligned_16(inp);
}

bool gemma4_weighted_rmsnorm_args_valid(const floatX *out,
                                        const floatX *inp,
                                        const floatX *__restrict__ weight,
                                        int rows,
                                        int width) {
  return gemma4_rmsnorm_args_valid(out, inp, rows, width) &&
         (rows == 0 || (weight != nullptr && is_aligned_16(weight)));
}

bool gemma4_residual_add_rmsnorm_args_valid(floatX *residual,
                                            floatX *normed,
                                            const floatX *inp1,
                                            const floatX *inp2,
                                            int rows,
                                            int width) {
  if (!gemma4_rmsnorm_args_valid(normed, inp1, rows, width)) {
    return false;
  }
  if (rows == 0) {
    return true;
  }
  return residual != nullptr && inp2 != nullptr &&
         is_aligned_16(residual) && is_aligned_16(inp2);
}

cudaError_t gemma4_rmsnorm_bf16_impl(floatX *out,
                                     float *__restrict__ rstd,
                                     const floatX *inp,
                                     const floatX *__restrict__ weight,
                                     int rows,
                                     int width,
                                     float eps,
                                     cudaStream_t stream) {
  if (!gemma4_weighted_rmsnorm_args_valid(out, inp, weight, rows, width)) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }
  if (rows == 1 && width == GEMMA4_HIDDEN_SIZE) {
    gemma4_rmsnorm_bf16_decode_kernel<kDecodeRmsnormThreads>
        <<<1, kDecodeRmsnormThreads, 0, stream>>>(out, rstd, inp, weight, eps);
    return cudaGetLastError();
  }

  const int packs_per_row = width / kFloatXPerPack;
  const int grid = div_up(rows, kRmsnormRowsPerBlock);
  const size_t smem = static_cast<size_t>(1 + kRmsnormRowsPerBlock) *
                      packs_per_row * sizeof(RmsnormPack);

  cudaError_t attr = cudaFuncSetAttribute(
      reinterpret_cast<const void *>(gemma4_rmsnorm_bf16_shared_kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  if (attr == cudaSuccess) {
    gemma4_rmsnorm_bf16_shared_kernel<<<
        grid, dim3(WARP_SIZE, kRmsnormRowsPerBlock), smem, stream>>>(
        out, rstd, inp, weight, rows, width, packs_per_row, eps);
  } else {
    gemma4_rmsnorm_bf16_warp_kernel<<<grid, kRmsnormBlockSize, 0, stream>>>(
        out, rstd, inp, weight, rows, width, eps);
  }
  return cudaGetLastError();
}

cudaError_t gemma4_rmsnorm_scale_free_bf16_impl(floatX *out,
                                                float *__restrict__ rstd,
                                                const floatX *inp,
                                                int rows,
                                                int width,
                                                float eps,
                                                cudaStream_t stream) {
  if (!gemma4_rmsnorm_args_valid(out, inp, rows, width)) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }
  if (rows == 1 && width == GEMMA4_HIDDEN_SIZE) {
    gemma4_rmsnorm_scale_free_bf16_decode_kernel<kDecodeRmsnormThreads>
        <<<1, kDecodeRmsnormThreads, 0, stream>>>(out, rstd, inp, eps);
    return cudaGetLastError();
  }

  const int packs_per_row = width / kFloatXPerPack;
  const int grid = div_up(rows, kRmsnormRowsPerBlock);
  const size_t smem = static_cast<size_t>(kRmsnormRowsPerBlock) *
                      packs_per_row * sizeof(RmsnormPack);

  cudaError_t attr = cudaFuncSetAttribute(
      reinterpret_cast<const void *>(
          gemma4_rmsnorm_scale_free_bf16_shared_kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  if (attr == cudaSuccess) {
    gemma4_rmsnorm_scale_free_bf16_shared_kernel<<<
        grid, dim3(WARP_SIZE, kRmsnormRowsPerBlock), smem, stream>>>(
        out, rstd, inp, rows, width, packs_per_row, eps);
  } else {
    gemma4_rmsnorm_scale_free_bf16_warp_kernel<<<
        grid, kRmsnormBlockSize, 0, stream>>>(out, rstd, inp, rows, width, eps);
  }
  return cudaGetLastError();
}

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
  if (!gemma4_weighted_rmsnorm_args_valid(normed, inp1, weight, rows, width)) {
    return cudaErrorInvalidValue;
  }
  if (!gemma4_residual_add_rmsnorm_args_valid(residual, normed, inp1, inp2,
                                              rows, width)) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }
  if (rows == 1 && width == GEMMA4_HIDDEN_SIZE) {
    gemma4_residual_add_rmsnorm_bf16_decode_kernel<kDecodeFusedThreads>
        <<<1, kDecodeFusedThreads, 0, stream>>>(
        residual, normed, rstd, inp1, inp2, weight, eps);
    return cudaGetLastError();
  }

  const int packs_per_row = width / kFloatXPerPack;
  const int grid = div_up(rows, kRmsnormRowsPerBlock);
  const size_t smem = static_cast<size_t>(1 + kRmsnormRowsPerBlock) *
                      packs_per_row * sizeof(RmsnormPack);

  cudaError_t attr = cudaFuncSetAttribute(
      reinterpret_cast<const void *>(
          gemma4_residual_add_rmsnorm_bf16_shared_kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  if (attr == cudaSuccess) {
    gemma4_residual_add_rmsnorm_bf16_shared_kernel<<<
        grid, dim3(WARP_SIZE, kRmsnormRowsPerBlock), smem, stream>>>(
        residual, normed, rstd, inp1, inp2, weight, rows, width,
        packs_per_row, eps);
  } else {
    gemma4_residual_add_rmsnorm_bf16_warp_kernel<<<
        grid, kRmsnormBlockSize, 0, stream>>>(
        residual, normed, rstd, inp1, inp2, weight, rows, width, eps);
  }
  return cudaGetLastError();
}

cudaError_t gemma4_residual_add_rmsnorm_scale_free_bf16_impl(
    floatX *residual,
    floatX *normed,
    float *__restrict__ rstd,
    const floatX *inp1,
    const floatX *inp2,
    int rows,
    int width,
    float eps,
    cudaStream_t stream) {
  if (!gemma4_residual_add_rmsnorm_args_valid(residual, normed, inp1, inp2,
                                              rows, width)) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }
  if (rows == 1 && width == GEMMA4_HIDDEN_SIZE) {
    gemma4_residual_add_rmsnorm_scale_free_bf16_decode_kernel<
        kDecodeFusedThreads><<<1, kDecodeFusedThreads, 0, stream>>>(
        residual, normed, rstd, inp1, inp2, eps);
    return cudaGetLastError();
  }

  const int packs_per_row = width / kFloatXPerPack;
  const int grid = div_up(rows, kRmsnormRowsPerBlock);
  const size_t smem = static_cast<size_t>(kRmsnormRowsPerBlock) *
                      packs_per_row * sizeof(RmsnormPack);

  cudaError_t attr = cudaFuncSetAttribute(
      reinterpret_cast<const void *>(
          gemma4_residual_add_rmsnorm_scale_free_bf16_shared_kernel),
      cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  if (attr == cudaSuccess) {
    gemma4_residual_add_rmsnorm_scale_free_bf16_shared_kernel<<<
        grid, dim3(WARP_SIZE, kRmsnormRowsPerBlock), smem, stream>>>(
        residual, normed, rstd, inp1, inp2, rows, width, packs_per_row, eps);
  } else {
    gemma4_residual_add_rmsnorm_scale_free_bf16_warp_kernel<<<
        grid, kRmsnormBlockSize, 0, stream>>>(
        residual, normed, rstd, inp1, inp2, rows, width, eps);
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
    return gemma4_rmsnorm_bf16_impl(
        out, rstd, inp, weight, rows, width, eps, stream);
  }
  return gemma4_rmsnorm_scale_free_bf16_impl(
      out, rstd, inp, rows, width, eps, stream);
}

cudaError_t gemma4_rmsnorm_scale_free_bf16(floatX *out,
                                           float *__restrict__ rstd,
                                           const floatX *inp,
                                           int rows,
                                           int width,
                                           float eps,
                                           cudaStream_t stream) {
  return gemma4_rmsnorm_scale_free_bf16_impl(
      out, rstd, inp, rows, width, eps, stream);
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
  if (out == nullptr || inp1 == nullptr || inp2 == nullptr ||
      (count % kFloatXPerPack) != 0 || !is_aligned_16(out) ||
      !is_aligned_16(inp1) || !is_aligned_16(inp2)) {
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
    return gemma4_residual_add_rmsnorm_bf16_impl(
        residual, normed, rstd, inp1, inp2, weight, rows, width, eps, stream);
  }
  return gemma4_residual_add_rmsnorm_scale_free_bf16_impl(
      residual, normed, rstd, inp1, inp2, rows, width, eps, stream);
}
