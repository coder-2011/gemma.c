// Gemma RMSNorm: y = x * rsqrt(mean(x * x) + eps) * weight.

#include "gemma4_rmsnorm.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

#include <cuda_pipeline_primitives.h>

#include <math.h>

namespace {

using RmsnormPack = Bf16Packed128;
constexpr int kFloatXPerPack = RmsnormPack::size;
constexpr int kDecodePacks = GEMMA4_HIDDEN_SIZE / kFloatXPerPack;
constexpr int kDecodeRmsnormThreads = kDecodePacks;
constexpr int kDecodeFusedThreads = kDecodePacks;
constexpr int kHiddenPrefillFusedThreads = kDecodePacks;
constexpr int kRmsnormRowsPerBlock = 2;
constexpr int kRmsnormThreads = WARP_SIZE * kRmsnormRowsPerBlock;
constexpr int kResidualAddThreads = 256;

#ifndef GEMMA4_HIDDEN_PREFILL_MIN_BLOCKS_PER_SM
#define GEMMA4_HIDDEN_PREFILL_MIN_BLOCKS_PER_SM 2
#endif

__device__ void gemma4_async_load_input_pack(RmsnormPack *__restrict__ dst,
                                             const floatX *__restrict__ src) {
  static_assert(sizeof(RmsnormPack) == 16,
                "LDGSTS path copies one aligned 16-byte RMSNorm pack");
  __pipeline_memcpy_async(dst, src, sizeof(RmsnormPack));
}

// -----------------------------------------------------------------------------
// CUDA kernels

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void
gemma4_rmsnorm_bf16_decode_kernel(floatX *out,
                                  const floatX *inp,
                                  const floatX *__restrict__ weight,
                                  float eps) {
  static_assert((GEMMA4_HIDDEN_SIZE % kFloatXPerPack) == 0,
                "decode RMSNorm width must be divisible by Packed128 width");
  __shared__ RmsnormPack s_in[kDecodePacks];
  __shared__ float s_warp_sums[Threads / WARP_SIZE];
  __shared__ float s_scale;
  const int thread = threadIdx.x;
  const int lane = thread & (WARP_SIZE - 1);
  const int warp = thread / WARP_SIZE;

  for (int pack = thread; pack < kDecodePacks; pack += Threads) {
    gemma4_async_load_input_pack(s_in + pack, inp + pack * kFloatXPerPack);
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);

  float sum_sq = 0.0f;
  for (int pack = thread; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values = s_in[pack];
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
  }

  float block_sum = gemma4_block_reduce_sum<Threads>(
      sum_sq, s_warp_sums, thread, lane, warp);
  if (thread == 0) {
    s_scale = rsqrtf(block_sum / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();
  float scale = s_scale;
  for (int pack = thread; pack < kDecodePacks; pack += Threads) {
    RmsnormPack values = s_in[pack];
    RmsnormPack gamma = load128g(weight + pack * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
    store128(out + pack * kFloatXPerPack, result);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void
gemma4_residual_add_rmsnorm_bf16_decode_kernel(floatX *residual,
                                               floatX *normed,
                                               const floatX *inp1,
                                               const floatX *inp2,
                                               const floatX *__restrict__ weight,
                                               float eps) {
  static_assert((GEMMA4_HIDDEN_SIZE % kFloatXPerPack) == 0,
                "decode RMSNorm width must be divisible by Packed128 width");
  __shared__ float s_warp_sums[Threads / WARP_SIZE];
  __shared__ float s_scale;
  const int thread = threadIdx.x;
  const int lane = thread & (WARP_SIZE - 1);
  const int warp = thread / WARP_SIZE;

  float sum_sq = 0.0f;
  RmsnormPack values;
  for (int pack = thread; pack < kDecodePacks; pack += Threads) {
    RmsnormPack a = load128cs(inp1 + pack * kFloatXPerPack);
    RmsnormPack b = load128cs(inp2 + pack * kFloatXPerPack);
    values = gemma4_bf16_pack_add(a, b);
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
    store128(residual + pack * kFloatXPerPack, values);
  }

  float block_sum = gemma4_block_reduce_sum<Threads>(
      sum_sq, s_warp_sums, thread, lane, warp);
  if (thread == 0) {
    s_scale = rsqrtf(block_sum / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();
  float scale = s_scale;
  for (int pack = thread; pack < kDecodePacks; pack += Threads) {
    RmsnormPack gamma = load128g(weight + pack * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
    store128(normed + pack * kFloatXPerPack, result);
  }
}

template <int Threads>
__global__ __launch_bounds__(
    Threads, GEMMA4_HIDDEN_PREFILL_MIN_BLOCKS_PER_SM) void
gemma4_residual_add_rmsnorm_bf16_hidden_prefill_kernel(
    floatX *residual,
    floatX *normed,
    const floatX *inp1,
    const floatX *inp2,
    const floatX *__restrict__ weight,
    float eps) {
  static_assert((GEMMA4_HIDDEN_SIZE % kFloatXPerPack) == 0,
                "hidden RMSNorm width must be divisible by Packed128 width");
  static_assert(Threads == kDecodePacks,
                "hidden prefill keeps one residual pack per thread");
  __shared__ float s_warp_sums[Threads / WARP_SIZE];
  __shared__ float s_scale;

  const int row = blockIdx.x;
  const int offset = row * GEMMA4_HIDDEN_SIZE;
  inp1 += offset;
  inp2 += offset;
  residual += offset;
  normed += offset;

  const int thread = threadIdx.x;
  const int lane = thread & (WARP_SIZE - 1);
  const int warp = thread / WARP_SIZE;
  const int pack = thread;
  RmsnormPack a = load128g(inp1 + pack * kFloatXPerPack);
  RmsnormPack b = load128g(inp2 + pack * kFloatXPerPack);
  RmsnormPack values = gemma4_bf16_pack_add(a, b);
  float sum_sq = 0.0f;
  gemma4_bf16_pack_accumulate_square(values, sum_sq);
  store128(residual + pack * kFloatXPerPack, values);

  float block_sum = gemma4_block_reduce_sum<Threads>(
      sum_sq, s_warp_sums, thread, lane, warp);
  if (thread == 0) {
    s_scale = rsqrtf(block_sum / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();
  RmsnormPack gamma = load128g(weight + pack * kFloatXPerPack);
  RmsnormPack result = gemma4_bf16_pack_apply_rmsnorm(values, gamma, s_scale);
  store128(normed + pack * kFloatXPerPack, result);
}

__global__ __launch_bounds__(kRmsnormThreads) void
gemma4_rmsnorm_bf16_shared_kernel(
    floatX *out,
    const floatX *inp,
    const floatX *__restrict__ weight,
    int rows,
    int width,
    int packs_per_row,
    float eps) {
  extern __shared__ int4 shared_packs[];
  auto *shared = reinterpret_cast<RmsnormPack *>(shared_packs);
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

  float sum_sq = 0.0f;
  if (threadIdx.x < packs_per_row) {
    int pack = threadIdx.x;
    gemma4_async_load_input_pack(s_in + pack,
                                 inp + pack * kFloatXPerPack);
    __pipeline_commit();
    while (pack < packs_per_row) {
      __pipeline_wait_prior(0);
      RmsnormPack values = s_in[pack];
      const int next_pack = pack + WARP_SIZE;
      if (next_pack < packs_per_row) {
        gemma4_async_load_input_pack(s_in + next_pack,
                                     inp + next_pack * kFloatXPerPack);
        __pipeline_commit();
      }
      gemma4_bf16_pack_accumulate_square(values, sum_sq);
      pack = next_pack;
    }
  }
  sum_sq = warp_reduce_sum(sum_sq);
  float scale = rsqrtf(sum_sq / static_cast<float>(width) + eps);

  for (int pack = threadIdx.x; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = s_in[pack];
    RmsnormPack result =
        gemma4_bf16_pack_apply_rmsnorm(values, s_weight[pack], scale);
    store128cs(out + pack * kFloatXPerPack, result);
  }
}

__global__ __launch_bounds__(kRmsnormThreads) void
gemma4_rmsnorm_bf16_direct_weight_kernel(
    floatX *out,
    const floatX *inp,
    const floatX *__restrict__ weight,
    int rows,
    int width,
    int packs_per_row,
    float eps) {
  extern __shared__ int4 shared_packs[];
  auto *s_in = reinterpret_cast<RmsnormPack *>(shared_packs) +
               threadIdx.y * packs_per_row;

  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) {
    return;
  }

  inp += row * width;
  out += row * width;

  float sum_sq = 0.0f;
  if (threadIdx.x < packs_per_row) {
    int pack = threadIdx.x;
    gemma4_async_load_input_pack(s_in + pack,
                                 inp + pack * kFloatXPerPack);
    __pipeline_commit();
    while (pack < packs_per_row) {
      __pipeline_wait_prior(0);
      RmsnormPack values = s_in[pack];
      const int next_pack = pack + WARP_SIZE;
      if (next_pack < packs_per_row) {
        gemma4_async_load_input_pack(s_in + next_pack,
                                     inp + next_pack * kFloatXPerPack);
        __pipeline_commit();
      }
      gemma4_bf16_pack_accumulate_square(values, sum_sq);
      pack = next_pack;
    }
  }
  sum_sq = warp_reduce_sum(sum_sq);
  float scale = rsqrtf(sum_sq / static_cast<float>(width) + eps);

  for (int pack = threadIdx.x; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = s_in[pack];
    RmsnormPack gamma = load128g(weight + pack * kFloatXPerPack);
    RmsnormPack result = gemma4_bf16_pack_apply_rmsnorm(values, gamma, scale);
    store128cs(out + pack * kFloatXPerPack, result);
  }
}

__global__ __launch_bounds__(kRmsnormThreads) void
gemma4_rmsnorm_scale_free_bf16_shared_kernel(
    floatX *out,
    const floatX *inp,
    int rows,
    int width,
    int packs_per_row,
    float eps) {
  extern __shared__ int4 shared_packs[];
  auto *s_in = reinterpret_cast<RmsnormPack *>(shared_packs) +
               threadIdx.y * packs_per_row;

  const int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= rows) {
    return;
  }

  inp += row * width;
  out += row * width;

  float sum_sq = 0.0f;
  if (threadIdx.x < packs_per_row) {
    int pack = threadIdx.x;
    gemma4_async_load_input_pack(s_in + pack,
                                 inp + pack * kFloatXPerPack);
    __pipeline_commit();
    while (pack < packs_per_row) {
      __pipeline_wait_prior(0);
      RmsnormPack values = s_in[pack];
      const int next_pack = pack + WARP_SIZE;
      if (next_pack < packs_per_row) {
        gemma4_async_load_input_pack(s_in + next_pack,
                                     inp + next_pack * kFloatXPerPack);
        __pipeline_commit();
      }
      gemma4_bf16_pack_accumulate_square(values, sum_sq);
      pack = next_pack;
    }
  }
  sum_sq = warp_reduce_sum(sum_sq);
  float scale = rsqrtf(sum_sq / static_cast<float>(width) + eps);

  for (int pack = threadIdx.x; pack < packs_per_row; pack += WARP_SIZE) {
    RmsnormPack values = s_in[pack];
    RmsnormPack result = gemma4_bf16_pack_apply_scale(values, scale);
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

bool gemma4_residual_add_rmsnorm_args_valid(floatX *residual,
                                            const floatX *inp2,
                                            int rows,
                                            int width) {
  if (width != GEMMA4_HIDDEN_SIZE) {
    return false;
  }
  if (rows == 0) {
    return true;
  }
  return residual != nullptr && inp2 != nullptr &&
         is_aligned_16(residual) && is_aligned_16(inp2);
}

cudaError_t gemma4_rmsnorm_bf16_impl(floatX *out,
                                     const floatX *inp,
                                     const floatX *__restrict__ weight,
                                     int rows,
                                     int width,
                                     float eps,
                                     cudaStream_t stream) {
  if (!gemma4_rmsnorm_args_valid(out, inp, rows, width) ||
      (rows != 0 && (weight == nullptr || !is_aligned_16(weight)))) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }
  if (rows == 1 && width == GEMMA4_HIDDEN_SIZE) {
    gemma4_rmsnorm_bf16_decode_kernel<kDecodeRmsnormThreads>
        <<<1, kDecodeRmsnormThreads, 0, stream>>>(out, inp, weight, eps);
    return cudaGetLastError();
  }

  const int packs_per_row = width / kFloatXPerPack;
  const int grid = div_up(rows, kRmsnormRowsPerBlock);
  if (width <= GEMMA4_GLOBAL_HEAD_DIM) {
    const size_t smem = static_cast<size_t>(kRmsnormRowsPerBlock) *
                        packs_per_row * sizeof(RmsnormPack);
    gemma4_rmsnorm_bf16_direct_weight_kernel<<<
        grid, dim3(WARP_SIZE, kRmsnormRowsPerBlock), smem, stream>>>(
        out, inp, weight, rows, width, packs_per_row, eps);
    return cudaGetLastError();
  }

  const size_t smem = static_cast<size_t>(1 + kRmsnormRowsPerBlock) *
                      packs_per_row * sizeof(RmsnormPack);
  gemma4_rmsnorm_bf16_shared_kernel<<<
      grid, dim3(WARP_SIZE, kRmsnormRowsPerBlock), smem, stream>>>(
      out, inp, weight, rows, width, packs_per_row, eps);
  return cudaGetLastError();
}

cudaError_t gemma4_rmsnorm_scale_free_bf16_impl(floatX *out,
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
  const int packs_per_row = width / kFloatXPerPack;
  const int grid = div_up(rows, kRmsnormRowsPerBlock);
  const size_t smem = static_cast<size_t>(kRmsnormRowsPerBlock) *
                      packs_per_row * sizeof(RmsnormPack);

  gemma4_rmsnorm_scale_free_bf16_shared_kernel<<<
      grid, dim3(WARP_SIZE, kRmsnormRowsPerBlock), smem, stream>>>(
      out, inp, rows, width, packs_per_row, eps);
  return cudaGetLastError();
}

cudaError_t gemma4_residual_add_rmsnorm_bf16_impl(
    floatX *residual,
    floatX *normed,
    const floatX *inp1,
    const floatX *inp2,
    const floatX *__restrict__ weight,
    int rows,
    int width,
    float eps,
    cudaStream_t stream) {
  if (!gemma4_rmsnorm_args_valid(normed, inp1, rows, width) ||
      (rows != 0 && (weight == nullptr || !is_aligned_16(weight)))) {
    return cudaErrorInvalidValue;
  }
  if (!gemma4_residual_add_rmsnorm_args_valid(residual, inp2, rows, width)) {
    return cudaErrorInvalidValue;
  }
  if (rows == 0) {
    return cudaSuccess;
  }
  if (rows == 1) {
    gemma4_residual_add_rmsnorm_bf16_decode_kernel<kDecodeFusedThreads>
        <<<1, kDecodeFusedThreads, 0, stream>>>(
        residual, normed, inp1, inp2, weight, eps);
    return cudaGetLastError();
  }

  gemma4_residual_add_rmsnorm_bf16_hidden_prefill_kernel<
      kHiddenPrefillFusedThreads><<<rows, kHiddenPrefillFusedThreads, 0,
                                    stream>>>(
      residual, normed, inp1, inp2, weight, eps);
  return cudaGetLastError();
}

}  // namespace

cudaError_t gemma4_rmsnorm_bf16(floatX *out,
                                const floatX *inp,
                                const floatX *__restrict__ weight,
                                int rows,
                                int width,
                                float eps,
                                cudaStream_t stream) {
  return gemma4_rmsnorm_bf16_impl(
      out, inp, weight, rows, width, eps, stream);
}

cudaError_t gemma4_rmsnorm_scale_free_bf16(floatX *out,
                                           const floatX *inp,
                                           int rows,
                                           int width,
                                           float eps,
                                           cudaStream_t stream) {
  return gemma4_rmsnorm_scale_free_bf16_impl(
      out, inp, rows, width, eps, stream);
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
                                             const floatX *inp1,
                                             const floatX *inp2,
                                             const floatX *__restrict__ weight,
                                             int rows,
                                             int width,
                                             float eps,
                                             cudaStream_t stream) {
  return gemma4_residual_add_rmsnorm_bf16_impl(
      residual, normed, inp1, inp2, weight, rows, width, eps, stream);
}
