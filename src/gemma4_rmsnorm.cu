// Gemma RMSNorm: y = x * rsqrt(mean(x * x) + eps) * weight.

#include "gemma4_rmsnorm.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

#include <cute/layout.hpp>

#include <math.h>

enum Gemma4RmsnormMode {
  kLearnedWeightRmsnorm = 0,
  kScaleFreeRmsnorm = 1,
  kResidualAddLearnedRmsnorm = 2,
};

struct Gemma4RmsnormRowArgs {
  floatX *out = nullptr;
  floatX *residual_out = nullptr;
  const floatX *input = nullptr;
  const floatX *residual_in = nullptr;
  const floatX *weight = nullptr;
  int rows = 0;
  int width = 0;
  int packs_per_row = 0;
  float eps = 0.0f;
  Gemma4RmsnormMode mode = kLearnedWeightRmsnorm;
};

namespace {

constexpr int kMaxRmsnormThreads = 512;
constexpr int kMaxRmsnormWarpSums = 16;
constexpr int kResidualAddThreads = 256;

// Builds a row argument object shared by decode and prefill launch wrappers.
__host__ __device__ inline Gemma4RmsnormRowArgs gemma4_rmsnorm_make_args(
    floatX *out,
    floatX *residual_out,
    const floatX *input,
    const floatX *residual_in,
    const floatX *weight,
    int rows,
    int width,
    float eps,
    Gemma4RmsnormMode mode) {
  Gemma4RmsnormRowArgs args = {};
  args.out = out;
  args.residual_out = residual_out;
  args.input = input;
  args.residual_in = residual_in;
  args.weight = weight;
  args.rows = rows;
  args.width = width;
  args.packs_per_row = width / (sizeof(Bf16Packed128) / sizeof(floatX));
  args.eps = eps;
  args.mode = mode;
  return args;
}

// Processes one row with caller-owned scratch and threading.
__device__ void gemma4_rmsnorm_row_bf16(
    const Gemma4RmsnormRowArgs &args,
    int row,
    Bf16Packed128 *__restrict__ cached_row,
    float *__restrict__ warp_sums,
    float &scale,
    int thread_idx,
    int thread_count) {
  const auto row_layout = cute::make_layout(
      cute::make_shape(args.rows, args.packs_per_row),
      cute::make_stride(args.width, sizeof(Bf16Packed128) / sizeof(floatX)));

  // Cache the row so the scale pass does not reread input from global memory.
  float sum_sq = 0.0f;
  for (int pack = thread_idx; pack < args.packs_per_row; pack += thread_count) {
    const int offset = row_layout(row, pack);
    Bf16Packed128 values = load128g(args.input + offset);
    if (args.mode == kResidualAddLearnedRmsnorm) {
      const Bf16Packed128 residual = load128g(args.residual_in + offset);
      values = gemma4_bf16_pack_add(values, residual);
      store128(args.residual_out + offset, values);
    }
    cached_row[pack] = values;
    gemma4_bf16_pack_accumulate_square(values, sum_sq);
  }

  const int lane = thread_idx % warpSize;
  const int warp = thread_idx / warpSize;
  const int warps = div_up(thread_count, warpSize);

  // Reduce one partial per warp, then let warp 0 finish the block sum.
  sum_sq = warp_reduce_sum(sum_sq);
  if (lane == 0) {
    warp_sums[warp] = sum_sq;
  }
  __syncthreads();

  sum_sq = thread_idx < warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    sum_sq = warp_reduce_sum(sum_sq);
  }
  if (thread_idx == 0) {
    scale = rsqrtf(sum_sq / args.width + args.eps);
  }
  __syncthreads();

  const auto pack_layout = cute::make_layout(
      cute::make_shape(args.packs_per_row),
      cute::make_stride(sizeof(Bf16Packed128) / sizeof(floatX)));
  for (int pack = thread_idx; pack < args.packs_per_row; pack += thread_count) {
    const int offset = row_layout(row, pack);
    const Bf16Packed128 values = cached_row[pack];
    Bf16Packed128 result;
    if (args.mode == kScaleFreeRmsnorm) {
      result = gemma4_bf16_pack_apply_scale(values, scale);
    } else {
      const Bf16Packed128 gamma = load128g(args.weight + pack_layout(pack));
      result = gemma4_bf16_pack_apply_scale_weight(values, gamma, scale);
    }
    store128(args.out + offset, result);
  }
}

}  // namespace

// Thin decode kernel wrapper with fixed hidden-row scratch.
__global__ __launch_bounds__(kMaxRmsnormThreads, 1) void
gemma4_rmsnorm_decode_bf16_kernel(Gemma4RmsnormRowArgs args) {
  __shared__ Bf16Packed128 cached_row[
      GEMMA4_HIDDEN_SIZE / (sizeof(Bf16Packed128) / sizeof(floatX))];
  __shared__ float warp_sums[kMaxRmsnormWarpSums];
  __shared__ float scale;

  gemma4_rmsnorm_row_bf16(
      args, 0, cached_row, warp_sums, scale, int(threadIdx.x),
      kMaxRmsnormThreads);
}

// Thin prefill kernel wrapper around the prefill row device function.
__global__ __launch_bounds__(kMaxRmsnormThreads, 1) void
gemma4_rmsnorm_prefill_bf16_kernel(Gemma4RmsnormRowArgs args) {
  extern __shared__ int4 shared_storage[];
  auto *cached_row = reinterpret_cast<Bf16Packed128 *>(shared_storage);
  auto *warp_sums = reinterpret_cast<float *>(cached_row + args.packs_per_row);
  __shared__ float scale;

  gemma4_rmsnorm_row_bf16(
      args, static_cast<int>(blockIdx.x), cached_row, warp_sums, scale,
      int(threadIdx.x), int(blockDim.x));
}

namespace {

// Adds two contiguous BF16 tensors using one 128-bit pack per thread.
__global__ __launch_bounds__(kResidualAddThreads) void
gemma4_residual_add_bf16_kernel(
    floatX *out,
    const floatX *inp1,
    const floatX *inp2,
    int packs) {
  const int pack = blockIdx.x * blockDim.x + threadIdx.x;
  if (pack >= packs) {
    return;
  }

  const auto layout = cute::make_layout(
      cute::make_shape(packs),
      cute::make_stride(sizeof(Bf16Packed128) / sizeof(floatX)));
  const int offset = layout(pack);
  const Bf16Packed128 a = load128g(inp1 + offset);
  const Bf16Packed128 b = load128g(inp2 + offset);
  const Bf16Packed128 result = gemma4_bf16_pack_add(a, b);
  store128(out + offset, result);
}

// Selects fixed hidden-row decode or generic prefill while preserving public APIs.
cudaError_t gemma4_rmsnorm_launch_bf16(const Gemma4RmsnormRowArgs &args,
                                       cudaStream_t stream) {
  if (args.rows == 1 && args.width == GEMMA4_HIDDEN_SIZE) {
    gemma4_rmsnorm_decode_bf16_kernel<<<1, kMaxRmsnormThreads, 0, stream>>>(
        args);
    return cudaGetLastError();
  }

  const size_t smem =
      static_cast<size_t>(args.packs_per_row) * sizeof(Bf16Packed128) +
      kMaxRmsnormWarpSums * sizeof(float);
  const dim3 grid_dim(args.rows);
  const dim3 block_dim(kMaxRmsnormThreads);
  gemma4_rmsnorm_prefill_bf16_kernel<<<grid_dim, block_dim, smem, stream>>>(
      args);
  return cudaGetLastError();
}

}  // namespace

// Exported device entry used by the decode megakernel without exposing logic
// through a header-only RMSNorm helper.
extern "C" __device__ void gemma4_rmsnorm_hidden_row_512_bf16_device(
    floatX *__restrict__ out,
    const floatX *__restrict__ in,
    const floatX *__restrict__ weight,
    float eps,
    Bf16Packed128 *__restrict__ cached_row,
    float *__restrict__ warp_sums,
    float *__restrict__ scale,
    int thread_idx) {
  const Gemma4RmsnormRowArgs args = gemma4_rmsnorm_make_args(
      out, nullptr, in, nullptr, weight, 1, GEMMA4_HIDDEN_SIZE, eps,
      kLearnedWeightRmsnorm);
  gemma4_rmsnorm_row_bf16(
      args, 0, cached_row, warp_sums, *scale, thread_idx, kMaxRmsnormThreads);
}

// Launches learned-weight RMSNorm for row-major BF16 tensors.
cudaError_t gemma4_rmsnorm_bf16(floatX *out,
                                const floatX *inp,
                                const floatX *__restrict__ weight,
                                int rows,
                                int width,
                                float eps,
                                cudaStream_t stream) {
  if ((width % (sizeof(Bf16Packed128) / sizeof(floatX))) != 0) {
    return cudaErrorInvalidValue;
  }

  const Gemma4RmsnormRowArgs args = gemma4_rmsnorm_make_args(
      out, nullptr, inp, nullptr, weight, rows, width, eps,
      kLearnedWeightRmsnorm);
  return gemma4_rmsnorm_launch_bf16(args, stream);
}

// Launches scale-free RMSNorm for row-major BF16 tensors.
cudaError_t gemma4_rmsnorm_scale_free_bf16(floatX *out,
                                           const floatX *inp,
                                           int rows,
                                           int width,
                                           float eps,
                                           cudaStream_t stream) {
  if ((width % (sizeof(Bf16Packed128) / sizeof(floatX))) != 0) {
    return cudaErrorInvalidValue;
  }

  const Gemma4RmsnormRowArgs args = gemma4_rmsnorm_make_args(
      out, nullptr, inp, nullptr, nullptr, rows, width, eps,
      kScaleFreeRmsnorm);
  return gemma4_rmsnorm_launch_bf16(args, stream);
}

// Launches a standalone row-major BF16 residual add.
cudaError_t gemma4_residual_add_bf16(floatX *out,
                                     const floatX *inp1,
                                     const floatX *inp2,
                                     int count,
                                     cudaStream_t stream) {
  if ((count % (sizeof(Bf16Packed128) / sizeof(floatX))) != 0) {
    return cudaErrorInvalidValue;
  }

  const int packs = count / (sizeof(Bf16Packed128) / sizeof(floatX));
  const dim3 grid_dim(div_up(packs, kResidualAddThreads));
  const dim3 block_dim(kResidualAddThreads);
  gemma4_residual_add_bf16_kernel<<<grid_dim, block_dim, 0, stream>>>(
      out, inp1, inp2, packs);
  return cudaGetLastError();
}

// Launches fused residual add plus learned-weight RMSNorm for hidden rows.
cudaError_t gemma4_residual_add_rmsnorm_bf16(floatX *residual,
                                             floatX *normed,
                                             const floatX *inp1,
                                             const floatX *inp2,
                                             const floatX *__restrict__ weight,
                                             int rows,
                                             int width,
                                             float eps,
                                             cudaStream_t stream) {
  if (width != GEMMA4_HIDDEN_SIZE) {
    return cudaErrorInvalidValue;
  }

  const Gemma4RmsnormRowArgs args = gemma4_rmsnorm_make_args(
      normed, residual, inp1, inp2, weight, rows, width, eps,
      kResidualAddLearnedRmsnorm);
  return gemma4_rmsnorm_launch_bf16(args, stream);
}
