// Gemma RMSNorm: y = x * rsqrt(mean(x * x) + eps) * weight.

#include "gemma4_rmsnorm.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

#include <math.h>

struct Gemma4RmsnormRowArgs {
  __nv_bfloat16 *out = nullptr;
  __nv_bfloat16 *residual_out = nullptr;
  const __nv_bfloat16 *input = nullptr;
  const __nv_bfloat16 *residual_in = nullptr;
  const __nv_bfloat16 *weight = nullptr;
  int rows = 0;
  int width = 0;
  float eps = 0.0f;
  bool add_residual = false;
};

namespace {

constexpr int kMaxRmsnormThreads = 512;
constexpr int kMaxRmsnormWarpSums = 16;
constexpr int kResidualAddThreads = 256;

// Builds the row argument bundle used by host launchers and device entries.
__host__ __device__ constexpr Gemma4RmsnormRowArgs gemma4_rmsnorm_make_args(
    __nv_bfloat16 *out,
    __nv_bfloat16 *residual_out,
    const __nv_bfloat16 *input,
    const __nv_bfloat16 *residual_in,
    const __nv_bfloat16 *weight,
    int rows,
    int width,
    float eps,
    bool add_residual) {
  Gemma4RmsnormRowArgs args = {};
  args.out = out;
  args.residual_out = residual_out;
  args.input = input;
  args.residual_in = residual_in;
  args.weight = weight;
  args.rows = rows;
  args.width = width;
  args.eps = eps;
  args.add_residual = add_residual;
  return args;
}

// Processes one row with caller-owned block threads and scratch, optionally adding a residual.
__device__ void gemma4_rmsnorm_row_bf16(
    const Gemma4RmsnormRowArgs &args,
    int row,
    Bf16Packed128 *__restrict__ cached_row,
    float *__restrict__ warp_sums,
    float &scale,
    int thread_idx,
    int thread_count) {
  const int packs_per_row = args.width / kBf16Packed128Elements;
  const int row_offset = row * args.width;

  // Cache the row while accumulating sum_sq so writeback avoids another global read.
  float sum_sq = 0.0f;
  for (int pack = thread_idx; pack < packs_per_row; pack += thread_count) {
    const int offset = row_offset + pack * kBf16Packed128Elements;
    Bf16Packed128 values =
        Bf16Packed128{*reinterpret_cast<const int4 *>(args.input + offset)};
    if (args.add_residual) {
      const Bf16Packed128 residual = Bf16Packed128{
          *reinterpret_cast<const int4 *>(args.residual_in + offset)};
      values = gemma4_bf16_pack_add(values, residual);
      *reinterpret_cast<int4 *>(args.residual_out + offset) = values.bits();
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

  for (int pack = thread_idx; pack < packs_per_row; pack += thread_count) {
    const int offset = row_offset + pack * kBf16Packed128Elements;
    const int weight_offset = pack * kBf16Packed128Elements;
    const Bf16Packed128 values = cached_row[pack];
    const Bf16Packed128 gamma = Bf16Packed128{
        *reinterpret_cast<const int4 *>(args.weight + weight_offset)};
    const Bf16Packed128 result =
        gemma4_bf16_pack_apply_scale_weight(values, gamma, scale);
    *reinterpret_cast<int4 *>(args.out + offset) = result.bits();
  }
}

}  // namespace

// Computes the RMSNorm scale for values owned collectively by one warp.
extern "C" __device__ float gemma4_rmsnorm_warp_scale_f32_device(
    const float *__restrict__ values,
    int values_per_lane,
    int width,
    float eps) {
  float sum_sq = 0.0f;
  for (int i = 0; i < values_per_lane; ++i) {
    sum_sq = fmaf(values[i], values[i], sum_sq);
  }
  sum_sq = warp_reduce_sum(sum_sq);
  return rsqrtf(sum_sq / float(width) + eps);
}

// Host-launched RMSNorm kernel; one CTA owns one row for all row counts.
__global__ __launch_bounds__(kMaxRmsnormThreads, 1) void
gemma4_rmsnorm_bf16_kernel(Gemma4RmsnormRowArgs args) {
  extern __shared__ int4 shared_storage[];
  auto *cached_row = reinterpret_cast<Bf16Packed128 *>(shared_storage);
  const int packs_per_row = args.width / kBf16Packed128Elements;
  auto *warp_sums = reinterpret_cast<float *>(cached_row + packs_per_row);
  __shared__ float scale;

  gemma4_rmsnorm_row_bf16(
      args, static_cast<int>(blockIdx.x), cached_row, warp_sums, scale,
      int(threadIdx.x), int(blockDim.x));
}

namespace {

// Adds two contiguous BF16 tensors using one 128-bit pack per thread.
__global__ __launch_bounds__(kResidualAddThreads) void
gemma4_residual_add_bf16_kernel(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *inp1,
    const __nv_bfloat16 *inp2,
    int packs) {
  const int pack = blockIdx.x * blockDim.x + threadIdx.x;
  if (pack >= packs) {
    return;
  }

  const int offset = pack * kBf16Packed128Elements;
  const Bf16Packed128 a =
      Bf16Packed128{*reinterpret_cast<const int4 *>(inp1 + offset)};
  const Bf16Packed128 b =
      Bf16Packed128{*reinterpret_cast<const int4 *>(inp2 + offset)};
  const Bf16Packed128 result = gemma4_bf16_pack_add(a, b);
  *reinterpret_cast<int4 *>(out + offset) = result.bits();
}

// Launches one CTA per RMSNorm row while preserving the public host APIs.
cudaError_t gemma4_rmsnorm_launch_bf16(const Gemma4RmsnormRowArgs &args,
                                       cudaStream_t stream) {
  const int packs_per_row = args.width / kBf16Packed128Elements;
  const size_t smem =
      static_cast<size_t>(packs_per_row) * sizeof(Bf16Packed128) +
      kMaxRmsnormWarpSums * sizeof(float);
  const dim3 grid_dim(args.rows);
  const dim3 block_dim(kMaxRmsnormThreads);
  gemma4_rmsnorm_bf16_kernel<<<grid_dim, block_dim, smem, stream>>>(args);
  return cudaGetLastError();
}

}  // namespace

// Exported device entry used by the decode megakernel with caller-owned 512-thread
// block scratch, keeping RMSNorm logic out of headers.
extern "C" __device__ void gemma4_rmsnorm_hidden_row_512_bf16_device(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ in,
    const __nv_bfloat16 *__restrict__ weight,
    float eps,
    Bf16Packed128 *__restrict__ cached_row,
    float *__restrict__ warp_sums,
    float *__restrict__ scale,
    int thread_idx) {
  const Gemma4RmsnormRowArgs args = gemma4_rmsnorm_make_args(
      out, nullptr, in, nullptr, weight, 1, GEMMA4_HIDDEN_SIZE, eps,
      false);
  gemma4_rmsnorm_row_bf16(
      args, 0, cached_row, warp_sums, *scale, thread_idx, kMaxRmsnormThreads);
}

// Launches learned-weight RMSNorm for row-major BF16 tensors.
cudaError_t gemma4_rmsnorm_bf16(__nv_bfloat16 *out,
                                const __nv_bfloat16 *inp,
                                const __nv_bfloat16 *__restrict__ weight,
                                int rows,
                                int width,
                                float eps,
                                cudaStream_t stream) {
  if ((width % kBf16Packed128Elements) != 0) {
    return cudaErrorInvalidValue;
  }

  const Gemma4RmsnormRowArgs args = gemma4_rmsnorm_make_args(
      out, nullptr, inp, nullptr, weight, rows, width, eps,
      false);
  return gemma4_rmsnorm_launch_bf16(args, stream);
}

// Launches a standalone row-major BF16 residual add.
cudaError_t gemma4_residual_add_bf16(__nv_bfloat16 *out,
                                     const __nv_bfloat16 *inp1,
                                     const __nv_bfloat16 *inp2,
                                     int count,
                                     cudaStream_t stream) {
  if ((count % kBf16Packed128Elements) != 0) {
    return cudaErrorInvalidValue;
  }

  const int packs = count / kBf16Packed128Elements;
  const dim3 grid_dim(div_up(packs, kResidualAddThreads));
  const dim3 block_dim(kResidualAddThreads);
  gemma4_residual_add_bf16_kernel<<<grid_dim, block_dim, 0, stream>>>(
      out, inp1, inp2, packs);
  return cudaGetLastError();
}

// Launches fused residual add plus learned-weight RMSNorm for hidden rows.
cudaError_t gemma4_residual_add_rmsnorm_bf16(__nv_bfloat16 *residual,
                                             __nv_bfloat16 *normed,
                                             const __nv_bfloat16 *inp1,
                                             const __nv_bfloat16 *inp2,
                                             const __nv_bfloat16 *__restrict__ weight,
                                             int rows,
                                             int width,
                                             float eps,
                                             cudaStream_t stream) {
  if (width != GEMMA4_HIDDEN_SIZE) {
    return cudaErrorInvalidValue;
  }

  const Gemma4RmsnormRowArgs args = gemma4_rmsnorm_make_args(
      normed, residual, inp1, inp2, weight, rows, width, eps,
      true);
  return gemma4_rmsnorm_launch_bf16(args, stream);
}
