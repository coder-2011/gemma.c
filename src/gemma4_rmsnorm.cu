// Gemma RMSNorm: y = x * rsqrt(mean(x * x) + eps) * weight.

#include "gemma4_rmsnorm.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

#include <cute/layout.hpp>

#include <math.h>
#include <stdint.h>

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
constexpr int kMaxRmsnormWarps = kMaxRmsnormThreads / WARP_SIZE;
constexpr int kResidualAddThreads = 256;

// Maps [row, pack] to a BF16 element offset in a row-major tensor.
__device__ inline auto gemma4_rmsnorm_gmem_pack_layout(
    int rows,
    int width,
    int packs_per_row) {
  return cute::make_layout(
      cute::make_shape(rows, packs_per_row),
      cute::make_stride(width, sizeof(Bf16Packed128) / sizeof(floatX)));
}

// Maps one contiguous pack index to its BF16 element offset.
__device__ inline auto gemma4_rmsnorm_vector_pack_layout(int packs) {
  return cute::make_layout(
      cute::make_shape(packs),
      cute::make_stride(sizeof(Bf16Packed128) / sizeof(floatX)));
}

// Returns the dynamic shared-memory bytes needed by one RMSNorm row block.
size_t gemma4_rmsnorm_shared_bytes(int packs_per_row) {
  return static_cast<size_t>(packs_per_row) * sizeof(Bf16Packed128) +
         kMaxRmsnormWarps * sizeof(float);
}

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

// Reduces one row's sum of squares across the active CTA threads.
__device__ inline float gemma4_rmsnorm_reduce_sum(
    float sum_sq,
    float *__restrict__ warp_sums,
    int thread_idx,
    int thread_count) {
  const int lane = thread_idx & (WARP_SIZE - 1);
  const int warp = thread_idx / WARP_SIZE;
  const int warps = div_up(thread_count, WARP_SIZE);

  sum_sq = warp_reduce_sum(sum_sq);
  if (lane == 0) {
    warp_sums[warp] = sum_sq;
  }
  __syncthreads();

  sum_sq = thread_idx < warps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    sum_sq = warp_reduce_sum(sum_sq);
  }
  return sum_sq;
}

// Applies either learned or scale-free RMSNorm to a cached row pack.
__device__ inline Bf16Packed128 gemma4_rmsnorm_apply_pack(
    const Gemma4RmsnormRowArgs &args,
    Bf16Packed128 values,
    int pack,
    float scale) {
  if (args.mode == kScaleFreeRmsnorm) {
    return gemma4_bf16_pack_apply_scale(values, scale);
  }

  const auto pack_layout = gemma4_rmsnorm_vector_pack_layout(
      args.packs_per_row);
  const Bf16Packed128 gamma = load128g(args.weight + pack_layout(pack));
  return gemma4_bf16_pack_apply_scale_weight(values, gamma, scale);
}

// Builds one RMSNorm row, using CuTe layouts for row and pack addressing.
__device__ void gemma4_rmsnorm_row_bf16(
    const Gemma4RmsnormRowArgs &args,
    int row,
    Bf16Packed128 *__restrict__ cached_row,
    float *__restrict__ warp_sums,
    float &scale,
    int thread_idx,
    int thread_count) {
  const auto row_layout = gemma4_rmsnorm_gmem_pack_layout(
      args.rows, args.width, args.packs_per_row);

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

  const float total = gemma4_rmsnorm_reduce_sum(
      sum_sq, warp_sums, thread_idx, thread_count);
  if (thread_idx == 0) {
    scale = rsqrtf(total / static_cast<float>(args.width) + args.eps);
  }
  __syncthreads();

  for (int pack = thread_idx; pack < args.packs_per_row; pack += thread_count) {
    const int offset = row_layout(row, pack);
    const Bf16Packed128 values = cached_row[pack];
    const Bf16Packed128 result =
        gemma4_rmsnorm_apply_pack(args, values, pack, scale);
    store128(args.out + offset, result);
  }
}

}  // namespace

// Thin decode kernel wrapper around the decode row device function.
__global__ __launch_bounds__(kMaxRmsnormThreads, 1) void
gemma4_rmsnorm_decode_bf16_kernel(Gemma4RmsnormRowArgs args) {
  extern __shared__ int4 shared_storage[];
  auto *cached_row = reinterpret_cast<Bf16Packed128 *>(shared_storage);
  auto *warp_sums = reinterpret_cast<float *>(cached_row + args.packs_per_row);
  __shared__ float scale;

  gemma4_rmsnorm_row_bf16(
      args, 0, cached_row, warp_sums, scale, int(threadIdx.x), int(blockDim.x));
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

  const auto layout = gemma4_rmsnorm_vector_pack_layout(packs);
  const int offset = layout(pack);
  const Bf16Packed128 a = load128g(inp1 + offset);
  const Bf16Packed128 b = load128g(inp2 + offset);
  const Bf16Packed128 result = gemma4_bf16_pack_add(a, b);
  store128(out + offset, result);
}

// Validates common row-shaped RMSNorm launcher arguments.
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
  if (out == nullptr || inp == nullptr) {
    return false;
  }
  return (width % (sizeof(Bf16Packed128) / sizeof(floatX))) == 0;
}

// Validates fused residual-add RMSNorm launcher arguments.
bool gemma4_residual_add_rmsnorm_args_valid(const floatX *residual,
                                            const floatX *inp2,
                                            const floatX *weight,
                                            int rows,
                                            int width) {
  if (rows < 0 || width != GEMMA4_HIDDEN_SIZE) {
    return false;
  }
  if (rows == 0) {
    return true;
  }
  return residual != nullptr && inp2 != nullptr && weight != nullptr;
}

// Selects decode or prefill by row count while preserving public APIs.
cudaError_t gemma4_rmsnorm_launch_bf16(const Gemma4RmsnormRowArgs &args,
                                       cudaStream_t stream) {
  if (args.rows == 0) {
    return cudaSuccess;
  }

  int threads = div_up(args.packs_per_row, WARP_SIZE) * WARP_SIZE;
  if (threads < WARP_SIZE) {
    threads = WARP_SIZE;
  }
  if (threads > kMaxRmsnormThreads) {
    threads = kMaxRmsnormThreads;
  }

  const size_t smem = gemma4_rmsnorm_shared_bytes(args.packs_per_row);
  if (args.rows == 1) {
    gemma4_rmsnorm_decode_bf16_kernel<<<1, threads, smem, stream>>>(args);
  } else {
    gemma4_rmsnorm_prefill_bf16_kernel<<<args.rows, threads, smem, stream>>>(
        args);
  }
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
  Gemma4RmsnormRowArgs args = gemma4_rmsnorm_make_args(
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
  if (!gemma4_rmsnorm_args_valid(out, inp, rows, width) ||
      (rows > 0 && weight == nullptr)) {
    return cudaErrorInvalidValue;
  }

  Gemma4RmsnormRowArgs args = gemma4_rmsnorm_make_args(
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
  if (!gemma4_rmsnorm_args_valid(out, inp, rows, width)) {
    return cudaErrorInvalidValue;
  }

  Gemma4RmsnormRowArgs args = gemma4_rmsnorm_make_args(
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
  if (count < 0) {
    return cudaErrorInvalidValue;
  }
  if (count == 0) {
    return cudaSuccess;
  }
  if (out == nullptr || inp1 == nullptr || inp2 == nullptr) {
    return cudaErrorInvalidValue;
  }
  if ((count % (sizeof(Bf16Packed128) / sizeof(floatX))) != 0) {
    return cudaErrorInvalidValue;
  }

  const int packs = count / (sizeof(Bf16Packed128) / sizeof(floatX));
  const int grid = div_up(packs, kResidualAddThreads);
  gemma4_residual_add_bf16_kernel<<<grid, kResidualAddThreads, 0, stream>>>(
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
  if (!gemma4_rmsnorm_args_valid(normed, inp1, rows, width)) {
    return cudaErrorInvalidValue;
  }
  if (!gemma4_residual_add_rmsnorm_args_valid(
          residual, inp2, weight, rows, width)) {
    return cudaErrorInvalidValue;
  }

  Gemma4RmsnormRowArgs args = gemma4_rmsnorm_make_args(
      normed, residual, inp1, inp2, weight, rows, width, eps,
      kResidualAddLearnedRmsnorm);
  return gemma4_rmsnorm_launch_bf16(args, stream);
}
