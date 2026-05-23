#include "gemma4_ffn_decode.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_matmul_device.cuh"

#include <algorithm>

#ifndef GEMMA4_FFN_DECODE_ACT_TILE
#define GEMMA4_FFN_DECODE_ACT_TILE 2
#endif

#ifndef GEMMA4_FFN_DECODE_SWIZZLE_X
#define GEMMA4_FFN_DECODE_SWIZZLE_X 1
#endif

namespace {

constexpr int kFfnThreads = 1024;
constexpr int kFfnWarps = kFfnThreads / WARP_SIZE;
constexpr int kIntermediateTile = 256;
constexpr int kIntermediateTiles =
    GEMMA4_INTERMEDIATE_SIZE / kIntermediateTile;
constexpr int kHiddenPacks = GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
constexpr int kActTile = GEMMA4_FFN_DECODE_ACT_TILE;
constexpr int kSwizzleX = GEMMA4_FFN_DECODE_SWIZZLE_X;
static_assert((GEMMA4_INTERMEDIATE_SIZE % kIntermediateTile) == 0,
              "FFN intermediate width must divide the decode tile width");
static_assert((GEMMA4_HIDDEN_SIZE % kBf16Packed128Elements) == 0,
              "FFN hidden width must divide the 128-bit bf16 pack width");
static_assert((kFfnThreads % WARP_SIZE) == 0,
              "FFN decode thread count must be a whole number of warps");
static_assert(kActTile >= 1, "FFN activation tile must be positive");
static_assert((kIntermediateTile % kActTile) == 0,
              "FFN activation tile must divide the CTA intermediate tile");
static_assert(kSwizzleX == 0 || kSwizzleX == 1,
              "FFN x shared-memory swizzle must be 0 or 1");

using FfnBf16Pack = Bf16Packed128;
using FfnFloatPack = Packed128<float>;

__device__ inline int shared_x_chunk_index(int chunk) {
  if constexpr (kSwizzleX) {
    constexpr int kSwizzleChunks = 8;  // 8 x 128-bit chunks = one 128-byte row.
    const unsigned u = static_cast<unsigned>(chunk);
    const unsigned col = u & (kSwizzleChunks - 1);
    const unsigned row = (u >> 3) & (kSwizzleChunks - 1);
    return static_cast<int>((u & ~(kSwizzleChunks - 1u)) | (col ^ row));
  }
  return chunk;
}

__device__ inline float gelu_tanh(float x) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
  constexpr float kGeluCubic = 0.044715f;
  const float x2 = x * x;
  const float inner = kSqrtTwoOverPi * (x + kGeluCubic * x * x2);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ inline void accumulate_scaled_pack(
    float scale,
    const FfnBf16Pack &pack,
    float (&values)[kBf16Packed128Elements]) {
  const __nv_bfloat162 *pairs = gemma4_bf16_pack_pairs(pack);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    const float2 packed = __bfloat1622float2(pairs[p]);
    values[2 * p] = fmaf(scale, packed.x, values[2 * p]);
    values[2 * p + 1] = fmaf(scale, packed.y, values[2 * p + 1]);
  }
}

__device__ inline void add_accum_pack(
    const float *__restrict__ accum,
    int col,
    float (&values)[kBf16Packed128Elements]) {
  const FfnFloatPack lo = load128(accum + col);
  const FfnFloatPack hi = load128(accum + col + FfnFloatPack::size);
#pragma unroll
  for (int i = 0; i < FfnFloatPack::size; ++i) {
    values[i] += lo[i];
    values[i + FfnFloatPack::size] += hi[i];
  }
}

__device__ inline void store_accum_pack(
    float *__restrict__ accum,
    int col,
    const float (&values)[kBf16Packed128Elements]) {
  FfnFloatPack lo;
  FfnFloatPack hi;
#pragma unroll
  for (int i = 0; i < FfnFloatPack::size; ++i) {
    lo[i] = values[i];
    hi[i] = values[i + FfnFloatPack::size];
  }
  store128(accum + col, lo);
  store128(accum + col + FfnFloatPack::size, hi);
}

__device__ inline void add_residual_store_pack(
    const FfnBf16Pack &residual_pack,
    float (&values)[kBf16Packed128Elements],
    FfnBf16Pack &out_pack,
    float &sum_sq) {
  const __nv_bfloat162 *residual_pairs =
      gemma4_bf16_pack_pairs(residual_pack);
  __nv_bfloat162 *out_pairs = gemma4_bf16_pack_pairs(out_pack);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    const float2 residual = __bfloat1622float2(residual_pairs[p]);
    const float x = values[2 * p] + residual.x;
    const float y = values[2 * p + 1] + residual.y;
    values[2 * p] = x;
    values[2 * p + 1] = y;
    sum_sq = fmaf(x, x, sum_sq);
    sum_sq = fmaf(y, y, sum_sq);
    out_pairs[p] = __floats2bfloat162_rn(x, y);
  }
}

__device__ inline FfnBf16Pack rmsnorm_store_pack(
    const float (&values)[kBf16Packed128Elements],
    const FfnBf16Pack &gamma_pack,
    float scale) {
  const __nv_bfloat162 *gamma_pairs = gemma4_bf16_pack_pairs(gamma_pack);
  FfnBf16Pack out_pack;
  __nv_bfloat162 *out_pairs = gemma4_bf16_pack_pairs(out_pack);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    const float2 gamma = __bfloat1622float2(gamma_pairs[p]);
    out_pairs[p] = __floats2bfloat162_rn(
        values[2 * p] * scale * gamma.x,
        values[2 * p + 1] * scale * gamma.y);
  }
  return out_pack;
}

__device__ inline float block_reduce_sum(float value,
                                         float *__restrict__ warp_sums) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;

  value = warp_reduce_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < kFfnWarps ? warp_sums[lane] : 0.0f;
  if (warp == 0) {
    value = warp_reduce_sum(value);
  }
  return value;
}

__device__ inline void wait_for_reduce_turn(const int *__restrict__ lock,
                                            int turn) {
  if (threadIdx.x == 0) {
    int state = -1;
    do {
      asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                   : "=r"(state)
                   : "l"(lock));
    } while (state != turn);
  }
  __syncthreads();

  int state = -1;
  do {
    asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
                 : "=r"(state)
                 : "l"(lock));
  } while (state != turn);
  __syncthreads();
}

__device__ inline void release_reduce_turn(int *__restrict__ lock) {
  __syncthreads();
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    constexpr int one = 1;
    asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n"
                 :
                 : "l"(lock), "r"(one));
  }
}

__global__ __launch_bounds__(kFfnThreads, 1) void
gemma4_ffn_decode_fused_bf16_kernel(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ x,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps) {
  __shared__ FfnBf16Pack s_x[kHiddenPacks];
  __shared__ float s_matmul_warp_sums[2][kActTile][kFfnWarps];
  __shared__ float s_rms_warp_sums[kFfnWarps];
  __shared__ float s_act[kActTile];
  __shared__ float s_scale;

  for (int pack = threadIdx.x; pack < kHiddenPacks; pack += kFfnThreads) {
    s_x[shared_x_chunk_index(pack)] =
        load128g(x + pack * kBf16Packed128Elements);
  }
  __syncthreads();

  float partial[kBf16Packed128Elements] = {};
  const int intermediate_begin =
      static_cast<int>(blockIdx.x) * kIntermediateTile;
  const int hidden_pack = threadIdx.x;
  const bool owns_hidden_pack = hidden_pack < kHiddenPacks;
  const int hidden_col = hidden_pack * kBf16Packed128Elements;

  for (int local_col = 0; local_col < kIntermediateTile;
       local_col += kActTile) {
    float gate[kActTile] = {};
    float up[kActTile] = {};
    const int gate_col0 = intermediate_begin + local_col;
    const int up_col0 = GEMMA4_INTERMEDIATE_SIZE + gate_col0;

    gemma4_matmul_device::dot_cols_pair_shared_x_reduce<
        GEMMA4_HIDDEN_SIZE, kActTile, kFfnThreads, (kSwizzleX != 0)>(
        s_x, w_gate_up_col_major, w_gate_up_col_major, gate_col0, up_col0,
        threadIdx.x, s_matmul_warp_sums[0], s_matmul_warp_sums[1], gate, up);

    if (threadIdx.x == 0) {
#pragma unroll
      for (int t = 0; t < kActTile; ++t) {
        s_act[t] = gate[t] * gelu_tanh(up[t]);
      }
    }
    __syncthreads();

    if (owns_hidden_pack) {
#pragma unroll
      for (int t = 0; t < kActTile; ++t) {
        const int intermediate_col = intermediate_begin + local_col + t;
        const floatX *down_row =
            w_down_row_major +
            static_cast<int64_t>(intermediate_col) * GEMMA4_HIDDEN_SIZE;
        const FfnBf16Pack down_pack = load128cs(down_row + hidden_col);
        accumulate_scaled_pack(s_act[t], down_pack, partial);
      }
    }
    __syncthreads();
  }

  const bool first = blockIdx.x == 0;
  const bool last = blockIdx.x == kIntermediateTiles - 1;
  wait_for_reduce_turn(&scratch->lock, static_cast<int>(blockIdx.x));

  if (owns_hidden_pack) {
    if (!first) {
      add_accum_pack(scratch->accum, hidden_col, partial);
    }
    if (!last) {
      store_accum_pack(scratch->accum, hidden_col, partial);
    }
  }

  if (!last) {
    release_reduce_turn(&scratch->lock);
    return;
  }

  float sum_sq = 0.0f;
  if (owns_hidden_pack) {
    FfnBf16Pack residual_pack = load128g(residual + hidden_col);
    FfnBf16Pack residual_out_pack;
    add_residual_store_pack(residual_pack, partial, residual_out_pack, sum_sq);
    store128(residual_out + hidden_col, residual_out_pack);
  }

  const float total = block_reduce_sum(sum_sq, s_rms_warp_sums);
  if (threadIdx.x == 0) {
    s_scale =
        rsqrtf(total / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();

  if (owns_hidden_pack) {
    const FfnBf16Pack gamma_pack = load128g(rms_weight + hidden_col);
    const FfnBf16Pack normed_pack =
        rmsnorm_store_pack(partial, gamma_pack, s_scale);
    store128(normed_out + hidden_col, normed_pack);
  }
}

bool ffn_decode_args_valid(const floatX *residual_out,
                           const floatX *normed_out,
                           const floatX *x,
                           const floatX *residual,
                           const floatX *rms_weight,
                           const floatX *w_gate_up_col_major,
                           const floatX *w_down_row_major,
                           const Gemma4FfnDecodeScratch *scratch) {
  return residual_out != nullptr && normed_out != nullptr && x != nullptr &&
         residual != nullptr && rms_weight != nullptr &&
         w_gate_up_col_major != nullptr && w_down_row_major != nullptr &&
         scratch != nullptr && is_aligned_16(residual_out) &&
         is_aligned_16(normed_out) && is_aligned_16(x) &&
         is_aligned_16(residual) && is_aligned_16(rms_weight) &&
         is_aligned_16(w_gate_up_col_major) &&
         is_aligned_16(w_down_row_major) && is_aligned_16(scratch);
}

cudaError_t check_resident_reduction_supported() {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  int sm_count = 0;
  status = cudaDeviceGetAttribute(
      &sm_count, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess) {
    return status;
  }

  int active_blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, gemma4_ffn_decode_fused_bf16_kernel,
      kFfnThreads, 0);
  if (status != cudaSuccess) {
    return status;
  }

  if (sm_count * active_blocks_per_sm < kIntermediateTiles) {
    return cudaErrorNotSupported;
  }
  return cudaSuccess;
}

}  // namespace

cudaError_t gemma4_ffn_decode_configure_scratch_l2(
    Gemma4FfnDecodeScratch *scratch,
    cudaStream_t stream) {
  if (scratch == nullptr || !is_aligned_16(scratch)) {
    return cudaErrorInvalidValue;
  }

  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  cudaDeviceProp prop = {};
  status = cudaGetDeviceProperties(&prop, device);
  if (status != cudaSuccess) {
    return status;
  }
  if (prop.persistingL2CacheMaxSize == 0 ||
      prop.accessPolicyMaxWindowSize == 0) {
    return cudaSuccess;
  }

  const size_t requested_set_aside =
      std::min(static_cast<size_t>(prop.l2CacheSize * 3 / 4),
               static_cast<size_t>(prop.persistingL2CacheMaxSize));
  status = cudaDeviceSetLimit(
      cudaLimitPersistingL2CacheSize, requested_set_aside);
  if (status != cudaSuccess) {
    return status;
  }

  cudaStreamAttrValue attr = {};
  attr.accessPolicyWindow.base_ptr = scratch;
  attr.accessPolicyWindow.num_bytes =
      std::min(sizeof(Gemma4FfnDecodeScratch),
               static_cast<size_t>(prop.accessPolicyMaxWindowSize));
  attr.accessPolicyWindow.hitRatio = 1.0;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  return cudaStreamSetAttribute(
      stream, cudaStreamAttributeAccessPolicyWindow, &attr);
}

cudaError_t gemma4_ffn_decode_fused_bf16(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ x,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps,
    cudaStream_t stream) {
  if (!ffn_decode_args_valid(residual_out, normed_out, x, residual,
                             rms_weight, w_gate_up_col_major,
                             w_down_row_major, scratch)) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = check_resident_reduction_supported();
  if (status != cudaSuccess) {
    return status;
  }

  status = cudaMemsetAsync(scratch, 0, sizeof(Gemma4FfnDecodeScratch), stream);
  if (status != cudaSuccess) {
    return status;
  }

  gemma4_ffn_decode_fused_bf16_kernel<<<
      kIntermediateTiles, kFfnThreads, 0, stream>>>(
      residual_out, normed_out, x, residual, rms_weight,
      w_gate_up_col_major, w_down_row_major, scratch, eps);
  return cudaGetLastError();
}
