#pragma once

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

#define GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS 8
#define GEMMA4_FFN_DECODE_FLOAT_PACK_ELEMENTS 4
#define GEMMA4_FFN_DECODE_HIDDEN_PACKS \
    (GEMMA4_HIDDEN_SIZE / GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS)

#ifndef GEMMA4_FFN_DECODE_REDUCTION_POLICY
static constexpr int GEMMA4_FFN_DECODE_REDUCTION_POLICY = 0;
#endif

#ifndef GEMMA4_FFN_DECODE_PARTIAL_GROUPS
static constexpr int GEMMA4_FFN_DECODE_PARTIAL_GROUPS =
    2 * GEMMA4_FFN_DECODE_HIDDEN_PACKS;
#endif

#ifndef GEMMA4_FFN_DECODE_ACT_TILE
static constexpr int GEMMA4_FFN_DECODE_ACT_TILE = 2;
#endif

#ifndef GEMMA4_FFN_DECODE_SWIZZLE_X
static constexpr int GEMMA4_FFN_DECODE_SWIZZLE_X = 1;
#endif

#ifndef GEMMA4_FFN_DECODE_THREADS
static constexpr int GEMMA4_FFN_DECODE_THREADS =
    GEMMA4_FFN_DECODE_HIDDEN_PACKS;
#endif

#ifndef GEMMA4_FFN_DECODE_INTERMEDIATE_TILE
static constexpr int GEMMA4_FFN_DECODE_INTERMEDIATE_TILE = 2;
#endif

#ifndef GEMMA4_FFN_DECODE_PRELOAD_DOWN
static constexpr int GEMMA4_FFN_DECODE_PRELOAD_DOWN = 0;
#endif

#ifndef GEMMA4_FFN_DECODE_ACCUM_BLOCKS
static constexpr int GEMMA4_FFN_DECODE_ACCUM_BLOCKS = 0;
#endif

struct alignas(128) Gemma4FfnDecodeScratch {
#if GEMMA4_FFN_DECODE_REDUCTION_POLICY == 1
  float partials[GEMMA4_FFN_DECODE_PARTIAL_GROUPS]
                [GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS]
                [GEMMA4_FFN_DECODE_HIDDEN_PACKS];
#endif
  float accum[GEMMA4_FFN_DECODE_BF16_PACK_ELEMENTS]
             [GEMMA4_FFN_DECODE_HIDDEN_PACKS];
};

struct Gemma4FfnPrefillScratch {
  __nv_bfloat16 *act = nullptr;
  // Reused as swizzled hidden input before gate/up and swizzled down output
  // before post-FFN RMSNorm + residual add.
  __nv_bfloat16 *down = nullptr;
  int capacity_rows = 0;
};

struct Gemma4FfnBf16Args {
  __nv_bfloat16 *residual_out = nullptr;
  __nv_bfloat16 *normed_out = nullptr;
  const __nv_bfloat16 *x = nullptr;
  const __nv_bfloat16 *residual = nullptr;
  const __nv_bfloat16 *rms_weight = nullptr;

  Gemma4FfnPrefillScratch prefill_scratch = {};

  const __nv_bfloat16 *w_gate_up_decode = nullptr;
  const __nv_bfloat16 *w_down_decode = nullptr;
  Gemma4FfnDecodeScratch *decode_scratch = nullptr;

  int rows = 0;
  float eps = GEMMA4_RMS_NORM_EPS;
  cudaStream_t stream = nullptr;
};

size_t gemma4_ffn_prefill_scratch_elements(int rows);

Gemma4FfnPrefillScratch gemma4_ffn_prefill_scratch_from_buffer(
    __nv_bfloat16 *buffer,
    int rows);

cudaError_t gemma4_ffn_bf16(const Gemma4FfnBf16Args &args);

// Runs prefill GeGLU FFN only and writes the natural-order down-projection row.
cudaError_t gemma4_ffn_prefill_mlp_bf16(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ w_gate_up_decode,
    const __nv_bfloat16 *__restrict__ w_down_decode,
    Gemma4FfnPrefillScratch scratch,
    int rows,
    cudaStream_t stream);

cudaError_t gemma4_ffn_decode_fused_bf16(
    __nv_bfloat16 *__restrict__ residual_out,
    __nv_bfloat16 *__restrict__ normed_out,
    const __nv_bfloat16 *__restrict__ x,
    const __nv_bfloat16 *__restrict__ residual,
    const __nv_bfloat16 *__restrict__ rms_weight,
    const __nv_bfloat16 *__restrict__ w_gate_up_col_major,
    const __nv_bfloat16 *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps,
    cudaStream_t stream);

cudaError_t gemma4_ffn_decode_swizzle_weights_bf16(
    __nv_bfloat16 *__restrict__ w_gate_up_swizzled,
    const __nv_bfloat16 *__restrict__ w_gate_up_col_major,
    __nv_bfloat16 *__restrict__ w_down_swizzled,
    const __nv_bfloat16 *__restrict__ w_down_row_major,
    cudaStream_t stream);

namespace gemma4_ffn_decode_device {

constexpr int kIntermediateTile = GEMMA4_FFN_DECODE_INTERMEDIATE_TILE;
constexpr int kIntermediateTiles =
    GEMMA4_INTERMEDIATE_SIZE / kIntermediateTile;
constexpr int kHiddenPacks = GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
constexpr int kActTile = GEMMA4_FFN_DECODE_ACT_TILE;
constexpr int kSwizzleX = GEMMA4_FFN_DECODE_SWIZZLE_X;
constexpr int kPreloadDown = GEMMA4_FFN_DECODE_PRELOAD_DOWN;
constexpr int kReductionPolicy = GEMMA4_FFN_DECODE_REDUCTION_POLICY;
constexpr int kPartialGroups = GEMMA4_FFN_DECODE_PARTIAL_GROUPS;
constexpr int kAccumBlocksOverride = GEMMA4_FFN_DECODE_ACCUM_BLOCKS;
constexpr int kDefaultAccumBlocks = kIntermediateTiles - kHiddenPacks;
constexpr int kAccumBlocks =
    kAccumBlocksOverride == 0 ? kDefaultAccumBlocks : kAccumBlocksOverride;

static_assert((GEMMA4_INTERMEDIATE_SIZE % kIntermediateTile) == 0,
              "FFN intermediate width must divide the decode tile width");
static_assert(kHiddenPacks == GEMMA4_FFN_DECODE_HIDDEN_PACKS,
              "FFN scratch hidden-pack shape must match bf16 pack width");
static_assert(kActTile >= 1, "FFN activation tile must be positive");
static_assert((kIntermediateTile % kActTile) == 0,
              "FFN activation tile must divide the CTA intermediate tile");
static_assert(kActTile <= WARP_SIZE,
              "FFN activation tile must fit the warp-local reduction helper");
static_assert(kSwizzleX == 0 || kSwizzleX == 1,
              "FFN hidden-pack swizzle must be 0 or 1");
static_assert(kPreloadDown == 0 || kPreloadDown == 1,
              "FFN down-weight preload must be 0 or 1");
static_assert(kReductionPolicy == 0 || kReductionPolicy == 1,
              "FFN reduction policy must be 0=atomic or 1=partial groups");
static_assert(kPartialGroups > 0 && kPartialGroups <= kIntermediateTiles,
              "FFN partial groups must be within the intermediate tile count");
static_assert(kDefaultAccumBlocks > 0,
              "FFN default accumulate grid must be positive");
static_assert(kAccumBlocks > 0 && kAccumBlocks <= kIntermediateTiles,
              "FFN accumulate grid must be within the intermediate tile count");

using FfnBf16Pack = Bf16Packed128;
using ::is_aligned_128;

// Reduces paired column accumulators across all warps in one block.
template <int ColsPerBlock, int Threads>
__device__ inline void reduce_cols_pair(
    int thread_idx,
    float (&a_warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&b_warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&a_sums)[ColsPerBlock],
    float (&b_sums)[ColsPerBlock]) {
  constexpr int warps = Threads / WARP_SIZE;
  const int lane = thread_idx & (WARP_SIZE - 1);
  const int warp = thread_idx / WARP_SIZE;

  warp_reduce_sum_to_lane0(a_sums);
  warp_reduce_sum_to_lane0(b_sums);

  if (lane == 0) {
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      a_warp_sums[col][warp] = a_sums[col];
      b_warp_sums[col][warp] = b_sums[col];
    }
  }
  __syncthreads();

#pragma unroll
  for (int col = 0; col < ColsPerBlock; ++col) {
    a_sums[col] = thread_idx < warps ? a_warp_sums[col][lane] : 0.0f;
    b_sums[col] = thread_idx < warps ? b_warp_sums[col][lane] : 0.0f;
  }

  if (warp == 0) {
    warp_reduce_sum_to_lane0(a_sums);
    warp_reduce_sum_to_lane0(b_sums);
  }
}

// Maps natural hidden packs into the swizzled decode weight layout.
__host__ __device__ inline int hidden_pack_swizzle_index(int chunk) {
  if constexpr (kSwizzleX) {
    constexpr int kSwizzleChunks = 8;
    const unsigned u = static_cast<unsigned>(chunk);
    const unsigned col = u & (kSwizzleChunks - 1);
    const unsigned row = (u >> 3) & (kSwizzleChunks - 1);
    return static_cast<int>((u & ~(kSwizzleChunks - 1u)) | (col ^ row));
  }
  return chunk;
}

// Applies the Gemma GeGLU tanh activation approximation.
__device__ inline float gelu_tanh(float x) {
  constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
  constexpr float kGeluCubic = 0.044715f;
  const float x2 = x * x;
  const float inner = kSqrtTwoOverPi * (x + kGeluCubic * x * x2);
  return 0.5f * x * (1.0f + tanhf(inner));
}

// Accumulates one BF16 down-projection pack scaled by one activation value.
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

// Loads this thread's accumulated FFN output pack from global scratch.
__device__ inline void load_accum_pack(
    const Gemma4FfnDecodeScratch *__restrict__ scratch,
    int hidden_pack,
    float (&values)[kBf16Packed128Elements]) {
#pragma unroll
  for (int i = 0; i < kBf16Packed128Elements; ++i) {
    values[i] = scratch->accum[i][hidden_pack];
  }
}

// Atomically adds one thread's hidden output pack into FFN scratch.
__device__ inline void atomic_add_accum_pack(
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    int hidden_pack,
    const float (&values)[kBf16Packed128Elements]) {
#pragma unroll
  for (int i = 0; i < kBf16Packed128Elements; ++i) {
    atomicAdd(&scratch->accum[i][hidden_pack], values[i]);
  }
}

// Stores one partial-group output pack when the partial reduction policy is on.
__device__ inline void store_partial_pack(
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    int group,
    int hidden_pack,
    const float (&values)[kBf16Packed128Elements]) {
#if GEMMA4_FFN_DECODE_REDUCTION_POLICY == 1
#pragma unroll
  for (int i = 0; i < kBf16Packed128Elements; ++i) {
    scratch->partials[group][i][hidden_pack] = values[i];
  }
#else
  (void)scratch;
  (void)group;
  (void)hidden_pack;
  (void)values;
#endif
}

// Sums partial-group scratch values for one hidden output pack.
__device__ inline void load_partial_sum_pack(
    const Gemma4FfnDecodeScratch *__restrict__ scratch,
    int hidden_pack,
    float (&values)[kBf16Packed128Elements]) {
#if GEMMA4_FFN_DECODE_REDUCTION_POLICY == 1
  for (int group = 0; group < kPartialGroups; ++group) {
#pragma unroll
    for (int i = 0; i < kBf16Packed128Elements; ++i) {
      values[i] += scratch->partials[group][i][hidden_pack];
    }
  }
#else
  (void)scratch;
  (void)hidden_pack;
  (void)values;
#endif
}

// Rounds one accumulated FFN output pack to BF16 before post-FFN RMSNorm.
__device__ inline FfnBf16Pack accum_to_bf16_pack(
    const float (&values)[kBf16Packed128Elements]) {
  FfnBf16Pack out_pack;
  __nv_bfloat162 *out_pairs = gemma4_bf16_pack_pairs(out_pack);
#pragma unroll
  for (int p = 0; p < kBf16Packed128Pairs; ++p) {
    out_pairs[p] = __floats2bfloat162_rn(
        values[2 * p], values[2 * p + 1]);
  }
  return out_pack;
}

// Applies post-FFN RMSNorm to the accumulated FFN output, then adds residual.
template <int Threads, bool GuardHiddenPack, bool UsePartialGroups>
__device__ inline void finalize_rmsnorm_residual(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    const Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps,
    float *__restrict__ warp_sums,
    float &scale,
    int hidden_pack) {
  float partial[kBf16Packed128Elements] = {};
  float sum_sq = 0.0f;
  FfnBf16Pack ffn_pack;
  const bool active_hidden_pack =
      !GuardHiddenPack || hidden_pack < kHiddenPacks;

  if (active_hidden_pack) {
    if constexpr (UsePartialGroups) {
      load_partial_sum_pack(scratch, hidden_pack, partial);
    } else {
      load_accum_pack(scratch, hidden_pack, partial);
    }
    ffn_pack = accum_to_bf16_pack(partial);
    gemma4_bf16_pack_accumulate_square(ffn_pack, sum_sq);
  }

  const float total =
      gemma4_block_reduce_sum<Threads>(sum_sq, warp_sums, threadIdx.x);
  if (threadIdx.x == 0) {
    scale = rsqrtf(total / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();

  if (active_hidden_pack) {
    const int hidden_col = hidden_pack * kBf16Packed128Elements;
    const FfnBf16Pack gamma_pack = load128g(rms_weight + hidden_col);
    const FfnBf16Pack normed_pack =
        gemma4_bf16_pack_apply_scale_weight(ffn_pack, gamma_pack, scale);
    const FfnBf16Pack residual_pack = load128g(residual + hidden_col);
    const FfnBf16Pack residual_out_pack =
        gemma4_bf16_pack_add(residual_pack, normed_pack);
    store128wb(normed_out + hidden_col, normed_pack);
    store128(residual_out + hidden_col, residual_out_pack);
  }
}

// Projects one gate/up tile from the normed hidden input for one hidden pack.
template <bool GuardHiddenPack>
__device__ __forceinline__ void dot_gate_up_pack(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    int gate_col0,
    int pack_idx,
    float (&gate)[kActTile],
    float (&up)[kActTile]) {
  if constexpr (GuardHiddenPack) {
    if (pack_idx >= kHiddenPacks) {
      return;
    }
  }

  const int x_col = pack_idx * kBf16Packed128Elements;
  const int weight_col =
      hidden_pack_swizzle_index(pack_idx) * kBf16Packed128Elements;
  const FfnBf16Pack x_pack = load128g(x + x_col);
  const floatX *gate_ptr =
      w_gate_up_col_major +
      static_cast<int64_t>(2 * gate_col0) * GEMMA4_HIDDEN_SIZE + weight_col;
  const floatX *up_ptr = gate_ptr + GEMMA4_HIDDEN_SIZE;
#pragma unroll
  for (int t = 0; t < kActTile; ++t) {
    const int64_t offset = static_cast<int64_t>(2 * t) * GEMMA4_HIDDEN_SIZE;
    const FfnBf16Pack gate_pack = load128weight(gate_ptr + offset);
    const FfnBf16Pack up_pack = load128weight(up_ptr + offset);
    gemma4_bf16_pack_accumulate_dot(x_pack, gate_pack, gate[t]);
    gemma4_bf16_pack_accumulate_dot(x_pack, up_pack, up[t]);
  }
}

// Computes and block-reduces gate/up dot products for one FFN tile.
template <int Threads, bool GuardHiddenPack>
__device__ inline void dot_gate_up_reduce(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    int gate_col0,
    float (&gate)[kActTile],
    float (&up)[kActTile],
    float (&gate_warp_sums)[kActTile][Threads / WARP_SIZE],
    float (&up_warp_sums)[kActTile][Threads / WARP_SIZE]) {
  dot_gate_up_pack<GuardHiddenPack>(
      x, w_gate_up_col_major, gate_col0, threadIdx.x, gate, up);

  reduce_cols_pair<kActTile, Threads>(
      threadIdx.x, gate_warp_sums, up_warp_sums, gate, up);
}

// Accumulates one intermediate tile into the calling thread's output pack.
template <int Threads, bool GuardHiddenPack>
__device__ inline void accumulate_intermediate_tile(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    int intermediate_begin,
    int swizzled_hidden_col,
    bool active_hidden_pack,
    float (&partial)[kBf16Packed128Elements],
    float (&s_matmul_warp_sums)[2][kActTile][Threads / WARP_SIZE],
    float (&s_act)[kActTile]) {
  for (int local_col = 0; local_col < kIntermediateTile;
       local_col += kActTile) {
    float gate[kActTile] = {};
    float up[kActTile] = {};
    const int gate_col0 = intermediate_begin + local_col;

    dot_gate_up_reduce<Threads, GuardHiddenPack>(
        x, w_gate_up_col_major, gate_col0, gate, up,
        s_matmul_warp_sums[0], s_matmul_warp_sums[1]);

    if (threadIdx.x == 0) {
#pragma unroll
      for (int t = 0; t < kActTile; ++t) {
        s_act[t] = gelu_tanh(gate[t]) * up[t];
      }
    }
    __syncthreads();

    if (!active_hidden_pack) {
      continue;
    }

    const floatX *down_row0 =
        w_down_row_major +
        static_cast<int64_t>(intermediate_begin + local_col) *
            GEMMA4_HIDDEN_SIZE +
        swizzled_hidden_col;
    if constexpr (kPreloadDown) {
      FfnBf16Pack down_packs[kActTile];
#pragma unroll
      for (int t = 0; t < kActTile; ++t) {
        down_packs[t] =
            load128weight(down_row0 + t * GEMMA4_HIDDEN_SIZE);
      }
#pragma unroll
      for (int t = 0; t < kActTile; ++t) {
        accumulate_scaled_pack(s_act[t], down_packs[t], partial);
      }
    } else {
#pragma unroll
      for (int t = 0; t < kActTile; ++t) {
        const FfnBf16Pack down_pack =
            load128weight(down_row0 + t * GEMMA4_HIDDEN_SIZE);
        accumulate_scaled_pack(s_act[t], down_pack, partial);
      }
    }
  }
}

// Clears the FFN accumulation scratch over a caller-owned work partition.
__device__ inline void zero_accum(
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    int linear_idx,
    int stride) {
  float *accum = &scratch->accum[0][0];
  constexpr int values = kBf16Packed128Elements * kHiddenPacks;
  for (int idx = linear_idx; idx < values; idx += stride) {
    accum[idx] = 0.0f;
  }
}


// Accumulates this CTA's assigned decode FFN intermediate tiles.
template <int Threads>
__device__ inline void decode_accumulate(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    int block_idx,
    int thread_idx,
    float (&s_matmul_warp_sums)[2][kActTile][Threads / WARP_SIZE],
    float (&s_act)[kActTile]) {
  float partial[kBf16Packed128Elements] = {};
  const int hidden_pack = thread_idx;
  const int swizzled_hidden_col =
      hidden_pack_swizzle_index(hidden_pack) * kBf16Packed128Elements;

  if constexpr (kAccumBlocks == kIntermediateTiles) {
    const int intermediate_begin = block_idx * kIntermediateTile;
    accumulate_intermediate_tile<Threads, false>(
        x, w_gate_up_col_major, w_down_row_major, intermediate_begin,
        swizzled_hidden_col, true, partial, s_matmul_warp_sums, s_act);
  } else if constexpr (kAccumBlocks * 2 > kIntermediateTiles) {
    const int tile0 = block_idx;
    accumulate_intermediate_tile<Threads, false>(
        x, w_gate_up_col_major, w_down_row_major, tile0 * kIntermediateTile,
        swizzled_hidden_col, true, partial, s_matmul_warp_sums, s_act);

    const int tile1 = tile0 + kAccumBlocks;
    if (tile1 < kIntermediateTiles) {
      accumulate_intermediate_tile<Threads, false>(
          x, w_gate_up_col_major, w_down_row_major, tile1 * kIntermediateTile,
          swizzled_hidden_col, true, partial, s_matmul_warp_sums, s_act);
    }
  } else {
    for (int tile = block_idx; tile < kIntermediateTiles;
         tile += kAccumBlocks) {
      const int intermediate_begin = tile * kIntermediateTile;
      accumulate_intermediate_tile<Threads, false>(
          x, w_gate_up_col_major, w_down_row_major, intermediate_begin,
          swizzled_hidden_col, true, partial, s_matmul_warp_sums, s_act);
    }
  }

  atomic_add_accum_pack(scratch, hidden_pack, partial);
}

// Accumulates this partial-reduction group for decode FFN.
template <int Threads>
__device__ inline void decode_accumulate_partials(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    int group,
    int thread_idx,
    float (&s_matmul_warp_sums)[2][kActTile][Threads / WARP_SIZE],
    float (&s_act)[kActTile]) {
  float partial[kBf16Packed128Elements] = {};
  const int hidden_pack = thread_idx;
  const int swizzled_hidden_col =
      hidden_pack_swizzle_index(hidden_pack) * kBf16Packed128Elements;

  for (int tile = group; tile < kIntermediateTiles; tile += kPartialGroups) {
    accumulate_intermediate_tile<Threads, false>(
        x, w_gate_up_col_major, w_down_row_major, tile * kIntermediateTile,
        swizzled_hidden_col, true, partial, s_matmul_warp_sums, s_act);
  }

  store_partial_pack(scratch, group, hidden_pack, partial);
}

// Finalizes standalone decode FFN accumulation into normalized residual output.
template <int Threads>
__device__ inline void decode_finalize(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    const Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps,
    float *__restrict__ warp_sums,
    float &scale,
    int thread_idx) {
  finalize_rmsnorm_residual<Threads, false, kReductionPolicy == 1>(
      residual_out, normed_out, residual, rms_weight, scratch, eps, warp_sums,
      scale, thread_idx);
}

// Swizzles one row of hidden-width BF16 packs for decode-friendly layout.
__device__ inline void swizzle_hidden_packs(
    floatX *__restrict__ dst,
    const floatX *__restrict__ src,
    int row,
    int block_idx,
    int grid_x,
    int block_dim,
    int thread_idx) {
  const int64_t row_offset = static_cast<int64_t>(row) * GEMMA4_HIDDEN_SIZE;
  const int pack_stride = grid_x * block_dim;
  for (int pack = block_idx * block_dim + thread_idx;
       pack < kHiddenPacks; pack += pack_stride) {
    const int src_col = pack * kBf16Packed128Elements;
    const int dst_col =
        hidden_pack_swizzle_index(pack) * kBf16Packed128Elements;
    const FfnBf16Pack pack_value = load128g(src + row_offset + src_col);
    store128(dst + row_offset + dst_col, pack_value);
  }
}

// RMS-normalizes a swizzled down-projection row, then adds the residual.
template <int Threads>
__device__ inline void rmsnorm_residual_from_swizzled_down(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ down_swizzled,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    float eps,
    float *__restrict__ warp_sums,
    float &scale,
    int row,
    int thread_idx) {
  const int hidden_pack = thread_idx;
  const int natural_col = hidden_pack * kBf16Packed128Elements;
  const int swizzled_col =
      hidden_pack_swizzle_index(hidden_pack) * kBf16Packed128Elements;
  const int64_t row_offset = static_cast<int64_t>(row) * GEMMA4_HIDDEN_SIZE;

  const FfnBf16Pack down_pack =
      load128g(down_swizzled + row_offset + swizzled_col);
  float sum_sq = 0.0f;
  gemma4_bf16_pack_accumulate_square(down_pack, sum_sq);

  const float total =
      gemma4_block_reduce_sum<Threads>(sum_sq, warp_sums, thread_idx);
  if (thread_idx == 0) {
    scale = rsqrtf(total / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();

  const FfnBf16Pack gamma_pack = load128g(rms_weight + natural_col);
  const FfnBf16Pack normed_pack =
      gemma4_bf16_pack_apply_scale_weight(down_pack, gamma_pack, scale);
  const FfnBf16Pack residual_pack =
      load128g(residual + row_offset + natural_col);
  const FfnBf16Pack residual_out_pack =
      gemma4_bf16_pack_add(residual_pack, normed_pack);
  store128wb(normed_out + row_offset + natural_col, normed_pack);
  store128(residual_out + row_offset + natural_col, residual_out_pack);
}

// Restores swizzled hidden packs to natural hidden-column order.
__device__ inline void unswizzle_hidden_packs(
    floatX *__restrict__ dst,
    const floatX *__restrict__ src,
    int row,
    int block_dim,
    int thread_idx) {
  const int64_t row_offset = static_cast<int64_t>(row) * GEMMA4_HIDDEN_SIZE;

  for (int hidden_pack = thread_idx; hidden_pack < kHiddenPacks;
       hidden_pack += block_dim) {
    const int natural_col = hidden_pack * kBf16Packed128Elements;
    const int swizzled_col =
        hidden_pack_swizzle_index(hidden_pack) * kBf16Packed128Elements;
    const FfnBf16Pack pack = load128g(src + row_offset + swizzled_col);
    store128(dst + row_offset + natural_col, pack);
  }
}

// Interleaves gate/up rows while applying the hidden-pack decode swizzle.
__device__ inline void swizzle_gate_up_interleaved(
    floatX *__restrict__ dst,
    const floatX *__restrict__ src,
    int dst_row,
    int block_idx,
    int grid_x,
    int block_dim,
    int thread_idx) {
  const int src_row =
      (dst_row & 1) == 0 ? dst_row / 2
                         : GEMMA4_INTERMEDIATE_SIZE + dst_row / 2;
  const int64_t src_row_offset =
      static_cast<int64_t>(src_row) * GEMMA4_HIDDEN_SIZE;
  const int64_t dst_row_offset =
      static_cast<int64_t>(dst_row) * GEMMA4_HIDDEN_SIZE;
  const int pack_stride = grid_x * block_dim;
  for (int pack = block_idx * block_dim + thread_idx;
       pack < kHiddenPacks; pack += pack_stride) {
    const int src_col = pack * kBf16Packed128Elements;
    const int dst_col =
        hidden_pack_swizzle_index(pack) * kBf16Packed128Elements;
    const FfnBf16Pack pack_value =
        load128g(src + src_row_offset + src_col);
    store128(dst + dst_row_offset + dst_col, pack_value);
  }
}

}  // namespace gemma4_ffn_decode_device
