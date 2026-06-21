#ifndef GEMMA4_FFN_DECODE_DEVICE_CUH
#define GEMMA4_FFN_DECODE_DEVICE_CUH

#include "gemma4_ffn.cuh"
#include "gemma4_cuda_utils.cuh"

#include <math.h>

#ifndef GEMMA4_FFN_DECODE_ACT_TILE
#define GEMMA4_FFN_DECODE_ACT_TILE 2
#endif

#ifndef GEMMA4_FFN_DECODE_SWIZZLE_X
#define GEMMA4_FFN_DECODE_SWIZZLE_X 1
#endif

#ifndef GEMMA4_FFN_DECODE_THREADS
#define GEMMA4_FFN_DECODE_THREADS GEMMA4_FFN_DECODE_HIDDEN_PACKS
#endif

#ifndef GEMMA4_FFN_DECODE_INTERMEDIATE_TILE
#define GEMMA4_FFN_DECODE_INTERMEDIATE_TILE 2
#endif

#ifndef GEMMA4_FFN_DECODE_PRELOAD_DOWN
#define GEMMA4_FFN_DECODE_PRELOAD_DOWN 0
#endif

#ifndef GEMMA4_FFN_DECODE_ACCUM_BLOCKS
#define GEMMA4_FFN_DECODE_ACCUM_BLOCKS 0
#endif

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
static_assert((GEMMA4_HIDDEN_SIZE % kBf16Packed128Elements) == 0,
              "FFN hidden width must divide the 128-bit bf16 pack width");
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

// Checks the 128-byte alignment required by Gemma4FfnDecodeScratch.
inline bool is_aligned_128(const void *ptr) {
  return is_aligned_to<128>(ptr);
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
        gemma4_bf16_pack_apply_rmsnorm(ffn_pack, gamma_pack, scale);
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

}  // namespace gemma4_ffn_decode_device

#endif
