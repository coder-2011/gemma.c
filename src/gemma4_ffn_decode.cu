#include "gemma4_ffn_decode.cuh"

#ifndef GEMMA4_WEIGHT_LOAD_POLICY
#define GEMMA4_WEIGHT_LOAD_POLICY 0
#endif

#include "gemma4_cuda_utils.cuh"
#include "gemma4_matmul_device.cuh"
#include "gemma4_rmsnorm.cuh"

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

#ifndef GEMMA4_FFN_DECODE_ACT_TILE
#define GEMMA4_FFN_DECODE_ACT_TILE 2
#endif

#ifndef GEMMA4_FFN_DECODE_SWIZZLE_X
#define GEMMA4_FFN_DECODE_SWIZZLE_X 1
#endif

#ifndef GEMMA4_FFN_DECODE_THREADS
#define GEMMA4_FFN_DECODE_THREADS 672
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

#ifndef GEMMA4_FFN_DECODE_REDUCTION_POLICY
#define GEMMA4_FFN_DECODE_REDUCTION_POLICY 0
#endif

#ifndef GEMMA4_FFN_DECODE_PARTIAL_GROUPS
#define GEMMA4_FFN_DECODE_PARTIAL_GROUPS 1344
#endif

namespace {

constexpr int kFfnThreads = GEMMA4_FFN_DECODE_THREADS;
constexpr int kFfnWarps = kFfnThreads / WARP_SIZE;
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
// Keep most CTAs one tile wide, while one hidden-pack wave folds a second tile.
constexpr int kDefaultAccumBlocks = kIntermediateTiles - kHiddenPacks;
constexpr int kAccumBlocks =
    kAccumBlocksOverride == 0 ? kDefaultAccumBlocks : kAccumBlocksOverride;
static_assert((GEMMA4_INTERMEDIATE_SIZE % kIntermediateTile) == 0,
              "FFN intermediate width must divide the decode tile width");
static_assert((GEMMA4_HIDDEN_SIZE % kBf16Packed128Elements) == 0,
              "FFN hidden width must divide the 128-bit bf16 pack width");
static_assert(kHiddenPacks == GEMMA4_FFN_DECODE_HIDDEN_PACKS,
              "FFN scratch hidden-pack shape must match bf16 pack width");
static_assert(kFfnThreads == kHiddenPacks,
              "FFN decode maps one CTA thread to one hidden bf16 pack");
static_assert((kFfnThreads % WARP_SIZE) == 0,
              "FFN decode thread count must be a whole number of warps");
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

inline bool is_aligned_128(const void *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0x7fu) == 0;
}

__device__ inline int hidden_pack_swizzle_index(int chunk) {
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

__device__ inline void load_accum_pack(
    const Gemma4FfnDecodeScratch *__restrict__ scratch,
    int hidden_pack,
    float (&values)[kBf16Packed128Elements]) {
#pragma unroll
  for (int i = 0; i < kBf16Packed128Elements; ++i) {
    values[i] = scratch->accum[i][hidden_pack];
  }
}

__device__ inline void atomic_add_accum_pack(
    Gemma4FfnDecodeScratch *__restrict__ scratch,
    int hidden_pack,
    const float (&values)[kBf16Packed128Elements]) {
#pragma unroll
  for (int i = 0; i < kBf16Packed128Elements; ++i) {
    atomicAdd(&scratch->accum[i][hidden_pack], values[i]);
  }
}

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

__device__ __forceinline__ void dot_gate_up_pack(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    int gate_col0,
    int pack_idx,
    float (&gate)[kActTile],
    float (&up)[kActTile]) {
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
    const FfnBf16Pack gate_pack =
        load128weight(gate_ptr + static_cast<int64_t>(2 * t) *
                                     GEMMA4_HIDDEN_SIZE);
    const FfnBf16Pack up_pack =
        load128weight(up_ptr + static_cast<int64_t>(2 * t) *
                                 GEMMA4_HIDDEN_SIZE);
    gemma4_bf16_pack_accumulate_dot(x_pack, gate_pack, gate[t]);
    gemma4_bf16_pack_accumulate_dot(x_pack, up_pack, up[t]);
  }
}

__device__ inline void dot_gate_up_swizzled_x_reduce(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    int gate_col0,
    float (&gate)[kActTile],
    float (&up)[kActTile],
    float (&gate_warp_sums)[kActTile][kFfnWarps],
    float (&up_warp_sums)[kActTile][kFfnWarps]) {
  dot_gate_up_pack(
      x, w_gate_up_col_major, gate_col0, threadIdx.x, gate, up);

  gemma4_matmul_device::reduce_cols_pair<kActTile, kFfnThreads>(
      threadIdx.x, gate_warp_sums, up_warp_sums, gate, up);
}

__device__ inline void accumulate_intermediate_tile(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    int intermediate_begin,
    int swizzled_hidden_col,
    float (&partial)[kBf16Packed128Elements],
    float (&s_matmul_warp_sums)[2][kActTile][kFfnWarps],
    float (&s_act)[kActTile]) {
  for (int local_col = 0; local_col < kIntermediateTile;
       local_col += kActTile) {
    float gate[kActTile] = {};
    float up[kActTile] = {};
    const int gate_col0 = intermediate_begin + local_col;

    dot_gate_up_swizzled_x_reduce(
        x, w_gate_up_col_major, gate_col0, gate, up,
        s_matmul_warp_sums[0], s_matmul_warp_sums[1]);

    if (threadIdx.x == 0) {
#pragma unroll
      for (int t = 0; t < kActTile; ++t) {
        s_act[t] = gate[t] * gelu_tanh(up[t]);
      }
    }
    __syncthreads();

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
    // The next gate/up reducer reaches a CTA barrier before rewriting s_act, so
    // down accumulation does not need a separate tail barrier here.
  }
}

__global__ __launch_bounds__(kFfnThreads, 1) void
gemma4_ffn_decode_accumulate_bf16_kernel(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch) {
  __shared__ float s_matmul_warp_sums[2][kActTile][kFfnWarps];
  __shared__ float s_act[kActTile];

  float partial[kBf16Packed128Elements] = {};
  const int hidden_pack = threadIdx.x;
  const int swizzled_hidden_col =
      hidden_pack_swizzle_index(hidden_pack) * kBf16Packed128Elements;

  if constexpr (kAccumBlocks == kIntermediateTiles) {
    const int intermediate_begin =
        static_cast<int>(blockIdx.x) * kIntermediateTile;
    accumulate_intermediate_tile(
        x, w_gate_up_col_major, w_down_row_major, intermediate_begin,
        swizzled_hidden_col, partial, s_matmul_warp_sums, s_act);
  } else if constexpr (kAccumBlocks * 2 > kIntermediateTiles) {
    const int tile0 = static_cast<int>(blockIdx.x);
    accumulate_intermediate_tile(
        x, w_gate_up_col_major, w_down_row_major, tile0 * kIntermediateTile,
        swizzled_hidden_col, partial, s_matmul_warp_sums, s_act);

    const int tile1 = tile0 + kAccumBlocks;
    if (tile1 < kIntermediateTiles) {
      accumulate_intermediate_tile(
          x, w_gate_up_col_major, w_down_row_major, tile1 * kIntermediateTile,
          swizzled_hidden_col, partial, s_matmul_warp_sums, s_act);
    }
  } else {
    for (int tile = static_cast<int>(blockIdx.x); tile < kIntermediateTiles;
         tile += kAccumBlocks) {
      const int intermediate_begin = tile * kIntermediateTile;
      accumulate_intermediate_tile(
          x, w_gate_up_col_major, w_down_row_major, intermediate_begin,
          swizzled_hidden_col, partial, s_matmul_warp_sums, s_act);
    }
  }

  atomic_add_accum_pack(scratch, hidden_pack, partial);
}

__global__ __launch_bounds__(kFfnThreads, 1) void
gemma4_ffn_decode_accumulate_partials_bf16_kernel(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    const floatX *__restrict__ w_down_row_major,
    Gemma4FfnDecodeScratch *__restrict__ scratch) {
  __shared__ float s_matmul_warp_sums[2][kActTile][kFfnWarps];
  __shared__ float s_act[kActTile];

  float partial[kBf16Packed128Elements] = {};
  const int hidden_pack = threadIdx.x;
  const int swizzled_hidden_col =
      hidden_pack_swizzle_index(hidden_pack) * kBf16Packed128Elements;
  const int group = static_cast<int>(blockIdx.x);

  for (int tile = group; tile < kIntermediateTiles; tile += kPartialGroups) {
    accumulate_intermediate_tile(
        x, w_gate_up_col_major, w_down_row_major, tile * kIntermediateTile,
        swizzled_hidden_col, partial, s_matmul_warp_sums, s_act);
  }

  store_partial_pack(scratch, group, hidden_pack, partial);
}

__global__ __launch_bounds__(kFfnThreads, 1) void
gemma4_ffn_decode_finalize_bf16_kernel(
    floatX *__restrict__ residual_out,
    floatX *__restrict__ normed_out,
    const floatX *__restrict__ residual,
    const floatX *__restrict__ rms_weight,
    const Gemma4FfnDecodeScratch *__restrict__ scratch,
    float eps) {
  float partial[kBf16Packed128Elements] = {};
  float sum_sq = 0.0f;
  __shared__ float s_rms_warp_sums[kFfnWarps];
  __shared__ float s_scale;

  const int hidden_pack = threadIdx.x;
  if constexpr (kReductionPolicy == 1) {
    load_partial_sum_pack(scratch, hidden_pack, partial);
  } else {
    load_accum_pack(scratch, hidden_pack, partial);
  }
  const int hidden_col = hidden_pack * kBf16Packed128Elements;
  FfnBf16Pack residual_pack = load128g(residual + hidden_col);
  FfnBf16Pack residual_out_pack;
  add_residual_store_pack(residual_pack, partial, residual_out_pack, sum_sq);
  store128(residual_out + hidden_col, residual_out_pack);

  const float total = block_reduce_sum(sum_sq, s_rms_warp_sums);
  if (threadIdx.x == 0) {
    s_scale =
        rsqrtf(total / static_cast<float>(GEMMA4_HIDDEN_SIZE) + eps);
  }
  __syncthreads();

  const FfnBf16Pack gamma_pack = load128g(rms_weight + hidden_col);
  const FfnBf16Pack normed_pack =
      rmsnorm_store_pack(partial, gamma_pack, s_scale);
  store128wb(normed_out + hidden_col, normed_pack);
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
         is_aligned_16(w_down_row_major) && is_aligned_128(scratch);
}

cudaError_t configure_dynamic_shared_memory() {
  return cudaSuccess;
}

__global__ void swizzle_hidden_packs_kernel(
    floatX *__restrict__ dst,
    const floatX *__restrict__ src,
    int rows) {
  const int pack = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int row = static_cast<int>(blockIdx.y);
  if (pack >= kHiddenPacks) {
    return;
  }

  const int src_col = pack * kBf16Packed128Elements;
  const int dst_col = hidden_pack_swizzle_index(pack) * kBf16Packed128Elements;
  const int64_t row_offset = static_cast<int64_t>(row) * GEMMA4_HIDDEN_SIZE;
  const FfnBf16Pack pack_value = load128g(src + row_offset + src_col);
  store128(dst + row_offset + dst_col, pack_value);
}

__global__ void swizzle_gate_up_interleaved_kernel(
    floatX *__restrict__ dst,
    const floatX *__restrict__ src) {
  const int pack = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int dst_row = static_cast<int>(blockIdx.y);
  if (pack >= kHiddenPacks) {
    return;
  }

  const int src_row =
      (dst_row & 1) == 0 ? dst_row / 2
                         : GEMMA4_INTERMEDIATE_SIZE + dst_row / 2;
  const int src_col = pack * kBf16Packed128Elements;
  const int dst_col = hidden_pack_swizzle_index(pack) * kBf16Packed128Elements;
  const int64_t src_row_offset =
      static_cast<int64_t>(src_row) * GEMMA4_HIDDEN_SIZE;
  const int64_t dst_row_offset =
      static_cast<int64_t>(dst_row) * GEMMA4_HIDDEN_SIZE;
  const FfnBf16Pack pack_value =
      load128g(src + src_row_offset + src_col);
  store128(dst + dst_row_offset + dst_col, pack_value);
}

template <typename LayoutB,
          int ThreadblockM,
          int ThreadblockN,
          int ThreadblockK,
          int WarpM,
          int WarpN,
          int Stages>
cudaError_t launch_cutlass_bf16_gemm(
    const floatX *__restrict__ a,
    const floatX *__restrict__ b,
    floatX *__restrict__ c,
    int m,
    int k,
    int n,
    int ldb,
    cudaStream_t stream) {
  using Element = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      LayoutB,
      Element,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
      cutlass::gemm::GemmShape<WarpM, WarpN, ThreadblockK>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Stages>;

  typename Gemm::Arguments args(
      {m, n, k},
      {reinterpret_cast<const Element *>(a), k},
      {reinterpret_cast<const Element *>(b), ldb},
      {reinterpret_cast<Element *>(c), n},
      {reinterpret_cast<Element *>(c), n},
      {1.0f, 0.0f});

  Gemm gemm;
  const cutlass::Status status = gemm(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t launch_prefill_gate_up_gemm(
    const floatX *__restrict__ x,
    const floatX *__restrict__ w_gate_up_col_major,
    floatX *__restrict__ gate_up,
    int rows,
    cudaStream_t stream) {
  return launch_cutlass_bf16_gemm<cutlass::layout::ColumnMajor,
                                  128, 128, 64, 64, 64, 3>(
      x, w_gate_up_col_major, gate_up, rows, GEMMA4_HIDDEN_SIZE,
      GEMMA4_PACKED_FFN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
}

cudaError_t launch_prefill_down_gemm(
    const floatX *__restrict__ act,
    const floatX *__restrict__ w_down_row_major,
    floatX *__restrict__ down,
    int rows,
    cudaStream_t stream) {
  if (rows <= 128) {
    return launch_cutlass_bf16_gemm<cutlass::layout::RowMajor,
                                    64, 128, 64, 32, 64, 3>(
        act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE,
        GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
  }
  return launch_cutlass_bf16_gemm<cutlass::layout::RowMajor,
                                  128, 128, 64, 64, 64, 3>(
      act, w_down_row_major, down, rows, GEMMA4_INTERMEDIATE_SIZE,
      GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, stream);
}

__global__ void gemma4_ffn_prefill_geglu_bf16_kernel(
    floatX *__restrict__ act,
    const floatX *__restrict__ gate_up,
    int rows) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int total = rows * GEMMA4_INTERMEDIATE_SIZE;
  if (idx >= total) {
    return;
  }

  const int row = idx / GEMMA4_INTERMEDIATE_SIZE;
  const int col = idx - row * GEMMA4_INTERMEDIATE_SIZE;
  const int64_t gate_up_row =
      static_cast<int64_t>(row) * GEMMA4_PACKED_FFN_SIZE;
  const float gate = __bfloat162float(gate_up[gate_up_row + col]);
  const float up =
      __bfloat162float(gate_up[gate_up_row + GEMMA4_INTERMEDIATE_SIZE + col]);
  act[idx] = __float2bfloat16_rn(gate * gelu_tanh(up));
}

bool ffn_common_args_valid(const Gemma4FfnBf16Args &args) {
  return args.rows >= 0 && args.residual_out != nullptr &&
         args.normed_out != nullptr && args.x != nullptr &&
         args.residual != nullptr && args.rms_weight != nullptr &&
         is_aligned_16(args.residual_out) && is_aligned_16(args.normed_out) &&
         is_aligned_16(args.x) && is_aligned_16(args.residual) &&
         is_aligned_16(args.rms_weight);
}

bool ffn_prefill_args_valid(const Gemma4FfnBf16Args &args) {
  return ffn_common_args_valid(args) && args.rows > 1 &&
         args.w_gate_up_prefill_col_major != nullptr &&
         args.w_down_prefill_row_major != nullptr &&
         args.prefill_scratch.gate_up != nullptr &&
         args.prefill_scratch.act != nullptr &&
         args.prefill_scratch.down != nullptr &&
         args.prefill_scratch.capacity_rows >= args.rows &&
         is_aligned_16(args.w_gate_up_prefill_col_major) &&
         is_aligned_16(args.w_down_prefill_row_major) &&
         is_aligned_16(args.prefill_scratch.gate_up) &&
         is_aligned_16(args.prefill_scratch.act) &&
         is_aligned_16(args.prefill_scratch.down);
}

cudaError_t gemma4_ffn_decode_fused_bf16_impl(
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

  cudaError_t status = configure_dynamic_shared_memory();
  if (status != cudaSuccess) {
    return status;
  }

  status = cudaMemsetAsync(scratch, 0, sizeof(*scratch), stream);
  if (status != cudaSuccess) {
    return status;
  }

  if constexpr (kReductionPolicy == 1) {
    gemma4_ffn_decode_accumulate_partials_bf16_kernel<<<
        kPartialGroups, kFfnThreads, 0, stream>>>(
        x, w_gate_up_col_major, w_down_row_major, scratch);
  } else {
    gemma4_ffn_decode_accumulate_bf16_kernel<<<
        kAccumBlocks, kFfnThreads, 0, stream>>>(
        x, w_gate_up_col_major, w_down_row_major, scratch);
  }
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  gemma4_ffn_decode_finalize_bf16_kernel<<<1, kFfnThreads, 0, stream>>>(
      residual_out, normed_out, residual, rms_weight, scratch, eps);
  return cudaGetLastError();
}

cudaError_t gemma4_ffn_prefill_bf16_impl(const Gemma4FfnBf16Args &args) {
  if (!ffn_prefill_args_valid(args)) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = launch_prefill_gate_up_gemm(
      args.x, args.w_gate_up_prefill_col_major, args.prefill_scratch.gate_up,
      args.rows, args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  constexpr int threads = 256;
  const int total = args.rows * GEMMA4_INTERMEDIATE_SIZE;
  const dim3 block_dim(threads);
  const dim3 grid_dim(div_up(total, threads));
  gemma4_ffn_prefill_geglu_bf16_kernel<<<grid_dim, block_dim, 0,
                                          args.stream>>>(
      args.prefill_scratch.act, args.prefill_scratch.gate_up, args.rows);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  status = launch_prefill_down_gemm(
      args.prefill_scratch.act, args.w_down_prefill_row_major,
      args.prefill_scratch.down, args.rows, args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  return gemma4_residual_add_rmsnorm_bf16(
      args.residual_out, args.normed_out, args.prefill_scratch.down,
      args.residual, args.rms_weight, args.rows, GEMMA4_HIDDEN_SIZE,
      args.eps, args.stream);
}

}  // namespace

cudaError_t gemma4_ffn_decode_swizzle_weights_bf16(
    floatX *__restrict__ w_gate_up_swizzled,
    const floatX *__restrict__ w_gate_up_col_major,
    floatX *__restrict__ w_down_swizzled,
    const floatX *__restrict__ w_down_row_major,
    cudaStream_t stream) {
  if (w_gate_up_swizzled == nullptr || w_gate_up_col_major == nullptr ||
      w_down_swizzled == nullptr || w_down_row_major == nullptr ||
      !is_aligned_16(w_gate_up_swizzled) ||
      !is_aligned_16(w_gate_up_col_major) ||
      !is_aligned_16(w_down_swizzled) ||
      !is_aligned_16(w_down_row_major)) {
    return cudaErrorInvalidValue;
  }

  constexpr int threads = 256;
  const dim3 block_dim(threads);
  const dim3 gate_up_grid_dim(div_up(kHiddenPacks, threads),
                              GEMMA4_PACKED_FFN_SIZE);
  swizzle_gate_up_interleaved_kernel<<<
      gate_up_grid_dim, block_dim, 0, stream>>>(
      w_gate_up_swizzled, w_gate_up_col_major);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  const dim3 down_grid_dim(div_up(kHiddenPacks, threads),
                           GEMMA4_INTERMEDIATE_SIZE);
  swizzle_hidden_packs_kernel<<<down_grid_dim, block_dim, 0, stream>>>(
      w_down_swizzled, w_down_row_major, GEMMA4_INTERMEDIATE_SIZE);
  return cudaGetLastError();
}

cudaError_t gemma4_ffn_decode_configure_scratch_l2(
    Gemma4FfnDecodeScratch *scratch,
    cudaStream_t stream) {
  (void)stream;
  if (scratch == nullptr || !is_aligned_128(scratch)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
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

  cudaError_t status = configure_dynamic_shared_memory();
  if (status != cudaSuccess) {
    return status;
  }

  status = cudaMemsetAsync(scratch, 0, sizeof(*scratch), stream);
  if (status != cudaSuccess) {
    return status;
  }

  if constexpr (kReductionPolicy == 1) {
    gemma4_ffn_decode_accumulate_partials_bf16_kernel<<<
        kPartialGroups, kFfnThreads, 0, stream>>>(
        x, w_gate_up_col_major, w_down_row_major, scratch);
  } else {
    gemma4_ffn_decode_accumulate_bf16_kernel<<<
        kAccumBlocks, kFfnThreads, 0, stream>>>(
        x, w_gate_up_col_major, w_down_row_major, scratch);
  }
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  gemma4_ffn_decode_finalize_bf16_kernel<<<1, kFfnThreads, 0, stream>>>(
      residual_out, normed_out, residual, rms_weight, scratch, eps);
  return cudaGetLastError();
}
