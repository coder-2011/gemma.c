#include <cuda_bf16.h>
#include <cuda/cmath>
#include <cuda_runtime.h>

#include <cub/block/block_reduce.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <cute/tensor.hpp>

#include "gemma4_cuda_utils.cuh"
#include "gemma4_megakernel.cuh"
#include "gemma4_ffn.cuh"
#include "gemma4_kv_cache.cuh"
#include "gemma4_rmsnorm.cuh"
#include "gemma4_rope.cuh"
#include "gemma4.h"

#include <cooperative_groups.h>

namespace gemma4_flash_attention {

using namespace cute;

namespace cg = cooperative_groups;

constexpr int kWarpSize = 32;
constexpr int kDecodeMegaThreads = 512;
constexpr int kDecodeMegaWarps = kDecodeMegaThreads / kWarpSize;
constexpr int kFfnHiddenPacks = gemma4_ffn_decode_device::kHiddenPacks;

// Projects the final attention row through O with hidden-column CTA ownership.
template <typename Traits>
__device__ inline void project_attention_out_post_attention(
    const Gemma4DecodeMegakernelLayerArgs &args,
    cg::grid_group grid) {
  constexpr int kColsPerBlock = kBf16Packed128Elements;
  constexpr int kHiddenBlocks = GEMMA4_HIDDEN_SIZE / kColsPerBlock;
  constexpr int kAttentionWidth = GEMMA4_NUM_QUERY_HEADS * Traits::kHeadDim;
  constexpr int kAttentionPacks = kAttentionWidth / kBf16Packed128Elements;
  static_assert(GEMMA4_HIDDEN_SIZE % kColsPerBlock == 0, "hidden size must be a whole BF16 pack tile");
  static_assert(kAttentionWidth % kBf16Packed128Elements == 0, "attention width must be a whole BF16 pack tile");

  __shared__ float warp_sums[kColsPerBlock][kDecodeMegaWarps];

  const int block = int(blockIdx.x);
  const int thread = int(threadIdx.x);
  const int lane = thread & (kWarpSize - 1);
  const int warp = thread / kWarpSize;

  for (int col_block = block; col_block < kHiddenBlocks;
       col_block += int(gridDim.x)) {
    float sums[kColsPerBlock] = {};
    for (int pack = thread; pack < kAttentionPacks;
         pack += kDecodeMegaThreads) {
      const int element = pack * kBf16Packed128Elements;
      const Bf16Packed128 x_pack =
          Bf16Packed128{*reinterpret_cast<const int4 *>(
              args.attention_out + element)};
#pragma unroll
      for (int col = 0; col < kColsPerBlock; ++col) {
        const int hidden_col = col_block * kColsPerBlock + col;
        const int64_t weight_offset =
            int64_t(hidden_col) * kAttentionWidth + element;
        const Bf16Packed128 w_pack =
            Bf16Packed128{*reinterpret_cast<const int4 *>(
                args.attention_o_proj_col_major + weight_offset)};
        gemma4_bf16_pack_accumulate_dot(x_pack, w_pack, sums[col]);
      }
    }

    warp_reduce_sum_to_lane0(sums);
    if (lane == 0) {
#pragma unroll
      for (int col = 0; col < kColsPerBlock; ++col) {
        warp_sums[col][warp] = sums[col];
      }
    }
    __syncthreads();

#pragma unroll
    for (int col = 0; col < kColsPerBlock; ++col) {
      sums[col] = thread < kDecodeMegaWarps ? warp_sums[col][lane] : 0.0f;
    }
    if (warp == 0) {
      warp_reduce_sum_to_lane0(sums);
    }
    if (thread == 0) {
      const int hidden_col0 = col_block * kColsPerBlock;
      Bf16Packed128 o_pack;
#pragma unroll
      for (int col = 0; col < kColsPerBlock; ++col) {
        o_pack[col] = __float2bfloat16_rn(sums[col]);
      }
      *reinterpret_cast<int4 *>(args.residual_out + hidden_col0) =
          o_pack.bits();
    }
    __syncthreads();
  }

  grid.sync();
  if (block == 0) {
    __shared__ Bf16Packed128 cached_input[kFfnHiddenPacks];
    __shared__ float rms_warp_sums[kDecodeMegaWarps];
    __shared__ float rms_scale;
    gemma4_rmsnorm_hidden_row_512_bf16_device(
        args.normed_out, args.residual_out, args.attention_post_norm_weight,
        GEMMA4_RMS_NORM_EPS, cached_input, rms_warp_sums, &rms_scale, thread);
    __syncthreads();

#if GEMMA4_FFN_FOLDED_PRE_NORM
    float pre_ffn_sum_sq = 0.0f;
#endif
    if (thread < kFfnHiddenPacks) {
      const int hidden_col0 = thread * kBf16Packed128Elements;
      const Bf16Packed128 residual =
          Bf16Packed128{*reinterpret_cast<const int4 *>(
              args.attention_x + hidden_col0)};
      const Bf16Packed128 normed =
          Bf16Packed128{*reinterpret_cast<const int4 *>(
              args.normed_out + hidden_col0)};
      const Bf16Packed128 ffn_residual =
          gemma4_bf16_pack_add(residual, normed);
      *reinterpret_cast<int4 *>(args.ffn_residual + hidden_col0) =
          ffn_residual.bits();
#if GEMMA4_FFN_FOLDED_PRE_NORM
      // Square the rounded packs so s2 matches the folded-away pre-FFN norm.
      gemma4_bf16_pack_accumulate_square(ffn_residual, pre_ffn_sum_sq);
      // Stage gamma_pre * y in the same pass so the FFN gate/up stream keeps
      // its baseline load pattern; the row scale s2 is applied post-reduction.
      const Bf16Packed128 pre_norm_gamma =
          Bf16Packed128{*reinterpret_cast<const int4 *>(
              args.attention_pre_ffn_norm_weight + hidden_col0)};
      const Bf16Packed128 gamma_scaled = gemma4_bf16_pack_apply_scale_weight(
          ffn_residual, pre_norm_gamma, 1.0f);
      *reinterpret_cast<int4 *>(args.ffn_x + hidden_col0) =
          gamma_scaled.bits();
#endif
    }
#if GEMMA4_FFN_FOLDED_PRE_NORM
    __shared__ float pre_ffn_warp_sums[kDecodeMegaWarps];
    pre_ffn_sum_sq = warp_reduce_sum(pre_ffn_sum_sq);
    if (lane == 0) {
      pre_ffn_warp_sums[warp] = pre_ffn_sum_sq;
    }
    __syncthreads();
    pre_ffn_sum_sq = thread < kDecodeMegaWarps
                         ? pre_ffn_warp_sums[lane]
                         : 0.0f;
    if (warp == 0) {
      pre_ffn_sum_sq = warp_reduce_sum(pre_ffn_sum_sq);
    }
    if (thread == 0) {
      *args.pre_ffn_scale = rsqrtf(
          pre_ffn_sum_sq / float(GEMMA4_HIDDEN_SIZE) + GEMMA4_RMS_NORM_EPS);
    }
#endif
  }
  // The remaining pre-FFN RMSNorm is CTA-0-owned and reads all residual tiles.
  grid.sync();
}

// Caches host-side cooperative launch geometry for one decode cooperative kernel.
struct DecodeCooperativeLaunchCache {
  void *kernel;
  int device = -1;
  int active_blocks = 0;
};

template <bool IsGlobal>
struct Gemma4AttentionTraits;

template <>
struct Gemma4AttentionTraits<false> {
  static constexpr bool kIsGlobal = false;
  static constexpr bool kHasVProjection = true;
  static constexpr int kKvHeads = GEMMA4_SLIDING_KV_HEADS;
  static constexpr int kHeadDim = GEMMA4_SLIDING_HEAD_DIM;
  static constexpr int kRotaryDim = GEMMA4_SLIDING_HEAD_DIM;
};

template <>
struct Gemma4AttentionTraits<true> {
  static constexpr bool kIsGlobal = true;
  static constexpr bool kHasVProjection = false;
  static constexpr int kKvHeads = GEMMA4_GLOBAL_KV_HEADS;
  static constexpr int kHeadDim = GEMMA4_GLOBAL_HEAD_DIM;
  static constexpr int kRotaryDim = GEMMA4_GLOBAL_HEAD_DIM / 4;
};

template <typename Traits>
struct Gemma4AttentionDerived {
  static constexpr int kPrepThreads = Traits::kHeadDim;
  static constexpr int kDecodeThreads = Traits::kHeadDim;
  static constexpr int kHeadsPerBlock = kPrepThreads / kWarpSize;
  static constexpr int kValuesPerLane = Traits::kHeadDim / kWarpSize;
  static constexpr int kRotaryHalf = Traits::kRotaryDim / 2;
  static constexpr int kRotaryPairsPerLane = kRotaryHalf / kWarpSize;
  static constexpr int kGqaRatio = GEMMA4_NUM_QUERY_HEADS / Traits::kKvHeads;
  static constexpr int kDecodeKWarp =
      kGqaRatio < kHeadsPerBlock ? kGqaRatio : 0;
  static constexpr int kDecodeVWarp = kDecodeKWarp + 1;

  static_assert(kHeadsPerBlock >= kGqaRatio);
  static_assert(!Traits::kHasVProjection || kDecodeVWarp < kHeadsPerBlock);
};

struct alignas(16) DecodeKvCpAsyncStorage {
  __nv_bfloat16 k[2][GEMMA4_SLIDING_HEAD_DIM];
  __nv_bfloat16 v[2][GEMMA4_SLIDING_HEAD_DIM];
};

// Starts one async K/V token copy; the caller owns wait and buffer lifetime.
__device__ __forceinline__ void decode_kv_cp_async_prefetch(
    DecodeKvCpAsyncStorage &storage,
    const __nv_bfloat16 *__restrict__ cache_k,
    const __nv_bfloat16 *__restrict__ cache_v,
    int64_t kv_base,
    int stage,
    int thread_idx) {
  constexpr int kPacks = GEMMA4_SLIDING_HEAD_DIM / kBf16Packed128Elements;
  if (thread_idx < kPacks) {
    const int dim0 = thread_idx * kBf16Packed128Elements;
    void *dst_k = storage.k[stage] + dim0;
    void *dst_v = storage.v[stage] + dim0;
    const void *src_k = cache_k + kv_base + dim0;
    const void *src_v = cache_v + kv_base + dim0;
    const uint32_t smem_k =
        static_cast<uint32_t>(__cvta_generic_to_shared(dst_k));
    const uint32_t smem_v =
        static_cast<uint32_t>(__cvta_generic_to_shared(dst_v));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_k),
                 "l"(src_k));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_v),
                 "l"(src_v));
  }
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ int32_t gemma4_warp_uniform_ldg_i32(
    const int32_t *__restrict__ ptr,
    int lane) {
  int32_t value = 0;
  if (lane == 0) value = __ldg(ptr);
  return __shfl_sync(0xffffffffu, value, 0);
}

template <typename Traits>
__device__ __forceinline__ void prep_load_head_values(
    const __nv_bfloat16 *__restrict__ in,
    int lane,
    float (&values)[Gemma4AttentionDerived<Traits>::kValuesPerLane]) {
  using Derived = Gemma4AttentionDerived<Traits>;
#pragma unroll
  for (int i = 0; i < Derived::kValuesPerLane; ++i) {
    // Lane-strided loads make one warp own one complete attention head.
    values[i] = __bfloat162float(in[lane + i * kWarpSize]);
  }
}

// Apply learned RMSNorm to one Q/K head and then rotate the configured rotary
// prefix. Global attention leaves the non-rotary tail as normalized channels.
template <typename Traits>
__device__ __forceinline__ void prep_weighted_rope_head_values(
    __nv_bfloat16 *__restrict__ out,
    const float (&values)[Gemma4AttentionDerived<Traits>::kValuesPerLane],
    const __nv_bfloat16 *__restrict__ weight,
    const float *__restrict__ cos_row,
    const float *__restrict__ sin_row,
    int lane) {
  using Derived = Gemma4AttentionDerived<Traits>;
  const float scale = gemma4_rmsnorm_warp_scale_f32_device(
      values, Derived::kValuesPerLane, Traits::kHeadDim, GEMMA4_RMS_NORM_EPS);
#pragma unroll
  for (int i = 0; i < Derived::kRotaryPairsPerLane; ++i) {
    const int dim = lane + i * kWarpSize;
    const float lo =
        values[i] * scale * __bfloat162float(loadg(weight + dim));
    const int hi_index = i + Derived::kRotaryPairsPerLane;
    const int hi_dim = Derived::kRotaryHalf + dim;
    const float hi = values[hi_index] * scale *
                     __bfloat162float(loadg(weight + hi_dim));
    gemma4_rope::store_rotated_pair_bf16(
        out, cos_row, sin_row, Derived::kRotaryHalf, dim, lo, hi);
  }

#pragma unroll
  for (int i = 0; i < Derived::kValuesPerLane; ++i) {
    const int dim = lane + i * kWarpSize;
    if (dim >= Traits::kRotaryDim) {
      const float value =
          values[i] * scale * __bfloat162float(loadg(weight + dim));
      out[dim] = __float2bfloat16_rn(value);
    }
  }
}

// Apply scale-free RMSNorm to one V head. Global attention derives V from the
// raw K projection and then applies this same scale-free normalization.
template <typename Traits>
__device__ __forceinline__ void prep_scale_free_head_values(
    __nv_bfloat16 *__restrict__ out,
    const float (&values)[Gemma4AttentionDerived<Traits>::kValuesPerLane],
    int lane) {
  using Derived = Gemma4AttentionDerived<Traits>;
  const float scale = gemma4_rmsnorm_warp_scale_f32_device(
      values, Derived::kValuesPerLane, Traits::kHeadDim, GEMMA4_RMS_NORM_EPS);
#pragma unroll
  for (int i = 0; i < Derived::kValuesPerLane; ++i) {
    out[lane + i * kWarpSize] = __float2bfloat16_rn(values[i] * scale);
  }
}

// Convert a decode token position into the Layout-A paged-cache row for one KV
// head. Page allocation stays on the host/runtime side; this kernel only writes
// if the page table already maps the requested position.
__device__ __forceinline__ int64_t decode_cache_head_offset(
    const Gemma4KvCacheConfig &config,
    const int32_t *__restrict__ page_table,
    int batch,
    int position,
    int cache_layer,
    int head,
    int lane) {
  const int logical_page = position / config.page_size;
  const int slot = logical_page % config.max_pages_per_seq;
  const int64_t page_table_offset =
      int64_t(batch) * config.max_pages_per_seq + slot;
  const int physical_page = gemma4_warp_uniform_ldg_i32(
      page_table + page_table_offset, lane);
  if (physical_page < 0 || physical_page >= config.num_pages) return -1;
  const int page_offset = position - logical_page * config.page_size;
  return gemma4_kv_cache_offset(config, cache_layer, physical_page, page_offset,
                                head, 0);
}

// Prepare one decode token's Q and paged K/V cache entry for one KV head.
template <typename Traits>
__device__ __forceinline__ void phase_decode_q_paged_kv_norm_rope(
    __nv_bfloat16 *__restrict__ q_prepared,
    __nv_bfloat16 *__restrict__ cache_k,
    __nv_bfloat16 *__restrict__ cache_v,
    Gemma4KvCacheConfig cache_config,
    const int32_t *__restrict__ page_table,
    const int32_t *__restrict__ token_position,
    int cache_layer,
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    const __nv_bfloat16 *__restrict__ q_norm_weight,
    const __nv_bfloat16 *__restrict__ k_norm_weight,
    const float *__restrict__ cos,
    const float *__restrict__ sin,
    int batch,
    int kv_head,
    int thread_idx) {
  using Derived = Gemma4AttentionDerived<Traits>;
  constexpr int kQHeads = GEMMA4_NUM_QUERY_HEADS;
  constexpr int kKvHeads = Traits::kKvHeads;
  constexpr int kHeadDim = Traits::kHeadDim;

  const int lane = thread_idx & (kWarpSize - 1);
  const int warp = thread_idx / kWarpSize;

  const int position =
      gemma4_warp_uniform_ldg_i32(token_position + batch, lane);
  if (position < 0) return;
  const float *cos_row = cos + int64_t(position) * Derived::kRotaryHalf;
  const float *sin_row = sin + int64_t(position) * Derived::kRotaryHalf;

  if (warp < Derived::kGqaRatio) {
    const int q_head = kv_head * Derived::kGqaRatio + warp;
    const int64_t q_offset = (int64_t(batch) * kQHeads + q_head) * kHeadDim;
    float q_values[Derived::kValuesPerLane];
    prep_load_head_values<Traits>(q + q_offset, lane, q_values);
    prep_weighted_rope_head_values<Traits>(
        q_prepared + q_offset, q_values, q_norm_weight, cos_row, sin_row,
        lane);
    if (warp != Derived::kDecodeKWarp) return;
  }

  // One decode producer owns one KV head and that head's GQA query group.
  if (warp != Derived::kDecodeKWarp &&
      (!Traits::kHasVProjection || warp != Derived::kDecodeVWarp)) {
    return;
  }
  const int64_t cache_offset = decode_cache_head_offset(
      cache_config, page_table, batch, position, cache_layer, kv_head, lane);
  if (cache_offset < 0) return;

  const int64_t kv_offset = (int64_t(batch) * kKvHeads + kv_head) * kHeadDim;
  if (warp == Derived::kDecodeKWarp) {
    float k_values[Derived::kValuesPerLane];
    prep_load_head_values<Traits>(k + kv_offset, lane, k_values);
    prep_weighted_rope_head_values<Traits>(
        cache_k + cache_offset, k_values, k_norm_weight, cos_row, sin_row,
        lane);
    if constexpr (!Traits::kHasVProjection) {
      prep_scale_free_head_values<Traits>(
          cache_v + cache_offset, k_values, lane);
    }
  } else if constexpr (Traits::kHasVProjection) {
    float v_values[Derived::kValuesPerLane];
    prep_load_head_values<Traits>(v + kv_offset, lane, v_values);
    prep_scale_free_head_values<Traits>(cache_v + cache_offset, v_values, lane);
  }
}

template <typename Traits>
bool gemma4_fa_valid_cache_config(
    const Gemma4KvCacheConfig &config,
    int32_t cache_layer) {
  const int64_t token_capacity =
      int64_t(config.max_pages_per_seq) * config.page_size;
  return cache_layer >= 0 && cache_layer < config.num_layers &&
         config.num_pages > 0 && config.page_size > 0 &&
         config.max_pages_per_seq > 0 &&
         config.batch_size > 0 &&
         config.num_heads == Traits::kKvHeads &&
         config.head_dim == Traits::kHeadDim &&
         (Traits::kIsGlobal ? config.window_size == 0
                            : config.window_size > 0 &&
                                  token_capacity >= config.window_size);
}

// Reduces one projection tile's partial sums across a CTA.
template <int ColsPerBlock, int Threads>
__device__ inline void project_reduce_cols(
    int thread_idx,
    float (&warp_sums)[ColsPerBlock][Threads / kWarpSize],
    float (&sums)[ColsPerBlock]) {
  constexpr int warps = Threads / kWarpSize;
  const int lane = thread_idx & (kWarpSize - 1);
  const int warp = thread_idx / kWarpSize;

  warp_reduce_sum_to_lane0(sums);
  if (lane == 0) {
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      warp_sums[col][warp] = sums[col];
    }
  }
  __syncthreads();

#pragma unroll
  for (int col = 0; col < ColsPerBlock; ++col) {
    sums[col] = thread_idx < warps ? warp_sums[col][lane] : 0.0f;
  }
  if (warp == 0) {
    warp_reduce_sum_to_lane0(sums);
  }
}

// Dots one output-column tile against this thread's cached BF16 RMSNorm pack.
template <int ColsPerBlock, int Threads>
__device__ inline void project_dot_cached_rmsnorm_cols(
    const Bf16Packed128 &normed_pack,
    const __nv_bfloat16 *__restrict__ w_col_major,
    int col0,
    int thread_idx,
    float (&sums)[ColsPerBlock]) {
  constexpr int packs_per_col = GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
  if (thread_idx >= packs_per_col) {
    return;
  }

  const int element_idx = thread_idx * kBf16Packed128Elements;
#pragma unroll
  for (int col = 0; col < ColsPerBlock; ++col) {
    const int weight_idx = (col0 + col) * GEMMA4_HIDDEN_SIZE + element_idx;
    const Bf16Packed128 w_pack =
        Bf16Packed128{
            *reinterpret_cast<const int4 *>(w_col_major + weight_idx)};
    gemma4_bf16_pack_accumulate_dot(normed_pack, w_pack, sums[col]);
  }
}

// Projects BF16(input RMSNorm(hidden)) into raw QKV/QK scratch for one layer.
template <int ProjectionWidth>
__device__ inline void phase_decode_rmsnorm_project_bf16(
    const Gemma4DecodeMegakernelLayerArgs &args,
    __nv_bfloat16 *__restrict__ raw_qkv_scratch,
    cg::grid_group grid) {
  constexpr int cols_per_block = 8;
  constexpr int tiles = ProjectionWidth / cols_per_block;
  static_assert(ProjectionWidth % cols_per_block == 0,
                "projection width must be divisible by the packed tile");

  const int thread_idx = int(threadIdx.x);
  if (blockIdx.x == 0) {
    __shared__ Bf16Packed128 cached_input[kFfnHiddenPacks];
    __shared__ float rms_warp_sums[kDecodeMegaWarps];
    __shared__ float rms_scale;
    gemma4_rmsnorm_hidden_row_512_bf16_device(
        args.normed_out, args.attention_x, args.attention_input_norm_weight,
        GEMMA4_RMS_NORM_EPS, cached_input, rms_warp_sums, &rms_scale,
        thread_idx);
  }
  grid.sync();

  __shared__ float warp_sums[cols_per_block][kDecodeMegaWarps];

  Bf16Packed128 normed_pack;
  if (thread_idx < kFfnHiddenPacks) {
    const int element = thread_idx * kBf16Packed128Elements;
    normed_pack = Bf16Packed128{
        *reinterpret_cast<const int4 *>(args.normed_out + element)};
  }

  for (int tile = int(blockIdx.x); tile < tiles; tile += int(gridDim.x)) {
    float sums[cols_per_block] = {};
    const int col0 = tile * cols_per_block;
    project_dot_cached_rmsnorm_cols<cols_per_block, kDecodeMegaThreads>(
        normed_pack, args.attention_qkv_proj_col_major, col0, thread_idx,
        sums);
    project_reduce_cols<cols_per_block, kDecodeMegaThreads>(
        thread_idx, warp_sums, sums);
    if (thread_idx == 0) {
      Bf16Packed128 out;
#pragma unroll
      for (int col = 0; col < cols_per_block; ++col) {
        out[col] = __float2bfloat16_rn(sums[col]);
      }
      *reinterpret_cast<int4 *>(raw_qkv_scratch + col0) = out.bits();
    }
    __syncthreads();
  }
}

// Projects raw attention inputs, then prepares Q and the paged K/V cache.
template <typename Traits, int ProjectionWidth>
__device__ inline void phase_megakernel_project_prepare_paged_kv(
    const Gemma4DecodeMegakernelLayerArgs &args,
    __nv_bfloat16 *__restrict__ raw_qkv_scratch,
    cg::grid_group grid) {
  phase_decode_rmsnorm_project_bf16<ProjectionWidth>(
      args, raw_qkv_scratch, grid);
  grid.sync();
  constexpr int q_width = GEMMA4_NUM_QUERY_HEADS * Traits::kHeadDim;
  constexpr int kv_width = Traits::kKvHeads * Traits::kHeadDim;
  const __nv_bfloat16 *raw_q = raw_qkv_scratch;
  const __nv_bfloat16 *raw_k = raw_q + q_width;
  const __nv_bfloat16 *raw_v =
      raw_k + (Traits::kHasVProjection ? kv_width : 0);

  for (int kv_head = int(blockIdx.x); kv_head < Traits::kKvHeads;
       kv_head += int(gridDim.x)) {
    phase_decode_q_paged_kv_norm_rope<Traits>(
        args.attention_q, args.attention_cache_k, args.attention_cache_v,
        args.attention_cache_config, args.attention_page_table,
        args.attention_token_position, args.attention_cache_layer, raw_q,
        raw_k, raw_v, args.attention_q_norm_weight,
        args.attention_k_norm_weight, args.attention_cos, args.attention_sin,
        0, kv_head, int(threadIdx.x));
  }
  grid.sync();
}

// Online softmax update for one output lane. `m` tracks the running max,
// `l` tracks the running denominator in that max's scale, and `acc` tracks the
// V-weighted numerator for the calling thread's head dimension.
__device__ __forceinline__ void decode_online_update(
    float score,
    float v_value,
    float &m,
    float &l,
    float &acc) {
  const float new_m = fmaxf(m, score);
  const float old_scale = __expf(m - new_m);
  const float new_scale = __expf(score - new_m);
  acc = acc * old_scale + v_value * new_scale;
  l = l * old_scale + new_scale;
  m = new_m;
}

// Processes one valid sliding-cache page span with a two-stage K/V copy pipe.
template <typename Traits,
          int DecodeThreads = Gemma4AttentionDerived<Traits>::kDecodeThreads>
__device__ __forceinline__ void phase_decode_paged_grouped_split_cp_async_span(
    DecodeKvCpAsyncStorage &storage,
    float (&acc)[Gemma4AttentionDerived<Traits>::kGqaRatio],
    float (&m)[Gemma4AttentionDerived<Traits>::kGqaRatio],
    float (&l)[Gemma4AttentionDerived<Traits>::kGqaRatio],
    const float (&q_values)[Gemma4AttentionDerived<Traits>::kGqaRatio],
    typename cub::BlockReduce<float, DecodeThreads>::TempStorage
        (&reduce_storage)[Gemma4AttentionDerived<Traits>::kGqaRatio],
    float (&s_score)[Gemma4AttentionDerived<Traits>::kGqaRatio],
    const __nv_bfloat16 *__restrict__ cache_k,
    const __nv_bfloat16 *__restrict__ cache_v,
    int64_t kv_base,
    int64_t kv_token_stride,
    int32_t &page_pos,
    int32_t span_end,
    float softmax_scale,
    int32_t dim,
    bool active_dim) {
  using Derived = Gemma4AttentionDerived<Traits>;
  static_assert(!Traits::kIsGlobal);
  static_assert(Traits::kHeadDim == GEMMA4_SLIDING_HEAD_DIM);

  int stage = 0;
  decode_kv_cp_async_prefetch(
      storage, cache_k, cache_v, kv_base, stage, dim);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();

  for (; page_pos < span_end; ++page_pos, kv_base += kv_token_stride) {
    const bool has_next = page_pos + 1 < span_end;
    if (has_next) {
      decode_kv_cp_async_prefetch(
          storage, cache_k, cache_v, kv_base + kv_token_stride, stage ^ 1,
          dim);
    }

    const float k_value =
        active_dim ? __bfloat162float(storage.k[stage][dim]) : 0.0f;
    const float v_value =
        active_dim ? __bfloat162float(storage.v[stage][dim]) : 0.0f;

    float scores[Derived::kGqaRatio];
#pragma unroll
    for (int i = 0; i < Derived::kGqaRatio; ++i) {
      scores[i] =
          cub::BlockReduce<float, DecodeThreads>(reduce_storage[i])
              .Sum(q_values[i] * k_value);
      if (dim == 0) s_score[i] = scores[i];
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Derived::kGqaRatio; ++i) {
      decode_online_update(
          s_score[i] * softmax_scale, v_value, m[i], l[i], acc[i]);
    }

    if (has_next) {
      asm volatile("cp.async.wait_group 0;\n" ::);
      __syncthreads();
      stage ^= 1;
    }
  }
}

// Computes one global-attention Q-head row when the live key range fits in one split.
template <int DecodeThreads>
__device__ __forceinline__ void phase_decode_global_q_head_direct(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ cache_k,
    const __nv_bfloat16 *__restrict__ cache_v,
    const int32_t *__restrict__ page_table,
    const int32_t *__restrict__ seq_lengths,
    Gemma4KvCacheConfig config,
    int32_t layer,
    float softmax_scale,
    int32_t split_size,
    int32_t q_head,
    int32_t dim,
    int32_t batch_count) {
  constexpr int kHeadDim = GEMMA4_GLOBAL_HEAD_DIM;
  const bool active_dim = dim < kHeadDim;
  const int32_t lane = dim & (kWarpSize - 1);
  const int32_t seq_len = gemma4_warp_uniform_ldg_i32(seq_lengths, lane);
  if (seq_len <= 0) {
    if (active_dim) {
      out[int64_t(q_head) * kHeadDim + dim] = __float2bfloat16_rn(0.0f);
    }
    return;
  }

  const int32_t split_end = std::min(seq_len, split_size);
  const int64_t q_base = int64_t(q_head) * kHeadDim;
  const float q_value =
      active_dim ? __bfloat162float(loadg(q + q_base + dim)) : 0.0f;
  float acc = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;

  using DecodeBlockReduce = cub::BlockReduce<float, DecodeThreads>;
  __shared__ typename DecodeBlockReduce::TempStorage reduce_storage;
  __shared__ float s_score;
  auto page_table_layout = make_layout(
      make_shape(batch_count, config.max_pages_per_seq),
      make_stride(config.max_pages_per_seq, 1));

  for (int32_t page_pos = 0; page_pos < split_end;) {
    const int32_t logical_page = page_pos / config.page_size;
    const int32_t page_offset0 = page_pos - logical_page * config.page_size;
    const int32_t span_end =
        std::min(split_end, page_pos + config.page_size - page_offset0);
    const int32_t page_slot = logical_page % config.max_pages_per_seq;
    const int32_t physical_page = gemma4_warp_uniform_ldg_i32(
        page_table + page_table_layout(0, page_slot), lane);
    if (physical_page < 0 || physical_page >= config.num_pages) {
      page_pos = span_end;
      continue;
    }

    int64_t kv_base = gemma4_kv_cache_offset(
        config, layer, physical_page, page_offset0, 0, 0);
    const int64_t kv_token_stride =
        int64_t(config.num_heads) * config.head_dim;
    for (; page_pos < span_end; ++page_pos, kv_base += kv_token_stride) {
      const float k_value =
          active_dim ? __bfloat162float(loadg(cache_k + kv_base + dim)) : 0.0f;
      const float v_value =
          active_dim ? __bfloat162float(loadg(cache_v + kv_base + dim)) : 0.0f;
      const float score =
          DecodeBlockReduce(reduce_storage).Sum(q_value * k_value);
      if (dim == 0) s_score = score;
      __syncthreads();
      decode_online_update(s_score * softmax_scale, v_value, m, l, acc);
    }
  }

  const float attention_value = active_dim && l > 0.0f ? acc / l : 0.0f;
  if (active_dim) {
    out[int64_t(q_head) * kHeadDim + dim] =
        __float2bfloat16_rn(attention_value);
  }
}

// Compute one paged-decode split for a caller-selected batch/KV-head/split CTA.
template <typename Traits,
          int DecodeThreads = Gemma4AttentionDerived<Traits>::kDecodeThreads>
__device__ __forceinline__ void phase_decode_paged_grouped_split(
    __nv_bfloat16 *__restrict__ direct_out,
    float *__restrict__ partial_m,
    float *__restrict__ partial_l,
    float *__restrict__ partial_acc,
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ cache_k,
    const __nv_bfloat16 *__restrict__ cache_v,
    const int32_t *__restrict__ page_table,
    const int32_t *__restrict__ seq_lengths,
    Gemma4KvCacheConfig config,
    int32_t layer,
    float softmax_scale,
    int32_t split_size,
    int32_t num_splits,
    int32_t batch,
    int32_t kv_head,
    int32_t split,
    int32_t dim,
    int32_t batch_count) {
  using Derived = Gemma4AttentionDerived<Traits>;
  // CTA mapping: x=batch row, y=KV head, z=split. Within the CTA, one thread
  // owns one head-dim lane, so the block-wide reductions cover the full dot.
  const bool active_dim = dim < Traits::kHeadDim;
  const int32_t lane = dim & (kWarpSize - 1);

  int32_t q_heads[Derived::kGqaRatio];
#pragma unroll
  for (int i = 0; i < Derived::kGqaRatio; ++i) {
    q_heads[i] = kv_head * Derived::kGqaRatio + i;
  }

  // The caller selects `num_splits`; graph callers may overprovision, while
  // no-graph decode can pass only the live split count for this step.
  const int32_t seq_len =
      gemma4_warp_uniform_ldg_i32(seq_lengths + batch, lane);
  const int32_t first_key =
      config.window_size > 0 ? std::max(0, seq_len - config.window_size) : 0;
  const int32_t key_count = std::max(0, seq_len - first_key);
  const int32_t actual_splits = div_up(key_count, split_size);
  const bool write_direct = num_splits == 1;
  if (key_count == 0 && write_direct) {
#pragma unroll
    for (int i = 0; i < Derived::kGqaRatio; ++i) {
      const int32_t row = batch * GEMMA4_NUM_QUERY_HEADS + q_heads[i];
      if (active_dim) {
        direct_out[int64_t(row) * Traits::kHeadDim + dim] =
            __float2bfloat16_rn(0.0f);
      }
    }
    return;
  }
  // Overprovisioned CTAs must not write neutral scratch; the reducer will also
  // stop at the same live split count.
  if (split >= actual_splits) return;

  // Keep the partial buffer stride as `num_splits` even if graph callers
  // overprovision beyond the row's live split count.
  const int32_t split_begin = first_key + split * split_size;
  const int32_t split_end = std::min(seq_len, split_begin + split_size);
  int32_t partials[Derived::kGqaRatio];
#pragma unroll
  for (int i = 0; i < Derived::kGqaRatio; ++i) {
    partials[i] =
        (batch * GEMMA4_NUM_QUERY_HEADS + q_heads[i]) * num_splits + split;
  }

  // Q is tiny for decode: each thread loads and keeps its GQA scalar Q lanes in
  // registers for every key in this split.
  float q_values[Derived::kGqaRatio];
#pragma unroll
  for (int i = 0; i < Derived::kGqaRatio; ++i) {
    const int64_t q_base =
        (int64_t(batch) * GEMMA4_NUM_QUERY_HEADS + q_heads[i]) *
        Traits::kHeadDim;
    q_values[i] =
        active_dim ? __bfloat162float(loadg(q + q_base + dim)) : 0.0f;
  }

  float acc[Derived::kGqaRatio];
  float m[Derived::kGqaRatio];
  float l[Derived::kGqaRatio];
#pragma unroll
  for (int i = 0; i < Derived::kGqaRatio; ++i) {
    acc[i] = 0.0f;
    m[i] = -INFINITY;
    l[i] = 0.0f;
  }

  using DecodeBlockReduce = cub::BlockReduce<float, DecodeThreads>;
  __shared__ typename DecodeBlockReduce::TempStorage
      reduce_storage[Derived::kGqaRatio];
  __shared__ float s_score[Derived::kGqaRatio];
  auto page_table_layout = make_layout(
      make_shape(batch_count, config.max_pages_per_seq),
      make_stride(config.max_pages_per_seq, 1));

  for (int32_t page_pos = split_begin; page_pos < split_end;) {
    // Resolve one logical page span at a time. This keeps page-table loads
    // read-only, but removes repeated page table, division/modulo, and full
    // cache-offset work for every token inside the same cache page.
    const int32_t logical_page = page_pos / config.page_size;
    const int32_t page_offset0 = page_pos - logical_page * config.page_size;
    const int32_t span_end =
        std::min(split_end, page_pos + config.page_size - page_offset0);
    const int32_t page_slot = logical_page % config.max_pages_per_seq;
    const int32_t physical_page = gemma4_warp_uniform_ldg_i32(
        page_table + page_table_layout(batch, page_slot), lane);
    if (physical_page < 0 || physical_page >= config.num_pages) {
      page_pos = span_end;
      continue;
    }

    // Cache layout is [layer, page, page_offset, kv_head, dim]. Advancing one
    // token inside a page is a fixed stride over all KV heads.
    int64_t kv_base = gemma4_kv_cache_offset(
        config, layer, physical_page, page_offset0, kv_head, 0);
    const int64_t kv_token_stride =
        int64_t(config.num_heads) * config.head_dim;
    if constexpr (!Traits::kIsGlobal &&
                  Traits::kHeadDim == GEMMA4_SLIDING_HEAD_DIM) {
      __shared__ DecodeKvCpAsyncStorage cp_async_storage;
      phase_decode_paged_grouped_split_cp_async_span<Traits, DecodeThreads>(
          cp_async_storage, acc, m, l, q_values, reduce_storage, s_score,
          cache_k, cache_v, kv_base, kv_token_stride, page_pos, span_end,
          softmax_scale, dim, active_dim);
    } else {
      for (; page_pos < span_end; ++page_pos, kv_base += kv_token_stride) {
        const float k_value = active_dim
                                  ? __bfloat162float(loadg(
                                        cache_k + kv_base + dim))
                                  : 0.0f;
        const float v_value = active_dim
                                  ? __bfloat162float(loadg(
                                        cache_v + kv_base + dim))
                                  : 0.0f;

        float scores[Derived::kGqaRatio];
#pragma unroll
        for (int i = 0; i < Derived::kGqaRatio; ++i) {
          scores[i] =
              DecodeBlockReduce(reduce_storage[i]).Sum(q_values[i] * k_value);
          if (dim == 0) s_score[i] = scores[i];
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < Derived::kGqaRatio; ++i) {
          decode_online_update(
              s_score[i] * softmax_scale, v_value, m[i], l[i], acc[i]);
        }
      }
    }
  }

  // A single split already covers the whole live key range, so it can write the
  // final BF16 attention row directly and avoid the partial scratch/reduce pass.
  if (write_direct) {
#pragma unroll
    for (int i = 0; i < Derived::kGqaRatio; ++i) {
      const int32_t row = batch * GEMMA4_NUM_QUERY_HEADS + q_heads[i];
      const float attention_value =
          active_dim && l[i] > 0.0f ? acc[i] / l[i] : 0.0f;
      if (active_dim) {
        direct_out[int64_t(row) * Traits::kHeadDim + dim] =
            __float2bfloat16_rn(attention_value);
      }
    }
    return;
  }

  if (dim == 0) {
#pragma unroll
    for (int i = 0; i < Derived::kGqaRatio; ++i) {
      partial_m[partials[i]] = m[i];
      partial_l[partials[i]] = l[i];
    }
  }
#pragma unroll
  for (int i = 0; i < Derived::kGqaRatio; ++i) {
    if (active_dim) {
      partial_acc[int64_t(partials[i]) * Traits::kHeadDim + dim] = acc[i];
    }
  }
}

// Reduce live paged-decode split partials for one batch/query-head row.
template <typename Traits,
          int DecodeThreads = Gemma4AttentionDerived<Traits>::kDecodeThreads>
__device__ __forceinline__ void phase_decode_paged_reduce(
    __nv_bfloat16 *__restrict__ out,
    const float *__restrict__ partial_m,
    const float *__restrict__ partial_l,
    const float *__restrict__ partial_acc,
    const int32_t *__restrict__ seq_lengths,
    Gemma4KvCacheConfig config,
    int32_t split_size,
    int32_t num_splits,
    int32_t batch,
    int32_t q_head,
    int32_t dim,
    int32_t thread_count) {
  using Derived = Gemma4AttentionDerived<Traits>;
  // One CTA reduces all live split partials for one [batch, query head] output
  // row; each thread writes one BF16 head-dim lane.
  const bool active_dim = dim < Traits::kHeadDim;
  const int32_t lane = dim & (kWarpSize - 1);
  const int32_t row = batch * GEMMA4_NUM_QUERY_HEADS + q_head;

  // Partial buffers are laid out with the caller-selected `num_splits` stride.
  const int32_t partial_row = row * num_splits;
  const int64_t partial_acc_row =
      int64_t(row) * num_splits * Traits::kHeadDim;

  // Recompute the live split count from device sequence lengths so
  // overprovisioned launches can ignore stale scratch beyond `actual_splits`.
  const int32_t seq_len =
      gemma4_warp_uniform_ldg_i32(seq_lengths + batch, lane);
  const int32_t first_key =
      config.window_size > 0 ? std::max(0, seq_len - config.window_size) : 0;
  const int32_t key_count = std::max(0, seq_len - first_key);
  const int32_t actual_splits = div_up(key_count, split_size);
  const int32_t reduce_splits = std::min(actual_splits, num_splits);

  // A zero-length row has no attention mass. This should be rare, but returning
  // zeros keeps the kernel total for defensive tests and future graph captures.
  if (reduce_splits == 0) {
    if (active_dim) {
      out[(int64_t(row) * Traits::kHeadDim) + dim] =
          __float2bfloat16_rn(0.0f);
    }
    return;
  }

  // First pass: find the global max across live splits for numerical stability.
  float local_m = -INFINITY;
  for (int32_t split = dim; split < reduce_splits; split += thread_count) {
    local_m = fmaxf(local_m, partial_m[partial_row + split]);
  }

  // Separate CUB storage mirrors the split kernel: max-reduce and sum-reduce
  // have distinct shared temp storage so CUB internals are not repurposed.
  using DecodeBlockReduce = cub::BlockReduce<float, DecodeThreads>;
  __shared__ typename DecodeBlockReduce::TempStorage reduce_m_storage;
  __shared__ typename DecodeBlockReduce::TempStorage reduce_l_storage;
  __shared__ float s_m;
  __shared__ float s_l;
  const float block_m = DecodeBlockReduce(reduce_m_storage).Reduce(
      local_m, [] __device__(float a, float b) { return fmaxf(a, b); });
  if (dim == 0) s_m = block_m;
  __syncthreads();

  // Second pass: rescale every split denominator into the global-max frame and
  // sum them into the final softmax denominator.
  float local_l = 0.0f;
  for (int32_t split = dim; split < reduce_splits; split += thread_count) {
    const float split_l = partial_l[partial_row + split];
    if (split_l > 0.0f) {
      local_l += split_l * __expf(partial_m[partial_row + split] - s_m);
    }
  }
  const float block_l = DecodeBlockReduce(reduce_l_storage).Sum(local_l);
  if (dim == 0) s_l = block_l;
  __syncthreads();

  // Final pass: every thread combines its own accumulator lane from all live
  // splits using the same max-rescale factor, then normalizes by `s_l`.
  float value = 0.0f;
  if (active_dim) {
    for (int32_t split = 0; split < reduce_splits; ++split) {
      const float split_l = partial_l[partial_row + split];
      if (split_l > 0.0f) {
        value += partial_acc[partial_acc_row + int64_t(split) *
                             Traits::kHeadDim + dim] *
                 __expf(partial_m[partial_row + split] - s_m);
      }
    }
  }
  const float attention_value = active_dim && s_l > 0.0f ? value / s_l : 0.0f;
  if (active_dim) {
    out[(int64_t(row) * Traits::kHeadDim) + dim] =
        __float2bfloat16_rn(attention_value);
  }
}

// Runs split-KV decode attention, then projects the final row through O.
template <typename Traits>
__device__ inline void phase_megakernel_flash_attention_o_projection(
    const Gemma4DecodeMegakernelLayerArgs &args,
    cg::grid_group grid) {
  const int block = int(blockIdx.x);
  const int stride = int(gridDim.x);
  if constexpr (Traits::kIsGlobal) {
    if (args.attention_num_splits == 1) {
      for (int q_head = block; q_head < GEMMA4_NUM_QUERY_HEADS;
           q_head += stride) {
        phase_decode_global_q_head_direct<kDecodeMegaThreads>(
            args.attention_out, args.attention_q,
            args.attention_cache_k, args.attention_cache_v,
            args.attention_page_table, args.attention_seq_lengths,
            args.attention_cache_config, args.attention_cache_layer,
            args.attention_softmax_scale, args.attention_split_size, q_head,
            int(threadIdx.x), 1);
      }
      grid.sync();
      project_attention_out_post_attention<Traits>(args, grid);
      return;
    }
  }

  const int split_tasks = Traits::kKvHeads * args.attention_num_splits;
  for (int task = block; task < split_tasks; task += stride) {
    const int kv_head = task / args.attention_num_splits;
    const int split = task - kv_head * args.attention_num_splits;
    phase_decode_paged_grouped_split<Traits, kDecodeMegaThreads>(
        args.attention_out, args.attention_partial_m, args.attention_partial_l,
        args.attention_partial_acc, args.attention_q,
        args.attention_cache_k, args.attention_cache_v,
        args.attention_page_table, args.attention_seq_lengths,
        args.attention_cache_config, args.attention_cache_layer,
        args.attention_softmax_scale, args.attention_split_size,
        args.attention_num_splits, 0, kv_head, split, int(threadIdx.x), 1);
  }
  grid.sync();

  if (args.attention_num_splits == 1) {
    project_attention_out_post_attention<Traits>(args, grid);
    return;
  }
  for (int q_head = block; q_head < GEMMA4_NUM_QUERY_HEADS;
       q_head += stride) {
    phase_decode_paged_reduce<Traits, kDecodeMegaThreads>(
        args.attention_out, args.attention_partial_m,
        args.attention_partial_l, args.attention_partial_acc,
        args.attention_seq_lengths, args.attention_cache_config,
        args.attention_split_size, args.attention_num_splits, 0, q_head,
        int(threadIdx.x), int(blockDim.x));
  }
  grid.sync();
  project_attention_out_post_attention<Traits>(args, grid);
}

// GEMMA4_DECODE_MEGA_MIN_BLOCKS=2 trades registers per thread (ptxas cap 64
// on sm_86) for a second resident CTA per SM, doubling latency-hiding warps.
#ifndef GEMMA4_DECODE_MEGA_MIN_BLOCKS
#define GEMMA4_DECODE_MEGA_MIN_BLOCKS 1
#endif

// Runs one fused decode layer from attention ingress through the FFN tail.
template <typename Traits>
__global__ __launch_bounds__(kDecodeMegaThreads, GEMMA4_DECODE_MEGA_MIN_BLOCKS)
void decode_megakernel_fused_layer_kernel(
    Gemma4DecodeMegakernelLayerArgs args) {
  cg::grid_group grid = cg::this_grid();
  if constexpr (Traits::kIsGlobal) {
    phase_megakernel_project_prepare_paged_kv<
        Traits, GEMMA4_GLOBAL_QK_PROJ_SIZE>(args, args.attention_q, grid);
  } else {
    phase_megakernel_project_prepare_paged_kv<
        Traits, GEMMA4_SLIDING_QKV_SIZE>(args, args.attention_out, grid);
  }

  phase_megakernel_flash_attention_o_projection<Traits>(args, grid);

#if GEMMA4_FFN_FOLDED_PRE_NORM
  // The pre-FFN norm scale is folded into the gate/up epilogue: the FFN
  // consumes the CTA0-staged gamma*y row plus the s2 scale stored before the
  // last grid sync, so the gate/up weight stream keeps its baseline pattern.
  gemma4_ffn_decode_fused_bf16_device(
      args.residual_out, args.attention_out, args.ffn_x,
      args.ffn_residual, args.ffn_norm_weight, args.ffn_gate_up_decode,
      args.ffn_down_decode, args.ffn_scratch, args.layer_scalar,
      GEMMA4_RMS_NORM_EPS, args.attention_pre_ffn_norm_weight,
      args.pre_ffn_scale);
#else
  // CTA 0 builds the pre-FFN normalized row after the post-attention residual.
  if (blockIdx.x == 0) {
    __shared__ Bf16Packed128 cached_input[kFfnHiddenPacks];
    __shared__ float warp_sums[kDecodeMegaWarps];
    __shared__ float scale;
    gemma4_rmsnorm_hidden_row_512_bf16_device(
        args.ffn_x, args.ffn_residual, args.attention_pre_ffn_norm_weight,
        GEMMA4_RMS_NORM_EPS, cached_input, warp_sums, &scale, int(threadIdx.x));
  }
  grid.sync();
  // Reuse attention scratch for FFN normed output; args.normed_out stays post-attn.
  gemma4_ffn_decode_fused_bf16_device(
      args.residual_out, args.attention_out, args.ffn_x, args.ffn_residual,
      args.ffn_norm_weight, args.ffn_gate_up_decode, args.ffn_down_decode,
      args.ffn_scratch, args.layer_scalar, GEMMA4_RMS_NORM_EPS,
      args.attention_pre_ffn_norm_weight, args.pre_ffn_scale);
#endif
}

// Launches the cooperative decode grid that owns the fused layer body.
template <typename Kernel>
cudaError_t launch_decode_layer(
    const Gemma4DecodeMegakernelLayerArgs &args,
    cudaStream_t stream,
    Kernel kernel) {
  static DecodeCooperativeLaunchCache launch_cache[2];
  int device = 0;
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaGetDevice(&device));
  void *kernel_ptr = reinterpret_cast<void *>(kernel);

  // Device properties and occupancy are invariant across all decode layers.
  DecodeCooperativeLaunchCache *cached = &launch_cache[0];
  bool cache_hit = false;
  for (DecodeCooperativeLaunchCache &entry : launch_cache) {
    if (entry.kernel == kernel_ptr && entry.device == device &&
        entry.active_blocks > 0) {
      cached = &entry;
      cache_hit = true;
    }
  }
  if (!cache_hit) {
    cached = launch_cache[0].active_blocks == 0 ? &launch_cache[0]
                                                : &launch_cache[1];
    cudaDeviceProp prop = {};
    GEMMA4_RETURN_IF_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));
    if (!prop.cooperativeLaunch) {
      return cudaErrorNotSupported;
    }

    int active_blocks_per_sm = 0;
    GEMMA4_RETURN_IF_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks_per_sm, kernel, kDecodeMegaThreads, 0));
    cached->kernel = kernel_ptr;
    cached->device = device;
    cached->active_blocks = active_blocks_per_sm * prop.multiProcessorCount;
    if (cached->active_blocks <= 0) {
      return cudaErrorInvalidValue;
    }
  }

  Gemma4DecodeMegakernelLayerArgs kernel_args_value = args;
  void *kernel_args[] = {&kernel_args_value};
  const cudaError_t status = cudaLaunchCooperativeKernel(
      kernel_ptr, cached->active_blocks, kDecodeMegaThreads, kernel_args, 0,
      stream);
  GEMMA4_RETURN_IF_CUDA_ERROR(status);
  return cudaGetLastError();
}

}  // namespace gemma4_flash_attention

// Launches one fused decode layer from attention ingress through FFN post-norm.
cudaError_t gemma4_decode_megakernel_flash_attention_layer_bf16(
    const Gemma4DecodeMegakernelLayerArgs &args,
    cudaStream_t stream) {
  const Gemma4KvCacheConfig &config = args.attention_cache_config;
  if (config.head_dim == GEMMA4_SLIDING_HEAD_DIM &&
      config.num_heads == GEMMA4_SLIDING_KV_HEADS) {
    using Traits = gemma4_flash_attention::Gemma4AttentionTraits<false>;
    if (!gemma4_flash_attention::gemma4_fa_valid_cache_config<Traits>(
            config, args.attention_cache_layer) ||
        args.attention_split_size <= 0 || args.attention_num_splits <= 0 ||
        args.attention_softmax_scale <= 0.0f) {
      return cudaErrorInvalidValue;
    }
    return gemma4_flash_attention::launch_decode_layer(
        args, stream,
        gemma4_flash_attention::
            decode_megakernel_fused_layer_kernel<
                gemma4_flash_attention::Gemma4AttentionTraits<false>>);
  }
  if (config.head_dim == GEMMA4_GLOBAL_HEAD_DIM &&
      config.num_heads == GEMMA4_GLOBAL_KV_HEADS) {
    using Traits = gemma4_flash_attention::Gemma4AttentionTraits<true>;
    if (!gemma4_flash_attention::gemma4_fa_valid_cache_config<Traits>(
            config, args.attention_cache_layer) ||
        args.attention_split_size <= 0 || args.attention_num_splits <= 0 ||
        args.attention_softmax_scale <= 0.0f) {
      return cudaErrorInvalidValue;
    }
    return gemma4_flash_attention::launch_decode_layer(
        args, stream,
        gemma4_flash_attention::
            decode_megakernel_fused_layer_kernel<
                gemma4_flash_attention::Gemma4AttentionTraits<true>>);
  }
  return cudaErrorInvalidValue;
}
