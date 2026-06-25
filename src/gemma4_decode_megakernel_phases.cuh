#pragma once

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_decode_megakernel.cuh"
#include "gemma4_embedding_gather.cuh"
#include "gemma4_ffn_decode_device.cuh"
#include "gemma4_rmsnorm_device.cuh"
#include "gemma4_sampling.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

namespace gemma4_decode_megakernel_phases {

namespace ffn_dev = gemma4_ffn_decode_device;
namespace rmsnorm_dev = gemma4_rmsnorm_device;

constexpr int kMegaThreads = 512;
constexpr int kMegaWarps = kMegaThreads / WARP_SIZE;
constexpr int kMegaColsPerBlock = 8;
constexpr int kMegaCandidateCount = GEMMA4_VOCAB_SIZE / kMegaColsPerBlock;
constexpr int kFfnIntermediateTile = ffn_dev::kIntermediateTile;
constexpr int kFfnIntermediateTiles = ffn_dev::kIntermediateTiles;
constexpr int kFfnHiddenPacks = ffn_dev::kHiddenPacks;
constexpr int kFfnActTile = ffn_dev::kActTile;
static_assert(ffn_dev::kReductionPolicy == 0,
              "megakernel FFN tail currently uses atomic accumulation");
static_assert((sizeof(Gemma4FfnDecodeScratch) %
               alignof(Gemma4SampleCandidate)) == 0,
              "FFN scratch must preserve candidate alignment");
static_assert(((kMegaCandidateCount * sizeof(Gemma4SampleCandidate)) % 16) ==
                  0,
              "candidate scratch must preserve 16-byte hidden-row alignment");

struct Gemma4DecodeSpineScratch {
  Gemma4SampleCandidate *candidates = nullptr;
  __nv_bfloat16 *normed_hidden = nullptr;
};

struct Gemma4DecodeFfnTailScratch {
  Gemma4FfnDecodeScratch *ffn = nullptr;
  Gemma4SampleCandidate *candidates = nullptr;
  __nv_bfloat16 *normed_hidden = nullptr;
};

// Returns scratch for all per-block candidates plus the normalized hidden row.
inline size_t spine_scratch_bytes(void) {
  const size_t candidates =
      static_cast<size_t>(kMegaCandidateCount) * sizeof(Gemma4SampleCandidate);
  const size_t normed_hidden =
      static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * sizeof(__nv_bfloat16);
  return candidates + normed_hidden + 16;
}

// Returns scratch for FFN accumulation plus the final sampling-tail scratch.
inline size_t ffn_tail_scratch_bytes(void) {
  const size_t candidates =
      static_cast<size_t>(kMegaCandidateCount) * sizeof(Gemma4SampleCandidate);
  const size_t normed_hidden =
      static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * sizeof(__nv_bfloat16);
  return sizeof(Gemma4FfnDecodeScratch) + candidates + normed_hidden;
}

// Splits the caller scratch buffer for the final decode spine.
inline Gemma4DecodeSpineScratch spine_scratch_from_buffer(void *scratch) {
  char *ptr = reinterpret_cast<char *>(scratch);
  auto *candidates = reinterpret_cast<Gemma4SampleCandidate *>(ptr);
  ptr += static_cast<size_t>(kMegaCandidateCount) *
         sizeof(Gemma4SampleCandidate);

  ptr = align_ptr_up<16>(ptr);
  auto *normed_hidden = reinterpret_cast<__nv_bfloat16 *>(ptr);
  return {candidates, normed_hidden};
}

// Splits the caller scratch buffer for the FFN phase and final decode spine.
inline Gemma4DecodeFfnTailScratch ffn_tail_scratch_from_buffer(void *scratch) {
  auto *ffn_scratch = reinterpret_cast<Gemma4FfnDecodeScratch *>(scratch);
  char *ptr = reinterpret_cast<char *>(ffn_scratch + 1);
  auto *candidates = reinterpret_cast<Gemma4SampleCandidate *>(ptr);
  ptr += static_cast<size_t>(kMegaCandidateCount) *
         sizeof(Gemma4SampleCandidate);

  auto *normed_hidden = reinterpret_cast<__nv_bfloat16 *>(ptr);
  return {ffn_scratch, candidates, normed_hidden};
}

// Validates the public spine launcher arguments before cooperative launch.
inline bool valid_args(const Gemma4DecodeMegakernelSpineArgs &args) {
  return args.state != nullptr && args.next_hidden != nullptr &&
         args.next_token != nullptr && args.final_norm_weight != nullptr &&
         args.lm_head_col_major != nullptr && is_aligned_16(args.state) &&
         is_aligned_16(args.next_hidden) &&
         is_aligned_16(args.final_norm_weight) &&
         is_aligned_16(args.lm_head_col_major);
}

// Returns true when the caller requested the optional FlashAttention phase.
inline bool ffn_tail_uses_flash_attention(
    const Gemma4DecodeMegakernelFfnTailArgs &args) {
  return (args.flags & GEMMA4_DECODE_MEGAKERNEL_FLAG_FLASH_ATTENTION) != 0;
}

// Validates the optional FlashAttention phase arguments.
inline bool valid_flash_attention_args(
    const Gemma4DecodeMegakernelFfnTailArgs &args) {
  if (!ffn_tail_uses_flash_attention(args)) {
    return true;
  }

  const Gemma4KvCacheConfig &config = args.attention_cache_config;
  const bool sliding = config.head_dim == GEMMA4_SLIDING_HEAD_DIM &&
                       config.num_heads == GEMMA4_SLIDING_KV_HEADS &&
                       config.window_size > 0;
  const bool global = config.head_dim == GEMMA4_GLOBAL_HEAD_DIM &&
                      config.num_heads == GEMMA4_GLOBAL_KV_HEADS &&
                      config.window_size == 0;
  const int64_t required_keys =
      global ? int64_t(config.max_pages_per_seq) * config.page_size
             : config.window_size;
  const int64_t cache_token_capacity =
      int64_t(config.max_pages_per_seq) * config.page_size;
  const int64_t split_capacity =
      int64_t(args.attention_split_size) * args.attention_num_splits;
  const bool needs_v_weight = sliding;
  const bool needs_partials = args.attention_num_splits > 1;

  return (sliding || global) &&
         args.attention_cache_layer >= 0 &&
         args.attention_cache_layer < config.num_layers &&
         config.num_layers > 0 && config.num_pages > 0 &&
         config.page_size > 0 && config.max_pages_per_seq > 0 &&
         config.num_pages % config.max_pages_per_seq == 0 &&
         args.attention_split_size > 0 &&
         args.attention_num_splits > 0 &&
         split_capacity >= required_keys &&
         (!sliding || cache_token_capacity >= config.window_size) &&
         args.attention_softmax_scale > 0.0f &&
         args.attention_q != nullptr &&
         args.attention_out != nullptr &&
         (!needs_partials ||
          (args.attention_partial_m != nullptr &&
           args.attention_partial_l != nullptr &&
           args.attention_partial_acc != nullptr)) &&
         args.attention_cache_k != nullptr &&
         args.attention_cache_v != nullptr &&
         args.attention_page_table != nullptr &&
         args.attention_token_position != nullptr &&
         args.attention_seq_lengths != nullptr &&
         args.attention_x != nullptr &&
         args.attention_input_norm_weight != nullptr &&
         args.attention_weights.d_q_col_major != nullptr &&
         args.attention_weights.d_k_col_major != nullptr &&
         (!needs_v_weight ||
          args.attention_weights.d_v_col_major != nullptr) &&
         args.attention_o_proj_col_major != nullptr &&
         args.attention_post_norm_weight != nullptr &&
         args.attention_pre_ffn_norm_weight != nullptr &&
         args.attention_q_norm_weight != nullptr &&
         args.attention_k_norm_weight != nullptr &&
         args.attention_cos != nullptr &&
         args.attention_sin != nullptr &&
         is_aligned_16(args.attention_q) &&
         is_aligned_16(args.attention_out) &&
         is_aligned_16(args.attention_cache_k) &&
         is_aligned_16(args.attention_cache_v) &&
         is_aligned_16(args.attention_x) &&
         is_aligned_16(args.attention_input_norm_weight) &&
         is_aligned_16(args.attention_weights.d_q_col_major) &&
         is_aligned_16(args.attention_weights.d_k_col_major) &&
         (!needs_v_weight ||
          is_aligned_16(args.attention_weights.d_v_col_major)) &&
         is_aligned_16(args.attention_o_proj_col_major) &&
         is_aligned_16(args.attention_post_norm_weight) &&
         is_aligned_16(args.attention_pre_ffn_norm_weight) &&
         is_aligned_16(args.attention_q_norm_weight) &&
         is_aligned_16(args.attention_k_norm_weight);
}

// Validates the public FFN-tail launcher arguments before cooperative launch.
inline bool valid_args(const Gemma4DecodeMegakernelFfnTailArgs &args) {
  constexpr uint32_t known_flags =
      GEMMA4_DECODE_MEGAKERNEL_FLAG_FLASH_ATTENTION;
  return args.residual_out != nullptr && args.normed_out != nullptr &&
         args.next_hidden != nullptr && args.next_token != nullptr &&
         args.ffn_x != nullptr && args.ffn_residual != nullptr &&
         args.ffn_norm_weight != nullptr &&
         args.ffn_gate_up_decode != nullptr &&
         args.ffn_down_decode != nullptr &&
         args.layer_scalar != nullptr &&
         args.final_norm_weight != nullptr &&
         args.lm_head_col_major != nullptr && args.eps > 0.0f &&
         (args.flags & ~known_flags) == 0 &&
         is_aligned_16(args.residual_out) &&
         is_aligned_16(args.normed_out) &&
         is_aligned_16(args.next_hidden) &&
         is_aligned_16(args.ffn_x) &&
         is_aligned_16(args.ffn_residual) &&
         is_aligned_16(args.ffn_norm_weight) &&
         is_aligned_16(args.ffn_gate_up_decode) &&
         is_aligned_16(args.ffn_down_decode) &&
         is_aligned_16(args.final_norm_weight) &&
         is_aligned_16(args.lm_head_col_major) &&
         valid_flash_attention_args(args);
}

// Clears FFN accumulation scratch across the resident cooperative grid.
__device__ inline void phase_ffn_zero_accum(
    Gemma4FfnDecodeScratch *__restrict__ scratch) {
  const int linear_idx = int(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride = int(gridDim.x) * blockDim.x;
  ffn_dev::zero_accum(scratch, linear_idx, stride);
}

// Runs the shared FFN decode accumulation inside the cooperative grid.
__device__ inline void phase_ffn_accumulate(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    Gemma4FfnDecodeScratch *__restrict__ scratch) {
  __shared__ float s_matmul_warp_sums[2][kFfnActTile][kMegaWarps];
  __shared__ float s_act[kFfnActTile];

  float partial[kBf16Packed128Elements] = {};
  const int hidden_pack = threadIdx.x;
  const bool active_hidden_pack = hidden_pack < kFfnHiddenPacks;
  const int swizzled_hidden_col =
      active_hidden_pack
          ? ffn_dev::hidden_pack_swizzle_index(hidden_pack) *
                kBf16Packed128Elements
          : 0;

  for (int tile = int(blockIdx.x); tile < kFfnIntermediateTiles;
       tile += int(gridDim.x)) {
    ffn_dev::accumulate_intermediate_tile<kMegaThreads, true>(
        args.ffn_x, args.ffn_gate_up_decode, args.ffn_down_decode,
        tile * kFfnIntermediateTile, swizzled_hidden_col, active_hidden_pack,
        partial, s_matmul_warp_sums, s_act);
  }

  if (active_hidden_pack) {
    ffn_dev::atomic_add_accum_pack(scratch, hidden_pack, partial);
  }
}

// Projects attention output to hidden width inside the resident decode grid.
template <int AttentionWidth>
__device__ inline void phase_attention_o_projection(
    const Gemma4DecodeMegakernelFfnTailArgs &args) {
  constexpr int kColsPerBlock = 8;
  constexpr int kHiddenBlocks = GEMMA4_HIDDEN_SIZE / kColsPerBlock;
  static_assert((GEMMA4_HIDDEN_SIZE % kColsPerBlock) == 0,
                "hidden width must divide O projection tile columns");
  static_assert((AttentionWidth % kBf16Packed128Elements) == 0,
                "attention width must divide bf16 pack width");

  __shared__ float s_warp_sums[kColsPerBlock][kMegaWarps];

  for (int col_block = int(blockIdx.x); col_block < kHiddenBlocks;
       col_block += int(gridDim.x)) {
    float sums[kColsPerBlock] = {};
    const int col0 = col_block * kColsPerBlock;
    constexpr int packs = AttentionWidth / kBf16Packed128Elements;

    for (int pack = int(threadIdx.x); pack < packs; pack += kMegaThreads) {
      const int element = pack * kBf16Packed128Elements;
      const Bf16Packed128 x_pack = load128g(args.attention_out + element);
#pragma unroll
      for (int col = 0; col < kColsPerBlock; ++col) {
        const __nv_bfloat16 *weight =
            args.attention_o_proj_col_major +
            int64_t(col0 + col) * AttentionWidth + element;
        const Bf16Packed128 w_pack = load128weight(weight);
        gemma4_bf16_pack_accumulate_dot(x_pack, w_pack, sums[col]);
      }
    }

    warp_reduce_sum_to_lane0(sums);
    const int lane = int(threadIdx.x) & (WARP_SIZE - 1);
    const int warp = int(threadIdx.x) / WARP_SIZE;
    if (lane == 0) {
#pragma unroll
      for (int col = 0; col < kColsPerBlock; ++col) {
        s_warp_sums[col][warp] = sums[col];
      }
    }
    __syncthreads();

#pragma unroll
    for (int col = 0; col < kColsPerBlock; ++col) {
      sums[col] = threadIdx.x < kMegaWarps ? s_warp_sums[col][lane] : 0.0f;
    }
    if (warp == 0) {
      warp_reduce_sum_to_lane0(sums);
    }
    if (threadIdx.x == 0) {
      Bf16Packed128 out_pack;
#pragma unroll
      for (int col = 0; col < kColsPerBlock; ++col) {
        out_pack[col] = __float2bfloat16_rn(sums[col]);
      }
      store128(args.residual_out + col0, out_pack);
    }
  }
}

// Builds the post-attention residual and pre-FFN normed row for one token.
__device__ inline void phase_attention_to_ffn(
    const Gemma4DecodeMegakernelFfnTailArgs &args) {
  if (blockIdx.x != 0) {
    return;
  }

  __shared__ Bf16Packed128 cached_input[kFfnHiddenPacks];
  __shared__ float warp_sums[kMegaWarps];
  __shared__ float scale;

  rmsnorm_dev::rmsnorm_hidden_row_bf16<kMegaThreads>(
      args.normed_out, args.residual_out, args.attention_post_norm_weight,
      args.eps, cached_input, warp_sums, scale, int(threadIdx.x));
  __syncthreads();

  for (int pack = int(threadIdx.x); pack < kFfnHiddenPacks;
       pack += kMegaThreads) {
    const int offset = pack * kBf16Packed128Elements;
    const Bf16Packed128 residual = load128g(args.attention_x + offset);
    const Bf16Packed128 delta = load128g(args.normed_out + offset);
    const Bf16Packed128 sum = gemma4_bf16_pack_add(residual, delta);
    store128(args.ffn_residual + offset, sum);
  }
  __syncthreads();

  rmsnorm_dev::rmsnorm_hidden_row_bf16<kMegaThreads>(
      args.ffn_x, args.ffn_residual, args.attention_pre_ffn_norm_weight,
      args.eps, cached_input, warp_sums, scale, int(threadIdx.x));
}

// Finalizes FFN as post-FFN RMSNorm followed by residual add.
__device__ inline void phase_ffn_finalize_rmsnorm_residual(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    const Gemma4FfnDecodeScratch *__restrict__ scratch) {
  if (blockIdx.x != 0) {
    return;
  }

  __shared__ float s_rms_warp_sums[kMegaWarps];
  __shared__ float s_scale;

  ffn_dev::finalize_rmsnorm_residual<kMegaThreads, true, false>(
      args.residual_out, args.normed_out, args.ffn_residual,
      args.ffn_norm_weight, scratch, args.eps, s_rms_warp_sums, s_scale,
      int(threadIdx.x));
}

// Applies the checkpoint layer scalar to the completed post-FFN residual row.
__device__ inline void phase_scale_layer_hidden(
    __nv_bfloat16 *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ layer_scalar) {
  const float scale = __bfloat162float(__ldg(layer_scalar));
  const int linear_idx = int(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride = int(gridDim.x) * blockDim.x;
  for (int pack = linear_idx; pack < kFfnHiddenPacks; pack += stride) {
    const int offset = pack * kBf16Packed128Elements;
    const Bf16Packed128 values = load128g(hidden + offset);
    const Bf16Packed128 scaled = gemma4_bf16_pack_apply_scale(values, scale);
    store128(hidden + offset, scaled);
  }
}

// Computes final RMSNorm in block 0 into the normalized hidden scratch row.
__device__ inline void phase_final_rmsnorm_hidden(
    __nv_bfloat16 *__restrict__ normed_hidden,
    const __nv_bfloat16 *__restrict__ state,
    const __nv_bfloat16 *__restrict__ final_norm_weight) {
  __shared__ float warp_sums[kMegaWarps];
  __shared__ float scale;
  __shared__ Bf16Packed128 cached_input[kFfnHiddenPacks];

  if (blockIdx.x != 0) {
    return;
  }

  rmsnorm_dev::rmsnorm_hidden_row_bf16<kMegaThreads>(
      normed_hidden, state, final_norm_weight, GEMMA4_RMS_NORM_EPS,
      cached_input, warp_sums, scale, int(threadIdx.x));
}

// Chooses the higher logit, breaking exact ties by the lower token id.
__device__ inline bool phase_better_candidate(float logit,
                                              int32_t token_id,
                                              float best_logit,
                                              int32_t best_token_id) {
  return logit > best_logit || (logit == best_logit && token_id < best_token_id);
}

// Reduces each LM-head output column's partial dot product across the CTA.
__device__ inline void phase_reduce_lm_head_cols(
    float (&warp_sums)[kMegaColsPerBlock][kMegaWarps],
    float (&sums)[kMegaColsPerBlock]) {
  const int lane = int(threadIdx.x) & (WARP_SIZE - 1);
  const int warp = int(threadIdx.x) / WARP_SIZE;

  warp_reduce_sum_to_lane0(sums);

  if (lane == 0) {
#pragma unroll
    for (int col = 0; col < kMegaColsPerBlock; ++col) {
      warp_sums[col][warp] = sums[col];
    }
  }
  __syncthreads();

#pragma unroll
  for (int col = 0; col < kMegaColsPerBlock; ++col) {
    sums[col] = threadIdx.x < kMegaWarps ? warp_sums[col][lane] : 0.0f;
  }

  if (warp == 0) {
    warp_reduce_sum_to_lane0(sums);
  }
}

// Computes one final LM-head vocab tile from the normalized hidden row.
__device__ inline void phase_lm_head_tile_logits(
    const __nv_bfloat16 *__restrict__ normed_hidden,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    int tile,
    float (&warp_sums)[kMegaColsPerBlock][kMegaWarps],
    float (&sums)[kMegaColsPerBlock]) {
  constexpr int packs_per_col =
      GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
  static_assert((GEMMA4_HIDDEN_SIZE % kBf16Packed128Elements) == 0,
                "hidden size must divide the 128-bit bf16 pack width");

  const int token0 = tile * kMegaColsPerBlock;
  for (int pack = int(threadIdx.x); pack < packs_per_col;
       pack += kMegaThreads) {
    const int element = pack * kBf16Packed128Elements;
    const Bf16Packed128 x_pack = load128g(normed_hidden + element);
#pragma unroll
    for (int col = 0; col < kMegaColsPerBlock; ++col) {
      const int token_id = token0 + col;
      const __nv_bfloat16 *weight =
          lm_head_col_major +
          int64_t(token_id) * GEMMA4_HIDDEN_SIZE + element;
      const Bf16Packed128 w_pack = load128weight(weight);
      gemma4_bf16_pack_accumulate_dot(x_pack, w_pack, sums[col]);
    }
  }

  phase_reduce_lm_head_cols(warp_sums, sums);
}

// Computes this CTA's final LM-head candidate.
__device__ inline Gemma4SampleCandidate phase_final_logits_block_candidate(
    const __nv_bfloat16 *__restrict__ normed_hidden,
    const __nv_bfloat16 *__restrict__ lm_head_col_major) {
  __shared__ float warp_sums[kMegaColsPerBlock][kMegaWarps];

  float best_logit = -INFINITY;
  int32_t best_token_id = GEMMA4_VOCAB_SIZE;

  for (int tile = int(blockIdx.x); tile < kMegaCandidateCount;
       tile += int(gridDim.x)) {
    float sums[kMegaColsPerBlock] = {};
    phase_lm_head_tile_logits(
        normed_hidden, lm_head_col_major, tile, warp_sums, sums);

    if (threadIdx.x == 0) {
      const int32_t token0 = tile * kMegaColsPerBlock;
#pragma unroll
      for (int col = 0; col < kMegaColsPerBlock; ++col) {
        const float logit = __bfloat162float(__float2bfloat16_rn(sums[col]));
        const int32_t token_id = token0 + col;
        if (phase_better_candidate(
                logit, token_id, best_logit, best_token_id)) {
          best_logit = logit;
          best_token_id = token_id;
        }
      }
    }
    __syncthreads();
  }

  return {best_logit, best_token_id};
}

// Reduces per-block candidates, writes the token, and gathers next hidden.
__device__ inline void phase_reduce_candidates_and_gather(
    __nv_bfloat16 *__restrict__ next_hidden,
    int32_t *__restrict__ next_token,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    Gemma4SampleCandidate *__restrict__ candidates,
    int32_t active_candidate_count) {
  __shared__ float s_logits[kMegaThreads];
  __shared__ int32_t s_token_ids[kMegaThreads];

  if (blockIdx.x != 0) {
    return;
  }

  Gemma4SampleCandidate best = {-INFINITY, GEMMA4_VOCAB_SIZE};
  for (int candidate_idx = int(threadIdx.x);
       candidate_idx < active_candidate_count;
       candidate_idx += kMegaThreads) {
    const Gemma4SampleCandidate candidate = candidates[candidate_idx];
    if (phase_better_candidate(candidate.logit, candidate.token_id,
                               best.logit, best.token_id)) {
      best = candidate;
    }
  }

  s_logits[threadIdx.x] = best.logit;
  s_token_ids[threadIdx.x] = best.token_id;
  __syncthreads();

  for (int stride = kMegaThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float other_logit = s_logits[threadIdx.x + stride];
      const int32_t other_token_id = s_token_ids[threadIdx.x + stride];
      if (phase_better_candidate(other_logit, other_token_id,
                                 s_logits[threadIdx.x],
                                 s_token_ids[threadIdx.x])) {
        s_logits[threadIdx.x] = other_logit;
        s_token_ids[threadIdx.x] = other_token_id;
      }
    }
    __syncthreads();
  }

  const int32_t selected_token_id = s_token_ids[0];
  if (threadIdx.x == 0) {
    *next_token = selected_token_id;
  }
  gemma4_embedding_gather::copy_embedding_row_bf16(
      next_hidden, lm_head_col_major, selected_token_id, int(threadIdx.x),
      kMegaThreads);
}

// Sizes the resident cooperative grid for a candidate-producing kernel.
template <typename Kernel>
inline cudaError_t active_candidate_count_for_kernel(
    Kernel kernel,
    int *active_candidate_count) {
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
  if (!prop.cooperativeLaunch) {
    return cudaErrorNotSupported;
  }

  int active_blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, kernel, kMegaThreads, 0);
  if (status != cudaSuccess) {
    return status;
  }

  int count = active_blocks_per_sm * prop.multiProcessorCount;
  if (count > kMegaCandidateCount) {
    count = kMegaCandidateCount;
  }
  if (count <= 0) {
    return cudaErrorInvalidValue;
  }
  *active_candidate_count = count;
  return cudaSuccess;
}

// Launches the cooperative final decode spine kernel.
template <typename Kernel>
inline cudaError_t launch_spine(
    const Gemma4DecodeMegakernelSpineArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream,
    Kernel kernel) {
  if (!valid_args(args) || scratch == nullptr ||
      !is_aligned_to<alignof(Gemma4SampleCandidate)>(scratch) ||
      scratch_bytes < spine_scratch_bytes()) {
    return cudaErrorInvalidValue;
  }

  int32_t active_candidate_count = 0;
  cudaError_t status =
      active_candidate_count_for_kernel(kernel, &active_candidate_count);
  if (status != cudaSuccess) {
    return status;
  }

  const Gemma4DecodeSpineScratch scratch_parts =
      spine_scratch_from_buffer(scratch);
  Gemma4DecodeMegakernelSpineArgs kernel_args_value = args;
  Gemma4SampleCandidate *candidates_arg = scratch_parts.candidates;
  __nv_bfloat16 *normed_hidden_arg = scratch_parts.normed_hidden;
  int32_t active_candidate_count_arg = active_candidate_count;
  void *kernel_args[] = {
      &kernel_args_value,
      &candidates_arg,
      &normed_hidden_arg,
      &active_candidate_count_arg,
  };

  status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(kernel),
      active_candidate_count,
      kMegaThreads,
      kernel_args,
      0,
      stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaGetLastError();
}

// Launches the cooperative FFN-tail kernel.
template <typename Kernel>
inline cudaError_t launch_ffn_tail(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    void *__restrict__ scratch,
    size_t scratch_bytes,
    cudaStream_t stream,
    Kernel kernel) {
  if (!valid_args(args) || scratch == nullptr || !is_aligned_128(scratch) ||
      scratch_bytes < ffn_tail_scratch_bytes()) {
    return cudaErrorInvalidValue;
  }

  int32_t active_candidate_count = 0;
  cudaError_t status =
      active_candidate_count_for_kernel(kernel, &active_candidate_count);
  if (status != cudaSuccess) {
    return status;
  }

  const Gemma4DecodeFfnTailScratch scratch_parts =
      ffn_tail_scratch_from_buffer(scratch);
  Gemma4DecodeMegakernelFfnTailArgs kernel_args_value = args;
  Gemma4FfnDecodeScratch *ffn_scratch_arg = scratch_parts.ffn;
  Gemma4SampleCandidate *candidates_arg = scratch_parts.candidates;
  __nv_bfloat16 *normed_hidden_arg = scratch_parts.normed_hidden;
  int32_t active_candidate_count_arg = active_candidate_count;
  void *kernel_args[] = {
      &kernel_args_value,
      &ffn_scratch_arg,
      &candidates_arg,
      &normed_hidden_arg,
      &active_candidate_count_arg,
  };

  status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(kernel),
      active_candidate_count,
      kMegaThreads,
      kernel_args,
      0,
      stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaGetLastError();
}

}  // namespace gemma4_decode_megakernel_phases
