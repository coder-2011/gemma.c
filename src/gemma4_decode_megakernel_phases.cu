// Include-only CUDA phase unit shared by decode and FlashAttention translation units.
#pragma once

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_decode_megakernel.cuh"
#include "gemma4_embedding_gather.cuh"
#include "gemma4_ffn.cuh"
#include "gemma4_sampling.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

extern "C" __device__ void gemma4_rmsnorm_hidden_row_512_bf16_device(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ in,
    const __nv_bfloat16 *__restrict__ weight,
    float eps,
    Bf16Packed128 *__restrict__ cached_row,
    float *__restrict__ warp_sums,
    float *__restrict__ scale,
    int thread_idx);

namespace gemma4_decode_megakernel_phases {

namespace ffn_dev = gemma4_ffn_decode_device;

constexpr int kMegaThreads = 512;
constexpr int kMegaWarps = kMegaThreads / 32;
constexpr int kMegaColsPerBlock = 8;
constexpr int kMegaCandidateCount = GEMMA4_VOCAB_SIZE / kMegaColsPerBlock;
constexpr int kFfnHiddenPacks = ffn_dev::kHiddenPacks;
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
constexpr size_t spine_scratch_bytes(void) {
  constexpr size_t candidates =
      static_cast<size_t>(kMegaCandidateCount) * sizeof(Gemma4SampleCandidate);
  constexpr size_t normed_hidden =
      static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * sizeof(__nv_bfloat16);
  return candidates + normed_hidden + 16;
}

// Returns scratch for CUTLASS FFN decode plus the final sampling-tail scratch.
constexpr size_t ffn_tail_scratch_bytes(void) {
  constexpr size_t candidates =
      static_cast<size_t>(kMegaCandidateCount) * sizeof(Gemma4SampleCandidate);
  constexpr size_t normed_hidden =
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

// Returns true when the caller requested the optional FlashAttention phase.
inline bool ffn_tail_uses_flash_attention(
    const Gemma4DecodeMegakernelFfnTailArgs &args) {
  return (args.flags & GEMMA4_DECODE_MEGAKERNEL_FLAG_FLASH_ATTENTION) != 0;
}

// Projects attention output to hidden width inside the cooperative prep grid.
template <int AttentionWidth>
__device__ inline void phase_attention_o_projection(
    const Gemma4DecodeMegakernelFfnTailArgs &args) {
  constexpr int kColsPerBlock = 8;
  constexpr int kHiddenBlocks = GEMMA4_HIDDEN_SIZE / kColsPerBlock;
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
      const Bf16Packed128 x_pack =
          Bf16Packed128{*reinterpret_cast<const int4 *>(
              args.attention_out + element)};
#pragma unroll
      for (int col = 0; col < kColsPerBlock; ++col) {
        const __nv_bfloat16 *weight =
            args.attention_o_proj_col_major +
            int64_t(col0 + col) * AttentionWidth + element;
        const Bf16Packed128 w_pack =
            Bf16Packed128{*reinterpret_cast<const int4 *>(weight)};
        gemma4_bf16_pack_accumulate_dot(x_pack, w_pack, sums[col]);
      }
    }

    warp_reduce_sum_to_lane0(sums);
    const int lane = int(threadIdx.x) & (warpSize - 1);
    const int warp = int(threadIdx.x) / warpSize;
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
      *reinterpret_cast<int4 *>(args.residual_out + col0) = out_pack.bits();
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

  gemma4_rmsnorm_hidden_row_512_bf16_device(
      args.normed_out, args.residual_out, args.attention_post_norm_weight,
      GEMMA4_RMS_NORM_EPS, cached_input, warp_sums, &scale,
      int(threadIdx.x));
  __syncthreads();

  const auto hidden_layout = cute::make_layout(
      cute::make_shape(kFfnHiddenPacks),
      cute::make_stride(kBf16Packed128Elements));
  for (int pack = int(threadIdx.x); pack < kFfnHiddenPacks;
       pack += kMegaThreads) {
    const int offset = hidden_layout(pack);
    const Bf16Packed128 residual =
        Bf16Packed128{*reinterpret_cast<const int4 *>(
            args.attention_x + offset)};
    const Bf16Packed128 delta =
        Bf16Packed128{*reinterpret_cast<const int4 *>(
            args.normed_out + offset)};
    const Bf16Packed128 sum = gemma4_bf16_pack_add(residual, delta);
    *reinterpret_cast<int4 *>(args.ffn_residual + offset) = sum.bits();
  }
  __syncthreads();

  gemma4_rmsnorm_hidden_row_512_bf16_device(
      args.ffn_x, args.ffn_residual, args.attention_pre_ffn_norm_weight,
      GEMMA4_RMS_NORM_EPS, cached_input, warp_sums, &scale,
      int(threadIdx.x));
}

// Applies the checkpoint layer scalar to the completed post-FFN residual row.
__device__ inline void phase_scale_layer_hidden(
    __nv_bfloat16 *__restrict__ hidden,
    const __nv_bfloat16 *__restrict__ layer_scalar) {
  const float scale = __bfloat162float(__ldg(layer_scalar));
  const int linear_idx = int(blockIdx.x) * blockDim.x + threadIdx.x;
  const int stride = int(gridDim.x) * blockDim.x;
  const auto hidden_layout = cute::make_layout(
      cute::make_shape(kFfnHiddenPacks),
      cute::make_stride(kBf16Packed128Elements));
  for (int pack = linear_idx; pack < kFfnHiddenPacks; pack += stride) {
    const int offset = hidden_layout(pack);
    const Bf16Packed128 values =
        Bf16Packed128{*reinterpret_cast<const int4 *>(hidden + offset)};
    const Bf16Packed128 scaled = gemma4_bf16_pack_apply_scale(values, scale);
    *reinterpret_cast<int4 *>(hidden + offset) = scaled.bits();
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

  gemma4_rmsnorm_hidden_row_512_bf16_device(
      normed_hidden, state, final_norm_weight, GEMMA4_RMS_NORM_EPS,
      cached_input, warp_sums, &scale, int(threadIdx.x));
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
  const int lane = int(threadIdx.x) & (warpSize - 1);
  const int warp = int(threadIdx.x) / warpSize;

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

  const int token0 = tile * kMegaColsPerBlock;
  for (int pack = int(threadIdx.x); pack < packs_per_col;
       pack += kMegaThreads) {
    const int element = pack * kBf16Packed128Elements;
    const Bf16Packed128 x_pack =
        Bf16Packed128{*reinterpret_cast<const int4 *>(
            normed_hidden + element)};
#pragma unroll
    for (int col = 0; col < kMegaColsPerBlock; ++col) {
      const int token_id = token0 + col;
      const __nv_bfloat16 *weight =
          lm_head_col_major +
          int64_t(token_id) * GEMMA4_HIDDEN_SIZE + element;
      const Bf16Packed128 w_pack =
          Bf16Packed128{*reinterpret_cast<const int4 *>(weight)};
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
  if (scratch_bytes < spine_scratch_bytes()) {
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

// Launches a cooperative pre-FFN preparation kernel.
template <typename Kernel>
inline cudaError_t launch_prepare(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    cudaStream_t stream,
    Kernel kernel) {
  int32_t active_block_count = 0;
  cudaError_t status =
      active_candidate_count_for_kernel(kernel, &active_block_count);
  if (status != cudaSuccess) {
    return status;
  }

  Gemma4DecodeMegakernelFfnTailArgs kernel_args_value = args;
  void *kernel_args[] = {
      &kernel_args_value,
  };

  status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(kernel),
      active_block_count,
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
