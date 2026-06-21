#ifndef GEMMA4_DECODE_MEGAKERNEL_PHASES_CUH
#define GEMMA4_DECODE_MEGAKERNEL_PHASES_CUH

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_decode_megakernel.cuh"
#include "gemma4_ffn_decode_device.cuh"
#include "gemma4_rmsnorm_device.cuh"
#include "gemma4_sampling_device.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace gemma4_decode_megakernel_phases {

namespace ffn_dev = gemma4_ffn_decode_device;
namespace rmsnorm_dev = gemma4_rmsnorm_device;
namespace sampling_dev = gemma4_sampling_device;

constexpr int kMegaThreads = 512;
constexpr int kMegaWarps = kMegaThreads / WARP_SIZE;
constexpr int kMegaColsPerBlock = 8;
constexpr int kMegaSwizzleTileBlocks = 1;
constexpr int kMegaCandidateCount = GEMMA4_VOCAB_SIZE / kMegaColsPerBlock;
constexpr int kFfnIntermediateTile = ffn_dev::kIntermediateTile;
constexpr int kFfnIntermediateTiles = ffn_dev::kIntermediateTiles;
constexpr int kFfnHiddenPacks = ffn_dev::kHiddenPacks;
constexpr int kFfnActTile = ffn_dev::kActTile;
static_assert(ffn_dev::kReductionPolicy == 0,
              "megakernel FFN tail currently uses atomic accumulation");
static_assert((sizeof(Gemma4FfnDecodeScratch) %
               alignof(Gemma4GreedyCandidate)) == 0,
              "FFN scratch must preserve candidate alignment");
static_assert(((kMegaCandidateCount * sizeof(Gemma4GreedyCandidate)) % 16) ==
                  0,
              "candidate scratch must preserve 16-byte hidden-row alignment");

using ffn_dev::is_aligned_128;

struct Gemma4DecodeSpineScratch {
  Gemma4GreedyCandidate *candidates = nullptr;
  __nv_bfloat16 *normed_hidden = nullptr;
};

struct Gemma4DecodeFfnTailScratch {
  Gemma4FfnDecodeScratch *ffn = nullptr;
  Gemma4GreedyCandidate *candidates = nullptr;
  __nv_bfloat16 *normed_hidden = nullptr;
};

// Returns scratch for all per-block candidates plus the normalized hidden row.
inline size_t spine_scratch_bytes(void) {
  const size_t candidates =
      static_cast<size_t>(kMegaCandidateCount) * sizeof(Gemma4GreedyCandidate);
  const size_t normed_hidden =
      static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * sizeof(__nv_bfloat16);
  return candidates + normed_hidden + 16;
}

// Returns scratch for FFN accumulation plus the final sampling-tail scratch.
inline size_t ffn_tail_scratch_bytes(void) {
  const size_t candidates =
      static_cast<size_t>(kMegaCandidateCount) * sizeof(Gemma4GreedyCandidate);
  const size_t normed_hidden =
      static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * sizeof(__nv_bfloat16);
  return sizeof(Gemma4FfnDecodeScratch) + candidates + normed_hidden;
}

// Splits the caller scratch buffer for the final decode spine.
inline Gemma4DecodeSpineScratch spine_scratch_from_buffer(void *scratch) {
  char *ptr = reinterpret_cast<char *>(scratch);
  auto *candidates = reinterpret_cast<Gemma4GreedyCandidate *>(ptr);
  ptr += static_cast<size_t>(kMegaCandidateCount) *
         sizeof(Gemma4GreedyCandidate);

  uintptr_t normed_address = reinterpret_cast<uintptr_t>(ptr);
  normed_address = (normed_address + 15u) & ~uintptr_t(15u);
  auto *normed_hidden = reinterpret_cast<__nv_bfloat16 *>(normed_address);
  return {candidates, normed_hidden};
}

// Splits the caller scratch buffer for the FFN phase and final decode spine.
inline Gemma4DecodeFfnTailScratch ffn_tail_scratch_from_buffer(void *scratch) {
  auto *ffn_scratch = reinterpret_cast<Gemma4FfnDecodeScratch *>(scratch);
  char *ptr = reinterpret_cast<char *>(ffn_scratch + 1);
  auto *candidates = reinterpret_cast<Gemma4GreedyCandidate *>(ptr);
  ptr += static_cast<size_t>(kMegaCandidateCount) *
         sizeof(Gemma4GreedyCandidate);

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
  const int64_t split_capacity =
      int64_t(args.attention_split_size) * args.attention_num_splits;
  const bool needs_v_weight = sliding;

  return (sliding || global) &&
         args.attention_cache_layer >= 0 &&
         args.attention_cache_layer < config.num_layers &&
         config.num_layers > 0 && config.num_pages > 0 &&
         config.page_size > 0 && config.max_pages_per_seq > 0 &&
         args.attention_split_size > 0 &&
         args.attention_num_splits > 0 &&
         split_capacity >= required_keys &&
         args.attention_softmax_scale > 0.0f &&
         args.attention_q != nullptr &&
         args.attention_out != nullptr &&
         args.attention_partial_m != nullptr &&
         args.attention_partial_l != nullptr &&
         args.attention_partial_acc != nullptr &&
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

// Finalizes the FFN accumulation into residual_out and normed_out.
__device__ inline void phase_ffn_finalize_residual_rmsnorm(
    const Gemma4DecodeMegakernelFfnTailArgs &args,
    const Gemma4FfnDecodeScratch *__restrict__ scratch) {
  if (blockIdx.x != 0) {
    return;
  }

  __shared__ float s_rms_warp_sums[kMegaWarps];
  __shared__ float s_scale;

  ffn_dev::finalize_residual_rmsnorm<kMegaThreads, true, false>(
      args.residual_out, args.normed_out, args.ffn_residual,
      args.ffn_norm_weight, scratch, args.eps, s_rms_warp_sums, s_scale,
      int(threadIdx.x));
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

// Computes this CTA's final greedy LM-head candidate.
__device__ inline Gemma4GreedyCandidate phase_final_logits_block_candidate(
    const __nv_bfloat16 *__restrict__ normed_hidden,
    const __nv_bfloat16 *__restrict__ lm_head_col_major) {
  return sampling_dev::greedy_lm_head_block_candidate<
      kMegaColsPerBlock,
      kMegaThreads,
      kMegaSwizzleTileBlocks,
      kMegaCandidateCount>(
      normed_hidden, lm_head_col_major, int(blockIdx.x), int(gridDim.x),
      int(threadIdx.x));
}

// Reduces per-block candidates, writes the token, and gathers next hidden.
__device__ inline void phase_reduce_greedy_and_gather(
    __nv_bfloat16 *__restrict__ next_hidden,
    int32_t *__restrict__ next_token,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    Gemma4GreedyCandidate *__restrict__ candidates,
    int32_t active_candidate_count) {
  sampling_dev::reduce_greedy_and_gather<kMegaThreads>(
      next_hidden, next_token, lm_head_col_major, candidates,
      active_candidate_count, int(blockIdx.x), int(threadIdx.x));
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
      !is_aligned_to<alignof(Gemma4GreedyCandidate)>(scratch) ||
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
  Gemma4GreedyCandidate *candidates_arg = scratch_parts.candidates;
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
  Gemma4GreedyCandidate *candidates_arg = scratch_parts.candidates;
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

#endif
