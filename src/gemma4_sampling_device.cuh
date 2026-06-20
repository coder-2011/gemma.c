#ifndef GEMMA4_SAMPLING_DEVICE_CUH
#define GEMMA4_SAMPLING_DEVICE_CUH

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_embedding_gather.cuh"
#include "gemma4_matmul_device.cuh"
#include "gemma4_sampling.cuh"

#include <cuda_bf16.h>
#include <stdint.h>

namespace gemma4_sampling_device {

constexpr float kNegativeInfinity = -3.4028234663852886e+38F;

// Chooses the higher logit, breaking exact ties by the lower token id.
__device__ inline bool better_candidate(float logit,
                                        int32_t token_id,
                                        float best_logit,
                                        int32_t best_token_id) {
  return logit > best_logit ||
         (logit == best_logit && token_id < best_token_id);
}

// Computes this CTA's greedy LM-head candidate over its vocab-tile stride.
template <int ColsPerBlock,
          int Threads,
          int SwizzleTileBlocks,
          int CandidateCount>
__device__ Gemma4GreedyCandidate greedy_lm_head_block_candidate(
    const __nv_bfloat16 *__restrict__ final_hidden,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    int block_idx,
    int grid_blocks,
    int thread_idx) {
  constexpr int warps = Threads / WARP_SIZE;
  __shared__ float warp_sums[ColsPerBlock][warps];

  float best_logit = kNegativeInfinity;
  int32_t best_token_id = GEMMA4_VOCAB_SIZE;

  for (int tile = block_idx; tile < CandidateCount; tile += grid_blocks) {
    float sums[ColsPerBlock] = {};
    gemma4_matmul_device::decode_gemv_cols_device<
        GEMMA4_HIDDEN_SIZE,
        GEMMA4_VOCAB_SIZE,
        ColsPerBlock,
        Threads,
        SwizzleTileBlocks,
        false>(
        final_hidden, lm_head_col_major, nullptr, tile, warp_sums, sums);

    if (thread_idx == 0) {
      const int32_t token0 = tile * ColsPerBlock;
#pragma unroll
      for (int col = 0; col < ColsPerBlock; ++col) {
        const float logit = __bfloat162float(__float2bfloat16_rn(sums[col]));
        const int32_t token_id = token0 + col;
        if (better_candidate(logit, token_id, best_logit, best_token_id)) {
          best_logit = logit;
          best_token_id = token_id;
        }
      }
    }
    __syncthreads();
  }

  return {best_logit, best_token_id};
}

// Reduces per-block greedy candidates and gathers the selected embedding row.
template <int Threads>
__device__ void reduce_greedy_and_gather(
    __nv_bfloat16 *__restrict__ next_hidden,
    int32_t *__restrict__ next_token,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    Gemma4GreedyCandidate *__restrict__ candidates,
    int32_t active_candidate_count,
    int block_idx,
    int thread_idx) {
  __shared__ float s_logits[Threads];
  __shared__ int32_t s_token_ids[Threads];

  if (block_idx != 0) {
    return;
  }

  Gemma4GreedyCandidate best = {kNegativeInfinity, GEMMA4_VOCAB_SIZE};
  for (int candidate_idx = thread_idx; candidate_idx < active_candidate_count;
       candidate_idx += Threads) {
    const Gemma4GreedyCandidate candidate = candidates[candidate_idx];
    if (better_candidate(candidate.logit, candidate.token_id, best.logit,
                         best.token_id)) {
      best = candidate;
    }
  }

  s_logits[thread_idx] = best.logit;
  s_token_ids[thread_idx] = best.token_id;
  __syncthreads();

  for (int stride = Threads / 2; stride > 0; stride >>= 1) {
    if (thread_idx < stride) {
      const float other_logit = s_logits[thread_idx + stride];
      const int32_t other_token_id = s_token_ids[thread_idx + stride];
      if (better_candidate(other_logit, other_token_id, s_logits[thread_idx],
                           s_token_ids[thread_idx])) {
        s_logits[thread_idx] = other_logit;
        s_token_ids[thread_idx] = other_token_id;
      }
    }
    __syncthreads();
  }

  const int32_t selected_token_id = s_token_ids[0];
  if (thread_idx == 0) {
    *next_token = selected_token_id;
  }
  gemma4_embedding_gather::copy_embedding_row_bf16(
      next_hidden, lm_head_col_major, selected_token_id, thread_idx,
      Threads);
}

}  // namespace gemma4_sampling_device

#endif
