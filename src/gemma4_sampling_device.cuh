#ifndef GEMMA4_SAMPLING_DEVICE_CUH
#define GEMMA4_SAMPLING_DEVICE_CUH

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_embedding_gather.cuh"
#include "gemma4_sampling.cuh"

#include <cuda_bf16.h>
#include <math.h>
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

// Mixes a 64-bit counter into a deterministic per-token RNG word.
__device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// Generates a reproducible uniform in the open interval (0, 1).
__device__ inline float uniform01_open(uint64_t seed,
                                       uint64_t step,
                                       int32_t token_id) {
  uint64_t state = seed;
  state ^= (step + 0x9e3779b97f4a7c15ull) * 0xbf58476d1ce4e5b9ull;
  state ^= (uint64_t(token_id) + 0x94d049bb133111ebull) *
           0x9e3779b97f4a7c15ull;
  const uint32_t bits = uint32_t(splitmix64(state) >> 40);
  return (float(bits) + 0.5f) * (1.0f / 16777216.0f);
}

// Converts the per-token uniform into standard Gumbel noise.
__device__ inline float gumbel_noise(uint64_t seed,
                                    uint64_t step,
                                    int32_t token_id) {
  const float u = uniform01_open(seed, step, token_id);
  return -logf(-logf(u));
}

// Applies Gemma's final logit softcap and sampling temperature.
__device__ inline float transformed_lm_head_score(float raw_logit,
                                                  float inv_temperature) {
  const float capped =
      tanhf(raw_logit / GEMMA4_FINAL_LOGIT_SOFTCAPPING) *
      GEMMA4_FINAL_LOGIT_SOFTCAPPING;
  return capped * inv_temperature;
}

template <int ColsPerBlock, int Threads>
__device__ inline void reduce_lm_head_cols(
    int thread_idx,
    float (&warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&sums)[ColsPerBlock]) {
  constexpr int warps = Threads / WARP_SIZE;
  const int lane = thread_idx & (WARP_SIZE - 1);
  const int warp = thread_idx / WARP_SIZE;

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

template <int ColsPerBlock, int Threads>
__device__ inline void lm_head_tile_logits(
    const __nv_bfloat16 *__restrict__ final_hidden,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    int tile,
    int thread_idx,
    float (&warp_sums)[ColsPerBlock][Threads / WARP_SIZE],
    float (&sums)[ColsPerBlock]) {
  constexpr int packs_per_col =
      GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
  static_assert((GEMMA4_HIDDEN_SIZE % kBf16Packed128Elements) == 0,
                "hidden size must divide the 128-bit bf16 pack width");

  const int token0 = tile * ColsPerBlock;
  for (int pack_idx = thread_idx; pack_idx < packs_per_col;
       pack_idx += Threads) {
    const int element_idx = pack_idx * kBf16Packed128Elements;
    const Bf16Packed128 x_pack = load128g(final_hidden + element_idx);
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const int token_id = token0 + col;
      const __nv_bfloat16 *weight =
          lm_head_col_major +
          static_cast<int64_t>(token_id) * GEMMA4_HIDDEN_SIZE +
          element_idx;
      const Bf16Packed128 w_pack = load128weight(weight);
      gemma4_bf16_pack_accumulate_dot(x_pack, w_pack, sums[col]);
    }
  }

  reduce_lm_head_cols<ColsPerBlock, Threads>(
      thread_idx, warp_sums, sums);
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
    static_assert(SwizzleTileBlocks == 1,
                  "fused sampling expects identity LM-head tile order");
    lm_head_tile_logits<ColsPerBlock, Threads>(
        final_hidden, lm_head_col_major, tile, thread_idx, warp_sums, sums);

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

// Computes this CTA's Gumbel-Max LM-head candidate over its vocab-tile stride.
template <int ColsPerBlock,
          int Threads,
          int SwizzleTileBlocks,
          int CandidateCount>
__device__ Gemma4GreedyCandidate gumbel_lm_head_block_candidate(
    const __nv_bfloat16 *__restrict__ final_hidden,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    int block_idx,
    int grid_blocks,
    int thread_idx,
    float inv_temperature,
    uint64_t seed,
    uint64_t step) {
  constexpr int warps = Threads / WARP_SIZE;
  __shared__ float warp_sums[ColsPerBlock][warps];

  float best_score = kNegativeInfinity;
  int32_t best_token_id = GEMMA4_VOCAB_SIZE;

  for (int tile = block_idx; tile < CandidateCount; tile += grid_blocks) {
    float sums[ColsPerBlock] = {};
    static_assert(SwizzleTileBlocks == 1,
                  "fused sampling expects identity LM-head tile order");
    lm_head_tile_logits<ColsPerBlock, Threads>(
        final_hidden, lm_head_col_major, tile, thread_idx, warp_sums, sums);

    if (thread_idx == 0) {
      const int32_t token0 = tile * ColsPerBlock;
#pragma unroll
      for (int col = 0; col < ColsPerBlock; ++col) {
        const float raw_logit =
            __bfloat162float(__float2bfloat16_rn(sums[col]));
        const int32_t token_id = token0 + col;
        const float score =
            transformed_lm_head_score(raw_logit, inv_temperature) +
            gumbel_noise(seed, step, token_id);
        if (better_candidate(score, token_id, best_score, best_token_id)) {
          best_score = score;
          best_token_id = token_id;
        }
      }
    }
    __syncthreads();
  }

  return {best_score, best_token_id};
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
