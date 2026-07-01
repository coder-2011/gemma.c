#include "gemma4_sampling.cuh"
#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_embedding_gather.cuh"

#include <cuda/atomic>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kFinalLogitsThreads = 1024;
constexpr int kFinalLogitsColsPerBlock = 8;
constexpr int kFinalLogitsMinBlocksPerSm = 1;
constexpr int kCandidateCount = GEMMA4_VOCAB_SIZE / kFinalLogitsColsPerBlock;

// Applies the Gemma final logit softcap for either first-token or decode sampling.
__device__ inline float softcapped_logit(float logit, Gemma4SamplingStage stage) {
  (void)stage;
  return tanhf(logit / GEMMA4_FINAL_LOGIT_SOFTCAPPING) *
         GEMMA4_FINAL_LOGIT_SOFTCAPPING;
}

// Chooses the higher logit, breaking exact ties by the lower token id.
__device__ inline bool better_candidate(float logit,
                                        int32_t token_id,
                                        float best_logit,
                                        int32_t best_token_id) {
  return logit > best_logit || (logit == best_logit && token_id < best_token_id);
}

// Reduces each column's partial dot product across the CTA.
template <int ColsPerBlock, int Threads>
__device__ inline void reduce_lm_head_cols(
    int thread_idx,
    float (&warp_sums)[ColsPerBlock][Threads / 32],
    float (&sums)[ColsPerBlock]) {
  constexpr int warps = Threads / 32;
  const int lane = thread_idx & (warpSize - 1);
  const int warp = thread_idx / warpSize;

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

// Computes one LM-head vocab tile from final hidden and column-major weights.
template <int ColsPerBlock, int Threads>
__device__ inline void lm_head_tile_logits(
    const __nv_bfloat16 *__restrict__ final_hidden,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    int tile,
    int thread_idx,
    float (&warp_sums)[ColsPerBlock][Threads / 32],
    float (&sums)[ColsPerBlock]) {
  constexpr int packs_per_col =
      GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;

  const int token0 = tile * ColsPerBlock;
  for (int pack_idx = thread_idx; pack_idx < packs_per_col;
       pack_idx += Threads) {
    const int element_idx = pack_idx * kBf16Packed128Elements;
    const Bf16Packed128 x_pack =
        Bf16Packed128{*reinterpret_cast<const int4 *>(final_hidden + element_idx)};
#pragma unroll
    for (int col = 0; col < ColsPerBlock; ++col) {
      const int token_id = token0 + col;
      const __nv_bfloat16 *weight =
          lm_head_col_major +
          static_cast<int64_t>(token_id) * GEMMA4_HIDDEN_SIZE +
          element_idx;
      const Bf16Packed128 w_pack =
          Bf16Packed128{*reinterpret_cast<const int4 *>(weight)};
      gemma4_bf16_pack_accumulate_dot(x_pack, w_pack, sums[col]);
    }
  }

  reduce_lm_head_cols<ColsPerBlock, Threads>(
      thread_idx, warp_sums, sums);
}

// Computes this CTA's best LM-head candidate over its vocab-tile stride.
template <int ColsPerBlock, int Threads, int CandidateCount>
__device__ Gemma4SampleCandidate lm_head_block_candidate(
    const __nv_bfloat16 *__restrict__ final_hidden,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    int block_idx,
    int grid_blocks,
    Gemma4SamplingStage stage,
    int thread_idx) {
  constexpr int warps = Threads / 32;
  __shared__ float warp_sums[ColsPerBlock][warps];

  float best_logit = -INFINITY;
  int32_t best_token_id = GEMMA4_VOCAB_SIZE;

  for (int tile = block_idx; tile < CandidateCount; tile += grid_blocks) {
    float sums[ColsPerBlock] = {};
    lm_head_tile_logits<ColsPerBlock, Threads>(
        final_hidden, lm_head_col_major, tile, thread_idx, warp_sums, sums);

    if (thread_idx == 0) {
      const int32_t token0 = tile * ColsPerBlock;
#pragma unroll
      for (int col = 0; col < ColsPerBlock; ++col) {
        const float raw_logit = __bfloat162float(__float2bfloat16_rn(sums[col]));
        const float logit = softcapped_logit(raw_logit, stage);
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

// Polls producer candidates from warp 0, then gathers the selected row.
template <int Threads>
__device__ void consume_and_gather(
    __nv_bfloat16 *__restrict__ next_hidden,
    int32_t *__restrict__ next_token,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    Gemma4SampleCandidate *__restrict__ candidates,
    uint32_t *__restrict__ ready_flags,
    int32_t producer_count,
    int thread_idx) {
  __shared__ int32_t selected_token_id;

  Gemma4SampleCandidate best = {-INFINITY, GEMMA4_VOCAB_SIZE};
  if (thread_idx < warpSize) {
    const int lane = thread_idx;
    for (int slot = lane; slot < producer_count; slot += warpSize) {
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(
          ready_flags[slot]);
      while (ready.load(cuda::memory_order_acquire) == 0u) {
        __nanosleep(64);
      }
      const Gemma4SampleCandidate candidate = candidates[slot];
      if (better_candidate(candidate.logit, candidate.token_id, best.logit,
                           best.token_id)) {
        best = candidate;
      }
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      const float other_logit =
          __shfl_down_sync(0xffffffffu, best.logit, offset);
      const int32_t other_token_id =
          __shfl_down_sync(0xffffffffu, best.token_id, offset);
      if (better_candidate(other_logit, other_token_id, best.logit,
                           best.token_id)) {
        best.logit = other_logit;
        best.token_id = other_token_id;
      }
    }
    if (lane == 0) {
      selected_token_id = best.token_id;
    }
  }
  __syncthreads();

  if (thread_idx == 0) {
    *next_token = selected_token_id;
  }
  gemma4_embedding_gather::copy_embedding_row_bf16(
      next_hidden, lm_head_col_major, selected_token_id, thread_idx,
      Threads);
}

// Lets block 0 consume producer candidates while producer blocks compute tiles.
__global__ __launch_bounds__(kFinalLogitsThreads, kFinalLogitsMinBlocksPerSm)
void final_logits_sample_kernel(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    const __nv_bfloat16 *__restrict__ d_final_hidden,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    Gemma4SampleCandidate *__restrict__ d_candidates,
    uint32_t *__restrict__ d_ready_flags,
    Gemma4SamplingStage stage,
    int producer_count) {
  if (blockIdx.x == 0) {
    consume_and_gather<kFinalLogitsThreads>(
        d_next_hidden, d_next_token, d_lm_head_col_major, d_candidates,
        d_ready_flags, producer_count, int(threadIdx.x));
    return;
  }

  const int producer_idx = int(blockIdx.x) - 1;
  const Gemma4SampleCandidate candidate =
      lm_head_block_candidate<
          kFinalLogitsColsPerBlock,
          kFinalLogitsThreads,
          kCandidateCount>(
          d_final_hidden, d_lm_head_col_major, producer_idx,
          producer_count, stage, int(threadIdx.x));

  if (threadIdx.x == 0) {
    d_candidates[producer_idx] = candidate;
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> ready(
        d_ready_flags[producer_idx]);
    ready.store(1u, cuda::memory_order_release);
  }
}

// Computes producer count while leaving one resident slot for the consumer CTA.
cudaError_t producer_count_for_kernel(const void *kernel, int *producer_count) {
  int device = 0;
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaGetDevice(&device));

  cudaDeviceProp prop = {};
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));
  int active_blocks_per_sm = 0;
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, kernel, kFinalLogitsThreads, 0));

  *producer_count = active_blocks_per_sm * prop.multiProcessorCount - 1;
  if (*producer_count > kCandidateCount) {
    *producer_count = kCandidateCount;
  }
  if (*producer_count <= 0) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

}  // namespace

// Returns the caller-owned scratch bytes needed by the fused sampling kernel.
size_t gemma4_sample_next_scratch_bytes(void) {
  const size_t candidate_bytes =
      static_cast<size_t>(kCandidateCount) * sizeof(Gemma4SampleCandidate);
  const size_t ready_bytes =
      static_cast<size_t>(kCandidateCount) * sizeof(uint32_t);
  return candidate_bytes + ready_bytes;
}

// Launches fused token selection from final hidden to next embedding.
cudaError_t gemma4_sample_next_bf16(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    void *__restrict__ d_scratch,
    size_t scratch_bytes,
    const __nv_bfloat16 *__restrict__ d_final_hidden,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    Gemma4SamplingStage stage,
    cudaStream_t stream) {
  const size_t candidate_bytes =
      static_cast<size_t>(kCandidateCount) * sizeof(Gemma4SampleCandidate);
  if (scratch_bytes < gemma4_sample_next_scratch_bytes()) {
    return cudaErrorInvalidValue;
  }

  auto *d_candidates =
      reinterpret_cast<Gemma4SampleCandidate *>(d_scratch);
  auto *d_ready_flags =
      reinterpret_cast<uint32_t *>(static_cast<char *>(d_scratch) +
                                   candidate_bytes);

  int producer_count = 0;
  GEMMA4_RETURN_IF_CUDA_ERROR(producer_count_for_kernel(
      reinterpret_cast<const void *>(final_logits_sample_kernel),
      &producer_count));

  const size_t active_ready_bytes =
      static_cast<size_t>(producer_count) * sizeof(uint32_t);
  GEMMA4_RETURN_IF_CUDA_ERROR(
      cudaMemsetAsync(d_ready_flags, 0, active_ready_bytes, stream));

  const dim3 grid_dim(producer_count + 1);
  const dim3 block_dim(kFinalLogitsThreads);
  final_logits_sample_kernel<<<grid_dim, block_dim, 0, stream>>>(
      d_next_hidden, d_next_token, d_final_hidden, d_lm_head_col_major,
      d_candidates, d_ready_flags, stage, producer_count);
  return cudaGetLastError();
}
