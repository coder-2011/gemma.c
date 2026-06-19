#include "gemma4_sampling.cuh"
#include "gemma4.h"

#ifndef GEMMA4_WEIGHT_LOAD_POLICY
#define GEMMA4_WEIGHT_LOAD_POLICY 0
#endif

#include "gemma4_cuda_utils.cuh"
#include "gemma4_embedding_gather.cuh"
#include "gemma4_matmul_device.cuh"

#include <cooperative_groups.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kFinalLogitsThreads = 1024;
constexpr int kFinalLogitsColsPerBlock = 8;
constexpr int kFinalLogitsMinBlocksPerSm = 1;
constexpr int kFinalLogitsSwizzleTileBlocks = 1;
constexpr int kCandidateCount = GEMMA4_VOCAB_SIZE / kFinalLogitsColsPerBlock;
constexpr int kLogitsSamplerThreads = 256;
constexpr int kMaxTopK = 64;
constexpr float kNegativeInfinity = -3.4028234663852886e+38F;

static_assert((GEMMA4_VOCAB_SIZE % kFinalLogitsColsPerBlock) == 0,
              "vocab size must divide final-logits columns per block");
static_assert((kFinalLogitsThreads % WARP_SIZE) == 0,
              "final logits thread count must be a warp multiple");
static_assert((kFinalLogitsThreads & (kFinalLogitsThreads - 1)) == 0,
              "final logits thread count must be a power of two");
static_assert((kLogitsSamplerThreads % WARP_SIZE) == 0,
              "logits sampler thread count must be a warp multiple");

inline bool is_aligned_8(const void *ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 0x7u) == 0;
}

__device__ inline bool better_candidate(float logit,
                                        int32_t token_id,
                                        float best_logit,
                                        int32_t best_token_id) {
  return logit > best_logit ||
         (logit == best_logit && token_id < best_token_id);
}

__device__ inline void insert_sorted_candidate(float *__restrict__ scores,
                                               int32_t *__restrict__ token_ids,
                                               int top_k,
                                               float score,
                                               int32_t token_id) {
  if (!better_candidate(score, token_id, scores[top_k - 1],
                        token_ids[top_k - 1])) {
    return;
  }

  int pos = top_k - 1;
  while (pos > 0 &&
         better_candidate(score, token_id, scores[pos - 1],
                          token_ids[pos - 1])) {
    scores[pos] = scores[pos - 1];
    token_ids[pos] = token_ids[pos - 1];
    --pos;
  }
  scores[pos] = score;
  token_ids[pos] = token_id;
}

__device__ inline float transformed_sampling_score(__nv_bfloat16 logit,
                                                   float inv_temperature) {
  const float raw = __bfloat162float(logit);
  const float capped =
      tanhf(raw / GEMMA4_FINAL_LOGIT_SOFTCAPPING) *
      GEMMA4_FINAL_LOGIT_SOFTCAPPING;
  return capped * inv_temperature;
}

__device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

__device__ inline float sampling_uniform01(uint64_t seed,
                                           uint64_t step,
                                           int32_t batch_row) {
  uint64_t state = seed;
  state ^= (step + 0x9e3779b97f4a7c15ull) * 0xbf58476d1ce4e5b9ull;
  state ^= (uint64_t(batch_row) + 0x94d049bb133111ebull) *
           0x9e3779b97f4a7c15ull;
  const uint32_t bits = uint32_t(splitmix64(state) >> 40);
  return float(bits) * (1.0f / 16777216.0f);
}

__global__ __launch_bounds__(kLogitsSamplerThreads)
void sample_from_logits_kernel(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    const __nv_bfloat16 *__restrict__ d_logits,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    Gemma4SamplingParams params) {
  __shared__ float s_candidate_scores[kLogitsSamplerThreads];
  __shared__ int32_t s_candidate_token_ids[kLogitsSamplerThreads];
  __shared__ int32_t s_candidate_owners[kLogitsSamplerThreads];
  __shared__ float s_top_scores[kMaxTopK];
  __shared__ int32_t s_top_token_ids[kMaxTopK];
  __shared__ int32_t s_selected_token_id;

  const int batch_row = int(blockIdx.x);
  const int thread_idx = threadIdx.x;
  const int top_k = params.top_k;
  const float inv_temperature = 1.0f / params.temperature;
  const __nv_bfloat16 *logits_row =
      d_logits + int64_t(batch_row) * GEMMA4_VOCAB_SIZE;

  float local_scores[kMaxTopK];
  int32_t local_token_ids[kMaxTopK];
  for (int i = 0; i < top_k; ++i) {
    local_scores[i] = kNegativeInfinity;
    local_token_ids[i] = GEMMA4_VOCAB_SIZE + i;
  }

  for (int token_id = thread_idx; token_id < GEMMA4_VOCAB_SIZE;
       token_id += blockDim.x) {
    const float score =
        transformed_sampling_score(logits_row[token_id], inv_temperature);
    insert_sorted_candidate(local_scores, local_token_ids, top_k, score,
                            token_id);
  }

  int local_pos = 0;
  for (int top_idx = 0; top_idx < top_k; ++top_idx) {
    s_candidate_scores[thread_idx] = local_scores[local_pos];
    s_candidate_token_ids[thread_idx] = local_token_ids[local_pos];
    s_candidate_owners[thread_idx] = thread_idx;
    __syncthreads();

    for (int stride = kLogitsSamplerThreads / 2; stride > 0; stride >>= 1) {
      if (thread_idx < stride) {
        const float other_score = s_candidate_scores[thread_idx + stride];
        const int32_t other_token_id =
            s_candidate_token_ids[thread_idx + stride];
        if (better_candidate(other_score, other_token_id,
                             s_candidate_scores[thread_idx],
                             s_candidate_token_ids[thread_idx])) {
          s_candidate_scores[thread_idx] = other_score;
          s_candidate_token_ids[thread_idx] = other_token_id;
          s_candidate_owners[thread_idx] =
              s_candidate_owners[thread_idx + stride];
        }
      }
      __syncthreads();
    }

    if (thread_idx == 0) {
      s_top_scores[top_idx] = s_candidate_scores[0];
      s_top_token_ids[top_idx] = s_candidate_token_ids[0];
    }
    __syncthreads();

    if (thread_idx == s_candidate_owners[0] && local_pos + 1 < top_k) {
      ++local_pos;
    }
    __syncthreads();
  }

  if (thread_idx == 0) {
    const float max_score = s_top_scores[0];
    float weights[kMaxTopK];
    float total_weight = 0.0f;
    for (int i = 0; i < top_k; ++i) {
      const float weight = expf(s_top_scores[i] - max_score);
      weights[i] = weight;
      total_weight += weight;
    }

    const float threshold = params.top_p * total_weight;
    float nucleus_weight = 0.0f;
    int nucleus_size = 0;
    for (int i = 0; i < top_k; ++i) {
      nucleus_weight += weights[i];
      nucleus_size = i + 1;
      if (nucleus_weight >= threshold) {
        break;
      }
    }

    const float target =
        sampling_uniform01(params.seed, params.step, batch_row) *
        nucleus_weight;
    float cumulative = 0.0f;
    int32_t selected_token_id = s_top_token_ids[nucleus_size - 1];
    for (int i = 0; i < nucleus_size; ++i) {
      cumulative += weights[i];
      if (cumulative > target) {
        selected_token_id = s_top_token_ids[i];
        break;
      }
    }

    s_selected_token_id = selected_token_id;
    d_next_token[batch_row] = selected_token_id;
  }
  __syncthreads();

  __nv_bfloat16 *next_hidden_row =
      d_next_hidden + int64_t(batch_row) * GEMMA4_HIDDEN_SIZE;
  gemma4_embedding_gather::copy_embedding_row_bf16(
      next_hidden_row, d_lm_head_col_major, s_selected_token_id, thread_idx,
      blockDim.x);
}

__global__ __launch_bounds__(kFinalLogitsThreads, kFinalLogitsMinBlocksPerSm)
void final_logits_greedy_fused_kernel(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    const __nv_bfloat16 *__restrict__ d_final_hidden,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    Gemma4GreedyCandidate *__restrict__ d_candidates,
    int active_candidate_count) {
  namespace cg = cooperative_groups;

  constexpr int warps = kFinalLogitsThreads / WARP_SIZE;
  __shared__ float warp_sums[kFinalLogitsColsPerBlock][warps];
  __shared__ float s_logits[kFinalLogitsThreads];
  __shared__ int32_t s_token_ids[kFinalLogitsThreads];

  const int thread_idx = threadIdx.x;
  float block_best_logit = kNegativeInfinity;
  int32_t block_best_token_id = GEMMA4_VOCAB_SIZE;

  for (int tile = int(blockIdx.x); tile < kCandidateCount;
       tile += int(gridDim.x)) {
    float sums[kFinalLogitsColsPerBlock] = {};
    gemma4_matmul_device::decode_gemv_cols_device<
        GEMMA4_HIDDEN_SIZE,
        GEMMA4_VOCAB_SIZE,
        kFinalLogitsColsPerBlock,
        kFinalLogitsThreads,
        kFinalLogitsSwizzleTileBlocks,
        false>(
        d_final_hidden, d_lm_head_col_major, nullptr, tile, warp_sums, sums);

    if (thread_idx == 0) {
      const int token0 = tile * kFinalLogitsColsPerBlock;
#pragma unroll
      for (int col = 0; col < kFinalLogitsColsPerBlock; ++col) {
        const float logit = __bfloat162float(__float2bfloat16_rn(sums[col]));
        const int32_t token_id = token0 + col;
        if (better_candidate(logit, token_id, block_best_logit,
                             block_best_token_id)) {
          block_best_logit = logit;
          block_best_token_id = token_id;
        }
      }
    }

    __syncthreads();
  }

  if (thread_idx == 0) {
    d_candidates[blockIdx.x] = {block_best_logit, block_best_token_id};
  }

  cg::this_grid().sync();

  if (blockIdx.x != 0) {
    return;
  }

  Gemma4GreedyCandidate best = {kNegativeInfinity, GEMMA4_VOCAB_SIZE};
  for (int candidate_idx = thread_idx; candidate_idx < active_candidate_count;
       candidate_idx += kFinalLogitsThreads) {
    const Gemma4GreedyCandidate candidate = d_candidates[candidate_idx];
    if (better_candidate(candidate.logit, candidate.token_id, best.logit,
                         best.token_id)) {
      best = candidate;
    }
  }

  s_logits[thread_idx] = best.logit;
  s_token_ids[thread_idx] = best.token_id;
  __syncthreads();

  for (int stride = kFinalLogitsThreads / 2; stride > 0; stride >>= 1) {
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
    *d_next_token = selected_token_id;
  }

  gemma4_embedding_gather::copy_embedding_row_bf16(
      d_next_hidden, d_lm_head_col_major, selected_token_id, thread_idx,
      blockDim.x);
}

}  // namespace

size_t gemma4_greedy_sample_next_scratch_bytes(void) {
  return static_cast<size_t>(kCandidateCount) * sizeof(Gemma4GreedyCandidate);
}

cudaError_t gemma4_greedy_sample_next_decode_bf16(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    void *__restrict__ d_scratch,
    size_t scratch_bytes,
    const __nv_bfloat16 *__restrict__ d_final_hidden,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    cudaStream_t stream) {
  const size_t required_scratch = gemma4_greedy_sample_next_scratch_bytes();
  if (scratch_bytes < required_scratch) {
    return cudaErrorInvalidValue;
  }
  if (d_next_hidden == nullptr || d_next_token == nullptr ||
      d_scratch == nullptr || d_final_hidden == nullptr ||
      d_lm_head_col_major == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (!is_aligned_16(d_next_hidden) || !is_aligned_16(d_final_hidden) ||
      !is_aligned_16(d_lm_head_col_major) || !is_aligned_8(d_scratch)) {
    return cudaErrorInvalidValue;
  }

  auto *d_candidates =
      reinterpret_cast<Gemma4GreedyCandidate *>(d_scratch);

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
      &active_blocks_per_sm,
      final_logits_greedy_fused_kernel,
      kFinalLogitsThreads,
      0);
  if (status != cudaSuccess) {
    return status;
  }
  int active_candidate_count = active_blocks_per_sm * prop.multiProcessorCount;
  if (active_candidate_count > kCandidateCount) {
    active_candidate_count = kCandidateCount;
  }
  if (active_candidate_count <= 0) {
    return cudaErrorInvalidValue;
  }

  __nv_bfloat16 *next_hidden_arg = d_next_hidden;
  int32_t *next_token_arg = d_next_token;
  const __nv_bfloat16 *final_hidden_arg = d_final_hidden;
  const __nv_bfloat16 *lm_head_arg = d_lm_head_col_major;
  Gemma4GreedyCandidate *candidates_arg = d_candidates;

  void *kernel_args[] = {
      &next_hidden_arg,
      &next_token_arg,
      &final_hidden_arg,
      &lm_head_arg,
      &candidates_arg,
      &active_candidate_count,
  };
  status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(final_logits_greedy_fused_kernel),
      active_candidate_count,
      kFinalLogitsThreads,
      kernel_args,
      0,
      stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaGetLastError();
}

size_t gemma4_sample_from_logits_scratch_bytes(int32_t batch_size,
                                               int32_t top_k) {
  (void)batch_size;
  (void)top_k;
  return 0;
}

cudaError_t gemma4_sample_from_logits_decode_bf16(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    void *__restrict__ d_scratch,
    size_t scratch_bytes,
    const __nv_bfloat16 *__restrict__ d_logits,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    int32_t batch_size,
    Gemma4SamplingParams params,
    cudaStream_t stream) {
  (void)d_scratch;
  (void)scratch_bytes;
  if (batch_size <= 0 || params.top_k < 1 || params.top_k > kMaxTopK ||
      !(params.temperature > 0.0f) || !(params.top_p > 0.0f) ||
      !(params.top_p <= 1.0f)) {
    return cudaErrorInvalidValue;
  }
  if (d_next_hidden == nullptr || d_next_token == nullptr ||
      d_logits == nullptr || d_lm_head_col_major == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (!is_aligned_16(d_next_hidden) || !is_aligned_16(d_logits) ||
      !is_aligned_16(d_lm_head_col_major)) {
    return cudaErrorInvalidValue;
  }

  sample_from_logits_kernel<<<batch_size, kLogitsSamplerThreads, 0, stream>>>(
      d_next_hidden, d_next_token, d_logits, d_lm_head_col_major, params);
  return cudaGetLastError();
}
