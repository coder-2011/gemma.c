#include "gemma4_sampling.cuh"
#include "gemma4.h"

#ifndef GEMMA4_WEIGHT_LOAD_POLICY
#define GEMMA4_WEIGHT_LOAD_POLICY 0
#endif

#include "gemma4_sampling_device.cuh"

#include <cooperative_groups.h>
#include <stdint.h>

namespace {

namespace sampling_dev = gemma4_sampling_device;

constexpr int kFinalLogitsThreads = 1024;
constexpr int kFinalLogitsColsPerBlock = 8;
constexpr int kFinalLogitsMinBlocksPerSm = 1;
constexpr int kFinalLogitsSwizzleTileBlocks = 1;
constexpr int kCandidateCount = GEMMA4_VOCAB_SIZE / kFinalLogitsColsPerBlock;

static_assert((GEMMA4_VOCAB_SIZE % kFinalLogitsColsPerBlock) == 0,
              "vocab size must divide final-logits columns per block");
static_assert((kFinalLogitsThreads % WARP_SIZE) == 0,
              "final logits thread count must be a warp multiple");
static_assert((kFinalLogitsThreads & (kFinalLogitsThreads - 1)) == 0,
              "final logits thread count must be a power of two");

__global__ __launch_bounds__(kFinalLogitsThreads, kFinalLogitsMinBlocksPerSm)
void final_logits_greedy_fused_kernel(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    const __nv_bfloat16 *__restrict__ d_final_hidden,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    Gemma4GreedyCandidate *__restrict__ d_candidates,
    int active_candidate_count) {
  namespace cg = cooperative_groups;

  const int thread_idx = threadIdx.x;
  const Gemma4GreedyCandidate candidate =
      sampling_dev::greedy_lm_head_block_candidate<
          kFinalLogitsColsPerBlock,
          kFinalLogitsThreads,
          kFinalLogitsSwizzleTileBlocks,
          kCandidateCount>(
          d_final_hidden, d_lm_head_col_major, int(blockIdx.x),
          int(gridDim.x), thread_idx);

  if (thread_idx == 0) {
    d_candidates[blockIdx.x] = candidate;
  }

  cg::this_grid().sync();

  sampling_dev::reduce_greedy_and_gather<kFinalLogitsThreads>(
      d_next_hidden, d_next_token, d_lm_head_col_major, d_candidates,
      active_candidate_count, int(blockIdx.x), thread_idx);
}

// Computes Gumbel-perturbed LM-head candidates and reduces to one sampled row.
__global__ __launch_bounds__(kFinalLogitsThreads, kFinalLogitsMinBlocksPerSm)
void final_logits_gumbel_fused_kernel(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    const __nv_bfloat16 *__restrict__ d_final_hidden,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    Gemma4GreedyCandidate *__restrict__ d_candidates,
    int active_candidate_count,
    Gemma4GumbelSamplingParams params) {
  namespace cg = cooperative_groups;

  const int thread_idx = threadIdx.x;
  const float inv_temperature = 1.0f / params.temperature;
  const Gemma4GreedyCandidate candidate =
      sampling_dev::gumbel_lm_head_block_candidate<
          kFinalLogitsColsPerBlock,
          kFinalLogitsThreads,
          kFinalLogitsSwizzleTileBlocks,
          kCandidateCount>(
          d_final_hidden, d_lm_head_col_major, int(blockIdx.x),
          int(gridDim.x), thread_idx, inv_temperature, params.seed,
          params.step);

  if (thread_idx == 0) {
    d_candidates[blockIdx.x] = candidate;
  }

  cg::this_grid().sync();

  sampling_dev::reduce_greedy_and_gather<kFinalLogitsThreads>(
      d_next_hidden, d_next_token, d_lm_head_col_major, d_candidates,
      active_candidate_count, int(blockIdx.x), thread_idx);
}

// Computes the resident cooperative grid size for fused final-logits kernels.
cudaError_t active_candidate_count_for_kernel(
    const void *kernel,
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
      &active_blocks_per_sm, kernel, kFinalLogitsThreads, 0);
  if (status != cudaSuccess) {
    return status;
  }

  *active_candidate_count =
      active_blocks_per_sm * prop.multiProcessorCount;
  if (*active_candidate_count > kCandidateCount) {
    *active_candidate_count = kCandidateCount;
  }
  if (*active_candidate_count <= 0) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
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
      !is_aligned_16(d_lm_head_col_major) ||
      !is_aligned_to<alignof(Gemma4GreedyCandidate)>(d_scratch)) {
    return cudaErrorInvalidValue;
  }

  auto *d_candidates =
      reinterpret_cast<Gemma4GreedyCandidate *>(d_scratch);

  int active_candidate_count = 0;
  cudaError_t status = active_candidate_count_for_kernel(
      reinterpret_cast<const void *>(final_logits_greedy_fused_kernel),
      &active_candidate_count);
  if (status != cudaSuccess) { return status; }

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

// Returns per-CTA candidate storage for the fused full-vocab sampler.
size_t gemma4_gumbel_sample_next_scratch_bytes(void) {
  return static_cast<size_t>(kCandidateCount) * sizeof(Gemma4GreedyCandidate);
}

// Launches full-vocab Gumbel-Max sampling fused into the decode LM-head GEMV.
cudaError_t gemma4_gumbel_sample_next_decode_bf16(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    void *__restrict__ d_scratch,
    size_t scratch_bytes,
    const __nv_bfloat16 *__restrict__ d_final_hidden,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    Gemma4GumbelSamplingParams params,
    cudaStream_t stream) {
  const size_t required_scratch = gemma4_gumbel_sample_next_scratch_bytes();
  if (scratch_bytes < required_scratch || !(params.temperature > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  if (d_next_hidden == nullptr || d_next_token == nullptr ||
      d_scratch == nullptr || d_final_hidden == nullptr ||
      d_lm_head_col_major == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (!is_aligned_16(d_next_hidden) || !is_aligned_16(d_final_hidden) ||
      !is_aligned_16(d_lm_head_col_major) ||
      !is_aligned_to<alignof(Gemma4GreedyCandidate)>(d_scratch)) {
    return cudaErrorInvalidValue;
  }

  auto *d_candidates =
      reinterpret_cast<Gemma4GreedyCandidate *>(d_scratch);

  int active_candidate_count = 0;
  cudaError_t status = active_candidate_count_for_kernel(
      reinterpret_cast<const void *>(final_logits_gumbel_fused_kernel),
      &active_candidate_count);
  if (status != cudaSuccess) { return status; }

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
      &params,
  };
  status = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(final_logits_gumbel_fused_kernel),
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
