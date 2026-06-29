#pragma once

#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

static constexpr int GEMMA4_SAMPLE_NEXT_COLS_PER_BLOCK = 8;
static constexpr int GEMMA4_SAMPLE_NEXT_CANDIDATE_COUNT =
    GEMMA4_VOCAB_SIZE / GEMMA4_SAMPLE_NEXT_COLS_PER_BLOCK;

// Carries one token candidate and its score through block-level reductions.
struct alignas(8) Gemma4SampleCandidate {
  float logit;
  int32_t token_id;
};

enum class Gemma4SamplingStage {
  kPrefill,
  kDecode,
};

// Returns the caller-owned scratch bytes needed by the fused sampling kernel.
size_t gemma4_sample_next_scratch_bytes(void);

// Runs fused LM-head GEMV, logit softcap, token selection, and embedding gather.
cudaError_t gemma4_sample_next_bf16(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    void *__restrict__ d_scratch,
    size_t scratch_bytes,
    const __nv_bfloat16 *__restrict__ d_final_hidden,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    Gemma4SamplingStage stage,
    cudaStream_t stream);

// Runs final RMSNorm, LM-head GEMV, token selection, and next embedding gather.
cudaError_t gemma4_sample_next_final_norm_bf16(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    void *__restrict__ d_scratch,
    size_t scratch_bytes,
    const __nv_bfloat16 *__restrict__ d_state,
    const __nv_bfloat16 *__restrict__ d_final_norm_weight,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    Gemma4SamplingStage stage,
    cudaStream_t stream);

// Runs fused LM-head GEMV, token selection, and next embedding gather for decode.
cudaError_t gemma4_sample_next_decode_bf16(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    void *__restrict__ d_scratch,
    size_t scratch_bytes,
    const __nv_bfloat16 *__restrict__ d_final_hidden,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    cudaStream_t stream);

// Runs sampling inside an already-resident decode token grid.
extern "C" __device__ __noinline__ void gemma4_sample_next_bf16_device(
    __nv_bfloat16 *__restrict__ next_hidden,
    int32_t *__restrict__ next_token,
    Gemma4SampleCandidate *__restrict__ candidates,
    uint32_t *__restrict__ ready_flags,
    const __nv_bfloat16 *__restrict__ final_hidden,
    const __nv_bfloat16 *__restrict__ lm_head_col_major,
    int producer_count,
    int thread_idx);
