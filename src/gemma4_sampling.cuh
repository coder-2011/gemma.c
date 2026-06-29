#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

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

// Runs fused LM-head GEMV, token selection, and next embedding gather for decode.
cudaError_t gemma4_sample_next_decode_bf16(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    void *__restrict__ d_scratch,
    size_t scratch_bytes,
    const __nv_bfloat16 *__restrict__ d_final_hidden,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    cudaStream_t stream);
