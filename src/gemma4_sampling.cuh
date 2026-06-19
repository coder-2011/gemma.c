#ifndef GEMMA4_SAMPLING_CUH
#define GEMMA4_SAMPLING_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

typedef struct alignas(8) Gemma4GreedyCandidate {
  float logit;
  int32_t token_id;
} Gemma4GreedyCandidate;

typedef struct {
  float temperature;
  float top_p;
  int32_t top_k;
  uint64_t seed;
  uint64_t step;
} Gemma4SamplingParams;

size_t gemma4_greedy_sample_next_scratch_bytes(void);

cudaError_t gemma4_greedy_sample_next_decode_bf16(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    void *__restrict__ d_scratch,
    size_t scratch_bytes,
    const __nv_bfloat16 *__restrict__ d_final_hidden,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    cudaStream_t stream);

size_t gemma4_sample_from_logits_scratch_bytes(int32_t batch_size,
                                               int32_t top_k);

cudaError_t gemma4_sample_from_logits_decode_bf16(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    void *__restrict__ d_scratch,
    size_t scratch_bytes,
    const __nv_bfloat16 *__restrict__ d_logits,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    int32_t batch_size,
    Gemma4SamplingParams params,
    cudaStream_t stream);

#endif
