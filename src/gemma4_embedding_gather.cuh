#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace gemma4_embedding_gather {

// Copies one tied embedding row into model-input space.
__device__ void copy_embedding_row_bf16(
    __nv_bfloat16 *__restrict__ out_row,
    const __nv_bfloat16 *__restrict__ embeddings,
    int32_t token_id,
    int lane,
    int thread_count);

}  // namespace gemma4_embedding_gather

cudaError_t gemma4_embedding_gather_bf16(
    __nv_bfloat16 *__restrict__ out,
    const int32_t *__restrict__ token_ids,
    const __nv_bfloat16 *__restrict__ embeddings,
    int32_t num_tokens,
    cudaStream_t stream);
