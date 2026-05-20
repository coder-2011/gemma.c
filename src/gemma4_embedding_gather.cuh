#ifndef GEMMA4_EMBEDDING_GATHER_CUH
#define GEMMA4_EMBEDDING_GATHER_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

cudaError_t gemma4_embedding_gather_bf16(
    __nv_bfloat16 *__restrict__ out,
    const int32_t *__restrict__ token_ids,
    const __nv_bfloat16 *__restrict__ embeddings,
    int32_t num_tokens,
    cudaStream_t stream);

#endif
