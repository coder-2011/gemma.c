#ifndef GEMMA4_EMBEDDING_GATHER_CUH
#define GEMMA4_EMBEDDING_GATHER_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

cudaError_t gemma4_embedding_gather_bf16(
    __nv_bfloat16* out,
    const int32_t* token_ids,
    const __nv_bfloat16* embeddings,
    int32_t num_tokens,
    cudaStream_t stream);

#endif
