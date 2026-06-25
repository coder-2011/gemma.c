#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"

namespace gemma4_embedding_gather {

constexpr int kEmbeddingGatherThreads = WARP_SIZE;
constexpr int kPacksPerEmbeddingRow =
    GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
static_assert((GEMMA4_HIDDEN_SIZE % kBf16Packed128Elements) == 0,
              "embedding width must be divisible by Packed128 bf16 width");

// Copies one tied embedding row into model-input space.
__device__ inline void copy_embedding_row_bf16(
    __nv_bfloat16 *__restrict__ out_row,
    const __nv_bfloat16 *__restrict__ embeddings,
    int32_t token_id,
    int lane,
    int thread_count) {
  const __nv_bfloat16 *embedding_row =
      embeddings + static_cast<int64_t>(token_id) * GEMMA4_HIDDEN_SIZE;

  for (int pack_idx = lane; pack_idx < kPacksPerEmbeddingRow;
       pack_idx += thread_count) {
    const int offset = pack_idx * kBf16Packed128Elements;
    Bf16Packed128 pack = load128g(embedding_row + offset);
    pack = gemma4_bf16_pack_apply_scale(pack, GEMMA4_EMBEDDING_SCALE);
    store128(out_row + offset, pack);
  }
}

}  // namespace gemma4_embedding_gather

cudaError_t gemma4_embedding_gather_bf16(
    __nv_bfloat16 *__restrict__ out,
    const int32_t *__restrict__ token_ids,
    const __nv_bfloat16 *__restrict__ embeddings,
    int32_t num_tokens,
    cudaStream_t stream);
