#include "gemma4_embedding_gather.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

#include <stdint.h>

namespace {

constexpr int kGemma4EmbeddingPacks =
    GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
static_assert((GEMMA4_HIDDEN_SIZE % kBf16Packed128Elements) == 0,
              "embedding width must be divisible by Packed128 bf16 width");

__global__ void gemma4_embedding_gather_bf16_warp_kernel(
    __nv_bfloat16* out,
    const int32_t* token_ids,
    const __nv_bfloat16* embeddings) {
    int32_t token_idx = blockIdx.x;
    int32_t lane = threadIdx.x;

    int32_t token_id = token_ids[token_idx];
    if (token_id < 0 || token_id >= GEMMA4_VOCAB_SIZE) {
        return;
    }

    const __nv_bfloat16* embedding_row =
        embeddings + (int64_t)token_id * GEMMA4_HIDDEN_SIZE;
    __nv_bfloat16* out_row = out + (int64_t)token_idx * GEMMA4_HIDDEN_SIZE;

    for (int32_t pack_idx = lane; pack_idx < kGemma4EmbeddingPacks;
         pack_idx += WARP_SIZE) {
        Bf16Packed128 pack =
            load128cs(embedding_row + pack_idx * kBf16Packed128Elements);
        store128(out_row + pack_idx * kBf16Packed128Elements, pack);
    }
}

}  // namespace

cudaError_t gemma4_embedding_gather_bf16(
    __nv_bfloat16* out,
    const int32_t* token_ids,
    const __nv_bfloat16* embeddings,
    int32_t num_tokens,
    cudaStream_t stream) {
    if (num_tokens < 0) {
        return cudaErrorInvalidValue;
    }
    if (num_tokens == 0) {
        return cudaSuccess;
    }
    if (out == nullptr || token_ids == nullptr || embeddings == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (!is_aligned_16(out) || !is_aligned_16(embeddings)) {
        return cudaErrorInvalidValue;
    }

    gemma4_embedding_gather_bf16_warp_kernel<<<num_tokens, WARP_SIZE, 0, stream>>>(
        out, token_ids, embeddings);
    return cudaGetLastError();
}
