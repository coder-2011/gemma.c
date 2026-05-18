#include "gemma4_embedding_gather.cuh"
#include "gemma4_cuda_utils.cuh"

#include <stdint.h>

namespace {

using EmbeddingPack = Packed128<floatX>;
constexpr int kFloatXPerPack = EmbeddingPack::size;

__global__ void embedding_gather_bf16_warp_kernel(
    __nv_bfloat16* out,
    const int32_t* token_ids,
    const __nv_bfloat16* embeddings,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t vocab_size) {
    int32_t token_idx = blockIdx.x;
    int32_t lane = threadIdx.x;

    if (token_idx >= num_tokens || lane >= GEMMA4_WARP_SIZE) {
        return;
    }

    int32_t token_id = token_ids[token_idx];
    if (token_id < 0 || token_id >= vocab_size) {
        return;
    }

    int32_t packs_per_row = hidden_size / kFloatXPerPack;
    const floatX* embedding_row = embeddings + (int64_t)token_id * hidden_size;
    floatX* out_row = out + (int64_t)token_idx * hidden_size;

    for (int32_t pack_idx = lane; pack_idx < packs_per_row;
         pack_idx += GEMMA4_WARP_SIZE) {
        EmbeddingPack pack =
            gemma4_load128cs(embedding_row + pack_idx * kFloatXPerPack);
        gemma4_store128(out_row + pack_idx * kFloatXPerPack, pack);
    }
}

}  // namespace

cudaError_t gemma4_embedding_gather_bf16(
    __nv_bfloat16* out,
    const int32_t* token_ids,
    const __nv_bfloat16* embeddings,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t vocab_size,
    cudaStream_t stream) {
    if (num_tokens < 0 || hidden_size <= 0 || vocab_size <= 0) {
        return cudaErrorInvalidValue;
    }
    if (num_tokens == 0) {
        return cudaSuccess;
    }
    if (out == nullptr || token_ids == nullptr || embeddings == nullptr) {
        return cudaErrorInvalidValue;
    }
    if ((hidden_size % kFloatXPerPack) != 0) {
        return cudaErrorInvalidValue;
    }
    if (!gemma4_is_aligned_16(out) || !gemma4_is_aligned_16(embeddings)) {
        return cudaErrorInvalidValue;
    }

    embedding_gather_bf16_warp_kernel<<<num_tokens, GEMMA4_WARP_SIZE, 0, stream>>>(
        out, token_ids, embeddings, num_tokens, hidden_size, vocab_size);
    return cudaGetLastError();
}
