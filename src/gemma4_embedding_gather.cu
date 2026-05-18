#include "gemma4_embedding_gather.cuh"

#include <stdint.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kBf16PerInt4 = sizeof(int4) / sizeof(__nv_bfloat16);

__global__ void embedding_gather_bf16_warp_kernel(
    __nv_bfloat16* out,
    const int32_t* token_ids,
    const __nv_bfloat16* embeddings,
    int32_t num_tokens,
    int32_t hidden_size,
    int32_t vocab_size) {
    int32_t token_idx = blockIdx.x;
    int32_t lane = threadIdx.x;

    if (token_idx >= num_tokens || lane >= kWarpSize) {
        return;
    }

    int32_t token_id = token_ids[token_idx];
    if (token_id < 0 || token_id >= vocab_size) {
        return;
    }

    int32_t vectors_per_row = hidden_size / kBf16PerInt4;
    const int4* embedding_row =
        reinterpret_cast<const int4*>(embeddings + (int64_t)token_id * hidden_size);
    int4* out_row = reinterpret_cast<int4*>(out + (int64_t)token_idx * hidden_size);

    for (int32_t vec = lane; vec < vectors_per_row; vec += kWarpSize) {
        out_row[vec] = __ldcs(embedding_row + vec);
    }
}

bool is_aligned_16(const void* ptr) {
    return (reinterpret_cast<uintptr_t>(ptr) & 0xfu) == 0;
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
    if ((hidden_size % kBf16PerInt4) != 0) {
        return cudaErrorInvalidValue;
    }
    if (!is_aligned_16(out) || !is_aligned_16(embeddings)) {
        return cudaErrorInvalidValue;
    }

    embedding_gather_bf16_warp_kernel<<<num_tokens, kWarpSize, 0, stream>>>(
        out, token_ids, embeddings, num_tokens, hidden_size, vocab_size);
    return cudaGetLastError();
}
