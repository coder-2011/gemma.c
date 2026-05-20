#include "gemma4_embedding_gather.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

#include <stdint.h>

namespace {

constexpr int kEmbeddingGatherThreads = WARP_SIZE;
constexpr int kPacksPerEmbeddingRow = GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
static_assert((GEMMA4_HIDDEN_SIZE % kBf16Packed128Elements) == 0,
              "embedding width must be divisible by Packed128 bf16 width");

__global__ __launch_bounds__(kEmbeddingGatherThreads) void embedding_gather_kernel(
    __nv_bfloat16 *__restrict__ out,
    const int32_t *__restrict__ token_ids,
    const __nv_bfloat16 *__restrict__ embeddings) {
  const int token_idx = blockIdx.x;
  const int lane = threadIdx.x;
  const int token_id = loadg(token_ids + token_idx);

  if (token_id < 0 || token_id >= GEMMA4_VOCAB_SIZE) {
    return;
  }

  const __nv_bfloat16 *embedding_row = embeddings + token_id * GEMMA4_HIDDEN_SIZE;
  __nv_bfloat16 *out_row = out + token_idx * GEMMA4_HIDDEN_SIZE;

  for (int pack_idx = lane; pack_idx < kPacksPerEmbeddingRow;
       pack_idx += kEmbeddingGatherThreads) {
    const int offset = pack_idx * kBf16Packed128Elements;
    Bf16Packed128 pack = load128g(embedding_row + offset);
    store128(out_row + offset, pack);
  }
}

}  // namespace

cudaError_t gemma4_embedding_gather_bf16(
    __nv_bfloat16 *__restrict__ out,
    const int32_t *__restrict__ token_ids,
    const __nv_bfloat16 *__restrict__ embeddings,
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

  embedding_gather_kernel<<<num_tokens, kEmbeddingGatherThreads, 0, stream>>>(
      out, token_ids, embeddings);
  return cudaGetLastError();
}
