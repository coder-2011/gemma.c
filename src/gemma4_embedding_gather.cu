#include "gemma4_embedding_gather.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

#include <stdint.h>

namespace {

__global__ __launch_bounds__(gemma4_embedding_gather::kEmbeddingGatherThreads)
void embedding_gather_kernel(
    __nv_bfloat16 *__restrict__ out,
    const int32_t *__restrict__ token_ids,
    const __nv_bfloat16 *__restrict__ embeddings) {
  const int token_idx = blockIdx.x;
  const int lane = threadIdx.x;
  const int token_id = loadg(token_ids + token_idx);

  if (token_id < 0 || token_id >= GEMMA4_VOCAB_SIZE) {
    return;
  }

  __nv_bfloat16 *out_row = out + token_idx * GEMMA4_HIDDEN_SIZE;
  gemma4_embedding_gather::copy_embedding_row_bf16(
      out_row, embeddings, token_id, lane, blockDim.x);
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

  embedding_gather_kernel<<<
      num_tokens, gemma4_embedding_gather::kEmbeddingGatherThreads, 0,
      stream>>>(out, token_ids, embeddings);
  return cudaGetLastError();
}
