#include "gemma4_embedding_gather.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4.h"

#include <stdint.h>

namespace gemma4_embedding_gather {

constexpr int kEmbeddingGatherThreads = 32;
constexpr int kPacksPerEmbeddingRow =
    GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;

// Copies one tied embedding row with coalesced 128-bit BF16 pack traffic.
__device__ void copy_embedding_row_bf16(
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
    Bf16Packed128 pack = Bf16Packed128{*reinterpret_cast<const int4 *>(embedding_row + offset)};
    pack = gemma4_bf16_pack_apply_scale(pack, GEMMA4_EMBEDDING_SCALE);
    *reinterpret_cast<int4 *>(out_row + offset) = pack.bits();
  }
}

}  // namespace gemma4_embedding_gather

namespace {

// Launches one CTA per token so lanes cooperatively copy one embedding row.
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

  const dim3 grid_dim(num_tokens);
  const dim3 block_dim(gemma4_embedding_gather::kEmbeddingGatherThreads);
  embedding_gather_kernel<<<grid_dim, block_dim, 0, stream>>>(
      out, token_ids, embeddings);
  return cudaGetLastError();
}
