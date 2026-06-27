#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <vector>

struct Gemma4KvCacheConfig {
  int32_t num_layers;
  int32_t num_pages;
  int32_t page_size;
  int32_t max_pages_per_seq;
  int32_t batch_size;
  int32_t num_heads;
  int32_t head_dim;
  int32_t window_size;
};

Gemma4KvCacheConfig gemma4_kv_cache_make_config(bool global,
                                                int32_t num_pages,
                                                int32_t page_size,
                                                int32_t max_pages_per_seq);

int32_t gemma4_kv_cache_layer_index(int32_t model_layer, bool global_cache);

// Returns the flat Layout-A cache offset: [layer, page, page_offset, head, dim].
__host__ __device__ constexpr inline int64_t gemma4_kv_cache_offset(
    const Gemma4KvCacheConfig &config,
    int32_t layer,
    int32_t page,
    int32_t page_offset,
    int32_t head,
    int32_t dim) {
  const int64_t layer_page = int64_t(layer) * config.num_pages + page;
  const int64_t token = layer_page * config.page_size + page_offset;
  return (token * config.num_heads + head) * config.head_dim + dim;
}

int32_t gemma4_kv_cache_ensure_page(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    const Gemma4KvCacheConfig &config,
    int32_t batch,
    int32_t position);

int32_t gemma4_kv_cache_ensure_range(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    const Gemma4KvCacheConfig &config,
    int32_t batch,
    int32_t first_position,
    int32_t token_count);

extern "C" cudaError_t gemma4_kv_cache_write_bf16(
    __nv_bfloat16 *__restrict__ d_cache_k,
    __nv_bfloat16 *__restrict__ d_cache_v,
    Gemma4KvCacheConfig config,
    const int32_t *__restrict__ d_page_table,
    const int32_t *__restrict__ d_token_batch,
    const int32_t *__restrict__ d_token_position,
    int32_t token_count,
    int32_t layer,
    const __nv_bfloat16 *__restrict__ d_k,
    const __nv_bfloat16 *__restrict__ d_v,
    cudaStream_t stream);
