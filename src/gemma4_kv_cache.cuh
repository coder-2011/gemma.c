#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/layout.hpp>

#include <stdint.h>
#include <vector>

struct Gemma4KvCacheConfig {
  int32_t num_layers;
  int32_t num_pages;
  int32_t page_size;
  int32_t max_pages_per_seq;
  int32_t num_heads;
  int32_t head_dim;
  int32_t window_size;
};

// Return the flat Layout-A cache mapping: [layer, page, page_offset, head, dim].
__host__ __device__ inline auto gemma4_kv_cache_layout(
    const Gemma4KvCacheConfig &config) {
  using namespace cute;
  int64_t dim_stride = 1;
  int64_t head_stride = config.head_dim;
  int64_t page_offset_stride = int64_t(config.num_heads) * head_stride;
  int64_t page_stride = int64_t(config.page_size) * page_offset_stride;
  int64_t layer_stride = int64_t(config.num_pages) * page_stride;
  return make_layout(
      make_shape(config.num_layers, config.num_pages, config.page_size,
                 config.num_heads, config.head_dim),
      make_stride(layer_stride, page_stride, page_offset_stride, head_stride,
                  dim_stride));
}

Gemma4KvCacheConfig gemma4_kv_cache_make_config(bool global,
                                                int32_t num_pages,
                                                int32_t page_size,
                                                int32_t max_pages_per_seq);

int32_t gemma4_kv_cache_layer_index(int32_t model_layer, bool global_cache);

int32_t gemma4_kv_cache_ensure_page(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    const Gemma4KvCacheConfig &config,
    int32_t batch_size,
    int32_t batch,
    int32_t position);

int32_t gemma4_kv_cache_ensure_range(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    const Gemma4KvCacheConfig &config,
    int32_t batch_size,
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
