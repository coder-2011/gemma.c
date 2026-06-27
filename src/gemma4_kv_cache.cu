#include "gemma4_kv_cache.cuh"

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"

namespace {

constexpr int kKvWriteVecThreads = 32;
constexpr int kKvWritePackElements = kBf16Packed128Elements;

// Copies the K/V packs assigned to one token/head for the kernel wrapper.
__device__ inline void kv_cache_write_vec_device(
    __nv_bfloat16 *__restrict__ cache_k,
    __nv_bfloat16 *__restrict__ cache_v,
    Gemma4KvCacheConfig config,
    const int32_t *__restrict__ token_batch,
    const int32_t *__restrict__ token_position,
    const int32_t *__restrict__ page_table,
    int32_t token_count,
    int32_t layer,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    int32_t token,
    int32_t head,
    int32_t vec,
    int32_t vec_stride) {
  const int batch = token_batch[token];
  const int position = token_position[token];
  if (batch < 0 || position < 0) return;
  if (batch >= config.batch_size) return;
  const int logical_page = position / config.page_size;
  if (config.window_size == 0 && logical_page >= config.max_pages_per_seq) return;
  const int slot = logical_page % config.max_pages_per_seq;
  const int page_table_index = batch * config.max_pages_per_seq + slot;
  const int physical_page = __ldg(page_table + page_table_index);
  if (physical_page < 0 || physical_page >= config.num_pages) return;
  const int page_offset = position - logical_page * config.page_size;

  const int vecs_per_head = config.head_dim / kBf16Packed128Elements;
  const int64_t src_base =
      ((int64_t)token * config.num_heads + head) * vecs_per_head;
  const int64_t dst_base =
      ((((int64_t)layer * config.num_pages + physical_page) *
        config.page_size + page_offset) *
       config.num_heads + head) * vecs_per_head;

  for (int i = vec; i < vecs_per_head; i += vec_stride) {
    const int64_t src = (src_base + i) * kKvWritePackElements;
    const int64_t dst = (dst_base + i) * kKvWritePackElements;
    const Bf16Packed128 k_pack =
        Bf16Packed128{*reinterpret_cast<const int4 *>(k + src)};
    const Bf16Packed128 v_pack =
        Bf16Packed128{*reinterpret_cast<const int4 *>(v + src)};
    *reinterpret_cast<int4 *>(cache_k + dst) = k_pack.bits();
    *reinterpret_cast<int4 *>(cache_v + dst) = v_pack.bits();
  }
}

// Vectorized writer for Gemma K/V heads, copying one 128-bit pack per lane step.
__global__ __launch_bounds__(kKvWriteVecThreads) void kv_cache_write_vec_kernel(
    __nv_bfloat16 *__restrict__ cache_k,
    __nv_bfloat16 *__restrict__ cache_v,
    Gemma4KvCacheConfig config,
    const int32_t *__restrict__ token_batch,
    const int32_t *__restrict__ token_position,
    const int32_t *__restrict__ page_table,
    int32_t token_count,
    int32_t layer,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v) {
  kv_cache_write_vec_device(cache_k, cache_v, config, token_batch, token_position,
                            page_table, token_count, layer, k, v,
                            blockIdx.x, blockIdx.y, threadIdx.x, blockDim.x);
}

}  // namespace

// Build the cache geometry for either sliding or global Gemma 4 layers.
Gemma4KvCacheConfig gemma4_kv_cache_make_config(bool global,
                                                int32_t num_pages,
                                                int32_t page_size,
                                                int32_t max_pages_per_seq) {
  const Gemma4KvCacheConfig config = {
      global ? GEMMA4_GLOBAL_LAYER_COUNT : GEMMA4_SLIDING_LAYER_COUNT,
      num_pages,
      page_size,
      max_pages_per_seq,
      num_pages / max_pages_per_seq,
      global ? GEMMA4_GLOBAL_KV_HEADS : GEMMA4_SLIDING_KV_HEADS,
      global ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_SLIDING_HEAD_DIM,
      global ? 0 : GEMMA4_SLIDING_WINDOW,
  };
  return config;
}

// Map a model-layer index into that cache's compact sliding/global layer index.
int32_t gemma4_kv_cache_layer_index(int32_t model_layer, bool global_cache) {
  if (model_layer < 0 || model_layer >= GEMMA4_NUM_LAYERS) return -1;
  if (gemma4_is_global_layer(model_layer) != global_cache) return -1;

  int32_t cache_layer = 0;
  for (int32_t layer = 0; layer < model_layer; ++layer) {
    if (gemma4_is_global_layer(layer) == global_cache) {
      ++cache_layer;
    }
  }
  return cache_layer;
}

// Record one batch/position's direct page-table slot and reject stale backwards use.
int32_t gemma4_kv_cache_ensure_page(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    const Gemma4KvCacheConfig &config,
    int32_t batch,
    int32_t position) {
  const int32_t logical_page = position / config.page_size;
  const int32_t slot = logical_page % config.max_pages_per_seq;
  const int64_t index = int64_t(batch) * config.max_pages_per_seq + slot;
  if (slot_logical_pages[index] > logical_page) return -1;

  const int32_t physical_page = batch * config.max_pages_per_seq + slot;
  page_table[index] = physical_page;
  slot_logical_pages[index] = logical_page;
  return physical_page;
}

// Ensure a contiguous token range has mapped cache pages.
int32_t gemma4_kv_cache_ensure_range(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    const Gemma4KvCacheConfig &config,
    int32_t batch,
    int32_t first_position,
    int32_t token_count) {
  if (token_count == 0) return 0;

  const int32_t first_page = first_position / config.page_size;
  const int32_t last_position = first_position + token_count - 1;
  const int32_t last_page = last_position / config.page_size;
  for (int32_t page = first_page; page <= last_page; ++page) {
    const int32_t position = page * config.page_size;
    const int32_t ensured_page = gemma4_kv_cache_ensure_page(
        page_table, slot_logical_pages, config, batch, position);
    if (ensured_page < 0) return -1;
  }
  return 0;
}

// Write already-prepared BF16 K/V rows into the paged cache.
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
    cudaStream_t stream) {
  if (token_count == 0) return cudaSuccess;
  if (token_count < 0 || layer < 0 || layer >= config.num_layers ||
      config.num_pages <= 0 ||
      config.page_size <= 0 || config.max_pages_per_seq <= 0 ||
      config.batch_size <= 0 ||
      config.window_size < 0 || config.num_heads <= 0 || config.head_dim <= 0 ||
      config.head_dim % kBf16Packed128Elements != 0 ||
      d_cache_k == nullptr || d_cache_v == nullptr ||
      d_page_table == nullptr ||
      d_token_batch == nullptr || d_token_position == nullptr ||
      d_k == nullptr || d_v == nullptr) {
    return cudaErrorInvalidValue;
  }

  const dim3 grid_dim(token_count, config.num_heads);
  kv_cache_write_vec_kernel<<<grid_dim, kKvWriteVecThreads, 0, stream>>>(
      d_cache_k, d_cache_v, config, d_token_batch, d_token_position, d_page_table,
      token_count, layer, d_k, d_v);
  return cudaGetLastError();
}
