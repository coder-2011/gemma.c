#include "gemma4_kv_cache.cuh"

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cute/tensor.hpp>

#include <algorithm>
#include <cmath>

namespace {
namespace cg = cooperative_groups;

constexpr int kKvWriteVecThreads = WARP_SIZE;
constexpr int kKvWritePackElements = kBf16Packed128Elements;
using KvWriteCopyAtom = cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, __nv_bfloat16>;

// Copies the K/V packs assigned to one token/head for the kernel wrapper.
__device__ inline void kv_cache_write_vec_device(
    __nv_bfloat16 *__restrict__ cache_k,
    __nv_bfloat16 *__restrict__ cache_v,
    Gemma4KvCacheConfig config,
    const int32_t *__restrict__ page_table,
    const int32_t *__restrict__ token_batch,
    const int32_t *__restrict__ token_position,
    int32_t token_count,
    int32_t layer,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v,
    int32_t token,
    int32_t head,
    int32_t vec,
    int32_t vec_stride) {
  if (token >= token_count || head >= config.num_heads) return;

  int batch = token_batch[token];
  int position = token_position[token];
  int logical_page = position / config.page_size;
  int slot = logical_page % config.max_pages_per_seq;
  int page_offset = position - logical_page * config.page_size;
  auto page_table_layout = cute::make_layout(
      cute::make_shape(token_count, config.max_pages_per_seq),
      cute::make_stride(config.max_pages_per_seq, 1));
  auto block = cg::tiled_partition<kKvWriteVecThreads>(cg::this_thread_block());
  int physical_page = cg::invoke_one_broadcast(block, [&] {
    return __ldcg(page_table + page_table_layout(batch, slot));
  });
  if (physical_page < 0 || physical_page >= config.num_pages) return;

  int vecs_per_head = config.head_dim / kBf16Packed128Elements;
  auto token_vec_layout = cute::make_layout(
      cute::make_shape(token_count, config.num_heads, vecs_per_head),
      cute::make_stride(int64_t(config.num_heads) * vecs_per_head,
                        vecs_per_head, 1));
  auto cache_vec_layout = cute::make_layout(
      cute::make_shape(config.num_layers, config.num_pages, config.page_size,
                       config.num_heads, vecs_per_head),
      cute::make_stride(int64_t(config.num_pages) * config.page_size * config.num_heads * vecs_per_head,
                        int64_t(config.page_size) * config.num_heads * vecs_per_head,
                        int64_t(config.num_heads) * vecs_per_head, vecs_per_head, 1));

  KvWriteCopyAtom copy_atom;
  for (int i = vec; i < vecs_per_head; i += vec_stride) {
    int64_t src = token_vec_layout(token, head, i) * kKvWritePackElements;
    int64_t dst = cache_vec_layout(layer, physical_page, page_offset, head, i) * kKvWritePackElements;
    auto src_k = cute::make_tensor(cute::make_gmem_ptr(k + src), cute::make_shape(cute::Int<kKvWritePackElements>{}));
    auto src_v = cute::make_tensor(cute::make_gmem_ptr(v + src), cute::make_shape(cute::Int<kKvWritePackElements>{}));
    auto dst_k = cute::make_tensor(cute::make_gmem_ptr(cache_k + dst), cute::make_shape(cute::Int<kKvWritePackElements>{}));
    auto dst_v = cute::make_tensor(cute::make_gmem_ptr(cache_v + dst), cute::make_shape(cute::Int<kKvWritePackElements>{}));
    cute::copy(copy_atom, src_k, dst_k);
    cute::copy(copy_atom, src_v, dst_v);
  }
}

// Vectorized writer for Gemma K/V heads, copying one 128-bit pack per lane step.
__global__ __launch_bounds__(kKvWriteVecThreads) void kv_cache_write_vec_kernel(
    __nv_bfloat16 *__restrict__ cache_k,
    __nv_bfloat16 *__restrict__ cache_v,
    Gemma4KvCacheConfig config,
    const int32_t *__restrict__ page_table,
    const int32_t *__restrict__ token_batch,
    const int32_t *__restrict__ token_position,
    int32_t token_count,
    int32_t layer,
    const __nv_bfloat16 *__restrict__ k,
    const __nv_bfloat16 *__restrict__ v) {
  kv_cache_write_vec_device(cache_k, cache_v, config, page_table, token_batch,
                            token_position, token_count, layer, k, v,
                            blockIdx.x, blockIdx.y, threadIdx.x, blockDim.x);
}

// Compute per-split online-softmax state over the paged K/V cache.
template <int BlockThreads>
__global__ __launch_bounds__(BlockThreads) void paged_decode_split_kernel(
    float *__restrict__ partial_m,
    float *__restrict__ partial_l,
    float *__restrict__ partial_acc,
    const __nv_bfloat16 *__restrict__ q,
    const __nv_bfloat16 *__restrict__ cache_k,
    const __nv_bfloat16 *__restrict__ cache_v,
    const int32_t *__restrict__ page_table,
    const int32_t *__restrict__ seq_lengths,
    Gemma4KvCacheConfig config,
    int32_t layer,
    int32_t q_heads,
    float softmax_scale,
    int32_t split_size,
    int32_t num_splits) {
  int batch = blockIdx.x;
  int q_head = blockIdx.y;
  int split = blockIdx.z;
  int dim = threadIdx.x;

  int group_size = q_heads / config.num_heads;
  int kv_head = q_head / group_size;
  int seq_len = __ldg(seq_lengths + batch);
  int first_key = config.window_size > 0 ? max(0, seq_len - config.window_size) : 0;
  int key_count = seq_len - first_key;
  int split_begin = first_key + split * split_size;
  int split_end = min(first_key + key_count, split_begin + split_size);
  auto page_table_layout = cute::make_layout(
      cute::make_shape(gridDim.x, config.max_pages_per_seq),
      cute::make_stride(config.max_pages_per_seq, 1));
  auto q_layout = cute::make_layout(
      cute::make_shape(gridDim.x, q_heads, config.head_dim),
      cute::make_stride(int64_t(q_heads) * config.head_dim, config.head_dim, 1));
  auto partial_layout = cute::make_layout(
      cute::make_shape(gridDim.x, q_heads, num_splits),
      cute::make_stride(q_heads * num_splits, num_splits, 1));
  auto partial_acc_layout = cute::make_layout(
      cute::make_shape(gridDim.x, q_heads, num_splits, config.head_dim),
      cute::make_stride(int64_t(q_heads) * num_splits * config.head_dim,
                        num_splits * config.head_dim, config.head_dim, 1));
  int64_t partial = partial_layout(batch, q_head, split);

  if (split_begin >= split_end) {
    if (dim == 0) {
      partial_m[partial] = -INFINITY;
      partial_l[partial] = 0.0f;
    }
    if (dim < config.head_dim) {
      partial_acc[partial_acc_layout(batch, q_head, split, dim)] = 0.0f;
    }
    return;
  }

  float acc = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;
  float q_value =
      dim < config.head_dim
          ? __bfloat162float(loadg(q + q_layout(batch, q_head, dim)))
          : 0.0f;

  auto block = cg::tiled_partition<BlockThreads>(cg::this_thread_block());
  auto cache_layout = gemma4_kv_cache_layout(config);
  int64_t kv_token_stride = cache_layout.stride<2>();
  for (int page_begin = split_begin; page_begin < split_end;) {
    int32_t logical_page = page_begin / config.page_size;
    int32_t page_offset = page_begin - logical_page * config.page_size;
    int32_t page_token_count =
        min(split_end - page_begin, config.page_size - page_offset);
    int32_t physical_page = cg::invoke_one_broadcast(block, [&] {
      int32_t slot = logical_page % config.max_pages_per_seq;
      return __ldg(page_table + page_table_layout(batch, slot));
    });

    if (physical_page >= 0 && physical_page < config.num_pages) {
      int64_t kv_page_base =
          cache_layout(layer, physical_page, page_offset, kv_head, 0);
      for (int page_token = 0; page_token < page_token_count; ++page_token) {
        int64_t kv_base = kv_page_base + int64_t(page_token) * kv_token_stride;
        float k_value =
            dim < config.head_dim
                ? __bfloat162float(loadg(cache_k + kv_base + dim))
                : 0.0f;
        float score =
            cg::reduce(block, q_value * k_value, cg::plus<float>{}) *
            softmax_scale;
        float new_m = fmaxf(m, score);
        float old_scale = __expf(m - new_m);
        float new_scale = __expf(score - new_m);

        if (dim < config.head_dim) {
          float v_value = __bfloat162float(loadg(cache_v + kv_base + dim));
          acc = acc * old_scale + v_value * new_scale;
        }
        l = l * old_scale + new_scale;
        m = new_m;
      }
    }

    page_begin += page_token_count;
  }

  if (dim == 0) {
    partial_m[partial] = m;
    partial_l[partial] = l;
  }
  if (dim < config.head_dim) {
    partial_acc[partial_acc_layout(batch, q_head, split, dim)] = acc;
  }
}

// Combine per-split softmax state into one BF16 attention output row.
template <int BlockThreads>
__global__ __launch_bounds__(BlockThreads) void paged_decode_reduce_kernel(
    __nv_bfloat16 *__restrict__ out,
    const float *__restrict__ partial_m,
    const float *__restrict__ partial_l,
    const float *__restrict__ partial_acc,
    int32_t batch_size,
    int32_t q_heads,
    int32_t head_dim,
    int32_t num_splits) {
  int batch = blockIdx.x;
  int q_head = blockIdx.y;
  int dim = threadIdx.x;
  auto partial_layout = cute::make_layout(
      cute::make_shape(batch_size, q_heads, num_splits),
      cute::make_stride(q_heads * num_splits, num_splits, 1));
  auto partial_acc_layout = cute::make_layout(
      cute::make_shape(batch_size, q_heads, num_splits, head_dim),
      cute::make_stride(int64_t(q_heads) * num_splits * head_dim,
                        num_splits * head_dim, head_dim, 1));
  auto out_layout = cute::make_layout(
      cute::make_shape(batch_size, q_heads, head_dim),
      cute::make_stride(int64_t(q_heads) * head_dim, head_dim, 1));
  auto block = cg::tiled_partition<BlockThreads>(cg::this_thread_block());

  float local_m = -INFINITY;
  for (int split = dim; split < num_splits; split += blockDim.x) {
    local_m = fmaxf(local_m, partial_m[partial_layout(batch, q_head, split)]);
  }
  float m = cg::reduce(block, local_m, cg::greater<float>{});
  __shared__ float s_m;
  __shared__ float s_l;
  if (dim == 0) {
    s_m = m;
  }
  __syncthreads();

  float local_l = 0.0f;
  for (int split = dim; split < num_splits; split += blockDim.x) {
    int64_t partial = partial_layout(batch, q_head, split);
    float split_m = partial_m[partial];
    float split_l = partial_l[partial];
    if (split_l > 0.0f) {
      local_l += split_l * __expf(split_m - s_m);
    }
  }
  float l = cg::reduce(block, local_l, cg::plus<float>{});
  if (dim == 0) {
    s_l = l;
  }
  __syncthreads();

  for (int d = dim; d < head_dim; d += blockDim.x) {
    float value = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
      int64_t partial = partial_layout(batch, q_head, split);
      float split_l = partial_l[partial];
      if (split_l > 0.0f) {
        float split_m = partial_m[partial];
        value += partial_acc[partial_acc_layout(batch, q_head, split, d)] *
                 __expf(split_m - s_m);
      }
    }
    out[out_layout(batch, q_head, d)] =
        s_l > 0.0f ? __float2bfloat16_rn(value / s_l) : __float2bfloat16_rn(0.0f);
  }
}

// Launch the generic split/reduce paged decode attention implementation.
template <int BlockThreads>
cudaError_t launch_paged_decode_attention(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_partial_m,
    float *__restrict__ d_partial_l,
    float *__restrict__ d_partial_acc,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_cache_k,
    const __nv_bfloat16 *__restrict__ d_cache_v,
    const int32_t *__restrict__ d_page_table,
    const int32_t *__restrict__ d_seq_lengths,
    Gemma4KvCacheConfig config,
    int32_t layer,
    int32_t batch_size,
    int32_t q_heads,
    float softmax_scale,
    int32_t split_size,
    int32_t num_splits,
    cudaStream_t stream) {
  dim3 split_grid(batch_size, q_heads, num_splits);
  paged_decode_split_kernel<BlockThreads>
      <<<split_grid, BlockThreads, 0, stream>>>(
          d_partial_m, d_partial_l, d_partial_acc, d_q, d_cache_k, d_cache_v,
          d_page_table, d_seq_lengths, config, layer, q_heads, softmax_scale,
          split_size, num_splits);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  dim3 reduce_grid(batch_size, q_heads);
  paged_decode_reduce_kernel<BlockThreads>
      <<<reduce_grid, BlockThreads, 0, stream>>>(
          d_out, d_partial_m, d_partial_l, d_partial_acc, batch_size, q_heads,
          config.head_dim, num_splits);
  return cudaGetLastError();
}

}  // namespace

// Allocate one physical cache page from the free-list or monotonic tail.
int32_t Gemma4KvPageAllocator::allocate() {
  if (!free_pages.empty()) {
    int32_t page = free_pages.back();
    free_pages.pop_back();
    return page;
  }
  if (next_page >= page_count) return -1;
  return next_page++;
}

// Return one physical page to the free-list.
void Gemma4KvPageAllocator::release(int32_t page) {
  if (page >= 0 && page < page_count) {
    free_pages.push_back(page);
  }
}

// Reset allocator state without clearing page contents.
void Gemma4KvPageAllocator::reset() {
  next_page = 0;
  free_pages.clear();
}

// Build the cache geometry for either sliding or global Gemma 4 layers.
Gemma4KvCacheConfig gemma4_kv_cache_make_config(bool global,
                                                int32_t num_pages,
                                                int32_t page_size,
                                                int32_t max_pages_per_seq) {
  Gemma4KvCacheConfig config = {
      global ? GEMMA4_GLOBAL_LAYER_COUNT : GEMMA4_SLIDING_LAYER_COUNT,
      num_pages,
      page_size,
      max_pages_per_seq,
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

// Ensure one batch/position has a page-table slot and reject stale backwards use.
int32_t gemma4_kv_cache_ensure_page(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    Gemma4KvPageAllocator &allocator,
    const Gemma4KvCacheConfig &config,
    int32_t batch_size,
    int32_t batch,
    int32_t position) {
  if (batch < 0 || batch >= batch_size || position < 0) {
    return -1;
  }

  int32_t logical_page = position / config.page_size;
  if (config.window_size == 0 && logical_page >= config.max_pages_per_seq) {
    return -1;
  }
  int32_t slot = logical_page % config.max_pages_per_seq;
  auto page_table_layout = cute::make_layout(
      cute::make_shape(batch_size, config.max_pages_per_seq),
      cute::make_stride(config.max_pages_per_seq, 1));
  int64_t index = page_table_layout(batch, slot);
  if (index < 0 || index >= static_cast<int64_t>(page_table.size()) ||
      index >= static_cast<int64_t>(slot_logical_pages.size())) {
    return -1;
  }

  if (slot_logical_pages[index] > logical_page) return -1;

  if (page_table[index] < 0) {
    page_table[index] = allocator.allocate();
  }
  if (page_table[index] < 0) return -1;

  slot_logical_pages[index] = logical_page;
  return page_table[index];
}

// Ensure a contiguous token range has mapped cache pages.
int32_t gemma4_kv_cache_ensure_range(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    Gemma4KvPageAllocator &allocator,
    const Gemma4KvCacheConfig &config,
    int32_t batch_size,
    int32_t batch,
    int32_t first_position,
    int32_t token_count) {
  for (int32_t i = 0; i < token_count; ++i) {
    int32_t page = gemma4_kv_cache_ensure_page(
        page_table, slot_logical_pages, allocator, config, batch_size, batch,
        first_position + i);
    if (page < 0) return -1;
  }
  return 0;
}

// Append one token position for a batch and advance its sequence length.
int32_t gemma4_kv_cache_append_position(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    std::vector<int32_t> &seq_lengths,
    Gemma4KvPageAllocator &allocator,
    const Gemma4KvCacheConfig &config,
    int32_t batch_size,
    int32_t batch) {
  if (batch < 0 || batch >= batch_size || batch >= static_cast<int32_t>(seq_lengths.size())) {
    return -1;
  }

  int32_t position = seq_lengths[batch];
  int32_t page = gemma4_kv_cache_ensure_page(
      page_table, slot_logical_pages, allocator, config, batch_size, batch, position);
  if (page < 0) return -1;

  seq_lengths[batch] = position + 1;
  return position;
}

// Return the element count for per-split softmax max/denominator scratch.
size_t gemma4_paged_decode_partial_m_elements(int32_t batch_size,
                                              int32_t q_heads,
                                              int32_t num_splits) {
  return size_t(batch_size) * q_heads * num_splits;
}

// Return the element count for per-split attention accumulator scratch.
size_t gemma4_paged_decode_partial_acc_elements(int32_t batch_size,
                                                int32_t q_heads,
                                                int32_t num_splits,
                                                int32_t head_dim) {
  return gemma4_paged_decode_partial_m_elements(batch_size, q_heads,
                                                num_splits) *
         size_t(head_dim);
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
      config.num_layers <= 0 || config.num_pages <= 0 ||
      config.page_size <= 0 || config.max_pages_per_seq <= 0 ||
      config.num_heads <= 0 || config.head_dim <= 0 ||
      config.head_dim % kBf16Packed128Elements != 0 ||
      d_cache_k == nullptr || d_cache_v == nullptr || d_page_table == nullptr ||
      d_token_batch == nullptr || d_token_position == nullptr ||
      d_k == nullptr || d_v == nullptr) {
    return cudaErrorInvalidValue;
  }

  dim3 grid_dim(token_count, config.num_heads);
  kv_cache_write_vec_kernel<<<grid_dim, kKvWriteVecThreads, 0, stream>>>(
      d_cache_k, d_cache_v, config, d_page_table, d_token_batch,
      d_token_position, token_count, layer, d_k, d_v);
  return cudaGetLastError();
}

// Run the generic paged decode attention reference path.
cudaError_t gemma4_paged_decode_attention_bf16(
    __nv_bfloat16 *__restrict__ d_out,
    float *__restrict__ d_partial_m,
    float *__restrict__ d_partial_l,
    float *__restrict__ d_partial_acc,
    const __nv_bfloat16 *__restrict__ d_q,
    const __nv_bfloat16 *__restrict__ d_cache_k,
    const __nv_bfloat16 *__restrict__ d_cache_v,
    const int32_t *__restrict__ d_page_table,
    const int32_t *__restrict__ d_seq_lengths,
    Gemma4KvCacheConfig config,
    int32_t layer,
    int32_t batch_size,
    int32_t q_heads,
    float softmax_scale,
    int32_t split_size,
    int32_t num_splits,
    cudaStream_t stream) {
  if (layer < 0 || layer >= config.num_layers ||
      batch_size <= 0 || q_heads < config.num_heads ||
      q_heads % config.num_heads != 0 ||
      split_size <= 0 || num_splits <= 0 ||
      (config.window_size > 0 &&
       int64_t(split_size) * num_splits < config.window_size) ||
      d_out == nullptr ||
      d_partial_m == nullptr || d_partial_l == nullptr ||
      d_partial_acc == nullptr || d_q == nullptr || d_cache_k == nullptr ||
      d_cache_v == nullptr || d_page_table == nullptr ||
      d_seq_lengths == nullptr) {
    return cudaErrorInvalidValue;
  }

  return config.head_dim <= 256
             ? launch_paged_decode_attention<256>(
                   d_out, d_partial_m, d_partial_l, d_partial_acc, d_q,
                   d_cache_k, d_cache_v, d_page_table, d_seq_lengths, config,
                   layer, batch_size, q_heads, softmax_scale, split_size,
                   num_splits, stream)
             : launch_paged_decode_attention<512>(
                   d_out, d_partial_m, d_partial_l, d_partial_acc, d_q,
                   d_cache_k, d_cache_v, d_page_table, d_seq_lengths, config,
                   layer, batch_size, q_heads, softmax_scale, split_size,
                   num_splits, stream);
}
