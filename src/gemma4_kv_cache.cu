#include "gemma4_kv_cache.cuh"

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"

#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_store.cuh>
#include <cuda/functional>

#include <algorithm>
#include <cmath>

namespace {

constexpr int kKvWriteThreads = 256;
constexpr int kKvWriteVecThreads = 128;

struct SumOp {
  __device__ float operator()(float a, float b) const { return a + b; }
};

int attention_threads_for_head_dim(int32_t head_dim) {
  return head_dim <= 256 ? 256 : 512;
}

bool valid_config(const Gemma4KvCacheConfig &config) {
  if (config.num_layers <= 0 || config.num_pages <= 0 ||
      config.page_size <= 0 || config.max_pages_per_seq <= 0 ||
      config.num_heads <= 0 || config.head_dim <= 0 ||
      config.head_dim > 512) {
    return false;
  }
  int32_t min_window_pages =
      config.window_size > 0 ? div_up(config.window_size, config.page_size) + 1 : 1;
  return config.max_pages_per_seq >= min_window_pages;
}

template <int BlockThreads, typename Op>
__device__ inline float block_reduce(float value, Op op) {
  using BlockReduce = cub::BlockReduce<float, BlockThreads>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ float result;
  float thread0_result = BlockReduce(temp_storage).Reduce(value, op);
  if (threadIdx.x == 0) result = thread0_result;
  __syncthreads();
  return result;
}

template <int BlockThreads>
__device__ inline float block_sum(float value) {
  return block_reduce<BlockThreads>(value, SumOp{});
}

template <int BlockThreads>
__device__ inline float block_max(float value) {
  return block_reduce<BlockThreads>(value, cuda::maximum<>{});
}

__device__ inline int32_t physical_page_for_position(
    const Gemma4KvCacheConfig &config,
    const int32_t *__restrict__ page_table,
    int32_t batch,
    int32_t position) {
  int32_t slot = gemma4_kv_cache_page_slot(config, position);
  return __ldg(page_table + batch * config.max_pages_per_seq + slot);
}

__global__ __launch_bounds__(kKvWriteThreads) void kv_cache_write_kernel(
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
  int token = blockIdx.x;
  int head = blockIdx.y;
  int dim = threadIdx.x;
  if (token >= token_count || head >= config.num_heads) return;

  int batch = token_batch[token];
  int position = token_position[token];
  int physical_page = physical_page_for_position(config, page_table, batch, position);
  if (physical_page < 0) return;

  int page_offset = gemma4_kv_cache_page_offset(config, position);
  int64_t src = (int64_t(token) * config.num_heads + head) * config.head_dim;
  int64_t dst = gemma4_kv_cache_offset(config, layer, physical_page, page_offset, head, 0);

  for (int d = dim; d < config.head_dim; d += blockDim.x) {
    cache_k[dst + d] = k[src + d];
    cache_v[dst + d] = v[src + d];
  }
}

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
  int token = blockIdx.x;
  int head = blockIdx.y;
  int vec = threadIdx.x;
  if (token >= token_count || head >= config.num_heads) return;

  int batch = token_batch[token];
  int position = token_position[token];
  int physical_page = physical_page_for_position(config, page_table, batch, position);
  if (physical_page < 0) return;

  int page_offset = gemma4_kv_cache_page_offset(config, position);
  int vecs_per_head = config.head_dim / kBf16Packed128Elements;
  int64_t src_vec = (int64_t(token) * config.num_heads + head) * vecs_per_head;
  int64_t dst_vec = gemma4_kv_cache_offset(config, layer, physical_page, page_offset, head, 0) /
                    kBf16Packed128Elements;
  const int4 *src_k = reinterpret_cast<const int4 *>(k);
  const int4 *src_v = reinterpret_cast<const int4 *>(v);
  int4 *dst_k = reinterpret_cast<int4 *>(cache_k);
  int4 *dst_v = reinterpret_cast<int4 *>(cache_v);

  for (int i = vec; i < vecs_per_head; i += blockDim.x) {
    cub::ThreadStore<cub::STORE_CG>(dst_k + dst_vec + i,
                                    cub::ThreadLoad<cub::LOAD_LDG>(src_k + src_vec + i));
    cub::ThreadStore<cub::STORE_CG>(dst_v + dst_vec + i,
                                    cub::ThreadLoad<cub::LOAD_LDG>(src_v + src_vec + i));
  }
}

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
  int partial = (batch * q_heads + q_head) * num_splits + split;

  if (split_begin >= split_end) {
    if (dim == 0) {
      partial_m[partial] = -INFINITY;
      partial_l[partial] = 0.0f;
    }
    if (dim < config.head_dim) {
      partial_acc[(int64_t)partial * config.head_dim + dim] = 0.0f;
    }
    return;
  }

  float acc = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;
  int64_t q_base = (int64_t(batch) * q_heads + q_head) * config.head_dim;
  int64_t partial_acc_base = (int64_t)partial * config.head_dim;
  float q_value = dim < config.head_dim ? __bfloat162float(loadg(q + q_base + dim)) : 0.0f;

  for (int pos = split_begin; pos < split_end; ++pos) {
    int physical_page = physical_page_for_position(config, page_table, batch, pos);
    if (physical_page < 0) continue;
    int page_offset = gemma4_kv_cache_page_offset(config, pos);
    int64_t kv_base = gemma4_kv_cache_offset(config, layer, physical_page, page_offset, kv_head, 0);
    float k_value = dim < config.head_dim ? __bfloat162float(loadg(cache_k + kv_base + dim)) : 0.0f;
    float score = block_sum<BlockThreads>(q_value * k_value) * softmax_scale;
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

  if (dim == 0) {
    partial_m[partial] = m;
    partial_l[partial] = l;
  }
  if (dim < config.head_dim) {
    partial_acc[partial_acc_base + dim] = acc;
  }
}

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
  int row = batch * q_heads + q_head;
  int partial_row = row * num_splits;
  int64_t partial_acc_row = int64_t(row) * num_splits * head_dim;

  float local_m = -INFINITY;
  for (int split = dim; split < num_splits; split += blockDim.x) {
    local_m = fmaxf(local_m, partial_m[partial_row + split]);
  }
  float m = block_max<BlockThreads>(local_m);
  __shared__ float s_m;
  __shared__ float s_l;
  if (dim == 0) {
    s_m = m;
  }
  __syncthreads();

  float local_l = 0.0f;
  for (int split = dim; split < num_splits; split += blockDim.x) {
    float split_m = partial_m[partial_row + split];
    float split_l = partial_l[partial_row + split];
    if (split_l > 0.0f) {
      local_l += split_l * __expf(split_m - s_m);
    }
  }
  float l = block_sum<BlockThreads>(local_l);
  if (dim == 0) {
    s_l = l;
  }
  __syncthreads();

  for (int d = dim; d < head_dim; d += blockDim.x) {
    float value = 0.0f;
    for (int split = 0; split < num_splits; ++split) {
      float split_l = partial_l[partial_row + split];
      if (split_l > 0.0f) {
        float split_m = partial_m[partial_row + split];
        value += partial_acc[partial_acc_row + int64_t(split) * head_dim + d] * __expf(split_m - s_m);
      }
    }
    out[(int64_t)row * head_dim + d] =
        s_l > 0.0f ? __float2bfloat16_rn(value / s_l) : __float2bfloat16_rn(0.0f);
  }
}

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

int32_t Gemma4KvPageAllocator::allocate() {
  if (!free_pages.empty()) {
    int32_t page = free_pages.back();
    free_pages.pop_back();
    return page;
  }
  if (next_page >= page_count) return -1;
  return next_page++;
}

void Gemma4KvPageAllocator::release(int32_t page) {
  if (page >= 0 && page < page_count) {
    free_pages.push_back(page);
  }
}

void Gemma4KvPageAllocator::reset() {
  next_page = 0;
  free_pages.clear();
}

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

int32_t gemma4_kv_cache_ensure_page(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    Gemma4KvPageAllocator &allocator,
    const Gemma4KvCacheConfig &config,
    int32_t batch_size,
    int32_t batch,
    int32_t position) {
  if (!valid_config(config) || batch < 0 || batch >= batch_size || position < 0) {
    return -1;
  }

  int32_t logical_page = gemma4_kv_cache_logical_page(config, position);
  if (config.window_size == 0 && logical_page >= config.max_pages_per_seq) {
    return -1;
  }
  int32_t slot = logical_page % config.max_pages_per_seq;
  int32_t index = batch * config.max_pages_per_seq + slot;
  if (index < 0 || index >= static_cast<int32_t>(page_table.size()) ||
      index >= static_cast<int32_t>(slot_logical_pages.size())) {
    return -1;
  }

  if (page_table[index] < 0) {
    page_table[index] = allocator.allocate();
  }
  if (page_table[index] < 0) return -1;

  // ponytail: sliding cache reuses the slot page; add physical-page recycling
  // only if the runtime wants to shrink cache memory below one full window.
  slot_logical_pages[index] = logical_page;
  return page_table[index];
}

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
        page_table, slot_logical_pages, allocator, config, batch_size, batch, first_position + i);
    if (page < 0) return -1;
  }
  return 0;
}

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

size_t gemma4_paged_decode_partial_m_elements(int32_t batch_size,
                                              int32_t q_heads,
                                              int32_t num_splits) {
  return size_t(batch_size) * q_heads * num_splits;
}

size_t gemma4_paged_decode_partial_acc_elements(int32_t batch_size,
                                                int32_t q_heads,
                                                int32_t num_splits,
                                                int32_t head_dim) {
  return gemma4_paged_decode_partial_m_elements(batch_size, q_heads, num_splits) * size_t(head_dim);
}

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
  if (!valid_config(config) || layer < 0 || layer >= config.num_layers ||
      d_cache_k == nullptr || d_cache_v == nullptr || d_page_table == nullptr ||
      d_token_batch == nullptr || d_token_position == nullptr ||
      d_k == nullptr || d_v == nullptr) {
    return cudaErrorInvalidValue;
  }

  dim3 grid_dim(token_count, config.num_heads);
  if (config.head_dim % kBf16Packed128Elements == 0) {
    kv_cache_write_vec_kernel<<<grid_dim, kKvWriteVecThreads, 0, stream>>>(
        d_cache_k, d_cache_v, config, d_page_table, d_token_batch,
        d_token_position, token_count, layer, d_k, d_v);
  } else {
    kv_cache_write_kernel<<<grid_dim, kKvWriteThreads, 0, stream>>>(
        d_cache_k, d_cache_v, config, d_page_table, d_token_batch,
        d_token_position, token_count, layer, d_k, d_v);
  }
  return cudaGetLastError();
}

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
  if (!valid_config(config) || layer < 0 || layer >= config.num_layers ||
      batch_size <= 0 || q_heads < config.num_heads ||
      q_heads % config.num_heads != 0 || split_size <= 0 ||
      num_splits <= 0 || d_out == nullptr ||
      d_partial_m == nullptr || d_partial_l == nullptr ||
      d_partial_acc == nullptr || d_q == nullptr || d_cache_k == nullptr ||
      d_cache_v == nullptr || d_page_table == nullptr ||
      d_seq_lengths == nullptr) {
    return cudaErrorInvalidValue;
  }

  const int threads = attention_threads_for_head_dim(config.head_dim);
  return threads == 256
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
