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
  // Addition functor for CUB reductions; using an explicit type keeps the sum
  // path symmetric with cuda::maximum for the max-reduction path below.
  __device__ float operator()(float a, float b) const { return a + b; }
};

// Validate the structural cache contract before any host launcher depends on
// the layout. The page table must be large enough to cover the active sliding
// window, and decode kernels in this file only specialize up to head_dim 512.
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

// Reduce one float across the whole CTA and broadcast the result back to every
// thread. The shared result slot avoids relying on CUB's thread-0 return value
// being visible to other lanes without an explicit block synchronization.
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

// Translate a logical token position into the physical page assigned to this
// batch. Sliding caches wrap logical pages onto a fixed slot ring; global caches
// use the same helper but callers prevent logical wraparound.
__device__ inline int32_t physical_page_for_position(
    const Gemma4KvCacheConfig &config,
    const int32_t *__restrict__ page_table,
    int32_t batch,
    int32_t position) {
  int32_t slot = gemma4_kv_cache_page_slot(config, position);
  return __ldg(page_table + batch * config.max_pages_per_seq + slot);
}

// Scalar KV-cache writer. One CTA owns one (token, KV head) row, and its threads
// stride across head_dim. This is the fallback for non-128-bit-aligned head
// widths; Gemma's 256/512 head dims normally use the vectorized writer below.
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
  // Grid shape: x selects the token being written, y selects the KV head, and
  // threadIdx.x selects the first dimension handled by this thread.
  int token = blockIdx.x;
  int head = blockIdx.y;
  int dim = threadIdx.x;
  if (token >= token_count || head >= config.num_heads) return;

  // The host supplies compact token metadata so prefill and decode can write
  // arbitrary token positions with the same kernel.
  int batch = token_batch[token];
  int position = token_position[token];
  int physical_page = physical_page_for_position(config, page_table, batch, position);
  if (physical_page < 0) return;

  // Source K/V rows are contiguous by token then head. Destination rows are
  // placed in Layout-A cache order: layer, physical page, page offset, head, dim.
  int page_offset = gemma4_kv_cache_page_offset(config, position);
  int64_t src = (int64_t(token) * config.num_heads + head) * config.head_dim;
  int64_t dst = gemma4_kv_cache_offset(config, layer, physical_page, page_offset, head, 0);

  // Stride by blockDim.x so the same kernel handles every supported head width.
  for (int d = dim; d < config.head_dim; d += blockDim.x) {
    cache_k[dst + d] = k[src + d];
    cache_v[dst + d] = v[src + d];
  }
}

// Vectorized KV-cache writer for Gemma's aligned head dims. Each thread moves
// one 128-bit int4 pack at a time, which is eight BF16 values, keeping global
// loads/stores coalesced and avoiding scalar per-element traffic.
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
  // Grid shape matches the scalar writer, but threadIdx.x indexes 128-bit packs
  // instead of individual BF16 dimensions.
  int token = blockIdx.x;
  int head = blockIdx.y;
  int vec = threadIdx.x;
  if (token >= token_count || head >= config.num_heads) return;

  // Resolve the destination physical page for this logical token position.
  int batch = token_batch[token];
  int position = token_position[token];
  int physical_page = physical_page_for_position(config, page_table, batch, position);
  if (physical_page < 0) return;

  // Convert element offsets into int4 pack offsets. This path is only launched
  // when head_dim is divisible by kBf16Packed128Elements.
  int page_offset = gemma4_kv_cache_page_offset(config, position);
  int vecs_per_head = config.head_dim / kBf16Packed128Elements;
  int64_t src_vec = (int64_t(token) * config.num_heads + head) * vecs_per_head;
  int64_t dst_vec = gemma4_kv_cache_offset(config, layer, physical_page, page_offset, head, 0) /
                    kBf16Packed128Elements;
  // Reinterpret as int4 only after the pack offsets are known; the public API
  // remains BF16 because callers reason in model element layout.
  const int4 *src_k = reinterpret_cast<const int4 *>(k);
  const int4 *src_v = reinterpret_cast<const int4 *>(v);
  int4 *dst_k = reinterpret_cast<int4 *>(cache_k);
  int4 *dst_v = reinterpret_cast<int4 *>(cache_v);

  // Read source projections through the read-only path and write cache rows with
  // cache-global stores, which fits the streaming nature of KV-cache updates.
  for (int i = vec; i < vecs_per_head; i += blockDim.x) {
    cub::ThreadStore<cub::STORE_CG>(dst_k + dst_vec + i,
                                    cub::ThreadLoad<cub::LOAD_LDG>(src_k + src_vec + i));
    cub::ThreadStore<cub::STORE_CG>(dst_v + dst_vec + i,
                                    cub::ThreadLoad<cub::LOAD_LDG>(src_v + src_vec + i));
  }
}

// First stage of generic paged decode attention. Each CTA handles one
// (batch, Q head, split) triple and computes numerically stable softmax partials
// over a contiguous slice of the key range.
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
  // Grid coordinates identify the output row and which segment of the attended
  // key range this CTA owns. threadIdx.x maps directly to the head dimension.
  int batch = blockIdx.x;
  int q_head = blockIdx.y;
  int split = blockIdx.z;
  int dim = threadIdx.x;

  // Gemma uses grouped-query attention. Several Q heads share one KV head; the
  // integer division maps this Q head to its backing KV cache row.
  int group_size = q_heads / config.num_heads;
  int kv_head = q_head / group_size;
  int seq_len = __ldg(seq_lengths + batch);
  // Sliding layers attend only to the latest window; global layers set
  // window_size=0 and therefore attend from token 0.
  int first_key = config.window_size > 0 ? max(0, seq_len - config.window_size) : 0;
  int key_count = seq_len - first_key;
  int split_begin = first_key + split * split_size;
  int split_end = min(first_key + key_count, split_begin + split_size);
  int partial = (batch * q_heads + q_head) * num_splits + split;

  // Empty splits can occur near the end of the key range. Write neutral softmax
  // state so the reduce stage can merge every split uniformly.
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

  // Per-thread state: each thread owns one output dimension's accumulator, while
  // m/l are block-wide online-softmax scalars computed identically by all lanes.
  float acc = 0.0f;
  float m = -INFINITY;
  float l = 0.0f;
  int64_t q_base = (int64_t(batch) * q_heads + q_head) * config.head_dim;
  int64_t partial_acc_base = (int64_t)partial * config.head_dim;
  float q_value = dim < config.head_dim ? __bfloat162float(loadg(q + q_base + dim)) : 0.0f;

  // Walk the split token by token. Each iteration loads K/V from the paged cache,
  // computes q dot k with a block reduction, and updates the online softmax.
  for (int pos = split_begin; pos < split_end; ++pos) {
    int physical_page = physical_page_for_position(config, page_table, batch, pos);
    if (physical_page < 0) continue;
    int page_offset = gemma4_kv_cache_page_offset(config, pos);
    int64_t kv_base = gemma4_kv_cache_offset(config, layer, physical_page, page_offset, kv_head, 0);
    float k_value = dim < config.head_dim ? __bfloat162float(loadg(cache_k + kv_base + dim)) : 0.0f;
    // All threads contribute one dimension to the dot product. Threads outside
    // head_dim contribute zero, which lets 512-thread launches handle dim 256.
    float score =
        block_reduce<BlockThreads>(q_value * k_value, SumOp{}) * softmax_scale;
    // Online softmax rescaling keeps the running accumulator stable as the
    // maximum score changes across the split.
    float new_m = fmaxf(m, score);
    float old_scale = __expf(m - new_m);
    float new_scale = __expf(score - new_m);

    // Only real head dimensions hold V data. Padding threads still participate
    // in reductions but do not touch partial_acc.
    if (dim < config.head_dim) {
      float v_value = __bfloat162float(loadg(cache_v + kv_base + dim));
      acc = acc * old_scale + v_value * new_scale;
    }
    l = l * old_scale + new_scale;
    m = new_m;
  }

  // Store the split-local softmax state and this dimension's weighted V sum for
  // the second-stage merge.
  if (dim == 0) {
    partial_m[partial] = m;
    partial_l[partial] = l;
  }
  if (dim < config.head_dim) {
    partial_acc[partial_acc_base + dim] = acc;
  }
}

// Second stage of generic paged decode attention. One CTA owns one output
// (batch, Q head) row and merges all split partials into the final BF16 vector.
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
  // Flatten batch and Q head into the row index used by the partial buffers.
  int batch = blockIdx.x;
  int q_head = blockIdx.y;
  int dim = threadIdx.x;
  int row = batch * q_heads + q_head;
  int partial_row = row * num_splits;
  int64_t partial_acc_row = int64_t(row) * num_splits * head_dim;

  // First merge softmax maxima. Every split accumulator must be rescaled into
  // this common max space before denominators or V sums are combined.
  float local_m = -INFINITY;
  for (int split = dim; split < num_splits; split += blockDim.x) {
    local_m = fmaxf(local_m, partial_m[partial_row + split]);
  }
  float m = block_reduce<BlockThreads>(local_m, cuda::maximum<>{});
  __shared__ float s_m;
  __shared__ float s_l;
  if (dim == 0) {
    s_m = m;
  }
  __syncthreads();

  // Merge denominators in the shared max space. Empty splits have l=0 and do not
  // affect the final normalization.
  float local_l = 0.0f;
  for (int split = dim; split < num_splits; split += blockDim.x) {
    float split_m = partial_m[partial_row + split];
    float split_l = partial_l[partial_row + split];
    if (split_l > 0.0f) {
      local_l += split_l * __expf(split_m - s_m);
    }
  }
  float l = block_reduce<BlockThreads>(local_l, SumOp{});
  if (dim == 0) {
    s_l = l;
  }
  __syncthreads();

  // Merge the per-dimension weighted V accumulators, normalize by the merged
  // denominator, and write the final attention output row.
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

// Launch the two-stage generic paged decode attention path for a fixed CTA size.
// The template parameter is selected by the public wrapper from head_dim.
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
  // Split stage exposes split as grid.z so long contexts can be divided into
  // independent CTAs before the reduction kernel merges them.
  dim3 split_grid(batch_size, q_heads, num_splits);
  paged_decode_split_kernel<BlockThreads>
      <<<split_grid, BlockThreads, 0, stream>>>(
          d_partial_m, d_partial_l, d_partial_acc, d_q, d_cache_k, d_cache_v,
          d_page_table, d_seq_lengths, config, layer, q_heads, softmax_scale,
          split_size, num_splits);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  // Reduce stage collapses all splits for each output row.
  dim3 reduce_grid(batch_size, q_heads);
  paged_decode_reduce_kernel<BlockThreads>
      <<<reduce_grid, BlockThreads, 0, stream>>>(
          d_out, d_partial_m, d_partial_l, d_partial_acc, batch_size, q_heads,
          config.head_dim, num_splits);
  return cudaGetLastError();
}

}  // namespace

// Allocate one physical page. Freed pages are reused first so long-running decode
// sessions can recycle released sequences before consuming never-used pages.
int32_t Gemma4KvPageAllocator::allocate() {
  if (!free_pages.empty()) {
    int32_t page = free_pages.back();
    free_pages.pop_back();
    return page;
  }
  if (next_page >= page_count) return -1;
  return next_page++;
}

// Return a physical page to the allocator's freelist. Invalid page ids are
// ignored so callers can release conditionally without extra guards.
void Gemma4KvPageAllocator::release(int32_t page) {
  if (page >= 0 && page < page_count) {
    free_pages.push_back(page);
  }
}

// Reset allocator state to the initial empty-cache condition.
void Gemma4KvPageAllocator::reset() {
  next_page = 0;
  free_pages.clear();
}

// Build a cache config for either the compact global-layer cache or the
// sliding-layer cache. The values come from the Gemma 4 31B architecture
// constants in gemma4.h.
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

// Map a model layer index into the corresponding compact KV-cache layer index.
// Sliding and global layers live in separate caches, so this counts only layers
// of the requested type before model_layer.
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

// Ensure the page-table entry for one (batch, position) exists and return its
// physical page. Sliding caches wrap logical pages onto slots; global caches
// reject positions that exceed max_pages_per_seq.
int32_t gemma4_kv_cache_ensure_page(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    Gemma4KvPageAllocator &allocator,
    const Gemma4KvCacheConfig &config,
    int32_t batch_size,
    int32_t batch,
    int32_t position) {
  // Reject impossible host-side metadata before touching the vectors.
  if (!valid_config(config) || batch < 0 || batch >= batch_size || position < 0) {
    return -1;
  }

  // Global cache has no wraparound, so the logical page must fit in the table.
  // Sliding cache wraps the logical page onto a ring slot below.
  int32_t logical_page = gemma4_kv_cache_logical_page(config, position);
  if (config.window_size == 0 && logical_page >= config.max_pages_per_seq) {
    return -1;
  }
  int32_t slot = logical_page % config.max_pages_per_seq;
  int32_t index = batch * config.max_pages_per_seq + slot;
  // Guard against malformed host vectors or batch/table-size mismatches.
  if (index < 0 || index >= static_cast<int32_t>(page_table.size()) ||
      index >= static_cast<int32_t>(slot_logical_pages.size())) {
    return -1;
  }

  // Allocate lazily: a page table slot receives a physical page only when the
  // first token that maps to it is appended or prefilled.
  if (page_table[index] < 0) {
    page_table[index] = allocator.allocate();
  }
  if (page_table[index] < 0) return -1;

  // ponytail: sliding cache reuses the slot page; add physical-page recycling
  // only if the runtime wants to shrink cache memory below one full window.
  slot_logical_pages[index] = logical_page;
  return page_table[index];
}

// Ensure every page touched by a contiguous token range exists. This is a simple
// host-side convenience used by prefill-style writes.
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

// Append one token position for a batch sequence. The returned value is the
// position assigned to the new token, and seq_lengths is advanced only after its
// page has been ensured.
int32_t gemma4_kv_cache_append_position(
    std::vector<int32_t> &page_table,
    std::vector<int32_t> &slot_logical_pages,
    std::vector<int32_t> &seq_lengths,
    Gemma4KvPageAllocator &allocator,
    const Gemma4KvCacheConfig &config,
    int32_t batch_size,
    int32_t batch) {
  // seq_lengths is indexed by batch; reject mismatched batch metadata.
  if (batch < 0 || batch >= batch_size || batch >= static_cast<int32_t>(seq_lengths.size())) {
    return -1;
  }

  // The current sequence length is the next decode position for this batch.
  int32_t position = seq_lengths[batch];
  int32_t page = gemma4_kv_cache_ensure_page(
      page_table, slot_logical_pages, allocator, config, batch_size, batch, position);
  if (page < 0) return -1;

  seq_lengths[batch] = position + 1;
  return position;
}

// Number of scalar max/denominator partials needed for split decode.
size_t gemma4_paged_decode_partial_m_elements(int32_t batch_size,
                                              int32_t q_heads,
                                              int32_t num_splits) {
  return size_t(batch_size) * q_heads * num_splits;
}

// Number of per-dimension accumulator partials needed for split decode.
size_t gemma4_paged_decode_partial_acc_elements(int32_t batch_size,
                                                int32_t q_heads,
                                                int32_t num_splits,
                                                int32_t head_dim) {
  return gemma4_paged_decode_partial_m_elements(batch_size, q_heads, num_splits) * size_t(head_dim);
}

// Public GPU entry point for writing contiguous K/V rows into the paged cache.
// The caller must have already populated d_page_table for every token position.
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
  // Empty writes are valid no-ops; everything else needs a usable config and
  // all required device pointers.
  if (token_count == 0) return cudaSuccess;
  if (!valid_config(config) || layer < 0 || layer >= config.num_layers ||
      d_cache_k == nullptr || d_cache_v == nullptr || d_page_table == nullptr ||
      d_token_batch == nullptr || d_token_position == nullptr ||
      d_k == nullptr || d_v == nullptr) {
    return cudaErrorInvalidValue;
  }

  // One CTA per token/head row. Prefer 128-bit vector traffic when the head dim
  // supports it, otherwise fall back to scalar striding.
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

// Public generic paged decode attention entry point. This is kept as a compact
// reference/coverage path; specialized FlashAttention decode paths can replace
// it for production sliding/global layers when available.
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
  // Validate both layout constraints and scratch/output pointers before any
  // kernel launch. q_heads must be an integer multiple of KV heads for GQA.
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

  // Inline dispatch: dim 256 uses 256 threads, dim 512 uses 512 threads. Padding
  // lanes safely contribute zero in the split kernel if head_dim is smaller.
  return config.head_dim <= 256
      ? launch_paged_decode_attention<256>(
            d_out, d_partial_m, d_partial_l, d_partial_acc, d_q, d_cache_k,
            d_cache_v, d_page_table, d_seq_lengths, config, layer, batch_size,
            q_heads, softmax_scale, split_size, num_splits, stream)
      : launch_paged_decode_attention<512>(
            d_out, d_partial_m, d_partial_l, d_partial_acc, d_q, d_cache_k,
            d_cache_v, d_page_table, d_seq_lengths, config, layer, batch_size,
            q_heads, softmax_scale, split_size, num_splits, stream);
}
