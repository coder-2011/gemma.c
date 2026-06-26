#include "gemma4_runtime.cuh"

#include <algorithm>
#include <cmath>

namespace {

// Returns ceil(value / divisor) for positive runtime geometry.
constexpr int32_t ceil_div_i32(int32_t value, int32_t divisor) {
  return (value + divisor - 1) / divisor;
}

// Counts BF16 elements in one Layout-A KV cache allocation.
constexpr size_t kv_cache_elements(const Gemma4KvCacheConfig &config) {
  return static_cast<size_t>(config.num_layers) * config.num_pages *
         config.page_size * config.num_heads * config.head_dim;
}

// Counts sliding page slots needed by any live window, including boundary straddles.
constexpr int32_t sliding_pages_per_sequence(int32_t max_seq_len,
                                             int32_t page_size) {
  if (max_seq_len <= GEMMA4_SLIDING_WINDOW) {
    return ceil_div_i32(max_seq_len, page_size);
  }
  return ceil_div_i32(GEMMA4_SLIDING_WINDOW + page_size - 1, page_size);
}

// Allocates a typed CUDA buffer and leaves the pointer null on failure.
template <typename T>
cudaError_t cuda_alloc_elements(T **ptr, size_t elements) {
  *ptr = nullptr;
  if (elements == 0) {
    return cudaSuccess;
  }
  cudaError_t status = cudaMalloc(ptr, elements * sizeof(T));
  if (status != cudaSuccess) {
    *ptr = nullptr;
  }
  return status;
}

// Copies a host vector prefix into its device mirror on the caller stream.
template <typename T>
cudaError_t copy_vector_prefix(
    T *dst,
    const std::vector<T> &src,
    size_t count,
    cudaStream_t stream) {
  if (count == 0) {
    return cudaSuccess;
  }
  return cudaMemcpyAsync(
      dst, src.data(), count * sizeof(T), cudaMemcpyHostToDevice, stream);
}

// Fills compact row-major cos/sin tables expected by the fused RoPE helpers.
void fill_rope_tables(
    std::vector<float> &cos_table,
    std::vector<float> &sin_table,
    int32_t max_seq_len,
    int32_t rotary_half,
    float theta,
    int32_t exponent_dim) {
  cos_table.resize(static_cast<size_t>(max_seq_len) * rotary_half);
  sin_table.resize(cos_table.size());
  for (int32_t pos = 0; pos < max_seq_len; ++pos) {
    for (int32_t i = 0; i < rotary_half; ++i) {
      const float exponent = -2.0f * static_cast<float>(i) /
                             static_cast<float>(exponent_dim);
      const float angle = static_cast<float>(pos) * std::pow(theta, exponent);
      const size_t offset = static_cast<size_t>(pos) * rotary_half + i;
      cos_table[offset] = std::cos(angle);
      sin_table[offset] = std::sin(angle);
    }
  }
}

// Uploads page tables, sequence lengths, and the live token row metadata.
cudaError_t upload_runtime_metadata(
    Gemma4RuntimeState *state,
    size_t token_count,
    cudaStream_t stream) {
  cudaError_t status = copy_vector_prefix(
      state->sliding_page_table, state->h_sliding_page_table,
      state->h_sliding_page_table.size(), stream);
  if (status != cudaSuccess) {
    return status;
  }
  status = copy_vector_prefix(
      state->global_page_table, state->h_global_page_table,
      state->h_global_page_table.size(), stream);
  if (status != cudaSuccess) {
    return status;
  }
  status = copy_vector_prefix(
      state->seq_lengths, state->h_seq_lengths,
      state->h_seq_lengths.size(), stream);
  if (status != cudaSuccess) {
    return status;
  }
  status = copy_vector_prefix(
      state->token_batch, state->h_token_batch, token_count, stream);
  if (status != cudaSuccess) {
    return status;
  }
  return copy_vector_prefix(
      state->token_position, state->h_token_position, token_count, stream);
}

// Ensures both cache classes have pages for one batch/position pair.
cudaError_t ensure_runtime_pages(
    Gemma4RuntimeState *state,
    int32_t batch,
    int32_t position) {
  int32_t page = gemma4_kv_cache_ensure_page(
      state->h_sliding_page_table, state->h_sliding_slot_logical_pages,
      state->sliding_cache_config, state->batch_size, batch, position);
  if (page < 0) {
    return cudaErrorInvalidValue;
  }
  page = gemma4_kv_cache_ensure_page(
      state->h_global_page_table, state->h_global_slot_logical_pages,
      state->global_cache_config, state->batch_size, batch, position);
  return page < 0 ? cudaErrorInvalidValue : cudaSuccess;
}

// Ensures both cache classes have pages for a contiguous prompt range.
cudaError_t ensure_runtime_range(
    Gemma4RuntimeState *state,
    int32_t batch,
    int32_t seq_len) {
  int32_t status = gemma4_kv_cache_ensure_range(
      state->h_sliding_page_table, state->h_sliding_slot_logical_pages,
      state->sliding_cache_config, state->batch_size, batch, 0, seq_len);
  if (status < 0) {
    return cudaErrorInvalidValue;
  }
  status = gemma4_kv_cache_ensure_range(
      state->h_global_page_table, state->h_global_slot_logical_pages,
      state->global_cache_config, state->batch_size, batch, 0, seq_len);
  return status < 0 ? cudaErrorInvalidValue : cudaSuccess;
}

}  // namespace

// Releases every CUDA allocation owned by the runtime state.
void gemma4_runtime_state_free(Gemma4RuntimeState *state) {
  if (state == nullptr) {
    return;
  }
  cudaFree(state->sliding_cache_k);
  cudaFree(state->sliding_cache_v);
  cudaFree(state->global_cache_k);
  cudaFree(state->global_cache_v);
  cudaFree(state->sliding_page_table);
  cudaFree(state->global_page_table);
  cudaFree(state->seq_lengths);
  cudaFree(state->token_batch);
  cudaFree(state->token_position);
  cudaFree(state->sliding_cos);
  cudaFree(state->sliding_sin);
  cudaFree(state->global_cos);
  cudaFree(state->global_sin);
  *state = Gemma4RuntimeState();
}

// Allocates device-owned KV caches, page tables, token positions, and RoPE tables.
cudaError_t gemma4_runtime_state_init(
    Gemma4RuntimeState *state,
    int32_t batch_size,
    int32_t max_seq_len,
    int32_t page_size,
    cudaStream_t stream) {
  if (state == nullptr || batch_size <= 0 || max_seq_len <= 0 ||
      page_size <= 0 || max_seq_len > GEMMA4_MAX_POSITION_EMBEDDINGS) {
    return cudaErrorInvalidValue;
  }

  const int32_t global_pages_per_seq = ceil_div_i32(max_seq_len, page_size);
  const int32_t sliding_pages_per_seq =
      sliding_pages_per_sequence(max_seq_len, page_size);
  const int32_t global_pages = batch_size * global_pages_per_seq;
  const int32_t sliding_pages = batch_size * sliding_pages_per_seq;
  const size_t token_rows = static_cast<size_t>(batch_size) * max_seq_len;

  *state = Gemma4RuntimeState();
  state->batch_size = batch_size;
  state->max_seq_len = max_seq_len;
  state->page_size = page_size;
  state->sliding_cache_config = gemma4_kv_cache_make_config(
      false, sliding_pages, page_size, sliding_pages_per_seq);
  state->sliding_cache_config.window_size =
      std::min(max_seq_len, GEMMA4_SLIDING_WINDOW);
  state->global_cache_config = gemma4_kv_cache_make_config(
      true, global_pages, page_size, global_pages_per_seq);

  state->h_sliding_page_table.assign(
      static_cast<size_t>(batch_size) * sliding_pages_per_seq, -1);
  state->h_global_page_table.assign(
      static_cast<size_t>(batch_size) * global_pages_per_seq, -1);
  state->h_sliding_slot_logical_pages.assign(
      state->h_sliding_page_table.size(), -1);
  state->h_global_slot_logical_pages.assign(
      state->h_global_page_table.size(), -1);
  state->h_seq_lengths.assign(batch_size, 0);
  state->h_token_batch.resize(token_rows);
  state->h_token_position.resize(token_rows);

  cudaError_t status = cuda_alloc_elements(
      &state->sliding_cache_k, kv_cache_elements(state->sliding_cache_config));
  if (status != cudaSuccess) goto fail;
  status = cuda_alloc_elements(
      &state->sliding_cache_v, kv_cache_elements(state->sliding_cache_config));
  if (status != cudaSuccess) goto fail;
  status = cuda_alloc_elements(
      &state->global_cache_k, kv_cache_elements(state->global_cache_config));
  if (status != cudaSuccess) goto fail;
  status = cuda_alloc_elements(
      &state->global_cache_v, kv_cache_elements(state->global_cache_config));
  if (status != cudaSuccess) goto fail;
  status = cuda_alloc_elements(
      &state->sliding_page_table, state->h_sliding_page_table.size());
  if (status != cudaSuccess) goto fail;
  status = cuda_alloc_elements(
      &state->global_page_table, state->h_global_page_table.size());
  if (status != cudaSuccess) goto fail;
  status = cuda_alloc_elements(&state->seq_lengths, state->h_seq_lengths.size());
  if (status != cudaSuccess) goto fail;
  status = cuda_alloc_elements(&state->token_batch, token_rows);
  if (status != cudaSuccess) goto fail;
  status = cuda_alloc_elements(&state->token_position, token_rows);
  if (status != cudaSuccess) goto fail;

  {
    constexpr int32_t kSlidingRotaryHalf = GEMMA4_SLIDING_HEAD_DIM / 2;
    constexpr int32_t kGlobalRotaryDim =
        static_cast<int32_t>(GEMMA4_GLOBAL_HEAD_DIM *
                             GEMMA4_PARTIAL_ROTARY_FACTOR_GLOBAL);
    constexpr int32_t kGlobalRotaryHalf = kGlobalRotaryDim / 2;
    std::vector<float> cos_table;
    std::vector<float> sin_table;

    fill_rope_tables(
        cos_table, sin_table, max_seq_len, kSlidingRotaryHalf,
        GEMMA4_ROPE_THETA_SLIDING, GEMMA4_SLIDING_HEAD_DIM);
    status = cuda_alloc_elements(&state->sliding_cos, cos_table.size());
    if (status != cudaSuccess) goto fail;
    status = cuda_alloc_elements(&state->sliding_sin, sin_table.size());
    if (status != cudaSuccess) goto fail;
    status = copy_vector_prefix(
        state->sliding_cos, cos_table, cos_table.size(), stream);
    if (status != cudaSuccess) goto fail;
    status = copy_vector_prefix(
        state->sliding_sin, sin_table, sin_table.size(), stream);
    if (status != cudaSuccess) goto fail;

    fill_rope_tables(
        cos_table, sin_table, max_seq_len, kGlobalRotaryHalf,
        GEMMA4_ROPE_THETA_GLOBAL, GEMMA4_GLOBAL_HEAD_DIM);
    status = cuda_alloc_elements(&state->global_cos, cos_table.size());
    if (status != cudaSuccess) goto fail;
    status = cuda_alloc_elements(&state->global_sin, sin_table.size());
    if (status != cudaSuccess) goto fail;
    status = copy_vector_prefix(
        state->global_cos, cos_table, cos_table.size(), stream);
    if (status != cudaSuccess) goto fail;
    status = copy_vector_prefix(
        state->global_sin, sin_table, sin_table.size(), stream);
    if (status != cudaSuccess) goto fail;
  }

  status = upload_runtime_metadata(state, 0, stream);
  if (status != cudaSuccess) goto fail;
  return cudaSuccess;

fail:
  gemma4_runtime_state_free(state);
  return status;
}

// Prepares page tables and absolute token positions for an initial prompt.
cudaError_t gemma4_runtime_prepare_prefill(
    Gemma4RuntimeState *state,
    int32_t seq_len,
    cudaStream_t stream) {
  if (state == nullptr || seq_len < 0 || seq_len > state->max_seq_len) {
    return cudaErrorInvalidValue;
  }

  std::fill(state->h_sliding_page_table.begin(),
            state->h_sliding_page_table.end(), -1);
  std::fill(state->h_global_page_table.begin(),
            state->h_global_page_table.end(), -1);
  std::fill(state->h_sliding_slot_logical_pages.begin(),
            state->h_sliding_slot_logical_pages.end(), -1);
  std::fill(state->h_global_slot_logical_pages.begin(),
            state->h_global_slot_logical_pages.end(), -1);

  for (int32_t batch = 0; batch < state->batch_size; ++batch) {
    cudaError_t status = ensure_runtime_range(state, batch, seq_len);
    if (status != cudaSuccess) {
      return status;
    }
    state->h_seq_lengths[batch] = seq_len;
    for (int32_t pos = 0; pos < seq_len; ++pos) {
      const size_t row = static_cast<size_t>(batch) * seq_len + pos;
      state->h_token_batch[row] = batch;
      state->h_token_position[row] = pos;
    }
  }

  state->token_count = state->batch_size * seq_len;
  return upload_runtime_metadata(state, state->token_count, stream);
}

// Appends one decode position per batch and uploads the new runtime metadata.
cudaError_t gemma4_runtime_prepare_decode_step(
    Gemma4RuntimeState *state,
    cudaStream_t stream) {
  if (state == nullptr) {
    return cudaErrorInvalidValue;
  }

  // Validate all batches before mutating page tables for this decode step.
  for (int32_t batch = 0; batch < state->batch_size; ++batch) {
    const int32_t pos = state->h_seq_lengths[batch];
    if (pos < 0 || pos >= state->max_seq_len) {
      return cudaErrorInvalidValue;
    }
  }

  for (int32_t batch = 0; batch < state->batch_size; ++batch) {
    const int32_t pos = state->h_seq_lengths[batch];
    cudaError_t status = ensure_runtime_pages(state, batch, pos);
    if (status != cudaSuccess) {
      return status;
    }
    state->h_token_batch[batch] = batch;
    state->h_token_position[batch] = pos;
  }

  for (int32_t batch = 0; batch < state->batch_size; ++batch) {
    ++state->h_seq_lengths[batch];
  }
  state->token_count = state->batch_size;
  return upload_runtime_metadata(state, state->token_count, stream);
}
