#include "gemma4_runtime.cuh"

#include "gemma4_cuda_utils.cuh"
#include "gemma4_rope.cuh"

#include <algorithm>

namespace {

// Counts BF16 elements in one Layout-A KV cache allocation.
constexpr size_t kv_cache_elements(const Gemma4KvCacheConfig &config) {
  return static_cast<size_t>(config.num_layers) * config.num_pages *
         config.page_size * config.num_heads * config.head_dim;
}

// Counts sliding page slots needed by any live window, including boundary straddles.
constexpr int32_t sliding_pages_per_sequence(int32_t max_seq_len,
                                             int32_t page_size) {
  if (max_seq_len <= GEMMA4_SLIDING_WINDOW) {
    return div_up(max_seq_len, page_size);
  }
  return div_up(GEMMA4_SLIDING_WINDOW + page_size - 1, page_size);
}

// Uploads page tables, sequence lengths, and the live token row metadata.
cudaError_t upload_runtime_metadata(
    Gemma4RuntimeState *state,
    size_t token_count,
    cudaStream_t stream) {
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      state->sliding_page_table, state->h_sliding_page_table.data(),
      state->h_sliding_page_table.size() * sizeof(*state->sliding_page_table),
      cudaMemcpyHostToDevice, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      state->global_page_table, state->h_global_page_table.data(),
      state->h_global_page_table.size() * sizeof(*state->global_page_table),
      cudaMemcpyHostToDevice, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      state->seq_lengths, state->h_seq_lengths.data(),
      state->h_seq_lengths.size() * sizeof(*state->seq_lengths),
      cudaMemcpyHostToDevice, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      state->token_batch, state->h_token_batch.data(),
      token_count * sizeof(*state->token_batch), cudaMemcpyHostToDevice,
      stream));
  return cudaMemcpyAsync(
      state->token_position, state->h_token_position.data(),
      token_count * sizeof(*state->token_position), cudaMemcpyHostToDevice,
      stream);
}

}  // namespace

// Releases every CUDA allocation owned by the runtime state.
void gemma4_runtime_state_free(Gemma4RuntimeState *state) {
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
  if (batch_size <= 0 || max_seq_len <= 0 || page_size <= 0 ||
      max_seq_len > GEMMA4_MAX_POSITION_EMBEDDINGS) {
    return cudaErrorInvalidValue;
  }

  const int32_t global_pages_per_seq = div_up(max_seq_len, page_size);
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
  state->sliding_cache_config.batch_size = batch_size;
  state->sliding_cache_config.window_size =
      std::min(max_seq_len, GEMMA4_SLIDING_WINDOW);
  state->global_cache_config = gemma4_kv_cache_make_config(
      true, global_pages, page_size, global_pages_per_seq);
  state->global_cache_config.batch_size = batch_size;

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

  cudaError_t status = cudaMalloc(
      &state->sliding_cache_k,
      kv_cache_elements(state->sliding_cache_config) *
          sizeof(*state->sliding_cache_k));
  if (status != cudaSuccess) goto fail;
  status = cudaMalloc(
      &state->sliding_cache_v,
      kv_cache_elements(state->sliding_cache_config) *
          sizeof(*state->sliding_cache_v));
  if (status != cudaSuccess) goto fail;
  status = cudaMalloc(
      &state->global_cache_k,
      kv_cache_elements(state->global_cache_config) *
          sizeof(*state->global_cache_k));
  if (status != cudaSuccess) goto fail;
  status = cudaMalloc(
      &state->global_cache_v,
      kv_cache_elements(state->global_cache_config) *
          sizeof(*state->global_cache_v));
  if (status != cudaSuccess) goto fail;
  status = cudaMalloc(
      &state->sliding_page_table,
      state->h_sliding_page_table.size() * sizeof(*state->sliding_page_table));
  if (status != cudaSuccess) goto fail;
  status = cudaMalloc(
      &state->global_page_table,
      state->h_global_page_table.size() * sizeof(*state->global_page_table));
  if (status != cudaSuccess) goto fail;
  status = cudaMalloc(
      &state->seq_lengths,
      state->h_seq_lengths.size() * sizeof(*state->seq_lengths));
  if (status != cudaSuccess) goto fail;
  status = cudaMalloc(
      &state->token_batch, token_rows * sizeof(*state->token_batch));
  if (status != cudaSuccess) goto fail;
  status = cudaMalloc(
      &state->token_position, token_rows * sizeof(*state->token_position));
  if (status != cudaSuccess) goto fail;

  {
    constexpr int32_t kSlidingRotaryHalf = GEMMA4_SLIDING_HEAD_DIM / 2;
    constexpr int32_t kGlobalRotaryDim =
        static_cast<int32_t>(GEMMA4_GLOBAL_HEAD_DIM *
                             GEMMA4_PARTIAL_ROTARY_FACTOR_GLOBAL);
    constexpr int32_t kGlobalRotaryHalf = kGlobalRotaryDim / 2;
    std::vector<float> cos_table;
    std::vector<float> sin_table;

    gemma4_rope::fill_tables(
        cos_table, sin_table, max_seq_len, kSlidingRotaryHalf,
        GEMMA4_ROPE_THETA_SLIDING, GEMMA4_SLIDING_HEAD_DIM);
    status = cudaMalloc(
        &state->sliding_cos, cos_table.size() * sizeof(*state->sliding_cos));
    if (status != cudaSuccess) goto fail;
    status = cudaMalloc(
        &state->sliding_sin, sin_table.size() * sizeof(*state->sliding_sin));
    if (status != cudaSuccess) goto fail;
    status = cudaMemcpyAsync(
        state->sliding_cos, cos_table.data(),
        cos_table.size() * sizeof(*state->sliding_cos),
        cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) goto fail;
    status = cudaMemcpyAsync(
        state->sliding_sin, sin_table.data(),
        sin_table.size() * sizeof(*state->sliding_sin),
        cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) goto fail;

    gemma4_rope::fill_tables(
        cos_table, sin_table, max_seq_len, kGlobalRotaryHalf,
        GEMMA4_ROPE_THETA_GLOBAL, GEMMA4_GLOBAL_HEAD_DIM);
    status = cudaMalloc(
        &state->global_cos, cos_table.size() * sizeof(*state->global_cos));
    if (status != cudaSuccess) goto fail;
    status = cudaMalloc(
        &state->global_sin, sin_table.size() * sizeof(*state->global_sin));
    if (status != cudaSuccess) goto fail;
    status = cudaMemcpyAsync(
        state->global_cos, cos_table.data(),
        cos_table.size() * sizeof(*state->global_cos),
        cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) goto fail;
    status = cudaMemcpyAsync(
        state->global_sin, sin_table.data(),
        sin_table.size() * sizeof(*state->global_sin),
        cudaMemcpyHostToDevice, stream);
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
  if (seq_len < 0 || seq_len > state->max_seq_len) {
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
    int32_t status = gemma4_kv_cache_ensure_range(
        state->h_sliding_page_table, state->h_sliding_slot_logical_pages,
        state->sliding_cache_config, batch, 0, seq_len);
    if (status < 0) return cudaErrorInvalidValue;
    status = gemma4_kv_cache_ensure_range(
        state->h_global_page_table, state->h_global_slot_logical_pages,
        state->global_cache_config, batch, 0, seq_len);
    if (status < 0) return cudaErrorInvalidValue;
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

// Appends one decode position per batch and uploads the changed runtime metadata.
cudaError_t gemma4_runtime_prepare_decode_step(
    Gemma4RuntimeState *state,
    cudaStream_t stream) {
  // Validate all batches before mutating page tables for this decode step.
  for (int32_t batch = 0; batch < state->batch_size; ++batch) {
    const int32_t pos = state->h_seq_lengths[batch];
    if (pos < 0 || pos >= state->max_seq_len) {
      return cudaErrorInvalidValue;
    }
  }

  for (int32_t batch = 0; batch < state->batch_size; ++batch) {
    const int32_t pos = state->h_seq_lengths[batch];
    const int32_t sliding_slot =
        (pos / state->sliding_cache_config.page_size) %
        state->sliding_cache_config.max_pages_per_seq;
    const int32_t global_slot =
        (pos / state->global_cache_config.page_size) %
        state->global_cache_config.max_pages_per_seq;
    const size_t sliding_index =
        static_cast<size_t>(batch) *
            state->sliding_cache_config.max_pages_per_seq +
        sliding_slot;
    const size_t global_index =
        static_cast<size_t>(batch) *
            state->global_cache_config.max_pages_per_seq +
        global_slot;
    if (gemma4_kv_cache_ensure_page(
            state->h_sliding_page_table, state->h_sliding_slot_logical_pages,
            state->sliding_cache_config, batch, pos) < 0) {
      return cudaErrorInvalidValue;
    }
    if (gemma4_kv_cache_ensure_page(
            state->h_global_page_table, state->h_global_slot_logical_pages,
            state->global_cache_config, batch, pos) < 0) {
      return cudaErrorInvalidValue;
    }
    state->h_token_batch[batch] = batch;
    state->h_token_position[batch] = pos;
    GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        state->sliding_page_table + sliding_index,
        state->h_sliding_page_table.data() + sliding_index,
        sizeof(*state->sliding_page_table), cudaMemcpyHostToDevice, stream));
    GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        state->global_page_table + global_index,
        state->h_global_page_table.data() + global_index,
        sizeof(*state->global_page_table), cudaMemcpyHostToDevice, stream));
  }

  for (int32_t batch = 0; batch < state->batch_size; ++batch) {
    ++state->h_seq_lengths[batch];
  }
  state->token_count = state->batch_size;
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      state->seq_lengths, state->h_seq_lengths.data(),
      state->h_seq_lengths.size() * sizeof(*state->seq_lengths),
      cudaMemcpyHostToDevice, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      state->token_batch, state->h_token_batch.data(),
      state->token_count * sizeof(*state->token_batch),
      cudaMemcpyHostToDevice, stream));
  return cudaMemcpyAsync(
      state->token_position, state->h_token_position.data(),
      state->token_count * sizeof(*state->token_position),
      cudaMemcpyHostToDevice, stream);
}
