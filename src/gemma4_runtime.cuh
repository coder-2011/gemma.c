#pragma once

#include "gemma4.h"
#include "gemma4_kv_cache.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <vector>

struct Gemma4RuntimeState {
  int32_t batch_size = 0;
  int32_t max_seq_len = 0;
  int32_t page_size = 0;
  int32_t token_count = 0;

  Gemma4KvCacheConfig sliding_cache_config = {};
  Gemma4KvCacheConfig global_cache_config = {};
  __nv_bfloat16 *sliding_cache_k = nullptr;
  __nv_bfloat16 *sliding_cache_v = nullptr;
  __nv_bfloat16 *global_cache_k = nullptr;
  __nv_bfloat16 *global_cache_v = nullptr;

  int32_t *sliding_page_table = nullptr;
  int32_t *global_page_table = nullptr;
  int32_t *seq_lengths = nullptr;
  int32_t *token_batch = nullptr;
  int32_t *token_position = nullptr;
  float *sliding_cos = nullptr;
  float *sliding_sin = nullptr;
  float *global_cos = nullptr;
  float *global_sin = nullptr;

  std::vector<int32_t> h_sliding_page_table;
  std::vector<int32_t> h_global_page_table;
  std::vector<int32_t> h_sliding_slot_logical_pages;
  std::vector<int32_t> h_global_slot_logical_pages;
  std::vector<int32_t> h_seq_lengths;
  std::vector<int32_t> h_token_batch;
  std::vector<int32_t> h_token_position;
};

// Allocates runtime buffers; `state` must be zero-initialized or freed.
cudaError_t gemma4_runtime_state_init(
    Gemma4RuntimeState *state,
    int32_t batch_size,
    int32_t max_seq_len,
    int32_t page_size,
    cudaStream_t stream);

// Releases every CUDA allocation owned by the runtime state.
void gemma4_runtime_state_free(Gemma4RuntimeState *state);

// Prepares page tables and absolute token positions for an initial prompt.
cudaError_t gemma4_runtime_prepare_prefill(
    Gemma4RuntimeState *state,
    int32_t seq_len,
    cudaStream_t stream);

// Appends one decode position per batch and uploads the new runtime metadata.
cudaError_t gemma4_runtime_prepare_decode_step(
    Gemma4RuntimeState *state,
    cudaStream_t stream);
