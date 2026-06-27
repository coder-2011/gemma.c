#include "gemma4_runtime.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

// Fails the test immediately when a CUDA API call returns an error.
void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

// Copies device int metadata into a host vector.
void copy_i32(std::vector<int32_t> &dst, const int32_t *src) {
  CHECK_CUDA(cudaMemcpy(dst.data(), src, dst.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));
}

// Verifies that the runtime owner advances prompt metadata into decode metadata.
void run_prefill_decode_metadata_case() {
  Gemma4RuntimeState state = {};
  CHECK_CUDA(gemma4_runtime_state_init(&state, 2, 66, 64, 0));

  CHECK_CUDA(gemma4_runtime_prepare_prefill(&state, 65, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<int32_t> seq_lengths(2);
  std::vector<int32_t> token_batch(state.token_count);
  std::vector<int32_t> token_position(state.token_count);
  copy_i32(seq_lengths, state.seq_lengths);
  copy_i32(token_batch, state.token_batch);
  copy_i32(token_position, state.token_position);

  if (seq_lengths[0] != 65 || seq_lengths[1] != 65) {
    std::fprintf(stderr, "prefill sequence lengths were not uploaded\n");
    std::exit(1);
  }
  if (token_batch[64] != 0 || token_position[64] != 64 ||
      token_batch[65] != 1 || token_position[65] != 0) {
    std::fprintf(stderr, "prefill token rows are not batch-major\n");
    std::exit(1);
  }
  const int32_t sliding_stride = state.sliding_cache_config.max_pages_per_seq;
  const int32_t global_stride = state.global_cache_config.max_pages_per_seq;
  if (state.h_sliding_page_table[0] < 0 ||
      state.h_sliding_page_table[sliding_stride] < 0 ||
      state.h_global_page_table[0] < 0 ||
      state.h_global_page_table[global_stride] < 0) {
    std::fprintf(stderr, "prefill did not allocate expected cache pages\n");
    std::exit(1);
  }

  CHECK_CUDA(gemma4_runtime_prepare_decode_step(&state, 0));
  CHECK_CUDA(cudaDeviceSynchronize());
  seq_lengths.assign(2, 0);
  token_batch.assign(state.token_count, 0);
  token_position.assign(state.token_count, 0);
  copy_i32(seq_lengths, state.seq_lengths);
  copy_i32(token_batch, state.token_batch);
  copy_i32(token_position, state.token_position);

  if (seq_lengths[0] != 66 || seq_lengths[1] != 66 ||
      token_batch[0] != 0 || token_batch[1] != 1 ||
      token_position[0] != 65 || token_position[1] != 65) {
    std::fprintf(stderr, "decode append metadata mismatch\n");
    std::exit(1);
  }

  state.h_seq_lengths[0] = 64;
  state.h_seq_lengths[1] = 66;
  const std::vector<int32_t> seq_before = state.h_seq_lengths;
  const std::vector<int32_t> batch_before = state.h_token_batch;
  const std::vector<int32_t> pos_before = state.h_token_position;
  const std::vector<int32_t> sliding_pages_before = state.h_sliding_page_table;
  const std::vector<int32_t> global_pages_before = state.h_global_page_table;
  const std::vector<int32_t> sliding_slots_before =
      state.h_sliding_slot_logical_pages;
  const std::vector<int32_t> global_slots_before =
      state.h_global_slot_logical_pages;
  cudaError_t too_long = gemma4_runtime_prepare_decode_step(&state, 0);
  if (too_long != cudaErrorInvalidValue ||
      state.h_seq_lengths != seq_before ||
      state.h_token_batch != batch_before ||
      state.h_token_position != pos_before ||
      state.h_sliding_page_table != sliding_pages_before ||
      state.h_global_page_table != global_pages_before ||
      state.h_sliding_slot_logical_pages != sliding_slots_before ||
      state.h_global_slot_logical_pages != global_slots_before) {
    std::fprintf(stderr, "failed decode step mutated runtime metadata\n");
    std::exit(1);
  }
  gemma4_runtime_state_free(&state);
}

// Verifies the compact RoPE buffers are initialized for sliding and p-RoPE.
void run_rope_table_case() {
  Gemma4RuntimeState state = {};
  CHECK_CUDA(gemma4_runtime_state_init(&state, 1, 4, 4, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  constexpr int kSlidingHalf = GEMMA4_SLIDING_HEAD_DIM / 2;
  constexpr int kGlobalHalf =
      static_cast<int>(GEMMA4_GLOBAL_HEAD_DIM *
                       GEMMA4_PARTIAL_ROTARY_FACTOR_GLOBAL) /
      2;
  float sliding_cos = 0.0f;
  float sliding_sin = 0.0f;
  float global_cos = 0.0f;
  CHECK_CUDA(cudaMemcpy(&sliding_cos, state.sliding_cos + kSlidingHalf,
                        sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&sliding_sin, state.sliding_sin + kSlidingHalf,
                        sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&global_cos, state.global_cos + kGlobalHalf + 1,
                        sizeof(float), cudaMemcpyDeviceToHost));

  const float expected_global = std::cos(
      std::pow(GEMMA4_ROPE_THETA_GLOBAL,
               -2.0f / static_cast<float>(GEMMA4_GLOBAL_HEAD_DIM)));
  if (std::fabs(sliding_cos - std::cos(1.0f)) > 1.0e-6f ||
      std::fabs(sliding_sin - std::sin(1.0f)) > 1.0e-6f ||
      std::fabs(global_cos - expected_global) > 1.0e-6f) {
    std::fprintf(stderr, "runtime RoPE table mismatch\n");
    std::exit(1);
  }
  gemma4_runtime_state_free(&state);
}

}  // namespace

// Runs the focused runtime-state regression suite.
int main() {
  run_prefill_decode_metadata_case();
  run_rope_table_case();
  std::printf("runtime state tests passed\n");
  return 0;
}
