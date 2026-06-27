#include "gemma4_megakernel.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

// Prints one failed condition and keeps the sentinel test compact.
bool expect_true(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message);
  }
  return condition;
}

// Exercises the shared runtime-prep wrapper on a tiny owned runtime state.
bool test_runtime_prep_smoke() {
  Gemma4RuntimeState runtime = {};
  cudaError_t status = gemma4_runtime_state_init(&runtime, 1, 4, 2, nullptr);
  if (status != cudaSuccess) {
    fprintf(stderr, "FAIL: runtime init: %s\n", cudaGetErrorString(status));
    return false;
  }

  bool ok = true;
  status = gemma4_megakernel_prepare_runtime(
      &runtime, Gemma4MegakernelPrepMode::kPrefill, 2, nullptr);
  ok &= expect_true(status == cudaSuccess, "prefill runtime prep succeeds");
  ok &= expect_true(runtime.token_count == 2, "prefill token count is prompt rows");
  ok &= expect_true(runtime.h_seq_lengths[0] == 2, "prefill sequence length is set");

  status = gemma4_megakernel_prepare_runtime(
      &runtime, Gemma4MegakernelPrepMode::kDecode, 0, nullptr);
  ok &= expect_true(status == cudaSuccess, "decode runtime prep succeeds");
  ok &= expect_true(runtime.token_count == 1, "decode prep uploads one live row");
  ok &= expect_true(runtime.h_seq_lengths[0] == 3, "decode sequence length advances");

  gemma4_runtime_state_free(&runtime);
  return ok;
}

// Checks that the rewritten decode owner exposes only the full-step API surface.
int main() {
  bool ok = true;
  ok &= expect_true(
      gemma4_decode_megakernel_scratch_bytes() >
          gemma4_sample_next_scratch_bytes(),
      "decode scratch includes FFN state plus final sampling state");

  const cudaError_t prep_status = gemma4_megakernel_prepare_runtime(
      nullptr, Gemma4MegakernelPrepMode::kDecode, 0, nullptr);
  ok &= expect_true(
      prep_status == cudaErrorInvalidValue,
      "null runtime is rejected by shared prep");

  Gemma4DecodeMegakernelArgs args = {};
  const cudaError_t decode_status = gemma4_decode_megakernel(args);
  ok &= expect_true(
      decode_status == cudaErrorInvalidValue,
      "empty decode args are rejected at the public boundary");
  ok &= test_runtime_prep_smoke();

  if (!ok) {
    return 1;
  }
  printf("test_decode_megakernel passed\n");
  return 0;
}
