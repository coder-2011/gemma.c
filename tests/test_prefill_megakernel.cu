#include "gemma4_megakernel.cuh"

#include <stdio.h>

// Prints one failed condition and gives the tiny API sentinel a single exit path.
bool expect_true(bool condition, const char *message) {
  if (!condition) {
    fprintf(stderr, "FAIL: %s\n", message);
  }
  return condition;
}

// Checks that the rewritten prefill owner exposes only the full-path API surface.
int main() {
  bool ok = true;
  ok &= expect_true(
      gemma4_prefill_megakernel_scratch_elements(0) == 0,
      "zero rows require no prefill scratch");
  ok &= expect_true(
      gemma4_prefill_megakernel_scratch_elements(1) > GEMMA4_HIDDEN_SIZE,
      "one row requires layer scratch beyond the hidden state");

  Gemma4PrefillMegakernelArgs args = {};
  const cudaError_t status = gemma4_prefill_megakernel(args);
  ok &= expect_true(
      status == cudaErrorInvalidValue,
      "empty prefill args are rejected at the public boundary");

  if (!ok) {
    return 1;
  }
  printf("test_prefill_megakernel passed\n");
  return 0;
}
