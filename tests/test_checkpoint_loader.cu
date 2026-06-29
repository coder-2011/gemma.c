#include "gemma4_checkpoint.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <initializer_list>
#include <string>

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

// Fails the test with the CUDA status and source location.
void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    fprintf(stderr, "%s:%d: %s failed: %s\n", file, line, expr,
            cudaGetErrorString(status));
    exit(1);
  }
}

// Converts BF16 values to their raw bits for exact checkpoint-layout checks.
uint16_t bf16_bits(__nv_bfloat16 value) {
  uint16_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

// Repeats the FFN decode hidden-pack swizzle used by the loader.
int hidden_pack_swizzle_index(int chunk) {
  constexpr int kSwizzleChunks = 8;
  const unsigned u = static_cast<unsigned>(chunk);
  const unsigned col = u & (kSwizzleChunks - 1);
  const unsigned row = (u >> 3) & (kSwizzleChunks - 1);
  return static_cast<int>((u & ~(kSwizzleChunks - 1u)) | (col ^ row));
}

// Copies one device BF16 value and compares it with the expected mapped value.
void expect_device_value(
    const __nv_bfloat16 *device,
    size_t offset,
    __nv_bfloat16 expected,
    const char *label) {
  __nv_bfloat16 actual;
  CHECK_CUDA(cudaMemcpy(&actual, device + offset, sizeof(actual),
                        cudaMemcpyDeviceToHost));
  if (bf16_bits(actual) != bf16_bits(expected)) {
    fprintf(stderr, "%s mismatch: got 0x%04x expected 0x%04x\n", label,
            bf16_bits(actual), bf16_bits(expected));
    exit(1);
  }
}

// Validates host tensor views and the full device-native weight preparation.
int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] :
      "models/gemma-4-12B/model.safetensors";

  Gemma4CheckpointHost checkpoint;
  if (!gemma4_checkpoint_open_text_bf16(&checkpoint, path)) {
    fprintf(stderr, "checkpoint open failed\n");
    return 1;
  }

  if (checkpoint.layers[0].v_proj_col_major == nullptr ||
      checkpoint.layers[5].v_proj_col_major != nullptr) {
    fprintf(stderr, "sliding/global V projection views are wrong\n");
    return 1;
  }

  Gemma4TextWeightsDevice weights;
  CHECK_CUDA(gemma4_load_text_weights_device_bf16(&weights, path, nullptr));

  expect_device_value(weights.token_embedding, 0,
                      checkpoint.token_embedding[0], "embedding");
  expect_device_value(weights.layers[0].ffn_gate_up_decode, 0,
                      checkpoint.layers[0].gate_proj_col_major[0],
                      "gate row 0");
  expect_device_value(weights.layers[0].ffn_gate_up_decode, GEMMA4_HIDDEN_SIZE,
                      checkpoint.layers[0].up_proj_col_major[0],
                      "up row 0");

  const int hidden = 9;
  const int row = 37;
  const int dst_col = hidden_pack_swizzle_index(hidden / 8) * 8 + (hidden & 7);
  const size_t down_dst = static_cast<size_t>(row) * GEMMA4_HIDDEN_SIZE +
                          dst_col;
  const size_t down_src = static_cast<size_t>(hidden) *
                          GEMMA4_INTERMEDIATE_SIZE + row;
  expect_device_value(weights.layers[0].ffn_down_decode, down_dst,
                      checkpoint.layers[0].down_proj_checkpoint[down_src],
                      "down transpose");

  for (__nv_bfloat16 *ptr : {weights.token_embedding, weights.final_norm_weight}) cudaFree(ptr);
  for (int layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
    Gemma4TextLayerWeightsDevice &w = weights.layers[layer];
    for (__nv_bfloat16 *ptr : {
        w.input_norm_weight, w.post_attention_norm_weight,
        w.pre_feedforward_norm_weight, w.post_feedforward_norm_weight,
        w.layer_scalar, w.q_norm_weight, w.k_norm_weight, w.q_proj_col_major,
        w.k_proj_col_major, w.v_proj_col_major, w.o_proj_col_major,
        w.ffn_gate_up_decode, w.ffn_down_decode}) {
      cudaFree(ptr);
    }
  }
  weights = Gemma4TextWeightsDevice();
  munmap(checkpoint.mapping, checkpoint.mapping_bytes);
  checkpoint = Gemma4CheckpointHost();
  puts("checkpoint loader tests passed");
  return 0;
}
