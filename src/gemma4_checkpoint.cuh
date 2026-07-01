#pragma once

#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stddef.h>
#include <string>

struct Gemma4CheckpointLayerHost {
  const __nv_bfloat16 *input_norm_weight = nullptr;
  const __nv_bfloat16 *post_attention_norm_weight = nullptr;
  const __nv_bfloat16 *pre_feedforward_norm_weight = nullptr;
  const __nv_bfloat16 *post_feedforward_norm_weight = nullptr;
  const __nv_bfloat16 *layer_scalar = nullptr;

  const __nv_bfloat16 *q_norm_weight = nullptr;
  const __nv_bfloat16 *k_norm_weight = nullptr;
  const __nv_bfloat16 *q_proj_col_major = nullptr;
  const __nv_bfloat16 *k_proj_col_major = nullptr;
  const __nv_bfloat16 *v_proj_col_major = nullptr;
  const __nv_bfloat16 *o_proj_col_major = nullptr;

  const __nv_bfloat16 *gate_proj_col_major = nullptr;
  const __nv_bfloat16 *up_proj_col_major = nullptr;
  const __nv_bfloat16 *down_proj_checkpoint = nullptr;
};

struct Gemma4CheckpointHost {
  void *mapping = nullptr;
  size_t mapping_bytes = 0;

  const __nv_bfloat16 *token_embedding = nullptr;
  const __nv_bfloat16 *final_norm_weight = nullptr;
  Gemma4CheckpointLayerHost layers[GEMMA4_NUM_LAYERS] = {};
};

struct Gemma4TextLayerWeightsDevice {
  __nv_bfloat16 *input_norm_weight = nullptr;
  __nv_bfloat16 *post_attention_norm_weight = nullptr;
  __nv_bfloat16 *pre_feedforward_norm_weight = nullptr;
  __nv_bfloat16 *post_feedforward_norm_weight = nullptr;
  __nv_bfloat16 *layer_scalar = nullptr;

  __nv_bfloat16 *q_norm_weight = nullptr;
  __nv_bfloat16 *k_norm_weight = nullptr;
  // Q/K/V are aliases into qkv_proj_col_major; only qkv_proj_col_major owns storage.
  __nv_bfloat16 *q_proj_col_major = nullptr;
  __nv_bfloat16 *k_proj_col_major = nullptr;
  __nv_bfloat16 *v_proj_col_major = nullptr;
  __nv_bfloat16 *qkv_proj_col_major = nullptr;
  __nv_bfloat16 *o_proj_col_major = nullptr;

  __nv_bfloat16 *ffn_gate_up_decode = nullptr;
  __nv_bfloat16 *ffn_down_decode = nullptr;
};

struct Gemma4TextWeightsDevice {
  __nv_bfloat16 *token_embedding = nullptr;
  __nv_bfloat16 *final_norm_weight = nullptr;
  Gemma4TextLayerWeightsDevice layers[GEMMA4_NUM_LAYERS] = {};
};

// Maps BF16 language tensor views from a Gemma 4 safetensors file.
bool gemma4_checkpoint_open_text_bf16(
    Gemma4CheckpointHost *checkpoint,
    const char *path);

// Loads text weights into device memory laid out for the existing decode paths.
cudaError_t gemma4_load_text_weights_device_bf16(
    Gemma4TextWeightsDevice *weights,
    const char *path,
    std::string *error);

// Releases every CUDA allocation owned by loaded text weights.
inline void gemma4_text_weights_device_free(Gemma4TextWeightsDevice *weights) {
  if (weights == nullptr) return;

  __nv_bfloat16 *top_level[] = {
      weights->token_embedding,
      weights->final_norm_weight,
  };
  for (__nv_bfloat16 *ptr : top_level) cudaFree(ptr);

  for (int layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
    Gemma4TextLayerWeightsDevice &w = weights->layers[layer];
    __nv_bfloat16 *owned_layer[] = {
        w.input_norm_weight,
        w.post_attention_norm_weight,
        w.pre_feedforward_norm_weight,
        w.post_feedforward_norm_weight,
        w.layer_scalar,
        w.q_norm_weight,
        w.k_norm_weight,
        w.qkv_proj_col_major,
        w.o_proj_col_major,
        w.ffn_gate_up_decode,
        w.ffn_down_decode,
    };
    for (__nv_bfloat16 *ptr : owned_layer) cudaFree(ptr);
  }
  *weights = Gemma4TextWeightsDevice();
}
