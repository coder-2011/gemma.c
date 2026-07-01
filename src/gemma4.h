#pragma once

#include <stdint.h>

static constexpr int GEMMA4_VOCAB_SIZE = 262144;
static constexpr int GEMMA4_HIDDEN_SIZE = 3840;
static constexpr int GEMMA4_INTERMEDIATE_SIZE = 15360;
static constexpr int GEMMA4_NUM_LAYERS = 48;
static constexpr int GEMMA4_MAX_POSITION_EMBEDDINGS = 262144;

static constexpr int GEMMA4_GLOBAL_LAYER_PERIOD = 6;
static constexpr int GEMMA4_GLOBAL_LAYER_COUNT =
    (GEMMA4_NUM_LAYERS / GEMMA4_GLOBAL_LAYER_PERIOD) +
    ((GEMMA4_NUM_LAYERS % GEMMA4_GLOBAL_LAYER_PERIOD) != 0 ? 1 : 0);
static constexpr int GEMMA4_SLIDING_LAYER_COUNT = GEMMA4_NUM_LAYERS - GEMMA4_GLOBAL_LAYER_COUNT;

static constexpr int GEMMA4_NUM_QUERY_HEADS = 16;
static constexpr int GEMMA4_SLIDING_KV_HEADS = 8;
static constexpr int GEMMA4_GLOBAL_KV_HEADS = 1;
static constexpr int GEMMA4_SLIDING_HEAD_DIM = 256;
static constexpr int GEMMA4_GLOBAL_HEAD_DIM = 512;
// Total live sliding keys, including the current token.
static constexpr int GEMMA4_SLIDING_WINDOW = 1024;
static constexpr int GEMMA4_SLIDING_DECODE_SPLIT_SIZE = 20;

static constexpr float GEMMA4_ROPE_THETA_SLIDING = 10000.0f;
static constexpr float GEMMA4_ROPE_THETA_GLOBAL = 1000000.0f;
static constexpr float GEMMA4_PARTIAL_ROTARY_FACTOR_GLOBAL = 0.25f;
static constexpr float GEMMA4_RMS_NORM_EPS = 1.0e-6f;
static constexpr float GEMMA4_FINAL_LOGIT_SOFTCAPPING = 30.0f;
// HF casts sqrt(hidden_size) to BF16 before scaling token embeddings.
static constexpr float GEMMA4_EMBEDDING_SCALE = 62.0f;

static constexpr int GEMMA4_PACKED_FFN_SIZE = 2 * GEMMA4_INTERMEDIATE_SIZE;
static constexpr int GEMMA4_SLIDING_Q_PROJ_SIZE =
    GEMMA4_NUM_QUERY_HEADS * GEMMA4_SLIDING_HEAD_DIM;
static constexpr int GEMMA4_SLIDING_KV_PROJ_SIZE =
    GEMMA4_SLIDING_KV_HEADS * GEMMA4_SLIDING_HEAD_DIM;
static constexpr int GEMMA4_SLIDING_QKV_SIZE =
    GEMMA4_SLIDING_Q_PROJ_SIZE + 2 * GEMMA4_SLIDING_KV_PROJ_SIZE;
static constexpr int GEMMA4_SLIDING_ATTENTION_OUT_SIZE =
    GEMMA4_SLIDING_Q_PROJ_SIZE;
static constexpr int GEMMA4_GLOBAL_Q_PROJ_SIZE =
    GEMMA4_NUM_QUERY_HEADS * GEMMA4_GLOBAL_HEAD_DIM;
static constexpr int GEMMA4_GLOBAL_K_PROJ_SIZE =
    GEMMA4_GLOBAL_KV_HEADS * GEMMA4_GLOBAL_HEAD_DIM;
static constexpr int GEMMA4_GLOBAL_QK_PROJ_SIZE =
    GEMMA4_GLOBAL_Q_PROJ_SIZE + GEMMA4_GLOBAL_K_PROJ_SIZE;
static constexpr int GEMMA4_GLOBAL_ATTENTION_OUT_SIZE =
    GEMMA4_GLOBAL_Q_PROJ_SIZE;

struct Gemma4DenseConfig {
  int32_t vocab_size;
  int32_t hidden_size;
  int32_t intermediate_size;
  int32_t num_layers;
  int32_t max_position_embeddings;

  int32_t num_query_heads;
  int32_t sliding_kv_heads;
  int32_t global_kv_heads;
  int32_t sliding_head_dim;
  int32_t global_head_dim;
  int32_t sliding_window;

  float rope_theta_sliding;
  float rope_theta_global;
  float partial_rotary_factor_global;
  float rms_norm_eps;
  float final_logit_softcapping;
};

struct Gemma4AttentionSpec {
  bool global;
  int32_t q_heads;
  int32_t kv_heads;
  int32_t head_dim;
  int32_t window;
  int32_t rotary_dims;
  float rope_theta;
};

inline constexpr Gemma4DenseConfig gemma4_config = {
    GEMMA4_VOCAB_SIZE,
    GEMMA4_HIDDEN_SIZE,
    GEMMA4_INTERMEDIATE_SIZE,
    GEMMA4_NUM_LAYERS,
    GEMMA4_MAX_POSITION_EMBEDDINGS,

    GEMMA4_NUM_QUERY_HEADS,
    GEMMA4_SLIDING_KV_HEADS,
    GEMMA4_GLOBAL_KV_HEADS,
    GEMMA4_SLIDING_HEAD_DIM,
    GEMMA4_GLOBAL_HEAD_DIM,
    GEMMA4_SLIDING_WINDOW,

    GEMMA4_ROPE_THETA_SLIDING,
    GEMMA4_ROPE_THETA_GLOBAL,
    GEMMA4_PARTIAL_ROTARY_FACTOR_GLOBAL,
    GEMMA4_RMS_NORM_EPS,
    GEMMA4_FINAL_LOGIT_SOFTCAPPING,
};

// Identifies the full-context layers in Gemma 4's five-sliding, one-global cadence.
constexpr bool gemma4_is_global_layer(int32_t layer_index) {
  return layer_index == gemma4_config.num_layers - 1 ||
         (layer_index + 1) % GEMMA4_GLOBAL_LAYER_PERIOD == 0;
}

// Builds the attention shape constants for one model layer.
constexpr Gemma4AttentionSpec gemma4_attention_spec(int32_t layer_index) {
  const bool global = gemma4_is_global_layer(layer_index);
  const int32_t head_dim = global ? gemma4_config.global_head_dim : gemma4_config.sliding_head_dim;
  const int32_t kv_heads = global ? gemma4_config.global_kv_heads : gemma4_config.sliding_kv_heads;

  return {
      global,
      gemma4_config.num_query_heads,
      kv_heads,
      head_dim,
      global ? 0 : gemma4_config.sliding_window,
      global ? static_cast<int32_t>(head_dim * gemma4_config.partial_rotary_factor_global)
             : head_dim,
      global ? gemma4_config.rope_theta_global
             : gemma4_config.rope_theta_sliding,
  };
}
