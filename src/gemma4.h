#pragma once

#include <stdbool.h>
#include <stdint.h>

#define GEMMA4_VOCAB_SIZE 262144
#define GEMMA4_HIDDEN_SIZE 3840
#define GEMMA4_INTERMEDIATE_SIZE 15360
#define GEMMA4_NUM_LAYERS 48
#define GEMMA4_MAX_POSITION_EMBEDDINGS 262144

#define GEMMA4_GLOBAL_LAYER_PERIOD 6
#define GEMMA4_GLOBAL_LAYER_COUNT \
    ((GEMMA4_NUM_LAYERS / GEMMA4_GLOBAL_LAYER_PERIOD) + \
     ((GEMMA4_NUM_LAYERS % GEMMA4_GLOBAL_LAYER_PERIOD) != 0 ? 1 : 0))
#define GEMMA4_SLIDING_LAYER_COUNT \
    (GEMMA4_NUM_LAYERS - GEMMA4_GLOBAL_LAYER_COUNT)

#define GEMMA4_NUM_QUERY_HEADS 16
#define GEMMA4_SLIDING_KV_HEADS 8
#define GEMMA4_GLOBAL_KV_HEADS 1
#define GEMMA4_SLIDING_HEAD_DIM 256
#define GEMMA4_GLOBAL_HEAD_DIM 512
// Total live sliding keys, including the current token.
#define GEMMA4_SLIDING_WINDOW 1024
#define GEMMA4_SLIDING_DECODE_SPLIT_SIZE 20

#define GEMMA4_ROPE_THETA_SLIDING 10000.0f
#define GEMMA4_ROPE_THETA_GLOBAL 1000000.0f
#define GEMMA4_PARTIAL_ROTARY_FACTOR_GLOBAL 0.25f
#define GEMMA4_RMS_NORM_EPS 1.0e-6f
#define GEMMA4_FINAL_LOGIT_SOFTCAPPING 30.0f
// HF casts sqrt(hidden_size) to BF16 before scaling token embeddings.
#define GEMMA4_EMBEDDING_SCALE 62.0f

#define GEMMA4_PACKED_FFN_SIZE (2 * GEMMA4_INTERMEDIATE_SIZE)
#define GEMMA4_SLIDING_Q_PROJ_SIZE \
    (GEMMA4_NUM_QUERY_HEADS * GEMMA4_SLIDING_HEAD_DIM)
#define GEMMA4_SLIDING_KV_PROJ_SIZE \
    (GEMMA4_SLIDING_KV_HEADS * GEMMA4_SLIDING_HEAD_DIM)
#define GEMMA4_SLIDING_QKV_SIZE \
    (GEMMA4_SLIDING_Q_PROJ_SIZE + 2 * GEMMA4_SLIDING_KV_PROJ_SIZE)
#define GEMMA4_SLIDING_ATTENTION_OUT_SIZE GEMMA4_SLIDING_Q_PROJ_SIZE
#define GEMMA4_GLOBAL_Q_PROJ_SIZE \
    (GEMMA4_NUM_QUERY_HEADS * GEMMA4_GLOBAL_HEAD_DIM)
#define GEMMA4_GLOBAL_K_PROJ_SIZE \
    (GEMMA4_GLOBAL_KV_HEADS * GEMMA4_GLOBAL_HEAD_DIM)
#define GEMMA4_GLOBAL_ATTENTION_OUT_SIZE GEMMA4_GLOBAL_Q_PROJ_SIZE

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

bool gemma4_is_global_layer(int32_t layer_index);
Gemma4AttentionSpec gemma4_attention_spec(int32_t layer_index);
