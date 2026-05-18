#ifndef GEMMA4_H
#define GEMMA4_H

#include <stdbool.h>
#include <stdint.h>

#define GEMMA4_VOCAB_SIZE 262144
#define GEMMA4_HIDDEN_SIZE 5376
#define GEMMA4_INTERMEDIATE_SIZE 21504
#define GEMMA4_NUM_LAYERS 60
#define GEMMA4_MAX_POSITION_EMBEDDINGS 262144

#define GEMMA4_NUM_QUERY_HEADS 32
#define GEMMA4_SLIDING_KV_HEADS 16
#define GEMMA4_GLOBAL_KV_HEADS 4
#define GEMMA4_SLIDING_HEAD_DIM 256
#define GEMMA4_GLOBAL_HEAD_DIM 512
#define GEMMA4_SLIDING_WINDOW 1024

#define GEMMA4_ROPE_THETA_SLIDING 10000.0f
#define GEMMA4_ROPE_THETA_GLOBAL 1000000.0f
#define GEMMA4_PARTIAL_ROTARY_FACTOR_GLOBAL 0.25f
#define GEMMA4_RMS_NORM_EPS 1.0e-6f
#define GEMMA4_FINAL_LOGIT_SOFTCAPPING 30.0f

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

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
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
} Gemma4DenseConfig;

typedef struct {
    bool global;
    int32_t q_heads;
    int32_t kv_heads;
    int32_t head_dim;
    int32_t window;
    int32_t rotary_dims;
    float rope_theta;
} Gemma4AttentionSpec;

extern const Gemma4DenseConfig gemma4_config;

bool gemma4_is_global_layer(int32_t layer_index);
Gemma4AttentionSpec gemma4_attention_spec(int32_t layer_index);

#ifdef __cplusplus
}
#endif

#endif
