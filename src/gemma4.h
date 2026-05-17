#ifndef GEMMA4_H
#define GEMMA4_H

#include <stdbool.h>
#include <stdint.h>

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
