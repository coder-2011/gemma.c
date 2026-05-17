#ifndef GEMMA4_H
#define GEMMA4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t gemma4_bfloat16;

typedef enum {
    GEMMA4_DTYPE_F32 = 0,
    GEMMA4_DTYPE_BF16 = 1,
} Gemma4DType;

typedef enum {
    GEMMA4_LAYER_SLIDING_ATTENTION = 0,
    GEMMA4_LAYER_FULL_ATTENTION = 1,
} Gemma4LayerType;

typedef enum {
    GEMMA4_ROPE_DEFAULT = 0,
    GEMMA4_ROPE_PROPORTIONAL = 1,
} Gemma4RopeType;

typedef struct {
    Gemma4LayerType layer_type;
    Gemma4RopeType rope_type;
    int32_t num_query_heads;
    int32_t num_key_value_heads;
    int32_t head_dim;
    int32_t sliding_window;
    int32_t rotary_dims;
    float rope_theta;
    bool has_v_projection;
    bool value_norm_has_weight;
} Gemma4AttentionSpec;

typedef struct {
    int32_t vocab_size;
    int32_t hidden_size;
    int32_t intermediate_size;
    int32_t num_hidden_layers;
    int32_t context_window;
    int32_t max_position_embeddings;

    int32_t num_attention_heads;
    int32_t num_key_value_heads;
    int32_t num_global_key_value_heads;
    int32_t head_dim;
    int32_t global_head_dim;

    int32_t sliding_window;
    int32_t sliding_layers_per_global;

    float rope_theta_sliding;
    float rope_theta_global;
    float partial_rotary_factor_global;
    float rms_norm_eps;
    float initializer_range;
    float final_logit_softcapping;
    float query_pre_attn_scalar;

    int32_t pad_token_id;
    int32_t eos_token_id;
    int32_t bos_token_id;

    bool attention_bias;
    bool attention_dropout_enabled;
    bool attention_k_eq_v;
    bool tie_word_embeddings;
    bool enable_moe_block;

    Gemma4DType weight_dtype;
} Gemma4DenseConfig;

typedef struct {
    const gemma4_bfloat16 *input_layernorm_weight;
    const gemma4_bfloat16 *post_attention_layernorm_weight;
    const gemma4_bfloat16 *pre_feedforward_layernorm_weight;
    const gemma4_bfloat16 *post_feedforward_layernorm_weight;

    const gemma4_bfloat16 *q_norm_weight;
    const gemma4_bfloat16 *k_norm_weight;

    const gemma4_bfloat16 *q_proj_weight;
    const gemma4_bfloat16 *k_proj_weight;
    const gemma4_bfloat16 *v_proj_weight;
    const gemma4_bfloat16 *o_proj_weight;

    const gemma4_bfloat16 *gate_proj_weight;
    const gemma4_bfloat16 *up_proj_weight;
    const gemma4_bfloat16 *down_proj_weight;
} Gemma4LayerWeights;

typedef struct {
    const gemma4_bfloat16 *token_embedding_table;
    const Gemma4LayerWeights *layers;
    const gemma4_bfloat16 *final_norm_weight;

    /* Null means lm_head is tied to token_embedding_table. */
    const gemma4_bfloat16 *lm_head_weight;
} Gemma4DenseWeights;

typedef struct {
    void *key;
    void *value;
    int32_t capacity_tokens;
    int32_t cursor;
} Gemma4LayerKVCache;

typedef struct {
    void *hidden;
    void *normed;
    void *q;
    void *k;
    void *v;
    void *attention_scores;
    void *attention_probs;
    void *attention_out;
    void *ffn_gate;
    void *ffn_up;
    void *ffn_out;
    void *logits;

    Gemma4LayerKVCache *kv_cache;
    int32_t kv_cache_layers;
} Gemma4RunState;

Gemma4DenseConfig gemma4_dense_config_31b(void);
Gemma4DenseConfig gemma4_dense_config_tiny(void);

Gemma4LayerType gemma4_layer_type(const Gemma4DenseConfig *config, int32_t layer_index);
Gemma4AttentionSpec gemma4_attention_spec(const Gemma4DenseConfig *config, int32_t layer_index);

int32_t gemma4_attention_q_width(const Gemma4AttentionSpec *spec);
int32_t gemma4_attention_kv_width(const Gemma4AttentionSpec *spec);
int32_t gemma4_count_layers_of_type(const Gemma4DenseConfig *config, Gemma4LayerType layer_type);

bool gemma4_validate_dense_config(const Gemma4DenseConfig *config, char *error, size_t error_size);

const char *gemma4_layer_type_name(Gemma4LayerType layer_type);
const char *gemma4_rope_type_name(Gemma4RopeType rope_type);

#ifdef __cplusplus
}
#endif

#endif
