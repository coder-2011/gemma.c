#include "gemma4.h"

const Gemma4DenseConfig gemma4_config = {
    .vocab_size = GEMMA4_VOCAB_SIZE,
    .hidden_size = GEMMA4_HIDDEN_SIZE,
    .intermediate_size = GEMMA4_INTERMEDIATE_SIZE,
    .num_layers = GEMMA4_NUM_LAYERS,
    .max_position_embeddings = GEMMA4_MAX_POSITION_EMBEDDINGS,

    .num_query_heads = GEMMA4_NUM_QUERY_HEADS,
    .sliding_kv_heads = GEMMA4_SLIDING_KV_HEADS,
    .global_kv_heads = GEMMA4_GLOBAL_KV_HEADS,
    .sliding_head_dim = GEMMA4_SLIDING_HEAD_DIM,
    .global_head_dim = GEMMA4_GLOBAL_HEAD_DIM,
    .sliding_window = GEMMA4_SLIDING_WINDOW,

    .rope_theta_sliding = GEMMA4_ROPE_THETA_SLIDING,
    .rope_theta_global = GEMMA4_ROPE_THETA_GLOBAL,
    .partial_rotary_factor_global = GEMMA4_PARTIAL_ROTARY_FACTOR_GLOBAL,
    .rms_norm_eps = GEMMA4_RMS_NORM_EPS,
    .final_logit_softcapping = GEMMA4_FINAL_LOGIT_SOFTCAPPING,
};

bool gemma4_is_global_layer(int32_t layer_index) {
    return layer_index == gemma4_config.num_layers - 1 || (layer_index + 1) % 6 == 0;
}

Gemma4AttentionSpec gemma4_attention_spec(int32_t layer_index) {
    bool global = gemma4_is_global_layer(layer_index);
    int32_t head_dim = global ? gemma4_config.global_head_dim : gemma4_config.sliding_head_dim;
    int32_t kv_heads = global ? gemma4_config.global_kv_heads : gemma4_config.sliding_kv_heads;

    Gemma4AttentionSpec spec = {
        .global = global,
        .q_heads = gemma4_config.num_query_heads,
        .kv_heads = kv_heads,
        .head_dim = head_dim,
        .window = global ? 0 : gemma4_config.sliding_window,
        .rotary_dims = global ? (int32_t)(head_dim * gemma4_config.partial_rotary_factor_global) : head_dim,
        .rope_theta = global ? gemma4_config.rope_theta_global : gemma4_config.rope_theta_sliding,
    };
    return spec;
}
