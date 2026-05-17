#include "gemma4.h"

const Gemma4DenseConfig gemma4_config = {
    .vocab_size = 262144,
    .hidden_size = 5376,
    .intermediate_size = 21504,
    .num_layers = 60,
    .context_window = 256000,
    .max_position_embeddings = 262144,

    .num_query_heads = 32,
    .sliding_kv_heads = 16,
    .global_kv_heads = 4,
    .sliding_head_dim = 256,
    .global_head_dim = 512,
    .sliding_window = 1024,

    .rope_theta_sliding = 10000.0f,
    .rope_theta_global = 1000000.0f,
    .partial_rotary_factor_global = 0.25f,
    .rms_norm_eps = 1.0e-6f,
    .final_logit_softcapping = 30.0f,
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
