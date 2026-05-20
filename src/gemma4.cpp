#include "gemma4.h"

bool gemma4_is_global_layer(int32_t layer_index) {
    return layer_index == gemma4_config.num_layers - 1 ||
           (layer_index + 1) % GEMMA4_GLOBAL_LAYER_PERIOD == 0;
}

Gemma4AttentionSpec gemma4_attention_spec(int32_t layer_index) {
    bool global = gemma4_is_global_layer(layer_index);
    int32_t head_dim = global ? gemma4_config.global_head_dim : gemma4_config.sliding_head_dim;
    int32_t kv_heads = global ? gemma4_config.global_kv_heads : gemma4_config.sliding_kv_heads;

    Gemma4AttentionSpec spec = {
        global,
        gemma4_config.num_query_heads,
        kv_heads,
        head_dim,
        global ? 0 : gemma4_config.sliding_window,
        global ? static_cast<int32_t>(head_dim * gemma4_config.partial_rotary_factor_global) : head_dim,
        global ? gemma4_config.rope_theta_global : gemma4_config.rope_theta_sliding,
    };
    return spec;
}
