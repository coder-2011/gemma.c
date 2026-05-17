#include "gemma4.h"

#include <stdio.h>

static bool set_error(char *error, size_t error_size, const char *message) {
    if (error != NULL && error_size > 0) {
        snprintf(error, error_size, "%s", message);
    }
    return false;
}

Gemma4DenseConfig gemma4_dense_config_31b(void) {
    Gemma4DenseConfig config = {
        .vocab_size = 262144,
        .hidden_size = 5376,
        .intermediate_size = 21504,
        .num_hidden_layers = 60,
        .context_window = 256000,
        .max_position_embeddings = 262144,

        .num_attention_heads = 32,
        .num_key_value_heads = 16,
        .num_global_key_value_heads = 4,
        .head_dim = 256,
        .global_head_dim = 512,

        .sliding_window = 1024,
        .sliding_layers_per_global = 5,

        .rope_theta_sliding = 10000.0f,
        .rope_theta_global = 1000000.0f,
        .partial_rotary_factor_global = 0.25f,
        .rms_norm_eps = 1.0e-6f,
        .initializer_range = 0.02f,
        .final_logit_softcapping = 30.0f,
        .query_pre_attn_scalar = 256.0f,

        .pad_token_id = 0,
        .eos_token_id = 1,
        .bos_token_id = 2,

        .attention_bias = false,
        .attention_dropout_enabled = false,
        .attention_k_eq_v = true,
        .tie_word_embeddings = true,
        .enable_moe_block = false,

        .weight_dtype = GEMMA4_DTYPE_BF16,
    };
    return config;
}

Gemma4DenseConfig gemma4_dense_config_tiny(void) {
    Gemma4DenseConfig config = gemma4_dense_config_31b();
    config.vocab_size = 256;
    config.hidden_size = 64;
    config.intermediate_size = 256;
    config.num_hidden_layers = 6;
    config.context_window = 64;
    config.max_position_embeddings = 64;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.num_global_key_value_heads = 1;
    config.head_dim = 16;
    config.global_head_dim = 32;
    config.sliding_window = 16;
    config.query_pre_attn_scalar = 16.0f;
    return config;
}

Gemma4LayerType gemma4_layer_type(const Gemma4DenseConfig *config, int32_t layer_index) {
    const int32_t period = config->sliding_layers_per_global + 1;

    if (layer_index == config->num_hidden_layers - 1) {
        return GEMMA4_LAYER_FULL_ATTENTION;
    }
    if (period > 0 && (layer_index + 1) % period == 0) {
        return GEMMA4_LAYER_FULL_ATTENTION;
    }
    return GEMMA4_LAYER_SLIDING_ATTENTION;
}

Gemma4AttentionSpec gemma4_attention_spec(const Gemma4DenseConfig *config, int32_t layer_index) {
    Gemma4LayerType layer_type = gemma4_layer_type(config, layer_index);
    bool is_full = layer_type == GEMMA4_LAYER_FULL_ATTENTION;
    int32_t head_dim = is_full ? config->global_head_dim : config->head_dim;
    float partial_rotary = is_full ? config->partial_rotary_factor_global : 1.0f;

    Gemma4AttentionSpec spec = {
        .layer_type = layer_type,
        .rope_type = is_full ? GEMMA4_ROPE_PROPORTIONAL : GEMMA4_ROPE_DEFAULT,
        .num_query_heads = config->num_attention_heads,
        .num_key_value_heads = is_full ? config->num_global_key_value_heads : config->num_key_value_heads,
        .head_dim = head_dim,
        .sliding_window = is_full ? 0 : config->sliding_window,
        .rotary_dims = (int32_t)((float)head_dim * partial_rotary),
        .rope_theta = is_full ? config->rope_theta_global : config->rope_theta_sliding,
        .has_v_projection = !(is_full && config->attention_k_eq_v),
        .value_norm_has_weight = false,
    };
    return spec;
}

int32_t gemma4_attention_q_width(const Gemma4AttentionSpec *spec) {
    return spec->num_query_heads * spec->head_dim;
}

int32_t gemma4_attention_kv_width(const Gemma4AttentionSpec *spec) {
    return spec->num_key_value_heads * spec->head_dim;
}

int32_t gemma4_count_layers_of_type(const Gemma4DenseConfig *config, Gemma4LayerType layer_type) {
    int32_t count = 0;
    for (int32_t layer = 0; layer < config->num_hidden_layers; ++layer) {
        if (gemma4_layer_type(config, layer) == layer_type) {
            ++count;
        }
    }
    return count;
}

bool gemma4_validate_dense_config(const Gemma4DenseConfig *config, char *error, size_t error_size) {
    if (config == NULL) {
        return set_error(error, error_size, "config is null");
    }
    if (config->vocab_size <= 0 || config->hidden_size <= 0 || config->intermediate_size <= 0) {
        return set_error(error, error_size, "model dimensions must be positive");
    }
    if (config->num_hidden_layers <= 0 || config->sliding_layers_per_global <= 0) {
        return set_error(error, error_size, "layer counts must be positive");
    }
    if (config->num_attention_heads <= 0 ||
        config->num_key_value_heads <= 0 ||
        config->num_global_key_value_heads <= 0) {
        return set_error(error, error_size, "attention head counts must be positive");
    }
    if (config->head_dim <= 0 || config->global_head_dim <= 0) {
        return set_error(error, error_size, "head dimensions must be positive");
    }
    if (config->num_attention_heads % config->num_key_value_heads != 0) {
        return set_error(error, error_size, "sliding KV heads must divide query heads");
    }
    if (config->num_attention_heads % config->num_global_key_value_heads != 0) {
        return set_error(error, error_size, "global KV heads must divide query heads");
    }
    if (config->sliding_window <= 0 ||
        config->context_window <= 0 ||
        config->max_position_embeddings <= 0) {
        return set_error(error, error_size, "context sizes must be positive");
    }
    if (config->context_window > config->max_position_embeddings) {
        return set_error(error, error_size, "context window exceeds max position embeddings");
    }
    if (config->partial_rotary_factor_global <= 0.0f || config->partial_rotary_factor_global > 1.0f) {
        return set_error(error, error_size, "global partial rotary factor must be in (0, 1]");
    }
    if (config->rms_norm_eps <= 0.0f) {
        return set_error(error, error_size, "rms norm epsilon must be positive");
    }
    if (config->rope_theta_sliding <= 0.0f || config->rope_theta_global <= 0.0f) {
        return set_error(error, error_size, "RoPE theta values must be positive");
    }
    if (config->query_pre_attn_scalar <= 0.0f) {
        return set_error(error, error_size, "query pre-attention scalar must be positive");
    }
    for (int32_t layer = 0; layer < config->num_hidden_layers; ++layer) {
        Gemma4AttentionSpec spec = gemma4_attention_spec(config, layer);
        if (spec.rotary_dims <= 0 || spec.rotary_dims > spec.head_dim) {
            return set_error(error, error_size, "invalid rotary dimension count");
        }
        if (spec.rotary_dims % 2 != 0) {
            return set_error(error, error_size, "rotary dimension count must be even");
        }
    }
    if (gemma4_layer_type(config, config->num_hidden_layers - 1) != GEMMA4_LAYER_FULL_ATTENTION) {
        return set_error(error, error_size, "final layer must be full attention");
    }

    if (error != NULL && error_size > 0) {
        error[0] = '\0';
    }
    return true;
}

const char *gemma4_layer_type_name(Gemma4LayerType layer_type) {
    switch (layer_type) {
    case GEMMA4_LAYER_SLIDING_ATTENTION:
        return "sliding_attention";
    case GEMMA4_LAYER_FULL_ATTENTION:
        return "full_attention";
    default:
        return "unknown";
    }
}

const char *gemma4_rope_type_name(Gemma4RopeType rope_type) {
    switch (rope_type) {
    case GEMMA4_ROPE_DEFAULT:
        return "default";
    case GEMMA4_ROPE_PROPORTIONAL:
        return "proportional";
    default:
        return "unknown";
    }
}
