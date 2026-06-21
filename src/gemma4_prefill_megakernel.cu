#include "gemma4_prefill_megakernel.cuh"

#include "gemma4.h"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_flash_attention.cuh"
#include "gemma4_matmul_kernels.cuh"
#include "gemma4_rmsnorm.cuh"

#include <cute/layout.hpp>

#include <limits.h>
#include <stdint.h>

namespace {

constexpr int kScaleThreads = 256;

struct PrefillAttentionShape {
  int32_t q_width;
  int32_t kv_width;
  int32_t attention_width;
  int32_t window_size;
};

// Returns the fixed Gemma 4 attention dimensions for the selected layer type.
PrefillAttentionShape prefill_attention_shape(bool global) {
  if (global) {
    return {GEMMA4_GLOBAL_Q_PROJ_SIZE,
            GEMMA4_GLOBAL_K_PROJ_SIZE,
            GEMMA4_GLOBAL_ATTENTION_OUT_SIZE,
            0};
  }
  return {GEMMA4_SLIDING_Q_PROJ_SIZE,
          GEMMA4_SLIDING_KV_PROJ_SIZE,
          GEMMA4_SLIDING_ATTENTION_OUT_SIZE,
          GEMMA4_SLIDING_WINDOW};
}

// Multiplies one hidden tensor by the per-layer scalar after both sublayers.
__global__ __launch_bounds__(kScaleThreads) void scale_hidden_bf16_kernel(
    floatX *__restrict__ out,
    const floatX *__restrict__ in,
    const floatX *__restrict__ layer_scalar,
    int packs) {
  const int pack = blockIdx.x * blockDim.x + threadIdx.x;
  if (pack >= packs) {
    return;
  }

  const float scale = __bfloat162float(__ldg(layer_scalar));
  const int offset = pack * kBf16Packed128Elements;
  Bf16Packed128 values = load128g(in + offset);
  Bf16Packed128 result = gemma4_bf16_pack_apply_scale(values, scale);
  store128(out + offset, result);
}

}  // namespace

// Returns the BF16 scratch elements needed for one prefill layer runner call.
size_t gemma4_prefill_megakernel_layer_scratch_elements(bool global,
                                                        int32_t rows) {
  if (rows <= 0) {
    return 0;
  }

  const PrefillAttentionShape shape = prefill_attention_shape(global);
  const size_t hidden = static_cast<size_t>(GEMMA4_HIDDEN_SIZE);
  const size_t q_width = static_cast<size_t>(shape.q_width);
  const size_t kv_width = static_cast<size_t>(shape.kv_width);
  const size_t attention_width = static_cast<size_t>(shape.attention_width);
  const auto hidden_layout = cute::make_layout(cute::make_shape(rows, hidden));
  const auto q_layout = cute::make_layout(cute::make_shape(rows, q_width));
  const auto kv_layout = cute::make_layout(cute::make_shape(rows, kv_width));
  const auto attention_layout =
      cute::make_layout(cute::make_shape(rows, attention_width));
  const size_t hidden_elements = cute::size(hidden_layout);
  const size_t q_elements = cute::size(q_layout);
  const size_t kv_elements = cute::size(kv_layout);
  const size_t attention_elements = cute::size(attention_layout);
  size_t total = 4 * hidden_elements + 2 * q_elements + 3 * kv_elements;
  if (!global) {
    total += kv_elements;
  }
  return total + attention_elements + gemma4_ffn_prefill_scratch_elements(rows);
}

// Splits one aligned BF16 scratch buffer into the layer runner work tensors.
Gemma4PrefillMegakernelLayerScratch
gemma4_prefill_megakernel_layer_scratch_from_buffer(
    __nv_bfloat16 *buffer,
    bool global,
    int32_t rows) {
  Gemma4PrefillMegakernelLayerScratch scratch = {};
  if (buffer == nullptr || rows <= 0) {
    return scratch;
  }

  scratch.capacity_rows = rows;
  scratch.global = global;

  const PrefillAttentionShape shape = prefill_attention_shape(global);
  __nv_bfloat16 *ptr = buffer;
  const auto hidden_layout =
      cute::make_layout(cute::make_shape(rows, GEMMA4_HIDDEN_SIZE));
  const auto q_layout =
      cute::make_layout(cute::make_shape(rows, shape.q_width));
  const auto kv_layout =
      cute::make_layout(cute::make_shape(rows, shape.kv_width));
  const auto attention_layout =
      cute::make_layout(cute::make_shape(rows, shape.attention_width));
  const size_t hidden = cute::size(hidden_layout);
  const size_t q = cute::size(q_layout);
  const size_t kv = cute::size(kv_layout);
  const size_t attention = cute::size(attention_layout);

  scratch.hidden_work = ptr;
  ptr += hidden;
  scratch.hidden_delta = ptr;
  ptr += hidden;
  scratch.post_attention_residual = ptr;
  ptr += hidden;
  scratch.pre_ffn_normed = ptr;
  ptr += hidden;
  scratch.q = ptr;
  ptr += q;
  scratch.k = ptr;
  ptr += kv;
  if (!global) {
    scratch.v = ptr;
    ptr += kv;
  }
  scratch.q_prepared = ptr;
  ptr += q;
  scratch.k_prepared = ptr;
  ptr += kv;
  scratch.v_prepared = ptr;
  ptr += kv;
  scratch.attention_out = ptr;
  ptr += attention;
  scratch.ffn = gemma4_ffn_prefill_scratch_from_buffer(ptr, rows);
  return scratch;
}

// Runs one Gemma 4 prefill transformer layer using the existing host APIs.
cudaError_t gemma4_prefill_megakernel_layer_bf16(
    const Gemma4PrefillMegakernelLayerArgs &args,
    const Gemma4PrefillMegakernelLayerScratch &scratch) {
  if (args.batch_size < 0 || args.seq_len < 0 ||
      args.layer_index < 0 || args.layer_index >= GEMMA4_NUM_LAYERS) {
    return cudaErrorInvalidValue;
  }

  const int64_t rows64 = int64_t(args.batch_size) * args.seq_len;
  if (rows64 == 0) {
    return cudaSuccess;
  }
  if (rows64 > INT_MAX ||
      rows64 > INT_MAX / GEMMA4_HIDDEN_SIZE) {
    return cudaErrorInvalidValue;
  }

  const bool global = gemma4_is_global_layer(args.layer_index);
  const PrefillAttentionShape shape = prefill_attention_shape(global);
  const int32_t rows = static_cast<int32_t>(rows64);
  if (args.out == nullptr || args.hidden == nullptr ||
      args.weights == nullptr || args.cos == nullptr || args.sin == nullptr ||
      args.softmax_scale <= 0.0f || !is_aligned_16(args.out)) {
    return cudaErrorInvalidValue;
  }

  const Gemma4TextLayerWeightsDevice *weights = args.weights;
  if (weights->layer_scalar == nullptr ||
      scratch.capacity_rows < rows || scratch.global != global) {
    return cudaErrorInvalidValue;
  }

  const int32_t expected_cache_layers =
      global ? GEMMA4_GLOBAL_LAYER_COUNT : GEMMA4_SLIDING_LAYER_COUNT;
  const int32_t expected_cache_heads =
      global ? GEMMA4_GLOBAL_KV_HEADS : GEMMA4_SLIDING_KV_HEADS;
  const int32_t expected_cache_head_dim =
      global ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_SLIDING_HEAD_DIM;
  const int32_t expected_cache_window = global ? 0 : GEMMA4_SLIDING_WINDOW;
  if (args.cache_config.num_layers != expected_cache_layers ||
      args.cache_config.num_pages <= 0 || args.cache_config.page_size <= 0 ||
      args.cache_config.max_pages_per_seq <= 0 ||
      args.cache_config.num_heads != expected_cache_heads ||
      args.cache_config.head_dim != expected_cache_head_dim ||
      args.cache_config.window_size != expected_cache_window) {
    return cudaErrorInvalidValue;
  }

  const int32_t cache_layer =
      gemma4_kv_cache_layer_index(args.layer_index, global);
  if (cache_layer < 0) {
    return cudaErrorInvalidValue;
  }

  cudaError_t status = gemma4_rmsnorm_bf16(
      scratch.hidden_work, args.hidden, weights->input_norm_weight, rows,
      GEMMA4_HIDDEN_SIZE, args.eps, args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  status = gemma4_prefill_gemm_bf16(
      scratch.hidden_work, weights->q_proj_col_major, scratch.q, rows,
      GEMMA4_HIDDEN_SIZE, shape.q_width, args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  status = gemma4_prefill_gemm_bf16(
      scratch.hidden_work, weights->k_proj_col_major, scratch.k, rows,
      GEMMA4_HIDDEN_SIZE, shape.kv_width, args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  if (!global) {
    status = gemma4_prefill_gemm_bf16(
        scratch.hidden_work, weights->v_proj_col_major, scratch.v, rows,
        GEMMA4_HIDDEN_SIZE, shape.kv_width, args.stream);
    if (status != cudaSuccess) {
      return status;
    }

    status = gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
        scratch.attention_out, nullptr, scratch.q_prepared,
        scratch.k_prepared, scratch.v_prepared, scratch.q, scratch.k,
        scratch.v, weights->q_norm_weight, weights->k_norm_weight, args.cos,
        args.sin, args.batch_size, args.seq_len, args.seq_len,
        shape.window_size, args.softmax_scale, args.stream);
  } else {
    status = gemma4_flash_attention_global_fwd_bf16_norm_rope(
        scratch.attention_out, nullptr, scratch.q_prepared,
        scratch.k_prepared, scratch.v_prepared, scratch.q, scratch.k,
        weights->q_norm_weight, weights->k_norm_weight, args.cos, args.sin,
        args.batch_size, args.seq_len, args.seq_len, args.softmax_scale,
        args.stream);
  }
  if (status != cudaSuccess) {
    return status;
  }

  if (global || args.seq_len <= GEMMA4_SLIDING_WINDOW) {
    status = gemma4_kv_cache_write_bf16(
        args.cache_k, args.cache_v, args.cache_config, args.page_table,
        args.token_batch, args.token_position, rows, cache_layer,
        scratch.k_prepared, scratch.v_prepared, args.stream);
    if (status != cudaSuccess) {
      return status;
    }
  } else {
    // Sliding layers keep only the live tail of each batch in the ring cache.
    const int32_t first_live_seq = args.seq_len - GEMMA4_SLIDING_WINDOW;
    for (int32_t batch = 0; batch < args.batch_size; ++batch) {
      const int32_t row_offset = batch * args.seq_len + first_live_seq;
      const int64_t source_offset = int64_t(row_offset) * shape.kv_width;
      status = gemma4_kv_cache_write_bf16(
          args.cache_k, args.cache_v, args.cache_config, args.page_table,
          args.token_batch + row_offset, args.token_position + row_offset,
          GEMMA4_SLIDING_WINDOW, cache_layer,
          scratch.k_prepared + source_offset, scratch.v_prepared + source_offset,
          args.stream);
      if (status != cudaSuccess) {
        return status;
      }
    }
  }

  status = gemma4_prefill_gemm_bf16(
      scratch.attention_out, weights->o_proj_col_major,
      scratch.hidden_delta, rows, shape.attention_width, GEMMA4_HIDDEN_SIZE,
      args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  status = gemma4_rmsnorm_bf16(
      scratch.hidden_work, scratch.hidden_delta,
      weights->post_attention_norm_weight, rows, GEMMA4_HIDDEN_SIZE,
      args.eps, args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  status = gemma4_residual_add_bf16(
      scratch.post_attention_residual, args.hidden, scratch.hidden_work,
      rows * GEMMA4_HIDDEN_SIZE, args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  status = gemma4_rmsnorm_bf16(
      scratch.pre_ffn_normed, scratch.post_attention_residual,
      weights->pre_feedforward_norm_weight, rows, GEMMA4_HIDDEN_SIZE,
      args.eps, args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  status = gemma4_ffn_prefill_mlp_bf16(
      scratch.hidden_delta, scratch.pre_ffn_normed,
      weights->ffn_gate_up_decode, weights->ffn_down_decode, scratch.ffn,
      rows, args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  status = gemma4_rmsnorm_bf16(
      scratch.hidden_work, scratch.hidden_delta,
      weights->post_feedforward_norm_weight, rows, GEMMA4_HIDDEN_SIZE,
      args.eps, args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  status = gemma4_residual_add_bf16(
      scratch.hidden_delta, scratch.post_attention_residual,
      scratch.hidden_work, rows * GEMMA4_HIDDEN_SIZE, args.stream);
  if (status != cudaSuccess) {
    return status;
  }

  const int scale_packs =
      rows * GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
  const int scale_grid = cute::ceil_div(scale_packs, kScaleThreads);
  scale_hidden_bf16_kernel<<<scale_grid, kScaleThreads, 0, args.stream>>>(
      args.out, scratch.hidden_delta, weights->layer_scalar, scale_packs);
  return cudaGetLastError();
}
