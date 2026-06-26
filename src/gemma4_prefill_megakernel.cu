#include "gemma4_decode_megakernel.cuh"

#include "gemma4_cuda_utils.cuh"
#include "gemma4_ffn.cuh"
#include "gemma4_matmul_kernels.cuh"
#include "gemma4_rmsnorm.cuh"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <utility>

namespace {

constexpr int kScaleThreads = 256;

struct PrefillAttentionShape {
  int32_t q_width;
  int32_t kv_width;
  int32_t attention_width;
  int32_t window_size;
};

struct PrefillScratch {
  __nv_bfloat16 *hidden_work = nullptr;
  __nv_bfloat16 *hidden_delta = nullptr;
  __nv_bfloat16 *post_attention_residual = nullptr;
  __nv_bfloat16 *pre_ffn_normed = nullptr;
  __nv_bfloat16 *q = nullptr;
  __nv_bfloat16 *k = nullptr;
  __nv_bfloat16 *v = nullptr;
  __nv_bfloat16 *q_prepared = nullptr;
  __nv_bfloat16 *k_prepared = nullptr;
  __nv_bfloat16 *v_prepared = nullptr;
  __nv_bfloat16 *attention_out = nullptr;
  Gemma4FfnPrefillScratch ffn = {};
};

// Returns the fixed attention widths for a sliding or global layer.
constexpr PrefillAttentionShape prefill_attention_shape(bool global) {
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

// Returns the BF16 scratch elements needed by one layer of this attention type.
size_t prefill_layer_scratch_elements(bool global, int32_t rows) {
  if (rows <= 0) {
    return 0;
  }

  const PrefillAttentionShape shape = prefill_attention_shape(global);
  const size_t hidden = static_cast<size_t>(rows) * GEMMA4_HIDDEN_SIZE;
  const size_t q = static_cast<size_t>(rows) * shape.q_width;
  const size_t kv = static_cast<size_t>(rows) * shape.kv_width;
  const size_t attention = static_cast<size_t>(rows) * shape.attention_width;
  size_t total = 4 * hidden + 2 * q + 3 * kv + attention;
  if (!global) {
    total += kv;
  }
  return total + gemma4_ffn_prefill_scratch_elements(rows);
}

// Splits one caller-owned BF16 buffer into the tensors reused by a prefill layer.
PrefillScratch prefill_scratch_from_buffer(
    __nv_bfloat16 *buffer,
    bool global,
    int32_t rows) {
  const PrefillAttentionShape shape = prefill_attention_shape(global);
  const size_t hidden = static_cast<size_t>(rows) * GEMMA4_HIDDEN_SIZE;
  const size_t q = static_cast<size_t>(rows) * shape.q_width;
  const size_t kv = static_cast<size_t>(rows) * shape.kv_width;
  const size_t attention = static_cast<size_t>(rows) * shape.attention_width;

  PrefillScratch scratch = {};
  __nv_bfloat16 *ptr = buffer;
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

// Scales 128-bit BF16 packs by the checkpoint layer scalar.
__global__ __launch_bounds__(kScaleThreads) void scale_hidden_bf16_kernel(
    __nv_bfloat16 *__restrict__ out,
    const __nv_bfloat16 *__restrict__ in,
    const __nv_bfloat16 *__restrict__ layer_scalar,
    int packs) {
  const int pack = blockIdx.x * blockDim.x + threadIdx.x;
  if (pack >= packs) {
    return;
  }

  const float scale = __bfloat162float(__ldg(layer_scalar));
  const int offset = pack * kBf16Packed128Elements;
  const Bf16Packed128 values =
      Bf16Packed128{*reinterpret_cast<const int4 *>(in + offset)};
  const Bf16Packed128 scaled = gemma4_bf16_pack_apply_scale(values, scale);
  *reinterpret_cast<int4 *>(out + offset) = scaled.bits();
}

// Runs one full prefill layer using the existing unfused kernels in model order.
cudaError_t run_prefill_layer(
    __nv_bfloat16 *out,
    const __nv_bfloat16 *hidden,
    const Gemma4TextLayerWeightsDevice &weights,
    Gemma4RuntimeState *runtime,
    const PrefillScratch &scratch,
    int32_t layer,
    int32_t rows,
    int32_t seq_len,
    cudaStream_t stream) {
  const bool global = gemma4_is_global_layer(layer);
  const int32_t batch_size = runtime->batch_size;
  const int32_t head_dim =
      global ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_SLIDING_HEAD_DIM;
  const PrefillAttentionShape shape = prefill_attention_shape(global);
  const int32_t cache_layer = gemma4_kv_cache_layer_index(layer, global);
  const float softmax_scale = 1.0f / sqrtf(float(head_dim));
  Gemma4KvCacheConfig cache_config =
      global ? runtime->global_cache_config : runtime->sliding_cache_config;
  __nv_bfloat16 *cache_k =
      global ? runtime->global_cache_k : runtime->sliding_cache_k;
  __nv_bfloat16 *cache_v =
      global ? runtime->global_cache_v : runtime->sliding_cache_v;
  const int32_t *page_table =
      global ? runtime->global_page_table : runtime->sliding_page_table;
  const float *cos = global ? runtime->global_cos : runtime->sliding_cos;
  const float *sin = global ? runtime->global_sin : runtime->sliding_sin;

  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_rmsnorm_bf16(
      scratch.hidden_work, hidden, weights.input_norm_weight, rows,
      GEMMA4_HIDDEN_SIZE, GEMMA4_RMS_NORM_EPS, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_prefill_gemm_bf16(
      scratch.hidden_work, weights.q_proj_col_major, scratch.q, rows,
      GEMMA4_HIDDEN_SIZE, shape.q_width, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_prefill_gemm_bf16(
      scratch.hidden_work, weights.k_proj_col_major, scratch.k, rows,
      GEMMA4_HIDDEN_SIZE, shape.kv_width, stream));

  if (!global) {
    GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_prefill_gemm_bf16(
        scratch.hidden_work, weights.v_proj_col_major, scratch.v, rows,
        GEMMA4_HIDDEN_SIZE, shape.kv_width, stream));
    GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_flash_attention_sliding_fwd_bf16_norm_rope(
        scratch.attention_out, nullptr, scratch.q_prepared,
        scratch.k_prepared, scratch.v_prepared, scratch.q, scratch.k,
        scratch.v, weights.q_norm_weight, weights.k_norm_weight, cos, sin,
        runtime->token_position, batch_size, seq_len, seq_len,
        shape.window_size, softmax_scale, stream));
  } else {
    GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_flash_attention_global_fwd_bf16_norm_rope(
        scratch.attention_out, nullptr, scratch.q_prepared,
        scratch.k_prepared, scratch.v_prepared, scratch.q, scratch.k,
        weights.q_norm_weight, weights.k_norm_weight, cos, sin,
        runtime->token_position, batch_size, seq_len, seq_len,
        softmax_scale, stream));
  }

  if (global || seq_len <= GEMMA4_SLIDING_WINDOW) {
    GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_kv_cache_write_bf16(
        cache_k, cache_v, cache_config, page_table, runtime->token_batch,
        runtime->token_position, rows, cache_layer, scratch.k_prepared,
        scratch.v_prepared, stream));
  } else {
    // Long sliding prefill writes only the live ring-cache tail.
    const int32_t first_live_seq = seq_len - GEMMA4_SLIDING_WINDOW;
    for (int32_t batch = 0; batch < batch_size; ++batch) {
      const int32_t row_offset = batch * seq_len + first_live_seq;
      const int64_t source_offset = int64_t(row_offset) * shape.kv_width;
      GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_kv_cache_write_bf16(
          cache_k, cache_v, cache_config, page_table,
          runtime->token_batch + row_offset,
          runtime->token_position + row_offset, GEMMA4_SLIDING_WINDOW,
          cache_layer, scratch.k_prepared + source_offset,
          scratch.v_prepared + source_offset, stream));
    }
  }

  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_prefill_gemm_bf16(
      scratch.attention_out, weights.o_proj_col_major, scratch.hidden_delta,
      rows, shape.attention_width, GEMMA4_HIDDEN_SIZE, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_rmsnorm_bf16(
      scratch.hidden_work, scratch.hidden_delta,
      weights.post_attention_norm_weight, rows, GEMMA4_HIDDEN_SIZE,
      GEMMA4_RMS_NORM_EPS, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_residual_add_bf16(
      scratch.post_attention_residual, hidden, scratch.hidden_work,
      rows * GEMMA4_HIDDEN_SIZE, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_rmsnorm_bf16(
      scratch.pre_ffn_normed, scratch.post_attention_residual,
      weights.pre_feedforward_norm_weight, rows, GEMMA4_HIDDEN_SIZE,
      GEMMA4_RMS_NORM_EPS, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_ffn_prefill_mlp_bf16(
      scratch.hidden_delta, scratch.pre_ffn_normed, weights.ffn_gate_up_decode,
      weights.ffn_down_decode, scratch.ffn, rows, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_rmsnorm_bf16(
      scratch.hidden_work, scratch.hidden_delta,
      weights.post_feedforward_norm_weight, rows, GEMMA4_HIDDEN_SIZE,
      GEMMA4_RMS_NORM_EPS, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_residual_add_bf16(
      scratch.hidden_delta, scratch.post_attention_residual,
      scratch.hidden_work, rows * GEMMA4_HIDDEN_SIZE, stream));

  const int packs = rows * GEMMA4_HIDDEN_SIZE / kBf16Packed128Elements;
  const dim3 grid_dim((packs + kScaleThreads - 1) / kScaleThreads);
  const dim3 block_dim(kScaleThreads);
  scale_hidden_bf16_kernel<<<grid_dim, block_dim, 0, stream>>>(
      out, scratch.hidden_delta, weights.layer_scalar, packs);
  return cudaGetLastError();
}

}  // namespace

// Returns caller-owned BF16 scratch elements needed by the full prefill path.
size_t gemma4_prefill_megakernel_scratch_elements(int32_t rows) {
  const size_t sliding = prefill_layer_scratch_elements(false, rows);
  const size_t global = prefill_layer_scratch_elements(true, rows);
  return sliding > global ? sliding : global;
}

// Runs all 48 prefill layers over caller-provided prompt embeddings.
cudaError_t gemma4_prefill_megakernel(const Gemma4PrefillMegakernelArgs &args) {
  if (args.hidden_a == nullptr || args.hidden_b == nullptr ||
      args.scratch == nullptr || args.weights == nullptr ||
      args.runtime == nullptr || args.seq_len <= 0) {
    return cudaErrorInvalidValue;
  }

  const int64_t rows64 = int64_t(args.runtime->batch_size) * args.seq_len;
  if (rows64 <= 0 || rows64 > INT32_MAX) {
    return cudaErrorInvalidValue;
  }
  const int32_t rows = static_cast<int32_t>(rows64);
  if (args.scratch_elements < gemma4_prefill_megakernel_scratch_elements(rows)) {
    return cudaErrorInvalidValue;
  }

  GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_megakernel_prepare_runtime(
      args.runtime, Gemma4MegakernelPrepMode::kPrefill, args.seq_len,
      args.stream));

  __nv_bfloat16 *hidden_in = args.hidden_a;
  __nv_bfloat16 *hidden_out = args.hidden_b;
  for (int32_t layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
    const bool global = gemma4_is_global_layer(layer);
    const PrefillScratch scratch =
        prefill_scratch_from_buffer(args.scratch, global, rows);
    GEMMA4_RETURN_IF_CUDA_ERROR(run_prefill_layer(
        hidden_out, hidden_in, args.weights->layers[layer], args.runtime,
        scratch, layer, rows, args.seq_len, args.stream));
    std::swap(hidden_in, hidden_out);
  }

  if (args.final_hidden != nullptr) {
    *args.final_hidden = hidden_in;
  }
  return cudaSuccess;
}
