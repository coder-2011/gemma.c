#include "test_decode_common.cuh"
#include "gemma4_flash_attention.cuh"

namespace {

using namespace gemma4_test;

void run_global_prefill_norm_rope_case() {
  constexpr int batch_size = 1;
  constexpr int seq_len = 4;
  constexpr int q_heads = GEMMA4_NUM_QUERY_HEADS;
  constexpr int kv_heads = GEMMA4_GLOBAL_KV_HEADS;
  constexpr int head_dim = GEMMA4_GLOBAL_HEAD_DIM;
  constexpr int rotary_half = (GEMMA4_GLOBAL_HEAD_DIM / 4) / 2;
  const float scale = 1.0f / std::sqrt(float(head_dim));

  std::vector<__nv_bfloat16> q(batch_size * seq_len * q_heads * head_dim);
  std::vector<__nv_bfloat16> k(batch_size * seq_len * kv_heads * head_dim);
  std::vector<__nv_bfloat16> q_weight(head_dim);
  std::vector<__nv_bfloat16> k_weight(head_dim);
  std::vector<float> cos(seq_len * rotary_half);
  std::vector<float> sin(cos.size());
  for (int i = 0; i < static_cast<int>(q.size()); ++i) {
    q[i] = make_value(37000 + i);
  }
  for (int i = 0; i < static_cast<int>(k.size()); ++i) {
    k[i] = make_value(47000 + i);
  }
  for (int d = 0; d < head_dim; ++d) {
    q_weight[d] = __float2bfloat16_rn(0.80f + 0.001f * float(d % 29));
    k_weight[d] = __float2bfloat16_rn(0.90f - 0.001f * float(d % 31));
  }
  fill_global_rope_tables(cos, sin, seq_len);

  std::vector<__nv_bfloat16> expected_q(q.size());
  std::vector<__nv_bfloat16> expected_k(k.size());
  std::vector<__nv_bfloat16> expected_v(k.size());
  for (int b = 0; b < batch_size; ++b) {
    for (int pos = 0; pos < seq_len; ++pos) {
      for (int h = 0; h < q_heads; ++h) {
        reference_global_weighted_rope_head(
            expected_q,
            token_offset(b, pos, h, 0, seq_len, q_heads, head_dim),
            q,
            token_offset(b, pos, h, 0, seq_len, q_heads, head_dim),
            q_weight, cos, sin, pos);
      }
      for (int h = 0; h < kv_heads; ++h) {
        reference_global_weighted_rope_head(
            expected_k,
            token_offset(b, pos, h, 0, seq_len, kv_heads, head_dim),
            k,
            token_offset(b, pos, h, 0, seq_len, kv_heads, head_dim),
            k_weight, cos, sin, pos);
        reference_global_scale_head(
            expected_v,
            token_offset(b, pos, h, 0, seq_len, kv_heads, head_dim),
            k,
            token_offset(b, pos, h, 0, seq_len, kv_heads, head_dim));
      }
    }
  }
  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  __nv_bfloat16 *d_q_weight = nullptr;
  __nv_bfloat16 *d_k_weight = nullptr;
  __nv_bfloat16 *d_q_prepared = nullptr;
  __nv_bfloat16 *d_k_prepared = nullptr;
  __nv_bfloat16 *d_v_prepared = nullptr;
  __nv_bfloat16 *d_out = nullptr;
  __nv_bfloat16 *d_direct_out = nullptr;
  float *d_cos = nullptr;
  float *d_sin = nullptr;
  CHECK_CUDA(cudaMalloc(&d_q, q.size() * sizeof(*d_q)));
  CHECK_CUDA(cudaMalloc(&d_k, k.size() * sizeof(*d_k)));
  CHECK_CUDA(cudaMalloc(&d_q_weight, q_weight.size() * sizeof(*d_q_weight)));
  CHECK_CUDA(cudaMalloc(&d_k_weight, k_weight.size() * sizeof(*d_k_weight)));
  CHECK_CUDA(cudaMalloc(&d_q_prepared, q.size() * sizeof(*d_q_prepared)));
  CHECK_CUDA(cudaMalloc(&d_k_prepared, k.size() * sizeof(*d_k_prepared)));
  CHECK_CUDA(cudaMalloc(&d_v_prepared, k.size() * sizeof(*d_v_prepared)));
  CHECK_CUDA(cudaMalloc(&d_out, q.size() * sizeof(*d_out)));
  CHECK_CUDA(cudaMalloc(&d_direct_out, q.size() * sizeof(*d_direct_out)));
  CHECK_CUDA(cudaMalloc(&d_cos, cos.size() * sizeof(*d_cos)));
  CHECK_CUDA(cudaMalloc(&d_sin, sin.size() * sizeof(*d_sin)));
  CHECK_CUDA(cudaMemcpy(d_q, q.data(), q.size() * sizeof(q[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k, k.data(), k.size() * sizeof(k[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_q_weight, q_weight.data(),
                        q_weight.size() * sizeof(q_weight[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k_weight, k_weight.data(),
                        k_weight.size() * sizeof(k_weight[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_cos, cos.data(), cos.size() * sizeof(cos[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_sin, sin.data(), sin.size() * sizeof(sin[0]),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(gemma4_flash_attention_global_fwd_bf16_norm_rope(
      d_out, nullptr, d_q_prepared, d_k_prepared, d_v_prepared, d_q, d_k,
      d_q_weight, d_k_weight, d_cos, d_sin, batch_size, seq_len, seq_len,
      scale, 0));
  CHECK_CUDA(gemma4_flash_attention_global_fwd_bf16(
      d_direct_out, nullptr, d_q_prepared, d_k_prepared, d_v_prepared,
      batch_size, seq_len, seq_len, scale, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> actual_q(q.size());
  std::vector<__nv_bfloat16> actual_k(k.size());
  std::vector<__nv_bfloat16> actual_v(k.size());
  std::vector<__nv_bfloat16> actual_out(q.size());
  std::vector<__nv_bfloat16> direct_out(q.size());
  CHECK_CUDA(cudaMemcpy(actual_q.data(), d_q_prepared,
                        actual_q.size() * sizeof(actual_q[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_k.data(), d_k_prepared,
                        actual_k.size() * sizeof(actual_k[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_v.data(), d_v_prepared,
                        actual_v.size() * sizeof(actual_v[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_out.data(), d_out,
                        actual_out.size() * sizeof(actual_out[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(direct_out.data(), d_direct_out,
                        direct_out.size() * sizeof(direct_out[0]),
                        cudaMemcpyDeviceToHost));
  compare_bf16(actual_q, expected_q, 0.03125f, "global prefill prepared Q");
  compare_bf16(actual_k, expected_k, 0.03125f, "global prefill prepared K");
  compare_bf16(actual_v, expected_v, 0.015625f, "global prefill prepared V");
  compare_bf16(actual_out, direct_out, 0.0f,
               "global prefill norm-rope direct attention");

  CHECK_CUDA(cudaFree(d_sin));
  CHECK_CUDA(cudaFree(d_cos));
  CHECK_CUDA(cudaFree(d_direct_out));
  CHECK_CUDA(cudaFree(d_out));
  CHECK_CUDA(cudaFree(d_v_prepared));
  CHECK_CUDA(cudaFree(d_k_prepared));
  CHECK_CUDA(cudaFree(d_q_prepared));
  CHECK_CUDA(cudaFree(d_k_weight));
  CHECK_CUDA(cudaFree(d_q_weight));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_q));
}

void run_sliding_decode_prep_cache_case() {
  Gemma4KvCacheConfig config = {
      2,
      12,
      4,
      4,
      GEMMA4_SLIDING_KV_HEADS,
      GEMMA4_SLIDING_HEAD_DIM,
      8,
  };
  int batch_size = 2;
  int layer = 1;
  std::vector<int32_t> token_position = {5, 11};
  std::vector<int32_t> page_table(batch_size * config.max_pages_per_seq, -1);
  page_table[gemma4_kv_cache_page_slot(config, token_position[0])] = 3;
  page_table[config.max_pages_per_seq +
             gemma4_kv_cache_page_slot(config, token_position[1])] = 7;

  std::vector<__nv_bfloat16> q(batch_size * GEMMA4_NUM_QUERY_HEADS *
                               GEMMA4_SLIDING_HEAD_DIM);
  std::vector<__nv_bfloat16> k(batch_size * GEMMA4_SLIDING_KV_HEADS *
                               GEMMA4_SLIDING_HEAD_DIM);
  std::vector<__nv_bfloat16> v(k.size());
  std::vector<__nv_bfloat16> q_weight(GEMMA4_SLIDING_HEAD_DIM);
  std::vector<__nv_bfloat16> k_weight(GEMMA4_SLIDING_HEAD_DIM);
  std::vector<float> cos((token_position[1] + 1) *
                         (GEMMA4_SLIDING_HEAD_DIM / 2));
  std::vector<float> sin(cos.size());

  for (int i = 0; i < static_cast<int>(q.size()); ++i) {
    q[i] = make_value(12000 + i);
  }
  for (int i = 0; i < static_cast<int>(k.size()); ++i) {
    k[i] = make_value(22000 + i);
    v[i] = make_value(32000 + i);
  }
  for (int d = 0; d < GEMMA4_SLIDING_HEAD_DIM; ++d) {
    q_weight[d] = __float2bfloat16_rn(0.75f + 0.001f * float(d % 23));
    k_weight[d] = __float2bfloat16_rn(0.95f - 0.001f * float(d % 19));
  }
  fill_sliding_rope_tables(cos, sin, token_position[1] + 1);

  std::vector<__nv_bfloat16> expected_q(q.size());
  std::vector<__nv_bfloat16> expected_cache_k(cache_elements(config));
  std::vector<__nv_bfloat16> expected_cache_v(cache_elements(config));
  for (int b = 0; b < batch_size; ++b) {
    int pos = token_position[b];
    int slot = gemma4_kv_cache_page_slot(config, pos);
    int page = page_table[b * config.max_pages_per_seq + slot];
    int page_offset = gemma4_kv_cache_page_offset(config, pos);
    for (int h = 0; h < GEMMA4_NUM_QUERY_HEADS; ++h) {
      reference_sliding_weighted_rope_head(
          expected_q, sliding_q_offset(b, h, 0), q,
          sliding_q_offset(b, h, 0), q_weight, cos, sin, pos);
    }
    for (int h = 0; h < GEMMA4_SLIDING_KV_HEADS; ++h) {
      int64_t cache_base =
          gemma4_kv_cache_offset(config, layer, page, page_offset, h, 0);
      reference_sliding_weighted_rope_head(
          expected_cache_k, cache_base, k, sliding_kv_offset(b, h, 0),
          k_weight, cos, sin, pos);
      reference_sliding_scale_head(expected_cache_v, cache_base, v,
                                   sliding_kv_offset(b, h, 0));
    }
  }

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  __nv_bfloat16 *d_v = nullptr;
  __nv_bfloat16 *d_q_prepared = nullptr;
  __nv_bfloat16 *d_cache_k = nullptr;
  __nv_bfloat16 *d_cache_v = nullptr;
  __nv_bfloat16 *d_q_weight = nullptr;
  __nv_bfloat16 *d_k_weight = nullptr;
  int32_t *d_page_table = nullptr;
  int32_t *d_token_position = nullptr;
  float *d_cos = nullptr;
  float *d_sin = nullptr;
  CHECK_CUDA(cudaMalloc(&d_q, q.size() * sizeof(*d_q)));
  CHECK_CUDA(cudaMalloc(&d_k, k.size() * sizeof(*d_k)));
  CHECK_CUDA(cudaMalloc(&d_v, v.size() * sizeof(*d_v)));
  CHECK_CUDA(cudaMalloc(&d_q_prepared, q.size() * sizeof(*d_q_prepared)));
  CHECK_CUDA(cudaMalloc(&d_cache_k, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMalloc(&d_cache_v, cache_elements(config) * sizeof(*d_cache_v)));
  CHECK_CUDA(cudaMalloc(&d_q_weight, q_weight.size() * sizeof(*d_q_weight)));
  CHECK_CUDA(cudaMalloc(&d_k_weight, k_weight.size() * sizeof(*d_k_weight)));
  CHECK_CUDA(cudaMalloc(&d_page_table, page_table.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_token_position,
                        token_position.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_cos, cos.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sin, sin.size() * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_q_prepared, 0, q.size() * sizeof(*d_q_prepared)));
  CHECK_CUDA(cudaMemset(d_cache_k, 0, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMemset(d_cache_v, 0, cache_elements(config) * sizeof(*d_cache_v)));
  CHECK_CUDA(cudaMemcpy(d_q, q.data(), q.size() * sizeof(q[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k, k.data(), k.size() * sizeof(k[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_v, v.data(), v.size() * sizeof(v[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_q_weight, q_weight.data(),
                        q_weight.size() * sizeof(q_weight[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k_weight, k_weight.data(),
                        k_weight.size() * sizeof(k_weight[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_page_table, page_table.data(),
                        page_table.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_token_position, token_position.data(),
                        token_position.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_cos, cos.data(), cos.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_sin, sin.data(), sin.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(gemma4_flash_attention_sliding_decode_prepare_q_paged_kv_bf16(
      d_q_prepared, d_cache_k, d_cache_v, config, d_page_table,
      d_token_position, batch_size, layer, d_q, d_k, d_v, d_q_weight,
      d_k_weight, d_cos, d_sin, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> actual_q(q.size());
  std::vector<__nv_bfloat16> actual_cache_k(cache_elements(config));
  std::vector<__nv_bfloat16> actual_cache_v(cache_elements(config));
  CHECK_CUDA(cudaMemcpy(actual_q.data(), d_q_prepared,
                        actual_q.size() * sizeof(actual_q[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_cache_k.data(), d_cache_k,
                        actual_cache_k.size() * sizeof(actual_cache_k[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_cache_v.data(), d_cache_v,
                        actual_cache_v.size() * sizeof(actual_cache_v[0]),
                        cudaMemcpyDeviceToHost));
  compare_bf16(actual_q, expected_q, 0.03125f, "sliding decode prepared Q");
  compare_bf16(actual_cache_k, expected_cache_k, 0.03125f,
               "sliding decode cache K");
  compare_bf16(actual_cache_v, expected_cache_v, 0.015625f,
               "sliding decode cache V");

  CHECK_CUDA(cudaFree(d_sin));
  CHECK_CUDA(cudaFree(d_cos));
  CHECK_CUDA(cudaFree(d_token_position));
  CHECK_CUDA(cudaFree(d_page_table));
  CHECK_CUDA(cudaFree(d_k_weight));
  CHECK_CUDA(cudaFree(d_q_weight));
  CHECK_CUDA(cudaFree(d_cache_v));
  CHECK_CUDA(cudaFree(d_cache_k));
  CHECK_CUDA(cudaFree(d_q_prepared));
  CHECK_CUDA(cudaFree(d_v));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_q));
}

void run_global_decode_prep_cache_case() {
  Gemma4KvCacheConfig config = {
      2,
      12,
      4,
      4,
      GEMMA4_GLOBAL_KV_HEADS,
      GEMMA4_GLOBAL_HEAD_DIM,
      0,
  };
  int batch_size = 2;
  int layer = 1;
  std::vector<int32_t> token_position = {5, 11};
  std::vector<int32_t> page_table(batch_size * config.max_pages_per_seq, -1);
  page_table[gemma4_kv_cache_page_slot(config, token_position[0])] = 3;
  page_table[config.max_pages_per_seq +
             gemma4_kv_cache_page_slot(config, token_position[1])] = 7;

  std::vector<__nv_bfloat16> q(batch_size * GEMMA4_NUM_QUERY_HEADS *
                               GEMMA4_GLOBAL_HEAD_DIM);
  std::vector<__nv_bfloat16> k(batch_size * GEMMA4_GLOBAL_KV_HEADS *
                               GEMMA4_GLOBAL_HEAD_DIM);
  std::vector<__nv_bfloat16> q_weight(GEMMA4_GLOBAL_HEAD_DIM);
  std::vector<__nv_bfloat16> k_weight(GEMMA4_GLOBAL_HEAD_DIM);
  std::vector<float> cos((token_position[1] + 1) *
                         (GEMMA4_GLOBAL_HEAD_DIM / 8));
  std::vector<float> sin(cos.size());

  for (int i = 0; i < static_cast<int>(q.size()); ++i) {
    q[i] = make_value(42000 + i);
  }
  for (int i = 0; i < static_cast<int>(k.size()); ++i) {
    k[i] = make_value(52000 + i);
  }
  for (int d = 0; d < GEMMA4_GLOBAL_HEAD_DIM; ++d) {
    q_weight[d] = __float2bfloat16_rn(0.80f + 0.001f * float(d % 29));
    k_weight[d] = __float2bfloat16_rn(0.90f - 0.001f * float(d % 31));
  }
  fill_global_rope_tables(cos, sin, token_position[1] + 1);

  std::vector<__nv_bfloat16> expected_q(q.size());
  std::vector<__nv_bfloat16> expected_cache_k(cache_elements(config));
  std::vector<__nv_bfloat16> expected_cache_v(cache_elements(config));
  for (int b = 0; b < batch_size; ++b) {
    int pos = token_position[b];
    int slot = gemma4_kv_cache_page_slot(config, pos);
    int page = page_table[b * config.max_pages_per_seq + slot];
    int page_offset = gemma4_kv_cache_page_offset(config, pos);
    for (int h = 0; h < GEMMA4_NUM_QUERY_HEADS; ++h) {
      reference_global_weighted_rope_head(
          expected_q, global_q_offset(b, h, 0), q,
          global_q_offset(b, h, 0), q_weight, cos, sin, pos);
    }
    for (int h = 0; h < GEMMA4_GLOBAL_KV_HEADS; ++h) {
      int64_t cache_base =
          gemma4_kv_cache_offset(config, layer, page, page_offset, h, 0);
      reference_global_weighted_rope_head(
          expected_cache_k, cache_base, k, global_kv_offset(b, h, 0),
          k_weight, cos, sin, pos);
      reference_global_scale_head(expected_cache_v, cache_base, k,
                                  global_kv_offset(b, h, 0));
    }
  }

  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_k = nullptr;
  __nv_bfloat16 *d_q_prepared = nullptr;
  __nv_bfloat16 *d_cache_k = nullptr;
  __nv_bfloat16 *d_cache_v = nullptr;
  __nv_bfloat16 *d_q_weight = nullptr;
  __nv_bfloat16 *d_k_weight = nullptr;
  int32_t *d_page_table = nullptr;
  int32_t *d_token_position = nullptr;
  float *d_cos = nullptr;
  float *d_sin = nullptr;
  CHECK_CUDA(cudaMalloc(&d_q, q.size() * sizeof(*d_q)));
  CHECK_CUDA(cudaMalloc(&d_k, k.size() * sizeof(*d_k)));
  CHECK_CUDA(cudaMalloc(&d_q_prepared, q.size() * sizeof(*d_q_prepared)));
  CHECK_CUDA(cudaMalloc(&d_cache_k, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMalloc(&d_cache_v, cache_elements(config) * sizeof(*d_cache_v)));
  CHECK_CUDA(cudaMalloc(&d_q_weight, q_weight.size() * sizeof(*d_q_weight)));
  CHECK_CUDA(cudaMalloc(&d_k_weight, k_weight.size() * sizeof(*d_k_weight)));
  CHECK_CUDA(cudaMalloc(&d_page_table, page_table.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_token_position,
                        token_position.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_cos, cos.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_sin, sin.size() * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_q_prepared, 0, q.size() * sizeof(*d_q_prepared)));
  CHECK_CUDA(cudaMemset(d_cache_k, 0, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMemset(d_cache_v, 0, cache_elements(config) * sizeof(*d_cache_v)));
  CHECK_CUDA(cudaMemcpy(d_q, q.data(), q.size() * sizeof(q[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k, k.data(), k.size() * sizeof(k[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_q_weight, q_weight.data(),
                        q_weight.size() * sizeof(q_weight[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_k_weight, k_weight.data(),
                        k_weight.size() * sizeof(k_weight[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_page_table, page_table.data(),
                        page_table.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_token_position, token_position.data(),
                        token_position.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_cos, cos.data(), cos.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_sin, sin.data(), sin.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(gemma4_flash_attention_global_decode_prepare_q_paged_kv_bf16(
      d_q_prepared, d_cache_k, d_cache_v, config, d_page_table,
      d_token_position, batch_size, layer, d_q, d_k, d_q_weight,
      d_k_weight, d_cos, d_sin, 0));
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> actual_q(q.size());
  std::vector<__nv_bfloat16> actual_cache_k(cache_elements(config));
  std::vector<__nv_bfloat16> actual_cache_v(cache_elements(config));
  CHECK_CUDA(cudaMemcpy(actual_q.data(), d_q_prepared,
                        actual_q.size() * sizeof(actual_q[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_cache_k.data(), d_cache_k,
                        actual_cache_k.size() * sizeof(actual_cache_k[0]),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(actual_cache_v.data(), d_cache_v,
                        actual_cache_v.size() * sizeof(actual_cache_v[0]),
                        cudaMemcpyDeviceToHost));
  compare_bf16(actual_q, expected_q, 0.03125f, "global decode prepared Q");
  compare_bf16(actual_cache_k, expected_cache_k, 0.03125f,
               "global decode cache K");
  compare_bf16(actual_cache_v, expected_cache_v, 0.015625f,
               "global decode cache V");

  CHECK_CUDA(cudaFree(d_sin));
  CHECK_CUDA(cudaFree(d_cos));
  CHECK_CUDA(cudaFree(d_token_position));
  CHECK_CUDA(cudaFree(d_page_table));
  CHECK_CUDA(cudaFree(d_k_weight));
  CHECK_CUDA(cudaFree(d_q_weight));
  CHECK_CUDA(cudaFree(d_cache_v));
  CHECK_CUDA(cudaFree(d_cache_k));
  CHECK_CUDA(cudaFree(d_q_prepared));
  CHECK_CUDA(cudaFree(d_k));
  CHECK_CUDA(cudaFree(d_q));
}

struct FlashDecodeCase {
  const char *label;
  bool global;
  int batch_size;
  int seq_len;
  int page_size;
  int max_pages_per_seq;
  int window_size;
  int split_size;
  int extra_num_splits;
  bool stagger_seq_lengths;
};

void run_flash_decode_attention_case(const FlashDecodeCase &test_case) {
  const int kv_heads =
      test_case.global ? GEMMA4_GLOBAL_KV_HEADS : GEMMA4_SLIDING_KV_HEADS;
  const int head_dim =
      test_case.global ? GEMMA4_GLOBAL_HEAD_DIM : GEMMA4_SLIDING_HEAD_DIM;
  const int window_size = test_case.global ? 0 : test_case.window_size;
  Gemma4KvCacheConfig config = {
      1,
      test_case.batch_size * test_case.max_pages_per_seq,
      test_case.page_size,
      test_case.max_pages_per_seq,
      kv_heads,
      head_dim,
      window_size,
  };
  int layer = 0;
  int q_heads = GEMMA4_NUM_QUERY_HEADS;

  std::vector<int32_t> target_seq_lengths(test_case.batch_size,
                                          test_case.seq_len);
  if (test_case.stagger_seq_lengths) {
    const int step = std::max(1, test_case.split_size + 1);
    for (int b = 0; b < test_case.batch_size; ++b) {
      target_seq_lengths[b] = std::max(1, test_case.seq_len - b * step);
    }
  }

  int num_splits = 0;
  for (int batch_seq_len : target_seq_lengths) {
    const int first_key =
        window_size > 0 ? std::max(0, batch_seq_len - window_size) : 0;
    const int key_count = std::max(0, batch_seq_len - first_key);
    num_splits =
        std::max(num_splits, (key_count + test_case.split_size - 1) /
                                 test_case.split_size);
  }
  if (!test_case.global) {
    num_splits = std::max(num_splits,
                          (window_size + test_case.split_size - 1) /
                              test_case.split_size);
  }
  num_splits = std::max(1, num_splits + test_case.extra_num_splits);
  float scale = 1.0f / std::sqrt(float(config.head_dim));

  std::vector<int32_t> page_table(test_case.batch_size *
                                      config.max_pages_per_seq,
                                  -1);
  std::vector<int32_t> slot_logical_pages(page_table.size(), -1);
  std::vector<int32_t> seq_lengths(test_case.batch_size, 0);
  Gemma4KvPageAllocator allocator(config.num_pages);

  __nv_bfloat16 *d_cache_k = nullptr;
  __nv_bfloat16 *d_cache_v = nullptr;
  __nv_bfloat16 *d_one_k = nullptr;
  __nv_bfloat16 *d_one_v = nullptr;
  __nv_bfloat16 *d_q = nullptr;
  __nv_bfloat16 *d_direct_out = nullptr;
  int32_t *d_page_table = nullptr;
  int32_t *d_token_batch = nullptr;
  int32_t *d_token_position = nullptr;
  int32_t *d_seq_lengths = nullptr;
  float *d_partial_m = nullptr;
  float *d_partial_l = nullptr;
  float *d_partial_acc = nullptr;

  const size_t partial_m_bytes =
      gemma4_paged_decode_partial_m_elements(test_case.batch_size, q_heads,
                                             num_splits) *
      sizeof(float);
  const size_t partial_acc_bytes =
      gemma4_paged_decode_partial_acc_elements(
          test_case.batch_size, q_heads, num_splits, config.head_dim) *
      sizeof(float);

  CHECK_CUDA(cudaMalloc(&d_cache_k, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMalloc(&d_cache_v, cache_elements(config) * sizeof(*d_cache_v)));
  CHECK_CUDA(cudaMalloc(&d_one_k,
                        config.num_heads * config.head_dim * sizeof(*d_one_k)));
  CHECK_CUDA(cudaMalloc(&d_one_v,
                        config.num_heads * config.head_dim * sizeof(*d_one_v)));
  CHECK_CUDA(cudaMalloc(&d_q,
                        test_case.batch_size * q_heads * config.head_dim *
                            sizeof(*d_q)));
  CHECK_CUDA(cudaMalloc(&d_direct_out,
                        test_case.batch_size * q_heads * config.head_dim *
                            sizeof(*d_direct_out)));
  CHECK_CUDA(cudaMalloc(&d_page_table, page_table.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_token_batch, sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_token_position, sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_seq_lengths, seq_lengths.size() * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc(&d_partial_m, partial_m_bytes));
  CHECK_CUDA(cudaMalloc(&d_partial_l, partial_m_bytes));
  CHECK_CUDA(cudaMalloc(&d_partial_acc, partial_acc_bytes));
  CHECK_CUDA(cudaMemset(d_cache_k, 0, cache_elements(config) * sizeof(*d_cache_k)));
  CHECK_CUDA(cudaMemset(d_cache_v, 0, cache_elements(config) * sizeof(*d_cache_v)));

  std::vector<__nv_bfloat16> by_pos_k(test_case.batch_size *
                                      test_case.seq_len * config.num_heads *
                                      config.head_dim);
  std::vector<__nv_bfloat16> by_pos_v(by_pos_k.size());
  std::vector<__nv_bfloat16> one_k(config.num_heads * config.head_dim);
  std::vector<__nv_bfloat16> one_v(one_k.size());
  const int k_seed = test_case.global ? 41000 : 17000;
  const int v_seed = test_case.global ? 51000 : 27000;

  for (int b = 0; b < test_case.batch_size; ++b) {
    for (int pos = 0; pos < target_seq_lengths[b]; ++pos) {
      int ensured_page = gemma4_kv_cache_ensure_page(
          page_table, slot_logical_pages, allocator, config,
          test_case.batch_size, b, pos);
      if (ensured_page < 0) {
        std::fprintf(stderr, "%s page allocation failed\n", test_case.label);
        std::exit(1);
      }
      seq_lengths[b] = pos + 1;
      for (int h = 0; h < config.num_heads; ++h) {
        for (int d = 0; d < config.head_dim; ++d) {
          __nv_bfloat16 kv =
              make_value(k_seed + 1000 * b + 101 * pos + 17 * h + d);
          __nv_bfloat16 vv =
              make_value(v_seed + 1000 * b + 101 * pos + 17 * h + d);
          one_k[h * config.head_dim + d] = kv;
          one_v[h * config.head_dim + d] = vv;
          by_pos_k[token_offset(b, pos, h, d, test_case.seq_len,
                                config.num_heads, config.head_dim)] = kv;
          by_pos_v[token_offset(b, pos, h, d, test_case.seq_len,
                                config.num_heads, config.head_dim)] = vv;
        }
      }

      int32_t h_batch = b;
      int32_t h_position = pos;
      CHECK_CUDA(cudaMemcpy(d_page_table, page_table.data(),
                            page_table.size() * sizeof(int32_t),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_token_batch, &h_batch, sizeof(int32_t),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_token_position, &h_position, sizeof(int32_t),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_one_k, one_k.data(),
                            one_k.size() * sizeof(one_k[0]),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(d_one_v, one_v.data(),
                            one_v.size() * sizeof(one_v[0]),
                            cudaMemcpyHostToDevice));
      CHECK_CUDA(gemma4_kv_cache_write_bf16(
          d_cache_k, d_cache_v, config, d_page_table, d_token_batch,
          d_token_position, 1, layer, d_one_k, d_one_v, 0));
    }
  }

  std::vector<__nv_bfloat16> q(test_case.batch_size * q_heads *
                               config.head_dim);
  const int q_seed = test_case.global ? 61000 : 37000;
  for (int i = 0; i < static_cast<int>(q.size()); ++i) {
    q[i] = make_value(q_seed + i);
  }
  std::vector<__nv_bfloat16> expected(q.size());
  reference_decode_attention(expected, q, by_pos_k, by_pos_v, seq_lengths,
                             config, test_case.batch_size, q_heads,
                             test_case.seq_len, scale);

  CHECK_CUDA(cudaMemcpy(d_q, q.data(), q.size() * sizeof(q[0]),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_page_table, page_table.data(),
                        page_table.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_seq_lengths, seq_lengths.data(),
                        seq_lengths.size() * sizeof(int32_t),
                        cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemset(d_partial_m, 0x7f, partial_m_bytes));
  CHECK_CUDA(cudaMemset(d_partial_l, 0x7f, partial_m_bytes));
  CHECK_CUDA(cudaMemset(d_partial_acc, 0x7f, partial_acc_bytes));
  cudaError_t status =
      test_case.global
          ? gemma4_flash_attention_global_decode_paged_bf16(
                d_direct_out, d_partial_m, d_partial_l, d_partial_acc, d_q,
                d_cache_k, d_cache_v, d_page_table, d_seq_lengths, config,
                layer, test_case.batch_size, scale, test_case.split_size,
                num_splits, 0)
          : gemma4_flash_attention_sliding_decode_paged_bf16(
                d_direct_out, d_partial_m, d_partial_l, d_partial_acc, d_q,
                d_cache_k, d_cache_v, d_page_table, d_seq_lengths, config,
                layer, test_case.batch_size, scale, test_case.split_size,
                num_splits, 0);
  CHECK_CUDA(status);
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<__nv_bfloat16> direct(q.size());
  CHECK_CUDA(cudaMemcpy(direct.data(), d_direct_out,
                        direct.size() * sizeof(direct[0]),
                        cudaMemcpyDeviceToHost));

  compare_bf16(direct, expected, test_case.global ? 0.03125f : 0.015625f,
               test_case.label);

  CHECK_CUDA(cudaFree(d_cache_k));
  CHECK_CUDA(cudaFree(d_cache_v));
  CHECK_CUDA(cudaFree(d_one_k));
  CHECK_CUDA(cudaFree(d_one_v));
  CHECK_CUDA(cudaFree(d_q));
  CHECK_CUDA(cudaFree(d_direct_out));
  CHECK_CUDA(cudaFree(d_page_table));
  CHECK_CUDA(cudaFree(d_token_batch));
  CHECK_CUDA(cudaFree(d_token_position));
  CHECK_CUDA(cudaFree(d_seq_lengths));
  CHECK_CUDA(cudaFree(d_partial_m));
  CHECK_CUDA(cudaFree(d_partial_l));
  CHECK_CUDA(cudaFree(d_partial_acc));
}

void run_sliding_flash_decode_invalid_args_case() {
  __nv_bfloat16 *d_bf16 = nullptr;
  float *d_float = nullptr;
  int32_t *d_i32 = nullptr;
  CHECK_CUDA(cudaMalloc(&d_bf16, sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_float, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_i32, sizeof(int32_t)));

  Gemma4KvCacheConfig config = {
      1,
      1,
      64,
      16,
      GEMMA4_SLIDING_KV_HEADS,
      GEMMA4_SLIDING_HEAD_DIM,
      0,
  };
  cudaError_t status = gemma4_flash_attention_sliding_decode_paged_bf16(
      d_bf16, d_float, d_float, d_float, d_bf16, d_bf16, d_bf16, d_i32,
      d_i32, config, 0, 1, 0.25f, 64, 16, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected invalid sliding window config\n");
    std::exit(1);
  }

  config.window_size = 1024;
  status = gemma4_flash_attention_sliding_decode_paged_bf16(
      d_bf16, d_float, d_float, d_float, d_bf16, d_bf16, d_bf16, d_i32,
      d_i32, config, 0, 1, 0.25f, 64, 15, 0);
  if (status != cudaErrorInvalidValue) {
    std::fprintf(stderr, "expected invalid underprovisioned decode splits\n");
    std::exit(1);
  }

  CHECK_CUDA(cudaFree(d_i32));
  CHECK_CUDA(cudaFree(d_float));
  CHECK_CUDA(cudaFree(d_bf16));
}

}  // namespace

int main() {
  run_global_prefill_norm_rope_case();
  run_sliding_decode_prep_cache_case();
  run_global_decode_prep_cache_case();

  run_flash_decode_attention_case(
      {"sliding flash decode short", false, 1, 5, 8, 2, 8, 3, 0, false});
  run_flash_decode_attention_case(
      {"sliding flash decode shorter than split", false, 1, 5, 8, 2, 8, 8, 2,
       false});
  run_flash_decode_attention_case(
      {"sliding flash decode exact splits", false, 1, 8, 4, 3, 8, 4, 0,
       false});
  run_flash_decode_attention_case(
      {"sliding flash decode overprovisioned", false, 1, 5, 8, 2, 8, 3, 3,
       false});
  run_flash_decode_attention_case(
      {"sliding flash decode boundary", false, 2, 10, 4, 3, 8, 3, 0, false});
  run_flash_decode_attention_case(
      {"sliding flash decode varlen overprovisioned", false, 2, 10, 4, 3, 8, 3,
       3, true});
  run_flash_decode_attention_case(
      {"sliding flash decode wrap", false, 1, 13, 4, 3, 8, 5, 0, false});

  run_flash_decode_attention_case(
      {"global flash decode short", true, 1, 5, 8, 2, 0, 3, 0, false});
  run_flash_decode_attention_case(
      {"global flash decode exact splits", true, 1, 8, 4, 2, 0, 4, 0, false});
  run_flash_decode_attention_case(
      {"global flash decode overprovisioned", true, 1, 5, 8, 2, 0, 3, 3,
       false});
  run_flash_decode_attention_case(
      {"global flash decode varlen overprovisioned", true, 2, 10, 4, 3, 0, 3,
       3, true});

  run_sliding_flash_decode_invalid_args_case();
  std::puts("flash decode tests passed");
  return 0;
}
