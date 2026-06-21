#ifndef GEMMA4_PREFILL_MEGAKERNEL_CUH
#define GEMMA4_PREFILL_MEGAKERNEL_CUH

#include "gemma4_checkpoint.cuh"
#include "gemma4_ffn.cuh"
#include "gemma4_kv_cache.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
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
} Gemma4PrefillMegakernelLayerScratch;

typedef struct {
  __nv_bfloat16 *out = nullptr;
  const __nv_bfloat16 *hidden = nullptr;
  const Gemma4TextLayerWeightsDevice *weights = nullptr;
  int32_t layer_index = 0;
  int32_t batch_size = 0;
  int32_t seq_len = 0;
  const float *cos = nullptr;
  const float *sin = nullptr;
  float softmax_scale = 0.0f;

  __nv_bfloat16 *cache_k = nullptr;
  __nv_bfloat16 *cache_v = nullptr;
  Gemma4KvCacheConfig cache_config = {};
  const int32_t *page_table = nullptr;
  const int32_t *token_batch = nullptr;
  const int32_t *token_position = nullptr;
  int32_t cache_layer = 0;

  float eps = GEMMA4_RMS_NORM_EPS;
  cudaStream_t stream = nullptr;
} Gemma4PrefillMegakernelLayerArgs;

// Returns the BF16 scratch elements needed for one prefill layer runner call.
size_t gemma4_prefill_megakernel_layer_scratch_elements(bool global,
                                                        int32_t rows);

// Splits one aligned BF16 scratch buffer into the layer runner work tensors.
Gemma4PrefillMegakernelLayerScratch
gemma4_prefill_megakernel_layer_scratch_from_buffer(
    __nv_bfloat16 *buffer,
    bool global,
    int32_t rows);

// Runs one Gemma 4 prefill transformer layer using the existing host APIs.
cudaError_t gemma4_prefill_megakernel_layer_bf16(
    const Gemma4PrefillMegakernelLayerArgs &args,
    const Gemma4PrefillMegakernelLayerScratch &scratch);

#endif
