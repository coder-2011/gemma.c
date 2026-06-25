#include "gemma4_checkpoint.cuh"
#include "gemma4_decode_megakernel.cuh"
#include "gemma4_embedding_gather.cuh"
#include "gemma4_prefill_megakernel.cuh"
#include "gemma4_runtime.cuh"
#include "gemma4_tokenizer.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int32_t kDefaultMaxNewTokens = 1;
constexpr int32_t kDefaultPageSize = 64;

struct PromptOptions {
  std::string checkpoint_path = "models/gemma-4-12B/model.safetensors";
  std::string tokenizer_path = "models/gemma-4-12B/tokenizer.json";
  std::string prompt = "Hello";
  int32_t max_new_tokens = kDefaultMaxNewTokens;
  int32_t page_size = kDefaultPageSize;
};

// Owns a CUDA allocation used by the prompt runner.
template <typename T>
struct DeviceBuffer {
  T *ptr = nullptr;
  size_t count = 0;

  // Releases the device buffer owned by this wrapper.
  ~DeviceBuffer() {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  }

  // Allocates `n` elements and frees any previous allocation first.
  cudaError_t allocate(size_t n) {
    if (ptr != nullptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
    count = n;
    if (n == 0) {
      return cudaSuccess;
    }
    return cudaMalloc(&ptr, n * sizeof(T));
  }

  // Returns the mutable device pointer.
  T *get() { return ptr; }
};

// Releases loaded device weights when leaving the prompt runner.
struct WeightOwner {
  Gemma4TextWeightsDevice value = {};

  // Frees all weight allocations owned by the checkpoint loader.
  ~WeightOwner() { gemma4_text_weights_device_free(&value); }
};

// Releases runtime KV/cache metadata when leaving the prompt runner.
struct RuntimeOwner {
  Gemma4RuntimeState value = {};

  // Frees all CUDA allocations owned by the runtime state.
  ~RuntimeOwner() { gemma4_runtime_state_free(&value); }
};

// Prints one CUDA failure and converts it to a process exit status.
int fail_cuda(cudaError_t status, const char *where) {
  std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(status));
  return 1;
}

// Parses the tiny prompt-runner CLI.
bool parse_args(int argc, char **argv, PromptOptions *options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--checkpoint" && i + 1 < argc) {
      options->checkpoint_path = argv[++i];
    } else if (arg == "--tokenizer" && i + 1 < argc) {
      options->tokenizer_path = argv[++i];
    } else if (arg == "--prompt" && i + 1 < argc) {
      options->prompt = argv[++i];
    } else if (arg == "--max-new" && i + 1 < argc) {
      options->max_new_tokens = std::atoi(argv[++i]);
    } else if (arg == "--page-size" && i + 1 < argc) {
      options->page_size = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: %s [--checkpoint path] [--tokenizer path] "
                   "[--prompt text] [--max-new n] [--page-size n]\n",
                   argv[0]);
      return false;
    }
  }
  return options->max_new_tokens > 0 && options->page_size > 0;
}

// Runs every transformer layer over the prompt tokens and fills the KV cache.
cudaError_t run_prefill_layers(
    const Gemma4TextWeightsDevice &weights,
    Gemma4RuntimeState *runtime,
    __nv_bfloat16 *hidden_a,
    __nv_bfloat16 *hidden_b,
    __nv_bfloat16 *scratch_buffer,
    int32_t seq_len,
    cudaStream_t stream,
    __nv_bfloat16 **final_hidden) {
  __nv_bfloat16 *hidden_in = hidden_a;
  __nv_bfloat16 *hidden_out = hidden_b;
  for (int32_t layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
    const bool global = gemma4_is_global_layer(layer);
    Gemma4PrefillMegakernelLayerScratch scratch =
        gemma4_prefill_megakernel_layer_scratch_from_buffer(
            scratch_buffer, global, seq_len);

    Gemma4PrefillMegakernelLayerArgs args = {};
    args.out = hidden_out;
    args.hidden = hidden_in;
    args.weights = &weights.layers[layer];
    args.layer_index = layer;
    args.batch_size = 1;
    args.seq_len = seq_len;
    args.cos = global ? runtime->global_cos : runtime->sliding_cos;
    args.sin = global ? runtime->global_sin : runtime->sliding_sin;
    args.softmax_scale = 1.0f;
    args.cache_k = global ? runtime->global_cache_k : runtime->sliding_cache_k;
    args.cache_v = global ? runtime->global_cache_v : runtime->sliding_cache_v;
    args.cache_config =
        global ? runtime->global_cache_config : runtime->sliding_cache_config;
    args.page_table =
        global ? runtime->global_page_table : runtime->sliding_page_table;
    args.token_batch = runtime->token_batch;
    args.token_position = runtime->token_position;
    args.stream = stream;

    cudaError_t status = gemma4_prefill_megakernel_layer_bf16(args, scratch);
    if (status != cudaSuccess) {
      return status;
    }
    std::swap(hidden_in, hidden_out);
  }
  *final_hidden = hidden_in;
  return cudaSuccess;
}

// Samples the first generated token from the final prefill row.
cudaError_t sample_from_prefill(
    const Gemma4TextWeightsDevice &weights,
    __nv_bfloat16 *prefill_final_rows,
    int32_t seq_len,
    __nv_bfloat16 *decode_hidden,
    int32_t *next_token,
    void *scratch,
    size_t scratch_bytes,
    cudaStream_t stream) {
  const size_t row_bytes =
      static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * sizeof(__nv_bfloat16);
  const __nv_bfloat16 *last_row =
      prefill_final_rows + static_cast<int64_t>(seq_len - 1) * GEMMA4_HIDDEN_SIZE;
  cudaError_t status =
      cudaMemcpyAsync(decode_hidden, last_row, row_bytes,
                      cudaMemcpyDeviceToDevice, stream);
  if (status != cudaSuccess) {
    return status;
  }

  Gemma4DecodeMegakernelSpineArgs args = {};
  args.state = decode_hidden;
  args.next_hidden = decode_hidden;
  args.next_token = next_token;
  args.final_norm_weight = weights.final_norm_weight;
  args.lm_head_col_major = weights.token_embedding;
  return gemma4_decode_megakernel_spine_bf16(
      args, scratch, scratch_bytes, stream);
}

// Fills the per-layer decode megakernel arguments for one layer.
Gemma4DecodeMegakernelFfnTailArgs make_decode_args(
    const Gemma4TextWeightsDevice &weights,
    Gemma4RuntimeState *runtime,
    int32_t layer,
    __nv_bfloat16 *hidden_in,
    __nv_bfloat16 *hidden_out,
    __nv_bfloat16 *normed,
    __nv_bfloat16 *sampled_hidden,
    int32_t *next_token,
    __nv_bfloat16 *attention_q,
    __nv_bfloat16 *attention_out,
    float *partial_m,
    float *partial_l,
    float *partial_acc,
    int32_t split_size,
    int32_t sliding_splits,
    int32_t global_splits) {
  const bool global = gemma4_is_global_layer(layer);
  const Gemma4TextLayerWeightsDevice &w = weights.layers[layer];

  Gemma4DecodeMegakernelFfnTailArgs args = {};
  args.residual_out = hidden_out;
  args.normed_out = normed;
  args.next_hidden = sampled_hidden;
  args.next_token = next_token;
  args.ffn_x = normed;
  args.ffn_residual = hidden_out;
  args.ffn_norm_weight = w.post_feedforward_norm_weight;
  args.ffn_gate_up_decode = w.ffn_gate_up_decode;
  args.ffn_down_decode = w.ffn_down_decode;
  args.layer_scalar = w.layer_scalar;
  args.final_norm_weight = weights.final_norm_weight;
  args.lm_head_col_major = weights.token_embedding;
  args.flags = GEMMA4_DECODE_MEGAKERNEL_FLAG_FLASH_ATTENTION;

  args.attention_q = attention_q;
  args.attention_out = attention_out;
  args.attention_partial_m = partial_m;
  args.attention_partial_l = partial_l;
  args.attention_partial_acc = partial_acc;
  args.attention_cache_k =
      global ? runtime->global_cache_k : runtime->sliding_cache_k;
  args.attention_cache_v =
      global ? runtime->global_cache_v : runtime->sliding_cache_v;
  args.attention_cache_config =
      global ? runtime->global_cache_config : runtime->sliding_cache_config;
  args.attention_page_table =
      global ? runtime->global_page_table : runtime->sliding_page_table;
  args.attention_token_position = runtime->token_position;
  args.attention_seq_lengths = runtime->seq_lengths;
  args.attention_cache_layer = gemma4_kv_cache_layer_index(layer, global);
  args.attention_split_size = split_size;
  args.attention_num_splits = global ? global_splits : sliding_splits;
  args.attention_softmax_scale = 1.0f;
  args.attention_x = hidden_in;
  args.attention_input_norm_weight = w.input_norm_weight;
  args.attention_weights = {w.q_proj_col_major, w.k_proj_col_major,
                            w.v_proj_col_major, 0, 0, 0};
  args.attention_o_proj_col_major = w.o_proj_col_major;
  args.attention_post_norm_weight = w.post_attention_norm_weight;
  args.attention_pre_ffn_norm_weight = w.pre_feedforward_norm_weight;
  args.attention_q_norm_weight = w.q_norm_weight;
  args.attention_k_norm_weight = w.k_norm_weight;
  args.attention_cos = global ? runtime->global_cos : runtime->sliding_cos;
  args.attention_sin = global ? runtime->global_sin : runtime->sliding_sin;
  args.eps = GEMMA4_RMS_NORM_EPS;
  return args;
}

// Runs all decode layers for one already-sampled token embedding.
cudaError_t run_decode_step(
    const Gemma4TextWeightsDevice &weights,
    Gemma4RuntimeState *runtime,
    __nv_bfloat16 *decode_hidden_a,
    __nv_bfloat16 *decode_hidden_b,
    __nv_bfloat16 *normed,
    __nv_bfloat16 *sampled_hidden,
    int32_t *next_token,
    __nv_bfloat16 *attention_q,
    __nv_bfloat16 *attention_out,
    float *partial_m,
    float *partial_l,
    float *partial_acc,
    void *scratch,
    size_t scratch_bytes,
    int32_t split_size,
    int32_t sliding_splits,
    int32_t global_splits,
    cudaStream_t stream) {
  cudaError_t status = gemma4_runtime_prepare_decode_step(runtime, stream);
  if (status != cudaSuccess) {
    return status;
  }

  __nv_bfloat16 *hidden_in = decode_hidden_a;
  __nv_bfloat16 *hidden_out = decode_hidden_b;
  for (int32_t layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
    Gemma4DecodeMegakernelFfnTailArgs args = make_decode_args(
        weights, runtime, layer, hidden_in, hidden_out, normed,
        sampled_hidden, next_token, attention_q, attention_out, partial_m,
        partial_l, partial_acc, split_size, sliding_splits, global_splits);
    status = gemma4_decode_megakernel_ffn_tail_bf16(
        args, scratch, scratch_bytes, stream);
    if (status != cudaSuccess) {
      return status;
    }
    std::swap(hidden_in, hidden_out);
  }

  const size_t row_bytes =
      static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * sizeof(__nv_bfloat16);
  return cudaMemcpyAsync(
      decode_hidden_a, sampled_hidden, row_bytes, cudaMemcpyDeviceToDevice,
      stream);
}

// Copies one sampled token from device to host.
cudaError_t copy_next_token(
    int32_t *d_next_token,
    std::vector<int32_t> *generated,
    cudaStream_t stream) {
  int32_t token = -1;
  cudaError_t status =
      cudaMemcpyAsync(&token, d_next_token, sizeof(token),
                      cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    return status;
  }
  generated->push_back(token);
  return cudaSuccess;
}

// Runs tokenization, prefill, decode, and detokenization for one prompt.
int run_prompt(const PromptOptions &options) {
  Gemma4Tokenizer tokenizer;
  std::string tokenizer_error;
  if (!tokenizer.load(options.tokenizer_path, &tokenizer_error)) {
    std::fprintf(stderr, "load tokenizer failed: %s\n",
                 tokenizer_error.c_str());
    return 1;
  }

  std::vector<int32_t> prompt_tokens;
  if (!tokenizer.encode(options.prompt, &prompt_tokens) ||
      prompt_tokens.empty()) {
    std::fprintf(stderr, "tokenization failed\n");
    return 1;
  }
  const int32_t prompt_len = static_cast<int32_t>(prompt_tokens.size());
  const int32_t max_seq_len = prompt_len + options.max_new_tokens;
  std::printf("prompt tokens: %d\n", prompt_len);

  WeightOwner weights;
  std::string error;
  cudaError_t status = gemma4_load_text_weights_device_bf16(
      &weights.value, options.checkpoint_path.c_str(), &error);
  if (status != cudaSuccess) {
    if (!error.empty()) {
      std::fprintf(stderr, "%s\n", error.c_str());
    }
    return fail_cuda(status, "load weights");
  }

  RuntimeOwner runtime;
  status = gemma4_runtime_state_init(
      &runtime.value, 1, max_seq_len, options.page_size, 0);
  if (status != cudaSuccess) {
    return fail_cuda(status, "runtime init");
  }
  status = gemma4_runtime_prepare_prefill(&runtime.value, prompt_len, 0);
  if (status != cudaSuccess) {
    return fail_cuda(status, "runtime prefill metadata");
  }

  DeviceBuffer<int32_t> d_prompt_tokens;
  DeviceBuffer<int32_t> d_next_token;
  DeviceBuffer<__nv_bfloat16> d_prefill_a;
  DeviceBuffer<__nv_bfloat16> d_prefill_b;
  DeviceBuffer<__nv_bfloat16> d_prefill_scratch;
  DeviceBuffer<__nv_bfloat16> d_decode_a;
  DeviceBuffer<__nv_bfloat16> d_decode_b;
  DeviceBuffer<__nv_bfloat16> d_normed;
  DeviceBuffer<__nv_bfloat16> d_sampled;
  DeviceBuffer<__nv_bfloat16> d_attention_q;
  DeviceBuffer<__nv_bfloat16> d_attention_out;
  DeviceBuffer<float> d_partial_m;
  DeviceBuffer<float> d_partial_l;
  DeviceBuffer<float> d_partial_acc;
  DeviceBuffer<unsigned char> d_spine_scratch;
  DeviceBuffer<unsigned char> d_decode_scratch;

  status = d_prompt_tokens.allocate(prompt_tokens.size());
  if (status != cudaSuccess) return fail_cuda(status, "alloc prompt tokens");
  status = d_next_token.allocate(1);
  if (status != cudaSuccess) return fail_cuda(status, "alloc next token");

  const size_t prompt_hidden_elements =
      static_cast<size_t>(prompt_len) * GEMMA4_HIDDEN_SIZE;
  status = d_prefill_a.allocate(prompt_hidden_elements);
  if (status != cudaSuccess) return fail_cuda(status, "alloc prefill a");
  status = d_prefill_b.allocate(prompt_hidden_elements);
  if (status != cudaSuccess) return fail_cuda(status, "alloc prefill b");
  const size_t prefill_scratch_elements = std::max(
      gemma4_prefill_megakernel_layer_scratch_elements(false, prompt_len),
      gemma4_prefill_megakernel_layer_scratch_elements(true, prompt_len));
  status = d_prefill_scratch.allocate(prefill_scratch_elements);
  if (status != cudaSuccess) return fail_cuda(status, "alloc prefill scratch");

  status = d_decode_a.allocate(GEMMA4_HIDDEN_SIZE);
  if (status != cudaSuccess) return fail_cuda(status, "alloc decode a");
  status = d_decode_b.allocate(GEMMA4_HIDDEN_SIZE);
  if (status != cudaSuccess) return fail_cuda(status, "alloc decode b");
  status = d_normed.allocate(GEMMA4_HIDDEN_SIZE);
  if (status != cudaSuccess) return fail_cuda(status, "alloc normed");
  status = d_sampled.allocate(GEMMA4_HIDDEN_SIZE);
  if (status != cudaSuccess) return fail_cuda(status, "alloc sampled");
  status = d_attention_q.allocate(GEMMA4_GLOBAL_Q_PROJ_SIZE);
  if (status != cudaSuccess) return fail_cuda(status, "alloc attention q");
  status = d_attention_out.allocate(GEMMA4_GLOBAL_ATTENTION_OUT_SIZE);
  if (status != cudaSuccess) return fail_cuda(status, "alloc attention out");

  const int32_t split_size = GEMMA4_SLIDING_DECODE_SPLIT_SIZE;
  const int32_t sliding_splits =
      (GEMMA4_SLIDING_WINDOW + split_size - 1) / split_size;
  const int32_t global_keys =
      runtime.value.global_cache_config.max_pages_per_seq *
      runtime.value.global_cache_config.page_size;
  const int32_t global_splits = (global_keys + split_size - 1) / split_size;
  const int32_t max_splits = std::max(sliding_splits, global_splits);
  status = d_partial_m.allocate(GEMMA4_NUM_QUERY_HEADS * max_splits);
  if (status != cudaSuccess) return fail_cuda(status, "alloc partial m");
  status = d_partial_l.allocate(GEMMA4_NUM_QUERY_HEADS * max_splits);
  if (status != cudaSuccess) return fail_cuda(status, "alloc partial l");
  status = d_partial_acc.allocate(
      static_cast<size_t>(GEMMA4_NUM_QUERY_HEADS) * max_splits *
      GEMMA4_GLOBAL_HEAD_DIM);
  if (status != cudaSuccess) return fail_cuda(status, "alloc partial acc");
  status = d_spine_scratch.allocate(
      gemma4_decode_megakernel_spine_scratch_bytes());
  if (status != cudaSuccess) return fail_cuda(status, "alloc spine scratch");
  status = d_decode_scratch.allocate(
      gemma4_decode_megakernel_ffn_tail_scratch_bytes());
  if (status != cudaSuccess) return fail_cuda(status, "alloc decode scratch");

  status = cudaMemcpyAsync(
      d_prompt_tokens.get(), prompt_tokens.data(),
      prompt_tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice, 0);
  if (status != cudaSuccess) return fail_cuda(status, "copy prompt tokens");
  status = gemma4_embedding_gather_bf16(
      d_prefill_a.get(), d_prompt_tokens.get(), weights.value.token_embedding,
      prompt_len, 0);
  if (status != cudaSuccess) return fail_cuda(status, "embedding gather");

  __nv_bfloat16 *prefill_final = nullptr;
  status = run_prefill_layers(
      weights.value, &runtime.value, d_prefill_a.get(), d_prefill_b.get(),
      d_prefill_scratch.get(), prompt_len, 0, &prefill_final);
  if (status != cudaSuccess) {
    return fail_cuda(status, "prefill layers");
  }

  status = sample_from_prefill(
      weights.value, prefill_final, prompt_len, d_decode_a.get(),
      d_next_token.get(), d_spine_scratch.get(),
      gemma4_decode_megakernel_spine_scratch_bytes(), 0);
  if (status != cudaSuccess) {
    return fail_cuda(status, "sample from prefill");
  }

  std::vector<int32_t> generated;
  status = copy_next_token(d_next_token.get(), &generated, 0);
  if (status != cudaSuccess) {
    return fail_cuda(status, "copy first token");
  }

  while (static_cast<int32_t>(generated.size()) < options.max_new_tokens) {
    status = run_decode_step(
        weights.value, &runtime.value, d_decode_a.get(), d_decode_b.get(),
        d_normed.get(), d_sampled.get(), d_next_token.get(),
        d_attention_q.get(), d_attention_out.get(), d_partial_m.get(),
        d_partial_l.get(), d_partial_acc.get(), d_decode_scratch.get(),
        gemma4_decode_megakernel_ffn_tail_scratch_bytes(), split_size,
        sliding_splits, global_splits, 0);
    if (status != cudaSuccess) {
      return fail_cuda(status, "decode step");
    }
    status = copy_next_token(d_next_token.get(), &generated, 0);
    if (status != cudaSuccess) {
      return fail_cuda(status, "copy decode token");
    }
  }

  std::string decoded;
  if (!tokenizer.decode(generated, &decoded)) {
    std::fprintf(stderr, "detokenization failed\n");
    return 1;
  }
  std::printf("generated token ids:");
  for (int32_t token : generated) {
    std::printf(" %d", token);
  }
  std::printf("\ntext: %s\n", decoded.c_str());
  return 0;
}

}  // namespace

// Runs the local Gemma 4 text prompt path.
int main(int argc, char **argv) {
  PromptOptions options;
  if (!parse_args(argc, argv, &options)) {
    return 1;
  }
  return run_prompt(options);
}
