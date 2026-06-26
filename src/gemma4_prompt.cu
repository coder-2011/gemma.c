#include "gemma4_checkpoint.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_decode_megakernel.cuh"
#include "gemma4_embedding_gather.cuh"
#include "gemma4_prefill_megakernel.cuh"
#include "gemma4_runtime.cuh"
#include "gemma4_tokenizer.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

namespace {

// Returns the raw CUDA pointer owned by a Thrust device vector.
template <typename T>
T *raw_ptr(thrust::device_vector<T> &v) {
  return thrust::raw_pointer_cast(v.data());
}

// Returns the raw const CUDA pointer owned by a Thrust device vector.
template <typename T>
const T *raw_ptr(const thrust::device_vector<T> &v) {
  return thrust::raw_pointer_cast(v.data());
}

constexpr int32_t kDefaultMaxNewTokens = 1;
constexpr int32_t kDefaultPageSize = 64;
constexpr int32_t kDefaultBenchWarmup = 1;
constexpr int32_t kDefaultBenchIters = 3;
constexpr int32_t kDefaultBenchSamples = 3;

using PromptClock = std::chrono::steady_clock;

enum class PromptBenchmarkMode {
  kNone,
  kDecodeStep,
  kWarmServing,
  kColdStart,
};

struct PromptOptions {
  std::string checkpoint_path = "models/gemma-4-12B/model.safetensors";
  std::string tokenizer_path = "models/gemma-4-12B/tokenizer.json";
  std::string prompt = "Hello";
  int32_t max_new_tokens = kDefaultMaxNewTokens;
  int32_t page_size = kDefaultPageSize;
  PromptBenchmarkMode benchmark_mode = PromptBenchmarkMode::kNone;
  int32_t bench_warmup = kDefaultBenchWarmup;
  int32_t bench_iters = kDefaultBenchIters;
  int32_t bench_samples = kDefaultBenchSamples;
};

struct PromptBenchmarkStats {
  int32_t count = 0;
  float mean_ms = 0.0f;
  float p50_ms = 0.0f;
  float p90_ms = 0.0f;
  float p95_ms = 0.0f;
  float p99_ms = 0.0f;
  float min_ms = 0.0f;
  float max_ms = 0.0f;
};

struct PromptRequestTiming {
  float ttft_ms = 0.0f;
  float e2e_ms = 0.0f;
  float tpot_ms = 0.0f;
  float per_user_tps = 0.0f;
  std::vector<float> itl_ms;
};

// Releases loaded device weights when leaving the prompt runner.
struct WeightOwner {
  Gemma4TextWeightsDevice value = {};

  // Frees all weight allocations owned by the checkpoint loader.
  ~WeightOwner() {
    for (__nv_bfloat16 *ptr : {value.token_embedding, value.final_norm_weight}) cudaFree(ptr);
    for (int layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
      Gemma4TextLayerWeightsDevice &w = value.layers[layer];
      for (__nv_bfloat16 *ptr : {
          w.input_norm_weight, w.post_attention_norm_weight,
          w.pre_feedforward_norm_weight, w.post_feedforward_norm_weight,
          w.layer_scalar, w.q_norm_weight, w.k_norm_weight, w.q_proj_col_major,
          w.k_proj_col_major, w.v_proj_col_major, w.o_proj_col_major,
          w.ffn_gate_up_decode, w.ffn_down_decode}) {
        cudaFree(ptr);
      }
    }
  }
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

// Converts a benchmark mode to the stable CLI/reporting name.
const char *benchmark_mode_name(PromptBenchmarkMode mode) {
  switch (mode) {
    case PromptBenchmarkMode::kNone:
      return "none";
    case PromptBenchmarkMode::kDecodeStep:
      return "decode-step";
    case PromptBenchmarkMode::kWarmServing:
      return "warm-serving";
    case PromptBenchmarkMode::kColdStart:
      return "cold-start";
  }
  return "unknown";
}

// Parses the explicit benchmark regime requested by the CLI.
bool parse_benchmark_mode(const std::string &value, PromptBenchmarkMode *mode) {
  if (value == "decode-step") {
    *mode = PromptBenchmarkMode::kDecodeStep;
    return true;
  }
  if (value == "warm-serving") {
    *mode = PromptBenchmarkMode::kWarmServing;
    return true;
  }
  if (value == "cold-start") {
    *mode = PromptBenchmarkMode::kColdStart;
    return true;
  }
  if (value == "none") {
    *mode = PromptBenchmarkMode::kNone;
    return true;
  }
  return false;
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
    } else if (arg == "--benchmark") {
      options->benchmark_mode = PromptBenchmarkMode::kDecodeStep;
    } else if (arg == "--benchmark-mode" && i + 1 < argc) {
      if (!parse_benchmark_mode(argv[++i], &options->benchmark_mode)) {
        return false;
      }
    } else if (arg == "--bench-warmup" && i + 1 < argc) {
      options->bench_warmup = std::atoi(argv[++i]);
    } else if (arg == "--bench-iters" && i + 1 < argc) {
      options->bench_iters = std::atoi(argv[++i]);
    } else if (arg == "--bench-samples" && i + 1 < argc) {
      options->bench_samples = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr,
                   "usage: %s [--checkpoint path] [--tokenizer path] "
                   "[--prompt text] [--max-new n] [--page-size n] "
                   "[--benchmark] [--benchmark-mode none|decode-step|"
                   "warm-serving|cold-start] [--bench-warmup n] "
                   "[--bench-iters n] [--bench-samples n]\n",
                   argv[0]);
      return false;
    }
  }
  return options->max_new_tokens > 0 && options->page_size > 0 &&
         options->bench_warmup >= 0 && options->bench_iters > 0 &&
         options->bench_samples > 0;
}

// Returns the nearest-rank percentile from sorted millisecond samples.
float percentile_ms(const std::vector<float> &sorted, float pct) {
  if (sorted.empty()) {
    return 0.0f;
  }
  const float scaled = (pct / 100.0f) * static_cast<float>(sorted.size() - 1);
  size_t index = static_cast<size_t>(scaled + 0.5f);
  if (index >= sorted.size()) {
    index = sorted.size() - 1;
  }
  return sorted[index];
}

// Summarizes raw milliseconds as mean, percentiles, min, and max.
PromptBenchmarkStats summarize_benchmark_samples(std::vector<float> samples) {
  if (samples.empty()) {
    return {};
  }

  float total_ms = 0.0f;
  for (float sample : samples) {
    total_ms += sample;
  }
  std::sort(samples.begin(), samples.end());
  PromptBenchmarkStats stats = {};
  stats.count = static_cast<int32_t>(samples.size());
  stats.mean_ms = total_ms / static_cast<float>(samples.size());
  stats.min_ms = samples.front();
  stats.p50_ms = percentile_ms(samples, 50.0f);
  stats.p90_ms = percentile_ms(samples, 90.0f);
  stats.p95_ms = percentile_ms(samples, 95.0f);
  stats.p99_ms = percentile_ms(samples, 99.0f);
  stats.max_ms = samples.back();
  return stats;
}

// Prints a one-line latency distribution with stable field names.
void print_stats_line(const char *name, const PromptBenchmarkStats &stats) {
  std::printf(
      "%s n=%d mean_ms=%.3f p50_ms=%.3f p90_ms=%.3f p95_ms=%.3f "
      "p99_ms=%.3f min_ms=%.3f max_ms=%.3f\n",
      name, stats.count, stats.mean_ms, stats.p50_ms, stats.p90_ms,
      stats.p95_ms, stats.p99_ms, stats.min_ms, stats.max_ms);
}

// Prints a one-line throughput distribution with stable field names.
void print_rate_stats_line(const char *name, const PromptBenchmarkStats &stats) {
  std::printf(
      "%s n=%d mean=%.3f p50=%.3f p90=%.3f p95=%.3f p99=%.3f "
      "min=%.3f max=%.3f\n",
      name, stats.count, stats.mean_ms, stats.p50_ms, stats.p90_ms,
      stats.p95_ms, stats.p99_ms, stats.min_ms, stats.max_ms);
}

// Returns elapsed host-visible milliseconds between two steady-clock timestamps.
float host_elapsed_ms(PromptClock::time_point start,
                      PromptClock::time_point stop) {
  return std::chrono::duration<float, std::milli>(stop - start).count();
}

// Prints the CUDA environment fields this executable can query directly.
cudaError_t print_cuda_environment() {
  int device = 0;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  cudaDeviceProp prop = {};
  status = cudaGetDeviceProperties(&prop, device);
  if (status != cudaSuccess) {
    return status;
  }

  int driver = 0;
  status = cudaDriverGetVersion(&driver);
  if (status != cudaSuccess) {
    return status;
  }

  int runtime = 0;
  status = cudaRuntimeGetVersion(&runtime);
  if (status != cudaSuccess) {
    return status;
  }

  std::printf(
      "environment gpu=\"%s\" sm=%d.%d memory_bytes=%zu cuda_driver=%d "
      "cuda_runtime=%d\n",
      prop.name, prop.major, prop.minor, prop.totalGlobalMem, driver, runtime);
  return cudaSuccess;
}

// Times stateful decode steps with CUDA events on the measured stream.
template <typename Fn>
cudaError_t measure_cuda_event_steps_ms(
    PromptBenchmarkStats *stats,
    int32_t warmup,
    int32_t iters,
    int32_t samples,
    cudaStream_t stream,
    Fn fn) {
  const int32_t timed_steps = iters * samples;
  std::vector<float> step_ms;
  step_ms.reserve(timed_steps);

  for (int32_t i = 0; i < warmup; ++i) {
    cudaError_t status = fn();
    if (status != cudaSuccess) {
      return status;
    }
  }

  cudaError_t status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    return status;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  status = cudaEventCreate(&start);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaEventCreate(&stop);
  if (status != cudaSuccess) {
    cudaEventDestroy(start);
    return status;
  }

  for (int32_t i = 0; i < timed_steps; ++i) {
    status = cudaEventRecord(start);
    if (status != cudaSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return status;
    }
    status = fn();
    if (status != cudaSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return status;
    }
    status = cudaEventRecord(stop);
    if (status != cudaSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return status;
    }
    status = cudaEventSynchronize(stop);
    if (status != cudaSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return status;
    }

    float total_ms = 0.0f;
    status = cudaEventElapsedTime(&total_ms, start, stop);
    if (status != cudaSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return status;
    }
    step_ms.push_back(total_ms);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  *stats = summarize_benchmark_samples(step_ms);
  return cudaSuccess;
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
    const Gemma4PrefillMegakernelLayerScratch scratch =
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
    const Gemma4DecodeMegakernelFfnTailArgs args = make_decode_args(
        weights, runtime, layer, hidden_in, hidden_out, normed,
        sampled_hidden, next_token, attention_q, attention_out, partial_m,
        partial_l, partial_acc, split_size, sliding_splits, global_splits);
    const bool final_layer = layer == GEMMA4_NUM_LAYERS - 1;
    status = final_layer
                 ? gemma4_decode_megakernel_ffn_tail_bf16(
                       args, scratch, scratch_bytes, stream)
                 : gemma4_decode_megakernel_attention_ffn_bf16(
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
                      cudaMemcpyDeviceToHost);
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
int run_prompt(const PromptOptions &options,
               PromptClock::time_point process_start) {
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
  const bool benchmark_decode =
      options.benchmark_mode == PromptBenchmarkMode::kDecodeStep;
  const int32_t benchmark_decode_steps =
      benchmark_decode ? options.bench_warmup +
                             options.bench_samples * options.bench_iters
                       : 0;
  const int32_t max_seq_len =
      prompt_len + std::max(options.max_new_tokens, benchmark_decode_steps);
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
  thrust::device_vector<int32_t> d_prompt_tokens(prompt_tokens.size());
  thrust::device_vector<int32_t> d_next_token(1);

  const size_t prompt_hidden_elements =
      static_cast<size_t>(prompt_len) * GEMMA4_HIDDEN_SIZE;
  thrust::device_vector<__nv_bfloat16> d_prefill_a(prompt_hidden_elements);
  thrust::device_vector<__nv_bfloat16> d_prefill_b(prompt_hidden_elements);
  const size_t prefill_scratch_elements = std::max(
      gemma4_prefill_megakernel_layer_scratch_elements(false, prompt_len),
      gemma4_prefill_megakernel_layer_scratch_elements(true, prompt_len));
  thrust::device_vector<__nv_bfloat16> d_prefill_scratch(
      prefill_scratch_elements);
  thrust::device_vector<__nv_bfloat16> d_decode_a(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_decode_b(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_normed(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_sampled(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_attention_q(
      GEMMA4_GLOBAL_Q_PROJ_SIZE);
  thrust::device_vector<__nv_bfloat16> d_attention_out(
      GEMMA4_GLOBAL_ATTENTION_OUT_SIZE);

  const int32_t split_size = GEMMA4_SLIDING_DECODE_SPLIT_SIZE;
  const int32_t sliding_splits =
      (GEMMA4_SLIDING_WINDOW + split_size - 1) / split_size;
  const int32_t global_keys =
      runtime.value.global_cache_config.max_pages_per_seq *
      runtime.value.global_cache_config.page_size;
  const int32_t global_splits = (global_keys + split_size - 1) / split_size;
  const int32_t max_splits = std::max(sliding_splits, global_splits);
  thrust::device_vector<float> d_partial_m(
      GEMMA4_NUM_QUERY_HEADS * max_splits);
  thrust::device_vector<float> d_partial_l(
      GEMMA4_NUM_QUERY_HEADS * max_splits);
  thrust::device_vector<float> d_partial_acc(
      static_cast<size_t>(GEMMA4_NUM_QUERY_HEADS) * max_splits *
      GEMMA4_GLOBAL_HEAD_DIM);
  const size_t spine_scratch_bytes =
      gemma4_decode_megakernel_spine_scratch_bytes();
  const size_t decode_scratch_bytes =
      gemma4_decode_megakernel_ffn_tail_scratch_bytes();
  thrust::device_vector<unsigned char> d_spine_scratch(
      spine_scratch_bytes);
  thrust::device_vector<unsigned char> d_decode_scratch(
      decode_scratch_bytes);

  auto copy_prompt_tokens_to_device =
      [&](const std::vector<int32_t> &tokens) -> cudaError_t {
    return cudaMemcpyAsync(
        raw_ptr(d_prompt_tokens), tokens.data(),
        tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice, 0);
  };

  status = copy_prompt_tokens_to_device(prompt_tokens);
  if (status != cudaSuccess) return fail_cuda(status, "copy prompt tokens");

  auto run_prefill_once = [&]() -> cudaError_t {
    cudaError_t inner_status =
        gemma4_runtime_prepare_prefill(&runtime.value, prompt_len, 0);
    if (inner_status != cudaSuccess) {
      return inner_status;
    }
    inner_status = gemma4_embedding_gather_bf16(
        raw_ptr(d_prefill_a), raw_ptr(d_prompt_tokens), weights.value.token_embedding,
        prompt_len, 0);
    if (inner_status != cudaSuccess) {
      return inner_status;
    }

    __nv_bfloat16 *prefill_final_row = nullptr;
    inner_status = run_prefill_layers(
        weights.value, &runtime.value, raw_ptr(d_prefill_a), raw_ptr(d_prefill_b),
        raw_ptr(d_prefill_scratch), prompt_len, 0, &prefill_final_row);
    if (inner_status != cudaSuccess) {
      return inner_status;
    }
    return sample_from_prefill(
        weights.value, prefill_final_row, prompt_len, raw_ptr(d_decode_a),
        raw_ptr(d_next_token), raw_ptr(d_spine_scratch), spine_scratch_bytes, 0);
  };

  auto run_decode_once = [&]() -> cudaError_t {
    return run_decode_step(
        weights.value, &runtime.value, raw_ptr(d_decode_a), raw_ptr(d_decode_b),
        raw_ptr(d_normed), raw_ptr(d_sampled), raw_ptr(d_next_token),
        raw_ptr(d_attention_q), raw_ptr(d_attention_out), raw_ptr(d_partial_m),
        raw_ptr(d_partial_l), raw_ptr(d_partial_acc), raw_ptr(d_decode_scratch),
        decode_scratch_bytes, split_size, sliding_splits, global_splits, 0);
  };

  auto run_serving_request = [&](PromptRequestTiming *timing) -> cudaError_t {
    const PromptClock::time_point request_start = PromptClock::now();
    std::vector<int32_t> request_prompt_tokens;
    if (!tokenizer.encode(options.prompt, &request_prompt_tokens) ||
        static_cast<int32_t>(request_prompt_tokens.size()) != prompt_len) {
      return cudaErrorInvalidValue;
    }

    cudaError_t inner_status =
        copy_prompt_tokens_to_device(request_prompt_tokens);
    if (inner_status != cudaSuccess) {
      return inner_status;
    }
    inner_status = run_prefill_once();
    if (inner_status != cudaSuccess) {
      return inner_status;
    }

    std::vector<int32_t> generated;
    inner_status = copy_next_token(raw_ptr(d_next_token), &generated, 0);
    if (inner_status != cudaSuccess) {
      return inner_status;
    }
    const PromptClock::time_point first_token_time = PromptClock::now();
    PromptClock::time_point previous_token_time = first_token_time;

    while (static_cast<int32_t>(generated.size()) < options.max_new_tokens) {
      inner_status = run_decode_once();
      if (inner_status != cudaSuccess) {
        return inner_status;
      }
      inner_status = copy_next_token(raw_ptr(d_next_token), &generated, 0);
      if (inner_status != cudaSuccess) {
        return inner_status;
      }
      const PromptClock::time_point token_time = PromptClock::now();
      timing->itl_ms.push_back(host_elapsed_ms(previous_token_time, token_time));
      previous_token_time = token_time;
    }

    std::string decoded;
    if (!tokenizer.decode(generated, &decoded)) {
      return cudaErrorInvalidValue;
    }

    const PromptClock::time_point request_stop = PromptClock::now();
    timing->ttft_ms = host_elapsed_ms(request_start, first_token_time);
    timing->e2e_ms = host_elapsed_ms(request_start, request_stop);
    if (generated.size() > 1) {
      const float decode_ms =
          host_elapsed_ms(first_token_time, previous_token_time);
      timing->tpot_ms = decode_ms / static_cast<float>(generated.size() - 1);
      if (decode_ms > 0.0f) {
        timing->per_user_tps =
            1000.0f * static_cast<float>(generated.size() - 1) / decode_ms;
      }
    }
    return cudaSuccess;
  };

  if (options.benchmark_mode == PromptBenchmarkMode::kDecodeStep) {
    status = run_prefill_once();
    if (status != cudaSuccess) {
      return fail_cuda(status, "benchmark decode setup");
    }
    status = cudaStreamSynchronize(0);
    if (status != cudaSuccess) {
      return fail_cuda(status, "benchmark decode setup sync");
    }

    PromptBenchmarkStats decode_stats;
    status = measure_cuda_event_steps_ms(
        &decode_stats, options.bench_warmup, options.bench_iters,
        options.bench_samples, 0, run_decode_once);
    if (status != cudaSuccess) {
      return fail_cuda(status, "benchmark decode step");
    }

    status = print_cuda_environment();
    if (status != cudaSuccess) {
      return fail_cuda(status, "benchmark environment");
    }
    std::printf(
        "benchmark_mode=%s prompt_len=%d warmup_steps=%d timed_steps=%d "
        "timing=cuda_events_same_stream cache=stateful_decode "
        "setup_excluded=tokenizer,weight_load,allocation,prefill\n",
        benchmark_mode_name(options.benchmark_mode), prompt_len,
        options.bench_warmup, options.bench_iters * options.bench_samples);
    print_stats_line("decode_step_ms", decode_stats);
    if (decode_stats.p50_ms > 0.0f) {
      std::printf("decode_step_tps_p50=%.3f\n", 1000.0f / decode_stats.p50_ms);
    }
    return 0;
  }

  if (options.benchmark_mode == PromptBenchmarkMode::kWarmServing) {
    const int32_t measured_requests = options.bench_iters * options.bench_samples;
    for (int32_t request = 0; request < options.bench_warmup; ++request) {
      PromptRequestTiming timing;
      status = run_serving_request(&timing);
      if (status != cudaSuccess) {
        return fail_cuda(status, "warm serving warmup");
      }
    }

    std::vector<float> ttft_ms;
    std::vector<float> e2e_ms;
    std::vector<float> tpot_ms;
    std::vector<float> itl_ms;
    std::vector<float> per_user_tps;
    ttft_ms.reserve(measured_requests);
    e2e_ms.reserve(measured_requests);
    tpot_ms.reserve(measured_requests);
    per_user_tps.reserve(measured_requests);
    for (int32_t request = 0; request < measured_requests; ++request) {
      PromptRequestTiming timing;
      status = run_serving_request(&timing);
      if (status != cudaSuccess) {
        return fail_cuda(status, "warm serving request");
      }
      ttft_ms.push_back(timing.ttft_ms);
      e2e_ms.push_back(timing.e2e_ms);
      if (options.max_new_tokens > 1) {
        tpot_ms.push_back(timing.tpot_ms);
        per_user_tps.push_back(timing.per_user_tps);
        itl_ms.insert(itl_ms.end(), timing.itl_ms.begin(), timing.itl_ms.end());
      }
    }

    status = print_cuda_environment();
    if (status != cudaSuccess) {
      return fail_cuda(status, "benchmark environment");
    }
    std::printf(
        "benchmark_mode=%s prompt_len=%d output_tokens=%d concurrency=1 "
        "warmup_requests=%d measured_requests=%d timing=host_wall_clock "
        "model_load=excluded cuda_context=excluded tokenization=included "
        "prompt_h2d=included streaming=host_token_copy detokenization=e2e_only\n",
        benchmark_mode_name(options.benchmark_mode), prompt_len,
        options.max_new_tokens, options.bench_warmup, measured_requests);
    print_stats_line("ttft_ms", summarize_benchmark_samples(ttft_ms));
    print_stats_line("tpot_ms", summarize_benchmark_samples(tpot_ms));
    print_stats_line("itl_ms", summarize_benchmark_samples(itl_ms));
    print_stats_line("e2e_ms", summarize_benchmark_samples(e2e_ms));
    print_rate_stats_line("per_user_tps",
                          summarize_benchmark_samples(per_user_tps));
    return 0;
  }

  if (options.benchmark_mode == PromptBenchmarkMode::kColdStart) {
    status = run_prefill_once();
    if (status != cudaSuccess) {
      return fail_cuda(status, "cold-start prefill");
    }

    std::vector<int32_t> generated;
    status = copy_next_token(raw_ptr(d_next_token), &generated, 0);
    if (status != cudaSuccess) {
      return fail_cuda(status, "cold-start first token");
    }

    const float process_to_first_token_ms =
        host_elapsed_ms(process_start, PromptClock::now());
    status = print_cuda_environment();
    if (status != cudaSuccess) {
      return fail_cuda(status, "benchmark environment");
    }
    std::printf(
        "benchmark_mode=%s prompt_len=%d output_tokens=1 "
        "timing=host_wall_clock includes=cli_parse,tokenizer_load,"
        "tokenization,weight_load,allocation,runtime_init,prefill,first_token "
        "process_to_first_token_ms=%.3f cold_start_scope=current_process\n",
        benchmark_mode_name(options.benchmark_mode), prompt_len,
        process_to_first_token_ms);
    return 0;
  }

  status = run_prefill_once();
  if (status != cudaSuccess) {
    return fail_cuda(status, "prefill");
  }

  std::vector<int32_t> generated;
  status = copy_next_token(raw_ptr(d_next_token), &generated, 0);
  if (status != cudaSuccess) {
    return fail_cuda(status, "copy first token");
  }

  while (static_cast<int32_t>(generated.size()) < options.max_new_tokens) {
    status = run_decode_step(
        weights.value, &runtime.value, raw_ptr(d_decode_a), raw_ptr(d_decode_b),
        raw_ptr(d_normed), raw_ptr(d_sampled), raw_ptr(d_next_token),
        raw_ptr(d_attention_q), raw_ptr(d_attention_out), raw_ptr(d_partial_m),
        raw_ptr(d_partial_l), raw_ptr(d_partial_acc), raw_ptr(d_decode_scratch),
        decode_scratch_bytes, split_size, sliding_splits, global_splits, 0);
    if (status != cudaSuccess) {
      return fail_cuda(status, "decode step");
    }
    status = copy_next_token(raw_ptr(d_next_token), &generated, 0);
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
  const PromptClock::time_point process_start = PromptClock::now();
  PromptOptions options;
  if (!parse_args(argc, argv, &options)) {
    return 1;
  }
  return run_prompt(options, process_start);
}
