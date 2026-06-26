#include "gemma4_checkpoint.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_decode_megakernel.cuh"
#include "gemma4_embedding_gather.cuh"
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
#include <string>
#include <vector>

namespace {

// Returns the raw CUDA pointer owned by a Thrust device vector.
template <typename T>
T *raw_ptr(thrust::device_vector<T> &v) {
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
  int32_t decode_split_size = GEMMA4_SLIDING_DECODE_SPLIT_SIZE;
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
    } else if (arg == "--decode-split-size" && i + 1 < argc) {
      options->decode_split_size = std::atoi(argv[++i]);
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
                   "[--decode-split-size n] "
                   "[--benchmark] [--benchmark-mode none|decode-step|"
                   "warm-serving|cold-start] [--bench-warmup n] "
                   "[--bench-iters n] [--bench-samples n]\n",
                   argv[0]);
      return false;
    }
  }
  return options->max_new_tokens > 0 && options->page_size > 0 &&
         options->decode_split_size > 0 &&
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
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaGetDevice(&device));

  cudaDeviceProp prop = {};
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaGetDeviceProperties(&prop, device));

  int driver = 0;
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaDriverGetVersion(&driver));

  int runtime = 0;
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaRuntimeGetVersion(&runtime));

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
    GEMMA4_RETURN_IF_CUDA_ERROR(fn());
  }

  GEMMA4_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaEventCreate(&start));
  cudaError_t status = cudaEventCreate(&stop);
  if (status != cudaSuccess) {
    cudaEventDestroy(start);
    return status;
  }

  for (int32_t i = 0; i < timed_steps; ++i) {
    status = cudaEventRecord(start, stream);
    if (status != cudaSuccess) goto done;
    status = fn();
    if (status != cudaSuccess) goto done;
    status = cudaEventRecord(stop, stream);
    if (status != cudaSuccess) goto done;
    status = cudaEventSynchronize(stop);
    if (status != cudaSuccess) goto done;

    float total_ms = 0.0f;
    status = cudaEventElapsedTime(&total_ms, start, stop);
    if (status != cudaSuccess) goto done;
    step_ms.push_back(total_ms);
  }

done:
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (status != cudaSuccess) {
    return status;
  }
  *stats = summarize_benchmark_samples(step_ms);
  return cudaGetLastError();
}

// Copies one sampled token from device to host.
cudaError_t copy_next_token(
    int32_t *d_next_token,
    std::vector<int32_t> *generated,
    cudaStream_t stream) {
  int32_t token = -1;
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
      &token, d_next_token, sizeof(token), cudaMemcpyDeviceToHost, stream));
  GEMMA4_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(stream));
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
  const size_t prefill_scratch_elements =
      gemma4_prefill_megakernel_scratch_elements(prompt_len);
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

  const int32_t split_size = options.decode_split_size;
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
  const size_t decode_scratch_bytes = gemma4_decode_megakernel_scratch_bytes();
  thrust::device_vector<unsigned char> d_decode_scratch(
      decode_scratch_bytes);

  status = cudaMemcpyAsync(
      raw_ptr(d_prompt_tokens), prompt_tokens.data(),
      prompt_tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice, 0);
  if (status != cudaSuccess) {
    return fail_cuda(status, "prompt tokens h2d");
  }

  auto run_prefill_once = [&]() -> cudaError_t {
    GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_embedding_gather_bf16(
        raw_ptr(d_prefill_a), raw_ptr(d_prompt_tokens), weights.value.token_embedding,
        prompt_len, 0));

    __nv_bfloat16 *prefill_final_row = nullptr;
    Gemma4PrefillMegakernelArgs args = {};
    args.hidden_a = raw_ptr(d_prefill_a);
    args.hidden_b = raw_ptr(d_prefill_b);
    args.final_hidden = &prefill_final_row;
    args.scratch = raw_ptr(d_prefill_scratch);
    args.scratch_elements = prefill_scratch_elements;
    args.weights = &weights.value;
    args.runtime = &runtime.value;
    args.seq_len = prompt_len;
    GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_prefill_megakernel(args));

    const size_t row_bytes =
        static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * sizeof(__nv_bfloat16);
    const __nv_bfloat16 *last_row =
        prefill_final_row + int64_t(prompt_len - 1) * GEMMA4_HIDDEN_SIZE;
    GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        raw_ptr(d_decode_a), last_row, row_bytes, cudaMemcpyDeviceToDevice, 0));
    return gemma4_megakernel_sample_final_bf16(
        raw_ptr(d_decode_a), raw_ptr(d_next_token), raw_ptr(d_normed),
        raw_ptr(d_decode_a), &weights.value, raw_ptr(d_decode_scratch),
        decode_scratch_bytes, Gemma4SamplingStage::kPrefill, 0);
  };

  auto run_decode_once = [&]() -> cudaError_t {
    Gemma4DecodeMegakernelArgs args = {};
    args.hidden_a = raw_ptr(d_decode_a);
    args.hidden_b = raw_ptr(d_decode_b);
    args.normed = raw_ptr(d_normed);
    args.sampled_hidden = raw_ptr(d_sampled);
    args.next_token = raw_ptr(d_next_token);
    args.attention_q = raw_ptr(d_attention_q);
    args.attention_out = raw_ptr(d_attention_out);
    args.partial_m = raw_ptr(d_partial_m);
    args.partial_l = raw_ptr(d_partial_l);
    args.partial_acc = raw_ptr(d_partial_acc);
    args.scratch = raw_ptr(d_decode_scratch);
    args.scratch_bytes = decode_scratch_bytes;
    args.weights = &weights.value;
    args.runtime = &runtime.value;
    args.split_size = split_size;
    args.sliding_splits = sliding_splits;
    args.global_splits = global_splits;
    return gemma4_decode_megakernel(args);
  };

  auto run_serving_request = [&](PromptRequestTiming *timing) -> cudaError_t {
    const PromptClock::time_point request_start = PromptClock::now();
    std::vector<int32_t> request_prompt_tokens;
    if (!tokenizer.encode(options.prompt, &request_prompt_tokens) ||
        request_prompt_tokens.empty()) {
      return cudaErrorInvalidValue;
    }

    GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        raw_ptr(d_prompt_tokens), request_prompt_tokens.data(),
        request_prompt_tokens.size() * sizeof(int32_t), cudaMemcpyHostToDevice,
        0));
    GEMMA4_RETURN_IF_CUDA_ERROR(run_prefill_once());

    std::vector<int32_t> generated;
    GEMMA4_RETURN_IF_CUDA_ERROR(
        copy_next_token(raw_ptr(d_next_token), &generated, 0));
    const PromptClock::time_point first_token_time = PromptClock::now();
    PromptClock::time_point previous_token_time = first_token_time;

    while (static_cast<int32_t>(generated.size()) < options.max_new_tokens) {
      GEMMA4_RETURN_IF_CUDA_ERROR(run_decode_once());
      GEMMA4_RETURN_IF_CUDA_ERROR(
          copy_next_token(raw_ptr(d_next_token), &generated, 0));
      const PromptClock::time_point token_time = PromptClock::now();
      timing->itl_ms.push_back(host_elapsed_ms(previous_token_time, token_time));
      previous_token_time = token_time;
    }

    timing->ttft_ms = host_elapsed_ms(request_start, first_token_time);
    timing->e2e_ms = host_elapsed_ms(request_start, previous_token_time);
    if (generated.size() > 1) {
      const float decode_ms = timing->e2e_ms - timing->ttft_ms;
      timing->tpot_ms = decode_ms / static_cast<float>(generated.size() - 1);
      if (decode_ms > 0.0f) {
        timing->per_user_tps =
            1000.0f * static_cast<float>(generated.size() - 1) / decode_ms;
      }
    }

    std::string decoded;
    if (!tokenizer.decode(generated, &decoded)) {
      return cudaErrorInvalidValue;
    }
    return cudaSuccess;
  };

  if (options.benchmark_mode == PromptBenchmarkMode::kDecodeStep) {
    status = run_prefill_once();
    if (status != cudaSuccess) {
      return fail_cuda(status, "benchmark prefill");
    }
    status = cudaStreamSynchronize(0);
    if (status != cudaSuccess) {
      return fail_cuda(status, "benchmark prefill sync");
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
      return fail_cuda(status, "cuda environment");
    }
    std::printf(
        "benchmark_mode=%s prompt_len=%d warmup_steps=%d timed_steps=%d "
        "decode_split_size=%d "
        "timing=cuda_events_same_stream cache=stateful_decode "
        "setup_excluded=tokenizer,weight_load,allocation,prefill\n",
        benchmark_mode_name(options.benchmark_mode), prompt_len,
        options.bench_warmup, options.bench_iters * options.bench_samples,
        split_size);
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
    const PromptClock::time_point benchmark_start = PromptClock::now();
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
    const PromptClock::time_point benchmark_stop = PromptClock::now();
    const float benchmark_duration_ms =
        host_elapsed_ms(benchmark_start, benchmark_stop);
    const float benchmark_duration_s = benchmark_duration_ms / 1000.0f;
    const int64_t total_input_tokens =
        int64_t(prompt_len) * measured_requests;
    const int64_t total_output_tokens =
        int64_t(options.max_new_tokens) * measured_requests;
    const int64_t total_tokens = total_input_tokens + total_output_tokens;
    const float request_throughput =
        static_cast<float>(measured_requests) / benchmark_duration_s;
    const float output_token_throughput =
        static_cast<float>(total_output_tokens) / benchmark_duration_s;
    const float total_token_throughput =
        static_cast<float>(total_tokens) / benchmark_duration_s;

    status = print_cuda_environment();
    if (status != cudaSuccess) {
      return fail_cuda(status, "cuda environment");
    }
    std::printf(
        "benchmark_mode=%s prompt_len=%d output_tokens=%d concurrency=1 "
        "warmup_requests=%d measured_requests=%d timing=host_wall_clock "
        "request_schedule=closed_loop_sequential http=excluded "
        "model_load=excluded cuda_context=excluded tokenization=included "
        "prompt_h2d=included streaming=host_token_copy "
        "detokenization=excluded\n",
        benchmark_mode_name(options.benchmark_mode), prompt_len,
        options.max_new_tokens, options.bench_warmup, measured_requests);
    std::printf(
        "benchmark_duration_ms=%.3f successful_requests=%d "
        "total_input_tokens=%lld total_output_tokens=%lld "
        "request_throughput=%.3f output_token_throughput=%.3f "
        "total_token_throughput=%.3f\n",
        benchmark_duration_ms, measured_requests,
        static_cast<long long>(total_input_tokens),
        static_cast<long long>(total_output_tokens), request_throughput,
        output_token_throughput, total_token_throughput);
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
      return fail_cuda(status, "cold prefill");
    }

    std::vector<int32_t> generated;
    status = copy_next_token(raw_ptr(d_next_token), &generated, 0);
    if (status != cudaSuccess) {
      return fail_cuda(status, "copy next token");
    }

    const float process_to_first_token_ms =
        host_elapsed_ms(process_start, PromptClock::now());
    status = print_cuda_environment();
    if (status != cudaSuccess) {
      return fail_cuda(status, "cuda environment");
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
    return fail_cuda(status, "copy next token");
  }

  while (static_cast<int32_t>(generated.size()) < options.max_new_tokens) {
    status = run_decode_once();
    if (status != cudaSuccess) {
      return fail_cuda(status, "decode step");
    }
    status = copy_next_token(raw_ptr(d_next_token), &generated, 0);
    if (status != cudaSuccess) {
      return fail_cuda(status, "copy next token");
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
