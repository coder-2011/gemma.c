#include "benches/gemma4_bench_utils.cuh"
#include "gemma4_checkpoint.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_megakernel.cuh"
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
#include <string_view>
#include <vector>

namespace {

constexpr const char *kDefaultCheckpointPath = "models/gemma-4-12B/model.safetensors";
constexpr const char *kDefaultTokenizerPath = "models/gemma-4-12B/tokenizer.json";
constexpr const char *kDefaultPrompt = "Hello";
constexpr int32_t kDefaultMaxNewTokens = 1;
constexpr int32_t kDefaultPageSize = 64;
constexpr int32_t kDefaultBenchWarmup = 1;
constexpr int32_t kDefaultBenchIters = 3;
constexpr int32_t kDefaultBenchSamples = 3;
constexpr int32_t kPromptBatchSize = 1;
constexpr size_t kHiddenRowBytes =
    static_cast<size_t>(GEMMA4_HIDDEN_SIZE) * sizeof(__nv_bfloat16);

using PromptClock = std::chrono::steady_clock;

enum class PromptBenchmarkMode {
  kNone,          // Run one prompt and print the generated tokens and decoded text.
  kDecodeStep,    // Time repeated steady-state decode megakernel steps after one prefill.
  kOfflineDecode, // Time full local generation requests with one final stream sync.
  kWarmServing,   // Time warm request latency with per-token host synchronization.
};

struct PromptOptions {
  std::string checkpoint_path = kDefaultCheckpointPath;
  std::string tokenizer_path = kDefaultTokenizerPath;
  std::string prompt = kDefaultPrompt;
  int32_t max_new_tokens = kDefaultMaxNewTokens;
  int32_t page_size = kDefaultPageSize;
  PromptBenchmarkMode benchmark_mode = PromptBenchmarkMode::kNone;
  int32_t bench_warmup = kDefaultBenchWarmup;
  int32_t bench_iters = kDefaultBenchIters;
  int32_t bench_samples = kDefaultBenchSamples;
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
  ~WeightOwner() { gemma4_text_weights_device_free(&value); }
};

// Releases runtime KV/cache metadata when leaving the prompt runner.
struct RuntimeOwner {
  Gemma4RuntimeState value = {};

  // Frees all CUDA allocations owned by the runtime state.
  ~RuntimeOwner() { gemma4_runtime_state_free(&value); }
};

// Parses the tiny prompt-runner CLI.
bool parse_args(int argc, char **argv, PromptOptions *options) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--benchmark") {
      options->benchmark_mode = PromptBenchmarkMode::kDecodeStep;
      continue;
    }

    // Every supported option except --benchmark consumes the next argv entry.
    if (i + 1 >= argc) return false;
    const char *value_arg = argv[++i];
    if (arg == "--checkpoint") options->checkpoint_path = value_arg;
    else if (arg == "--tokenizer") options->tokenizer_path = value_arg;
    else if (arg == "--prompt") options->prompt = value_arg;
    else if (arg == "--max-new") options->max_new_tokens = std::atoi(value_arg);
    else if (arg == "--page-size") options->page_size = std::atoi(value_arg);
    else if (arg == "--benchmark-mode") {
      const std::string_view value = value_arg;
      if (value == "decode-step") {
        options->benchmark_mode = PromptBenchmarkMode::kDecodeStep;
      }
      else if (value == "offline-decode") {
        options->benchmark_mode = PromptBenchmarkMode::kOfflineDecode;
      }
      else if (value == "warm-serving") {
        options->benchmark_mode = PromptBenchmarkMode::kWarmServing;
      }
      else if (value == "none") options->benchmark_mode = PromptBenchmarkMode::kNone;
      else return false;
    }
    else if (arg == "--bench-warmup") options->bench_warmup = std::atoi(value_arg);
    else if (arg == "--bench-iters") options->bench_iters = std::atoi(value_arg);
    else if (arg == "--bench-samples") options->bench_samples = std::atoi(value_arg);
    else return false;
  }
  return options->max_new_tokens > 0 && options->page_size > 0 &&
         options->bench_warmup >= 0 && options->bench_iters > 0 &&
         options->bench_samples > 0;
}

// Prints one compact distribution line for latency or throughput samples.
void print_stats_line(
    const char *name, const TimingStats &stats, const char *unit) {
  std::printf(
      "%s n=%zu mean%s=%.3f p50%s=%.3f p95%s=%.3f p99%s=%.3f "
      "min%s=%.3f max%s=%.3f\n",
      name, stats.samples_ms.size(), unit, stats.avg_ms, unit,
      stats.median_ms, unit, stats.p95_ms, unit, stats.p99_ms, unit,
      stats.min_ms, unit, stats.max_ms);
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
int run_prompt(const PromptOptions &options) {
  Gemma4Tokenizer tokenizer;
  if (!tokenizer.load(options.tokenizer_path, nullptr)) return 1;

  std::vector<int32_t> prompt_tokens;
  if (!tokenizer.encode(options.prompt, &prompt_tokens)) return 1;
  if (prompt_tokens.empty()) return 1;
  const int32_t prompt_len = static_cast<int32_t>(prompt_tokens.size());
  const int32_t benchmark_decode_steps = options.benchmark_mode == PromptBenchmarkMode::kDecodeStep  ? options.bench_samples * (options.bench_warmup + options.bench_iters) : 0;
  const int32_t max_seq_len = prompt_len + std::max(options.max_new_tokens, benchmark_decode_steps);
  std::printf("prompt tokens: %d\n", prompt_len);

  WeightOwner weights;
  CUDA_CHECK(gemma4_load_text_weights_device_bf16(&weights.value, options.checkpoint_path.c_str(), nullptr));

  RuntimeOwner runtime;
  CUDA_CHECK(gemma4_runtime_state_init(&runtime.value, kPromptBatchSize, max_seq_len, options.page_size, 0));
  thrust::device_vector<int32_t> d_prompt_tokens(prompt_tokens.size());
  thrust::device_vector<int32_t> d_next_token(kPromptBatchSize);

  const size_t prompt_hidden_elements =
      static_cast<size_t>(prompt_len) * GEMMA4_HIDDEN_SIZE;
  thrust::device_vector<__nv_bfloat16> d_prefill_a(prompt_hidden_elements);
  thrust::device_vector<__nv_bfloat16> d_prefill_b(prompt_hidden_elements);
  const size_t prefill_scratch_elements = gemma4_prefill_megakernel_scratch_elements(prompt_len);

  thrust::device_vector<__nv_bfloat16> d_prefill_scratch(prefill_scratch_elements);
  thrust::device_vector<__nv_bfloat16> d_decode_a(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_decode_b(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_normed(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_sampled(GEMMA4_HIDDEN_SIZE);

  thrust::device_vector<__nv_bfloat16> d_attention_q(GEMMA4_GLOBAL_Q_PROJ_SIZE);
  thrust::device_vector<__nv_bfloat16> d_attention_out(GEMMA4_GLOBAL_ATTENTION_OUT_SIZE);

  constexpr int32_t split_size = GEMMA4_SLIDING_DECODE_SPLIT_SIZE;
  const int32_t sliding_keys = runtime.value.sliding_cache_config.window_size;
  const int32_t sliding_splits = div_up(sliding_keys, split_size);
  const int32_t global_keys =
      runtime.value.global_cache_config.max_pages_per_seq *
      runtime.value.global_cache_config.page_size;
  const int32_t global_splits = div_up(global_keys, split_size);
  const int32_t max_splits = std::max(sliding_splits, global_splits);
  thrust::device_vector<float> d_partial_m(GEMMA4_NUM_QUERY_HEADS * max_splits);
  thrust::device_vector<float> d_partial_l(GEMMA4_NUM_QUERY_HEADS * max_splits);
  thrust::device_vector<float> d_partial_acc(
      static_cast<size_t>(GEMMA4_NUM_QUERY_HEADS) * max_splits * GEMMA4_GLOBAL_HEAD_DIM);
  const size_t decode_scratch_bytes = gemma4_decode_megakernel_scratch_bytes();
  thrust::device_vector<unsigned char> d_decode_scratch(decode_scratch_bytes);

  auto copy_prompt_tokens = [&](const std::vector<int32_t> &tokens) {
    return cudaMemcpyAsync(raw_ptr(d_prompt_tokens), tokens.data(),
                           tokens.size() * sizeof(int32_t),
                           cudaMemcpyHostToDevice, 0);
  };
  CUDA_CHECK(copy_prompt_tokens(prompt_tokens));

  auto run_prefill_once = [&]() -> cudaError_t {
    GEMMA4_RETURN_IF_CUDA_ERROR(gemma4_embedding_gather_bf16(
        raw_ptr(d_prefill_a), raw_ptr(d_prompt_tokens), weights.value.token_embedding, prompt_len, 0));

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

    const __nv_bfloat16 *last_row =
        prefill_final_row + int64_t(prompt_len - 1) * GEMMA4_HIDDEN_SIZE;
    GEMMA4_RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        raw_ptr(d_decode_a), last_row, kHiddenRowBytes, cudaMemcpyDeviceToDevice, 0));
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

  auto generate_tokens = [&](int32_t max_tokens, std::vector<int32_t> *generated) -> cudaError_t {
    GEMMA4_RETURN_IF_CUDA_ERROR(run_prefill_once());

    generated->clear();
    GEMMA4_RETURN_IF_CUDA_ERROR( copy_next_token(raw_ptr(d_next_token), generated, 0));

    while (static_cast<int32_t>(generated->size()) < max_tokens) {
      GEMMA4_RETURN_IF_CUDA_ERROR(run_decode_once());
      GEMMA4_RETURN_IF_CUDA_ERROR(copy_next_token(raw_ptr(d_next_token), generated, 0));
    }
    return cudaSuccess;
  };

  auto run_serving_request = [&](PromptRequestTiming *timing) -> cudaError_t {
    const PromptClock::time_point request_start = PromptClock::now();
    GEMMA4_RETURN_IF_CUDA_ERROR(copy_prompt_tokens(prompt_tokens));
    std::vector<int32_t> generated;
    GEMMA4_RETURN_IF_CUDA_ERROR(run_prefill_once());
    GEMMA4_RETURN_IF_CUDA_ERROR(copy_next_token(raw_ptr(d_next_token), &generated, 0));

    const PromptClock::time_point first_token_time = PromptClock::now();
    PromptClock::time_point previous_token_time = first_token_time;
    while (static_cast<int32_t>(generated.size()) < options.max_new_tokens) {
      GEMMA4_RETURN_IF_CUDA_ERROR(run_decode_once());
      GEMMA4_RETURN_IF_CUDA_ERROR(copy_next_token(raw_ptr(d_next_token), &generated, 0));
      const PromptClock::time_point token_time = PromptClock::now();
      timing->itl_ms.push_back(
          std::chrono::duration<float, std::milli>(
              token_time - previous_token_time)
              .count());
      previous_token_time = token_time;
    }

    timing->ttft_ms =
        std::chrono::duration<float, std::milli>(
            first_token_time - request_start)
            .count();
    timing->e2e_ms =
        std::chrono::duration<float, std::milli>(
            previous_token_time - request_start)
            .count();
    if (generated.size() > 1) {
      const float decode_ms = timing->e2e_ms - timing->ttft_ms;
      timing->tpot_ms = decode_ms / static_cast<float>(generated.size() - 1);
      if (decode_ms > 0.0f) {
        timing->per_user_tps =
            1000.0f * static_cast<float>(generated.size() - 1) / decode_ms;
      }
    }
    return cudaSuccess;
  };

  // Times one offline generation request with only a final stream sync.
  auto run_offline_decode_request = [&](float *e2e_ms) -> cudaError_t {
    const PromptClock::time_point request_start = PromptClock::now();
    GEMMA4_RETURN_IF_CUDA_ERROR(copy_prompt_tokens(prompt_tokens));
    GEMMA4_RETURN_IF_CUDA_ERROR(run_prefill_once());
    for (int32_t token = 1; token < options.max_new_tokens; ++token) {
      GEMMA4_RETURN_IF_CUDA_ERROR(run_decode_once());
    }
    GEMMA4_RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(0));
    const PromptClock::time_point request_stop = PromptClock::now();
    *e2e_ms =
        std::chrono::duration<float, std::milli>(request_stop - request_start)
            .count();
    return cudaSuccess;
  };
  const int32_t measured_requests = options.bench_iters * options.bench_samples;

  if (options.benchmark_mode == PromptBenchmarkMode::kDecodeStep) {
    CUDA_CHECK(run_prefill_once());
    CUDA_CHECK(cudaStreamSynchronize(0));

    auto checked_decode = [&]() { CUDA_CHECK(run_decode_once()); };
    TimingStats decode_stats = time_ms(
        checked_decode, 0, options.bench_warmup, options.bench_iters,
        options.bench_samples);
    std::printf(
        "benchmark mode=decode-step prompt_len=%d warmup=%d iters=%d "
        "samples=%d split=%d\n",
        prompt_len, options.bench_warmup, options.bench_iters,
        options.bench_samples, split_size);
    print_stats_line("decode_step_ms", decode_stats, "_ms");
    if (decode_stats.median_ms > 0.0f) {
      std::printf("decode_step_tps_p50=%.3f\n", 1000.0f / decode_stats.median_ms);
    }
    return 0;
  }

  if (options.benchmark_mode == PromptBenchmarkMode::kOfflineDecode) {
    for (int32_t request = 0; request < options.bench_warmup; ++request) {
      float e2e_ms = 0.0f;
      CUDA_CHECK(run_offline_decode_request(&e2e_ms));
    }

    std::vector<float> e2e_ms;
    e2e_ms.reserve(measured_requests);
    for (int32_t request = 0; request < measured_requests; ++request) {
      float sample_ms = 0.0f;
      CUDA_CHECK(run_offline_decode_request(&sample_ms));
      e2e_ms.push_back(sample_ms);
    }

    std::printf(
        "benchmark mode=offline-decode prompt_len=%d output_tokens=%d warmup=%d "
        "measured=%d sync=end\n",
        prompt_len, options.max_new_tokens, options.bench_warmup,
        measured_requests);
    print_stats_line("e2e_ms", summarize_timing_samples(std::move(e2e_ms)), "_ms");
    return 0;
  }

  if (options.benchmark_mode == PromptBenchmarkMode::kWarmServing) {
    for (int32_t request = 0; request < options.bench_warmup; ++request) {
      PromptRequestTiming timing;
      CUDA_CHECK(run_serving_request(&timing));
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
      CUDA_CHECK(run_serving_request(&timing));
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
        std::chrono::duration<float, std::milli>(
            benchmark_stop - benchmark_start)
            .count();

    const float benchmark_duration_s = benchmark_duration_ms / 1000.0f;
    const int64_t total_input_tokens = int64_t(prompt_len) * measured_requests;
    const int64_t total_output_tokens = int64_t(options.max_new_tokens) * measured_requests;
    const int64_t total_tokens = total_input_tokens + total_output_tokens;
    const float request_throughput = static_cast<float>(measured_requests) / benchmark_duration_s;
    const float output_token_throughput = static_cast<float>(total_output_tokens) / benchmark_duration_s;
    const float total_token_throughput = static_cast<float>(total_tokens) / benchmark_duration_s;

    std::printf(
        "benchmark mode=warm-serving prompt_len=%d output_tokens=%d warmup=%d measured=%d\n",
        prompt_len, options.max_new_tokens, options.bench_warmup,
        measured_requests);
    std::printf(
        "benchmark_duration_ms=%.3f successful_requests=%d "
        "total_input_tokens=%lld total_output_tokens=%lld "
        "request_throughput=%.3f output_token_throughput=%.3f "
        "total_token_throughput=%.3f\n",
        benchmark_duration_ms, measured_requests,
        static_cast<long long>(total_input_tokens),
        static_cast<long long>(total_output_tokens), request_throughput,
        output_token_throughput, total_token_throughput);
    print_stats_line("ttft_ms", summarize_timing_samples(std::move(ttft_ms)), "_ms");
    print_stats_line("tpot_ms", summarize_timing_samples(std::move(tpot_ms)), "_ms");
    print_stats_line("itl_ms", summarize_timing_samples(std::move(itl_ms)), "_ms");
    print_stats_line("e2e_ms", summarize_timing_samples(std::move(e2e_ms)), "_ms");
    print_stats_line("per_user_tps", summarize_timing_samples(std::move(per_user_tps)), "");
    return 0;
  }

  std::vector<int32_t> generated;
  CUDA_CHECK(generate_tokens(options.max_new_tokens, &generated));

  std::string decoded;
  if (!tokenizer.decode(generated, &decoded)) {
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
  }
int main(int argc, char **argv) {
  PromptOptions options;
  if (!parse_args(argc, argv, &options)) {
    return 1;
  return run_prompt(options);
}
