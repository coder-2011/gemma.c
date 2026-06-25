#include "gemma4_sampling.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_embedding_gather.cuh"
#include "gemma4_matmul_kernels.cuh"
#include "gemma4_bench_utils.cuh"
#include "gemma4.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/version.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <numeric>
#include <vector>

namespace {

constexpr int kFinalLogitsColsPerBlock = 8;

struct SamplingBenchArgs {
  int warmup = 25;
  int iters = 100;
  int samples = 21;
};

void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

// Parses positional args plus the deleted Python wrapper's named aliases.
bool parse_args(int argc, char **argv, SamplingBenchArgs *args) {
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    const bool has_value = i + 1 < argc;
    if (std::strcmp(argv[i], "--warmup") == 0 && has_value) {
      args->warmup = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--iters") == 0 && has_value) {
      args->iters = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--samples") == 0 && has_value) {
      args->samples = std::atoi(argv[++i]);
    } else if (positional == 0) {
      args->warmup = std::atoi(argv[i]);
      ++positional;
    } else if (positional == 1) {
      args->iters = std::atoi(argv[i]);
      ++positional;
    } else if (positional == 2) {
      args->samples = std::atoi(argv[i]);
      ++positional;
    } else {
      return false;
    }
  }
  return args->warmup >= 0 && args->iters > 0 && args->samples > 0;
}

__device__ inline bool better_candidate(float logit,
                                        int32_t token_id,
                                        float best_logit,
                                        int32_t best_token_id) {
  return logit > best_logit ||
         (logit == best_logit && token_id < best_token_id);
}

// Runs the native LibTorch LM-head, argmax, gather, and embedding-scale path.
void run_libtorch_sampling_ops(at::Tensor &logits,
                               at::Tensor &values,
                               at::Tensor &token,
                               at::Tensor &selected,
                               at::Tensor &next_hidden,
                               const at::Tensor &hidden,
                               const at::Tensor &lm_head,
                               const at::Tensor &lm_head_t) {
  at::mm_out(logits, hidden, lm_head_t);
  at::max_out(values, token, logits, 1, false);
  at::index_select_out(selected, lm_head, 0, token);
  at::mul_out(next_hidden, selected, at::Scalar(GEMMA4_EMBEDDING_SCALE));
}

// Selects from materialized final logits and gathers the tied embedding row.
__global__ __launch_bounds__(1024) void materialized_logits_sample_embed_kernel(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    const __nv_bfloat16 *__restrict__ d_logits) {
  __shared__ float s_logits[1024];
  __shared__ int32_t s_token_ids[1024];

  const int thread_idx = threadIdx.x;
  float best_logit = -INFINITY;
  int32_t best_token_id = GEMMA4_VOCAB_SIZE;

  for (int token_id = thread_idx; token_id < GEMMA4_VOCAB_SIZE;
       token_id += blockDim.x) {
    const float logit = __bfloat162float(d_logits[token_id]);
    if (better_candidate(logit, token_id, best_logit, best_token_id)) {
      best_logit = logit;
      best_token_id = token_id;
    }
  }

  s_logits[thread_idx] = best_logit;
  s_token_ids[thread_idx] = best_token_id;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread_idx < stride) {
      const float other_logit = s_logits[thread_idx + stride];
      const int32_t other_token_id = s_token_ids[thread_idx + stride];
      if (better_candidate(other_logit, other_token_id, s_logits[thread_idx],
                           s_token_ids[thread_idx])) {
        s_logits[thread_idx] = other_logit;
        s_token_ids[thread_idx] = other_token_id;
      }
    }
    __syncthreads();
  }

  const int32_t selected_token_id = s_token_ids[0];
  if (thread_idx == 0) {
    *d_next_token = selected_token_id;
  }
  gemma4_embedding_gather::copy_embedding_row_bf16(
      d_next_hidden, d_lm_head_col_major, selected_token_id, thread_idx,
      blockDim.x);
}

// Runs untimed warmup, then records per-iteration CUDA-event samples.
template <typename Fn>
std::vector<float> time_samples(Fn &&fn,
                                cudaStream_t stream,
                                int warmup,
                                int iters,
                                int samples) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> values;
  values.reserve(samples);
  for (int sample = 0; sample < samples; ++sample) {
    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
      fn();
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    values.push_back(total_ms / float(iters));
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return values;
}

float percentile(std::vector<float> sorted, float p) {
  std::sort(sorted.begin(), sorted.end());
  const float scaled = p * float(sorted.size() - 1);
  const size_t lo = size_t(scaled);
  const size_t hi = std::min(lo + 1, sorted.size() - 1);
  const float t = scaled - float(lo);
  return sorted[lo] * (1.0f - t) + sorted[hi] * t;
}

// Returns a lightly trimmed mean to reduce one-sample launch or clock outliers.
float trimmed_mean(std::vector<float> sorted) {
  std::sort(sorted.begin(), sorted.end());
  const size_t trim = sorted.size() >= 10 ? sorted.size() / 10 : 0;
  const size_t begin = trim;
  const size_t end = sorted.size() - trim;
  const float sum = std::accumulate(sorted.begin() + begin,
                                    sorted.begin() + end, 0.0f);
  return sum / float(end - begin);
}

// Returns sample standard deviation in the same units as the input samples.
float sample_stddev(const std::vector<float> &samples, float mean) {
  if (samples.size() < 2) {
    return 0.0f;
  }
  float sum_sq = 0.0f;
  for (float sample : samples) {
    const float delta = sample - mean;
    sum_sq += delta * delta;
  }
  return sqrtf(sum_sq / float(samples.size() - 1));
}

// Times a captured LibTorch CUDA graph with events on the graph replay stream.
std::vector<float> time_libtorch_graph_samples(at::cuda::CUDAGraph &graph,
                                               c10::cuda::CUDAStream stream,
                                               int warmup,
                                               int iters,
                                               int samples) {
  c10::cuda::CUDAStreamGuard guard(stream);
  for (int i = 0; i < warmup; ++i) {
    graph.replay();
  }
  CHECK_CUDA(cudaStreamSynchronize(stream.stream()));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  std::vector<float> values;
  values.reserve(samples);
  for (int sample = 0; sample < samples; ++sample) {
    CHECK_CUDA(cudaEventRecord(start, stream.stream()));
    for (int i = 0; i < iters; ++i) {
      graph.replay();
    }
    CHECK_CUDA(cudaEventRecord(stop, stream.stream()));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    values.push_back(total_ms / float(iters));
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return values;
}

// Verifies the LibTorch row gather and embedding scale against an eager result.
int64_t check_libtorch_sampling_correctness(const at::Tensor &lm_head,
                                            const at::Tensor &token,
                                            const at::Tensor &next_hidden) {
  const at::Tensor expected =
      at::index_select(lm_head, 0, token) * GEMMA4_EMBEDDING_SCALE;
  if (!at::equal(next_hidden, expected)) {
    throw std::runtime_error("native LibTorch selected-row gather mismatch");
  }
  return token.cpu().item<int64_t>();
}

// Prints summary stats plus raw samples in a line-oriented parseable format.
void print_stats(const char *variant,
                 int batch_size,
                 int top_k,
                 float top_p,
                 float temperature,
                 const std::vector<float> &samples) {
  const float sum = std::accumulate(samples.begin(), samples.end(), 0.0f);
  const float mean = sum / float(samples.size());
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  std::printf("variant=%s batch_size=%d top_k=%d top_p=%.3f temp=%.3f "
              "min_us=%.3f median_us=%.3f trimmed_mean_us=%.3f "
              "mean_us=%.3f stddev_us=%.3f iqr_us=%.3f p95_us=%.3f "
              "p99_us=%.3f max_us=%.3f samples_us=[",
              variant, batch_size, top_k, top_p, temperature,
              sorted.front() * 1000.0f, percentile(samples, 0.50f) * 1000.0f,
              trimmed_mean(samples) * 1000.0f, mean * 1000.0f,
              sample_stddev(samples, mean) * 1000.0f,
              (percentile(samples, 0.75f) - percentile(samples, 0.25f)) *
                  1000.0f,
              percentile(samples, 0.95f) * 1000.0f,
              percentile(samples, 0.99f) * 1000.0f,
              sorted.back() * 1000.0f);
  for (size_t i = 0; i < samples.size(); ++i) {
    std::printf("%s%.3f", i == 0 ? "" : ",", samples[i] * 1000.0f);
  }
  std::printf("]\n");
}

}  // namespace

int main(int argc, char **argv) {
  SamplingBenchArgs args;
  if (!parse_args(argc, argv, &args)) {
    std::fprintf(stderr,
                 "usage: %s [warmup=25] [iters=100] [samples=21]\n"
                 "       %s [--warmup N] [--iters N] [--samples N]\n",
                 argv[0],
                 argv[0]);
    return 1;
  }
  const int warmup = args.warmup;
  const int iters = args.iters;
  const int samples = args.samples;

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  std::printf("benchmark_guardrail stability_scope=single_process "
              "minimum_effect_us=25 close_call_requires_process_rerun=true "
              "candidate_order=fixed\n");
  std::printf("benchmark_threats clocks=unlocked cache=warm_repeated_buffers "
              "contention=not_exclusively_verified ncu=not_run "
              "process_repeats=not_run\n");

  cudaDeviceProp prop = {};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("env=gpu=\"%s\" global_mem_bytes=%zu\n",
              prop.name, static_cast<size_t>(prop.totalGlobalMem));
  std::printf("contract=sampling_decode timing=cuda_events "
              "stream=nonblocking cache=warm_repeated_buffers "
              "launch_overhead=cpu_enqueue_excluded_by_cuda_events "
              "warmup=%d iters=%d samples=%d dtype=bf16 "
              "vocab=%d hidden=%d\n",
              warmup, iters, samples, GEMMA4_VOCAB_SIZE, GEMMA4_HIDDEN_SIZE);
  std::printf("benchmark_inputs lm_head_seed=0x5678 hidden_seed=0x2468 "
              "lm_head_scale=0.05 hidden_scale=0.05\n");
  std::printf("benchmark_libtorch torch_version=%s\n", TORCH_VERSION);
  {
    at::manual_seed(0x5678u);
    const auto torch_options =
        at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
    at::Tensor torch_lm_head =
        at::empty({GEMMA4_VOCAB_SIZE, GEMMA4_HIDDEN_SIZE}, torch_options);
    torch_lm_head.uniform_(-0.05, 0.05);
    at::Tensor torch_lm_head_t = torch_lm_head.t();

    at::manual_seed(0x2468u);
    at::Tensor torch_hidden =
        at::empty({1, GEMMA4_HIDDEN_SIZE}, torch_options);
    torch_hidden.uniform_(-0.05, 0.05);

    at::Tensor torch_logits = at::empty({1, GEMMA4_VOCAB_SIZE}, torch_options);
    at::Tensor torch_values = at::empty({1}, torch_options);
    at::Tensor torch_token =
        at::empty({1}, at::TensorOptions().device(at::kCUDA).dtype(at::kLong));
    at::Tensor torch_selected = at::empty_like(torch_hidden);
    at::Tensor torch_next_hidden = at::empty_like(torch_hidden);

    const c10::cuda::CUDAStream torch_stream =
        c10::cuda::getStreamFromPool(false, 0);
    at::cuda::CUDAGraph torch_graph;
    {
      c10::cuda::CUDAStreamGuard torch_guard(torch_stream);
      for (int i = 0; i < 3; ++i) {
        run_libtorch_sampling_ops(torch_logits, torch_values, torch_token,
                                  torch_selected, torch_next_hidden,
                                  torch_hidden, torch_lm_head,
                                  torch_lm_head_t);
      }
      CHECK_CUDA(cudaStreamSynchronize(torch_stream.stream()));

      torch_graph.capture_begin();
      run_libtorch_sampling_ops(torch_logits, torch_values, torch_token,
                                torch_selected, torch_next_hidden,
                                torch_hidden, torch_lm_head, torch_lm_head_t);
      torch_graph.capture_end();
      torch_graph.replay();
      CHECK_CUDA(cudaStreamSynchronize(torch_stream.stream()));
    }
    const int64_t torch_token_id = check_libtorch_sampling_correctness(
        torch_lm_head, torch_token, torch_next_hidden);
    std::printf("benchmark_correctness_libtorch token=%ld status=passed\n",
                static_cast<long>(torch_token_id));
    print_stats("native_pytorch_cuda_graph", 1, 1, 1.0f, 0.0f,
                time_libtorch_graph_samples(torch_graph, torch_stream, warmup,
                                            iters, samples));
  }

  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  DeviceBuffer<__nv_bfloat16> d_lm_head(lm_head_elems);
  DeviceBuffer<__nv_bfloat16> d_final_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_fused_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_materialized_hidden(GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_materialized_logits(GEMMA4_VOCAB_SIZE);
  DeviceBuffer<int32_t> d_fused_token(1);
  DeviceBuffer<int32_t> d_materialized_token(1);
  const size_t fused_scratch_bytes =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE / kFinalLogitsColsPerBlock) *
      (sizeof(Gemma4SampleCandidate) + sizeof(uint32_t));
  DeviceBuffer<unsigned char> d_fused_scratch(fused_scratch_bytes);

  fill_random_bf16(d_lm_head.get(), lm_head_elems, 0x5678u, 0.05f, stream);
  fill_random_bf16(d_final_hidden.get(), GEMMA4_HIDDEN_SIZE, 0x2468u, 0.05f,
                   stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  auto fused_sample = [&]() {
    CHECK_CUDA(gemma4_sample_next_decode_bf16(
        d_fused_hidden.get(), d_fused_token.get(),
        d_fused_scratch.get(), fused_scratch_bytes, d_final_hidden.get(),
        d_lm_head.get(), stream));
  };
  auto materialized_sample = [&]() {
    CHECK_CUDA(gemma4_projection_decode(
        GEMMA4_PROJECTION_FINAL_LOGITS, d_final_hidden.get(),
        d_lm_head.get(), d_materialized_logits.get(), stream));
    materialized_logits_sample_embed_kernel<<<1, 1024, 0, stream>>>(
        d_materialized_hidden.get(),
        d_materialized_token.get(), d_lm_head.get(),
        d_materialized_logits.get());
    CHECK_CUDA(cudaGetLastError());
  };
  fused_sample();
  materialized_sample();
  CHECK_CUDA(cudaStreamSynchronize(stream));

  int32_t fused_token = -1;
  int32_t materialized_token = -2;
  CHECK_CUDA(cudaMemcpy(&fused_token, d_fused_token.get(),
                        sizeof(fused_token), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(&materialized_token,
                        d_materialized_token.get(),
                        sizeof(materialized_token),
                        cudaMemcpyDeviceToHost));
  if (fused_token != materialized_token) {
    std::fprintf(stderr,
                 "sample mismatch fused=%d materialized=%d\n",
                 fused_token, materialized_token);
    return 1;
  }
  std::printf("benchmark_correctness token=%d status=passed\n", fused_token);

  print_stats("fused_lm_head_sample_full_vocab", 1, 1, 1.0f, 0.0f,
              time_samples(fused_sample, stream, warmup, iters, samples));
  print_stats("materialized_lm_head_sample_full_vocab", 1, 1, 1.0f, 0.0f,
              time_samples(materialized_sample, stream, warmup, iters,
                           samples));
  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
