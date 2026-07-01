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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <math.h>
#include <vector>

namespace {

constexpr int kFinalLogitsColsPerBlock = 8;
constexpr int kMaterializedSampleThreads = 1024;

struct SamplingBenchArgs {
  int warmup = 25;
  int iters = 100;
  int samples = 21;
};

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
__global__ __launch_bounds__(kMaterializedSampleThreads)
void materialized_logits_sample_embed_kernel(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    const __nv_bfloat16 *__restrict__ d_logits) {
  __shared__ float s_logits[kMaterializedSampleThreads];
  __shared__ int32_t s_token_ids[kMaterializedSampleThreads];

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

// Runs untimed warmup, then records per-iteration CUDA-event samples on the work stream.
template <typename Fn>
std::vector<float> time_samples(Fn &&fn,
                                cudaStream_t stream,
                                int warmup,
                                int iters,
                                int samples) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> values;
  values.reserve(samples);
  for (int sample = 0; sample < samples; ++sample) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
      fn();
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    values.push_back(total_ms / float(iters));
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return values;
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
  CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> values;
  values.reserve(samples);
  for (int sample = 0; sample < samples; ++sample) {
    CUDA_CHECK(cudaEventRecord(start, stream.stream()));
    for (int i = 0; i < iters; ++i) {
      graph.replay();
    }
    CUDA_CHECK(cudaEventRecord(stop, stream.stream()));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    values.push_back(total_ms / float(iters));
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
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
  char context[160];
  std::snprintf(context, sizeof(context),
                "variant=%s batch_size=%d top_k=%d top_p=%.3f temp=%.3f",
                variant, batch_size, top_k, top_p, temperature);
  gemma4_bench_print_timing_stats(
      "sampling_bench", context, summarize_timing_samples(samples));
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
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  gemma4_bench_print_common_metadata("sampling_bench");
  std::printf("benchmark_guardrail stability_scope=single_process "
              "minimum_effect_us=25 close_call_requires_process_rerun=true "
              "candidate_order=fixed\n");
  std::printf("benchmark_threats clocks=unlocked cache=warm_repeated_buffers "
              "contention=not_exclusively_verified profiler_counters=not_run "
              "process_repeats=not_run\n");

  std::printf("benchmark_contract name=sampling_bench measurement=sampling_decode "
              "timing=cuda_events "
              "stream=nonblocking cache=warm_repeated_buffers "
              "launch_overhead=cpu_enqueue_excluded_by_cuda_events "
              "aggregation=raw_samples warmup=%d iters=%d samples=%d dtype=bf16 "
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
      CUDA_CHECK(cudaStreamSynchronize(torch_stream.stream()));

      torch_graph.capture_begin();
      run_libtorch_sampling_ops(torch_logits, torch_values, torch_token,
                                torch_selected, torch_next_hidden,
                                torch_hidden, torch_lm_head, torch_lm_head_t);
      torch_graph.capture_end();
      torch_graph.replay();
      CUDA_CHECK(cudaStreamSynchronize(torch_stream.stream()));
    }
    const int64_t torch_token_id = check_libtorch_sampling_correctness(
        torch_lm_head, torch_token, torch_next_hidden);
    std::printf("benchmark_correctness_libtorch token=%ld status=passed\n",
                static_cast<long>(torch_token_id));
    print_stats("native_pytorch_cuda_graph", 1, 1, 1.0f, 0.0f,
                time_libtorch_graph_samples(torch_graph, torch_stream, warmup,
                                            iters, samples));
  }

  constexpr size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  thrust::device_vector<__nv_bfloat16> d_lm_head(lm_head_elems);
  thrust::device_vector<__nv_bfloat16> d_final_hidden(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_fused_hidden(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_materialized_hidden(GEMMA4_HIDDEN_SIZE);
  thrust::device_vector<__nv_bfloat16> d_materialized_logits(GEMMA4_VOCAB_SIZE);
  thrust::device_vector<int32_t> d_fused_token(1);
  thrust::device_vector<int32_t> d_materialized_token(1);
  constexpr size_t fused_scratch_bytes =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE / kFinalLogitsColsPerBlock) *
      (sizeof(Gemma4SampleCandidate) + sizeof(uint32_t));
  thrust::device_vector<unsigned char> d_fused_scratch(fused_scratch_bytes);

  fill_random_bf16(raw_ptr(d_lm_head), lm_head_elems, 0x5678u, 0.05f, stream);
  fill_random_bf16(
      raw_ptr(d_final_hidden), GEMMA4_HIDDEN_SIZE, 0x2468u, 0.05f, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto fused_sample = [&]() {
    CUDA_CHECK(gemma4_sample_next_bf16(
        raw_ptr(d_fused_hidden), raw_ptr(d_fused_token),
        raw_ptr(d_fused_scratch), fused_scratch_bytes, raw_ptr(d_final_hidden),
        raw_ptr(d_lm_head), Gemma4SamplingStage::kDecode, stream));
  };
  auto materialized_sample = [&]() {
    CUDA_CHECK(gemma4_projection_decode(
        GEMMA4_PROJECTION_FINAL_LOGITS, raw_ptr(d_final_hidden),
        raw_ptr(d_lm_head), raw_ptr(d_materialized_logits), stream));
    materialized_logits_sample_embed_kernel<<<
        1, kMaterializedSampleThreads, 0, stream>>>(
        raw_ptr(d_materialized_hidden),
        raw_ptr(d_materialized_token), raw_ptr(d_lm_head),
        raw_ptr(d_materialized_logits));
    CUDA_CHECK(cudaGetLastError());
  };
  fused_sample();
  materialized_sample();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  int32_t fused_token = -1;
  int32_t materialized_token = -2;
  CUDA_CHECK(cudaMemcpy(&fused_token, raw_ptr(d_fused_token),
                        sizeof(fused_token), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&materialized_token,
                        raw_ptr(d_materialized_token),
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
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
