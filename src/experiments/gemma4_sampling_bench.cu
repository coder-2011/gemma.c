#include "gemma4_sampling.cuh"
#include "gemma4_cuda_utils.cuh"
#include "gemma4_embedding_gather.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *expr, const char *file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n", file, line, expr,
                 cudaGetErrorString(status));
    std::exit(1);
  }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) {
    if (count > 0) {
      CHECK_CUDA(cudaMalloc(&ptr_, count * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *get() { return ptr_; }
  const T *get() const { return ptr_; }

 private:
  T *ptr_ = nullptr;
};

__device__ uint32_t mix_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

__global__ void fill_random_bf16_kernel(__nv_bfloat16 *ptr,
                                        size_t count,
                                        uint64_t seed,
                                        float scale) {
  const size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
  if (i >= count) {
    return;
  }

  uint32_t x = uint32_t(i) ^ uint32_t(i >> 32) ^ uint32_t(seed) ^
               uint32_t(seed >> 32);
  x = mix_u32(x);
  const float u = float(x >> 8) * (1.0f / 16777216.0f);
  ptr[i] = __float2bfloat16_rn((u * 2.0f - 1.0f) * scale);
}

void fill_random_bf16(__nv_bfloat16 *ptr,
                      size_t count,
                      uint64_t seed,
                      float scale,
                      cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = int((count + threads - 1) / threads);
  fill_random_bf16_kernel<<<blocks, threads, 0, stream>>>(
      ptr, count, seed, scale);
  CHECK_CUDA(cudaGetLastError());
}

__device__ inline bool better_candidate(float logit,
                                        int32_t token_id,
                                        float best_logit,
                                        int32_t best_token_id) {
  return logit > best_logit ||
         (logit == best_logit && token_id < best_token_id);
}

__global__ __launch_bounds__(1024) void logits_argmax_embed_kernel(
    __nv_bfloat16 *__restrict__ d_next_hidden,
    int32_t *__restrict__ d_next_token,
    const __nv_bfloat16 *__restrict__ d_lm_head_col_major,
    const __nv_bfloat16 *__restrict__ d_logits) {
  __shared__ float s_logits[1024];
  __shared__ int32_t s_token_ids[1024];

  const int batch_row = int(blockIdx.x);
  const int thread_idx = threadIdx.x;
  const __nv_bfloat16 *logits_row =
      d_logits + int64_t(batch_row) * GEMMA4_VOCAB_SIZE;
  float best_logit = __bfloat162float(logits_row[thread_idx]);
  int32_t best_token_id = thread_idx;

  for (int token_id = thread_idx + blockDim.x; token_id < GEMMA4_VOCAB_SIZE;
       token_id += blockDim.x) {
    const float logit = __bfloat162float(logits_row[token_id]);
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
    d_next_token[batch_row] = selected_token_id;
  }

  __nv_bfloat16 *next_hidden_row =
      d_next_hidden + int64_t(batch_row) * GEMMA4_HIDDEN_SIZE;
  gemma4_embedding_gather::copy_embedding_row_bf16(
      next_hidden_row, d_lm_head_col_major, selected_token_id, thread_idx,
      blockDim.x);
}

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
              "min_us=%.3f median_us=%.3f mean_us=%.3f p95_us=%.3f "
              "max_us=%.3f samples_us=[",
              variant, batch_size, top_k, top_p, temperature,
              sorted.front() * 1000.0f, percentile(samples, 0.50f) * 1000.0f,
              mean * 1000.0f, percentile(samples, 0.95f) * 1000.0f,
              sorted.back() * 1000.0f);
  for (size_t i = 0; i < samples.size(); ++i) {
    std::printf("%s%.3f", i == 0 ? "" : ",", samples[i] * 1000.0f);
  }
  std::printf("]\n");
}

}  // namespace

int main(int argc, char **argv) {
  const int warmup = argc > 1 ? std::atoi(argv[1]) : 25;
  const int iters = argc > 2 ? std::atoi(argv[2]) : 100;
  const int samples = argc > 3 ? std::atoi(argv[3]) : 21;
  const int max_batch = argc > 4 ? std::atoi(argv[4]) : 32;
  if (warmup < 0 || iters <= 0 || samples <= 0 || max_batch <= 0) {
    std::fprintf(stderr,
                 "usage: %s [warmup=25] [iters=100] [samples=21] "
                 "[max_batch=32]\n",
                 argv[0]);
    return 1;
  }

  cudaStream_t stream = nullptr;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cudaDeviceProp prop = {};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("env=gpu=\"%s\" global_mem_bytes=%zu\n",
              prop.name, static_cast<size_t>(prop.totalGlobalMem));
  std::printf("contract=sampling_from_materialized_logits timing=cuda_events "
              "stream=nonblocking cache=warm-ish launch_overhead=included "
              "warmup=%d iters=%d samples=%d max_batch=%d dtype=bf16 "
              "vocab=%d hidden=%d\n",
              warmup, iters, samples, max_batch, GEMMA4_VOCAB_SIZE,
              GEMMA4_HIDDEN_SIZE);

  const size_t lm_head_elems =
      static_cast<size_t>(GEMMA4_VOCAB_SIZE) * GEMMA4_HIDDEN_SIZE;
  DeviceBuffer<__nv_bfloat16> d_lm_head(lm_head_elems);
  DeviceBuffer<__nv_bfloat16> d_prob_logits(
      static_cast<size_t>(max_batch) * GEMMA4_VOCAB_SIZE);
  DeviceBuffer<__nv_bfloat16> d_prob_hidden(
      static_cast<size_t>(max_batch) * GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<__nv_bfloat16> d_argmax_hidden(
      static_cast<size_t>(max_batch) * GEMMA4_HIDDEN_SIZE);
  DeviceBuffer<int32_t> d_prob_token(max_batch);
  DeviceBuffer<int32_t> d_argmax_token(max_batch);

  fill_random_bf16(d_lm_head.get(), lm_head_elems, 0x5678u, 0.05f, stream);
  fill_random_bf16(d_prob_logits.get(),
                   static_cast<size_t>(max_batch) * GEMMA4_VOCAB_SIZE,
                   0x9abcu, 4.0f, stream);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  const int batch_sizes[] = {1, 8, 32, 128};
  const Gemma4SamplingParams variants[] = {
      {1.0f, 0.95f, 64, 0x12345678u, 5u},
      {1.0f, 1.0f, 1, 0x12345678u, 5u},
      {1.0f, 1.0f, 64, 0x12345678u, 5u},
  };
  const char *variant_names[] = {
      "prob_sampler_topk64_topp095",
      "prob_sampler_topk1_topp1",
      "prob_sampler_topk64_topp1",
  };

  for (int batch_size : batch_sizes) {
    if (batch_size > max_batch) {
      continue;
    }

    auto argmax_embed = [&]() {
      logits_argmax_embed_kernel<<<batch_size, 1024, 0, stream>>>(
          d_argmax_hidden.get(), d_argmax_token.get(), d_lm_head.get(),
          d_prob_logits.get());
      CHECK_CUDA(cudaGetLastError());
    };
    argmax_embed();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    print_stats("logits_argmax_embed_baseline", batch_size, 1, 1.0f, 1.0f,
                time_samples(argmax_embed, stream, warmup, iters, samples));

    for (int variant_idx = 0; variant_idx < 3; ++variant_idx) {
      const Gemma4SamplingParams params = variants[variant_idx];
      auto prob_sampler = [&]() {
        CHECK_CUDA(gemma4_sample_from_logits_decode_bf16(
            d_prob_hidden.get(), d_prob_token.get(), nullptr, 0,
            d_prob_logits.get(), d_lm_head.get(),
            batch_size, params, stream));
      };

      prob_sampler();
      CHECK_CUDA(cudaStreamSynchronize(stream));
      print_stats(variant_names[variant_idx], batch_size, params.top_k,
                  params.top_p, params.temperature,
                  time_samples(prob_sampler, stream, warmup, iters, samples));
    }
  }

  CHECK_CUDA(cudaStreamDestroy(stream));
  return 0;
}
