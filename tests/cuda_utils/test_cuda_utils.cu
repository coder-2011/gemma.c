#include "gemma4_cuda_utils.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

__global__ void warp_reduce_sum_real_kernel(const float *inp, float *out,
                                            int warps_per_block) {
  int lane = threadIdx.x & (WARP_SIZE - 1);
  int warp = threadIdx.x / WARP_SIZE;
  int global_warp = blockIdx.x * warps_per_block + warp;
  int index = global_warp * WARP_SIZE + lane;

  float value = inp[index];
  out[index] = warp_reduce_sum(value);
}

float make_real_value(int index) {
  int centered = ((index * 37 + 11) % 113) - 56;
  float base = static_cast<float>(centered) / 7.0f;
  return base + 0.001f * static_cast<float>((index % 5) - 2);
}

}  // namespace

int main() {
  constexpr int blocks = 2;
  constexpr int warps_per_block = 3;
  constexpr int total_warps = blocks * warps_per_block;
  constexpr int total_values = total_warps * WARP_SIZE;
  constexpr int threads = warps_per_block * WARP_SIZE;

  std::vector<float> inp(total_values);
  std::vector<float> out(total_values);
  std::vector<float> expected(total_warps);

  for (int i = 0; i < total_values; ++i) {
    inp[i] = make_real_value(i);
  }

  for (int warp = 0; warp < total_warps; ++warp) {
    float sum = 0.0f;
    for (int lane = 0; lane < WARP_SIZE; ++lane) {
      sum += inp[warp * WARP_SIZE + lane];
    }
    expected[warp] = sum;
  }

  float *d_inp = nullptr;
  float *d_out = nullptr;
  CHECK_CUDA(cudaMalloc(&d_inp, inp.size() * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out, out.size() * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(d_inp, inp.data(), inp.size() * sizeof(float), cudaMemcpyHostToDevice));

  warp_reduce_sum_real_kernel<<<blocks, threads>>>(d_inp, d_out, warps_per_block);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  for (int warp = 0; warp < total_warps; ++warp) {
    for (int lane = 0; lane < WARP_SIZE; ++lane) {
      int index = warp * WARP_SIZE + lane;
      float diff = std::fabs(out[index] - expected[warp]);
      if (diff > 1.0e-5f) {
        std::fprintf(stderr,
                     "warp_reduce_sum mismatch warp=%d lane=%d actual=%0.9g "
                     "expected=%0.9g diff=%0.9g\n",
                     warp, lane, out[index], expected[warp], diff);
        return 1;
      }
    }
  }

  CHECK_CUDA(cudaFree(d_inp));
  CHECK_CUDA(cudaFree(d_out));

  std::printf("cuda utils tests passed\n");
  return 0;
}
