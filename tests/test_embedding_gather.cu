#include "gemma4_embedding_gather.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* expr, const char* file, int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s:%d: CUDA error for %s: %s\n",
                     file, line, expr, cudaGetErrorString(status));
        std::exit(1);
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr, __FILE__, __LINE__)

__nv_bfloat16 make_value(int token, int channel) {
    float value = static_cast<float>((token * 131 + channel * 17) % 4096) / 16.0f;
    return __float2bfloat16(value);
}

void run_case(int vocab_size, int hidden_size, const std::vector<int32_t>& token_ids) {
    int num_tokens = static_cast<int>(token_ids.size());
    std::vector<__nv_bfloat16> embeddings(static_cast<size_t>(vocab_size) * hidden_size);
    std::vector<__nv_bfloat16> out(static_cast<size_t>(num_tokens) * hidden_size);

    for (int token = 0; token < vocab_size; ++token) {
        for (int c = 0; c < hidden_size; ++c) {
            embeddings[static_cast<size_t>(token) * hidden_size + c] = make_value(token, c);
        }
    }

    __nv_bfloat16* d_embeddings = nullptr;
    __nv_bfloat16* d_out = nullptr;
    int32_t* d_token_ids = nullptr;

    CHECK_CUDA(cudaMalloc(&d_embeddings, embeddings.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_out, out.size() * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_token_ids, token_ids.size() * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_embeddings, embeddings.data(),
                          embeddings.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_token_ids, token_ids.data(),
                          token_ids.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(gemma4_embedding_gather_bf16(
        d_out, d_token_ids, d_embeddings, num_tokens, hidden_size, vocab_size, 0));
    CHECK_CUDA(cudaMemcpy(out.data(), d_out, out.size() * sizeof(__nv_bfloat16),
                          cudaMemcpyDeviceToHost));

    const uint16_t* out_bits = reinterpret_cast<const uint16_t*>(out.data());
    const uint16_t* embedding_bits = reinterpret_cast<const uint16_t*>(embeddings.data());
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
        int32_t token_id = token_ids[token_idx];
        for (int c = 0; c < hidden_size; ++c) {
            uint16_t actual = out_bits[static_cast<size_t>(token_idx) * hidden_size + c];
            uint16_t expected = embedding_bits[static_cast<size_t>(token_id) * hidden_size + c];
            if (actual != expected) {
                std::fprintf(stderr,
                             "mismatch token_idx=%d token_id=%d channel=%d actual=0x%04x expected=0x%04x\n",
                             token_idx, token_id, c, actual, expected);
                std::exit(1);
            }
        }
    }

    CHECK_CUDA(cudaFree(d_embeddings));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_token_ids));
}

}  // namespace

int main() {
    run_case(257, GEMMA4_HIDDEN_SIZE,
             {0, 1, 42, 42, 128, 256, 7, 19, 0, 255, 3});
    run_case(17, 256, {16, 0, 8, 8, 3});

    cudaError_t invalid = gemma4_embedding_gather_bf16(
        nullptr, nullptr, nullptr, 1, GEMMA4_HIDDEN_SIZE + 1, 17, 0);
    if (invalid != cudaErrorInvalidValue) {
        std::fprintf(stderr, "expected cudaErrorInvalidValue for invalid arguments\n");
        return 1;
    }

    std::printf("embedding gather tests passed\n");
    return 0;
}
