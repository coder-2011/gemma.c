#include "gemma4_embedding_gather.cuh"
#include "gemma4_bench_utils.cuh"
#include "gemma4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

std::vector<int> token_counts_up_to(int max_tokens) {
    std::vector<int> counts;
    for (int count : {1, 4, 16, 64, 256, 1024, 4096, 8192}) {
        if (count <= max_tokens) {
            counts.push_back(count);
        }
    }
    if (counts.empty() || counts.back() != max_tokens) {
        counts.push_back(max_tokens);
    }
    return counts;
}

void fill_token_ids(std::vector<int32_t>& token_ids, int count, int vocab_size) {
    uint32_t state = 0x12345678u;
    for (int i = 0; i < count; ++i) {
        state = state * 1664525u + 1013904223u;
        token_ids[i] = static_cast<int32_t>(state % static_cast<uint32_t>(vocab_size));
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 200;
    const int warmup = argc > 2 ? std::atoi(argv[2]) : 30;
    const int trials = argc > 3 ? std::atoi(argv[3]) : 5;
    const int max_tokens = argc > 4 ? std::atoi(argv[4]) : 4096;

    if (iters <= 0 || warmup < 0 || trials <= 0 || max_tokens <= 0) {
        std::fprintf(stderr, "usage: %s [iters=200] [warmup=30] [trials=5] [max_tokens=4096]\n", argv[0]);
        return 1;
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    __nv_bfloat16* d_embeddings = nullptr;
    __nv_bfloat16* d_out = nullptr;
    int32_t* d_token_ids = nullptr;

    const int hidden_size = GEMMA4_HIDDEN_SIZE;
    const int vocab_size = GEMMA4_VOCAB_SIZE;

    const size_t embedding_elems = static_cast<size_t>(vocab_size) * hidden_size;
    const size_t embedding_bytes = embedding_elems * sizeof(__nv_bfloat16);
    const size_t out_elems = static_cast<size_t>(max_tokens) * hidden_size;
    const size_t out_bytes = out_elems * sizeof(__nv_bfloat16);

    CUDA_CHECK(cudaMalloc(&d_embeddings, embedding_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_token_ids, static_cast<size_t>(max_tokens) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemsetAsync(d_embeddings, 0, embedding_bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, out_bytes, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> h_token_ids(max_tokens);

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("device=%s\n", prop.name);
    std::printf("shape=hidden%d,vocab%d,embedding_bytes=%zu,out_max_tokens=%d\n",
                hidden_size, vocab_size, embedding_bytes, max_tokens);
    std::printf("iters=%d,warmup_iters=%d,trials=%d\n", iters, warmup, trials);
    std::printf("tokens,best_ms,avg_ms,best_effective_gib_s,avg_effective_gib_s,effective_mib\n");

    for (int token_count : token_counts_up_to(max_tokens)) {
        fill_token_ids(h_token_ids, token_count, vocab_size);
        CUDA_CHECK(cudaMemcpyAsync(d_token_ids, h_token_ids.data(),
                                   static_cast<size_t>(token_count) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto run_gather = [&]() {
            CUDA_CHECK(gemma4_embedding_gather_bf16(
                d_out, d_token_ids, d_embeddings, token_count, hidden_size, vocab_size, stream));
        };

        run_gather();
        CUDA_CHECK(cudaStreamSynchronize(stream));

        TimingStats stats = time_ms(run_gather, stream, warmup, iters, trials);
        const double moved_bytes =
            2.0 * static_cast<double>(token_count) * hidden_size * sizeof(__nv_bfloat16);
        const double moved_gib = moved_bytes / (1024.0 * 1024.0 * 1024.0);
        const double best_gib_s = moved_gib / (static_cast<double>(stats.best_ms) / 1000.0);
        const double avg_gib_s = moved_gib / (static_cast<double>(stats.avg_ms) / 1000.0);
        const double moved_mib = moved_bytes / (1024.0 * 1024.0);

        std::printf("%d,%.6f,%.6f,%.3f,%.3f,%.3f\n",
                    token_count, stats.best_ms, stats.avg_ms, best_gib_s, avg_gib_s, moved_mib);
    }

    CUDA_CHECK(cudaFree(d_embeddings));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_token_ids));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
