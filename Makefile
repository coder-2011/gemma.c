CC ?= clang
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Isrc
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -arch=sm_86
CUDNN_FRONTEND_INCLUDE ?= /tmp/cudnn-frontend/include

BUILD_DIR := build

.PHONY: all cuda-kernels decode-bench embedding-gather-bench rmsnorm-bench test-embedding-gather test-rmsnorm clean

all: $(BUILD_DIR)/gemma4.o

cuda-kernels: $(BUILD_DIR)/gemma4_matmul_kernels.o $(BUILD_DIR)/gemma4_rmsnorm.o
decode-bench: $(BUILD_DIR)/experiments/gemma4_decode_bench
embedding-gather-bench: $(BUILD_DIR)/experiments/gemma4_embedding_gather_bench
rmsnorm-bench: $(BUILD_DIR)/experiments/gemma4_rmsnorm_bench
test-embedding-gather: $(BUILD_DIR)/tests/test_embedding_gather
	./$<
test-rmsnorm: $(BUILD_DIR)/tests/test_rmsnorm
	./$<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/gemma4.o: src/gemma4.c src/gemma4.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/gemma4.c -o $@

$(BUILD_DIR)/gemma4_matmul_kernels.o: src/gemma4_matmul_kernels.cu src/gemma4_matmul_kernels.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c src/gemma4_matmul_kernels.cu -o $@

$(BUILD_DIR)/gemma4_rmsnorm.o: src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c src/gemma4_rmsnorm.cu -o $@

$(BUILD_DIR)/experiments:
	mkdir -p $(BUILD_DIR)/experiments

$(BUILD_DIR)/tests:
	mkdir -p $(BUILD_DIR)/tests

$(BUILD_DIR)/experiments/gemma4_decode_bench: src/experiments/gemma4_decode_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_matmul_kernels.cu src/gemma4_matmul_kernels.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu -lcublas -o $@

$(BUILD_DIR)/experiments/gemma4_embedding_gather_bench: src/experiments/gemma4_embedding_gather_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_embedding_gather_bench.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/experiments/gemma4_rmsnorm_bench: src/experiments/gemma4_rmsnorm_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(CUDNN_FRONTEND_INCLUDE) src/experiments/gemma4_rmsnorm_bench.cu src/gemma4_rmsnorm.cu -lcudnn -lnvrtc -lcuda -o $@

$(BUILD_DIR)/tests/test_embedding_gather: tests/test_embedding_gather.cu src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_embedding_gather.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/tests/test_rmsnorm: tests/test_rmsnorm.cu src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_rmsnorm.cu src/gemma4_rmsnorm.cu -o $@

clean:
	rm -rf $(BUILD_DIR)
