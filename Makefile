CC ?= clang
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Isrc
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -arch=sm_86

BUILD_DIR := build

.PHONY: all cuda-kernels decode-bench embedding-gather-bench test-embedding-gather clean

all: $(BUILD_DIR)/gemma4.o

cuda-kernels: $(BUILD_DIR)/gemma4_matmul_kernels.o
decode-bench: $(BUILD_DIR)/experiments/gemma4_decode_bench
embedding-gather-bench: $(BUILD_DIR)/experiments/gemma4_embedding_gather_bench
test-embedding-gather: $(BUILD_DIR)/tests/test_embedding_gather
	./$<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/gemma4.o: src/gemma4.c src/gemma4.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/gemma4.c -o $@

$(BUILD_DIR)/gemma4_matmul_kernels.o: src/gemma4_matmul_kernels.cu src/gemma4_matmul_kernels.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c src/gemma4_matmul_kernels.cu -o $@

$(BUILD_DIR)/experiments:
	mkdir -p $(BUILD_DIR)/experiments

$(BUILD_DIR)/tests:
	mkdir -p $(BUILD_DIR)/tests

$(BUILD_DIR)/experiments/gemma4_decode_bench: src/experiments/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu src/gemma4_matmul_kernels.cuh | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu -lcublas -o $@

$(BUILD_DIR)/experiments/gemma4_embedding_gather_bench: src/experiments/gemma4_embedding_gather_bench.cu src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_embedding_gather_bench.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/tests/test_embedding_gather: tests/test_embedding_gather.cu src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_embedding_gather.cu src/gemma4_embedding_gather.cu -o $@

clean:
	rm -rf $(BUILD_DIR)
