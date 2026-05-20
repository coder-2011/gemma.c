CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Isrc
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -arch=sm_86
CUDNN_FRONTEND_INCLUDE ?= /tmp/cudnn-frontend/include

BUILD_DIR := build

.PHONY: all cuda-kernels decode-bench embedding-gather-bench rmsnorm-bench rmsnorm-hidden-fused-bench rope-bench tuna-prefill-bench sgemm-prefill-bench sgemm-bf16-prefill-bench test-cuda-utils test-embedding-gather test-rmsnorm test-rope clean

all: $(BUILD_DIR)/gemma4.o

cuda-kernels: $(BUILD_DIR)/gemma4_matmul_kernels.o $(BUILD_DIR)/gemma4_rmsnorm.o $(BUILD_DIR)/gemma4_rope.o
decode-bench: $(BUILD_DIR)/experiments/gemma4_decode_bench
embedding-gather-bench: $(BUILD_DIR)/experiments/gemma4_embedding_gather_bench
rmsnorm-bench: $(BUILD_DIR)/experiments/gemma4_rmsnorm_bench
rmsnorm-hidden-fused-bench: $(BUILD_DIR)/experiments/gemma4_rmsnorm_hidden_fused_bench
rope-bench: $(BUILD_DIR)/experiments/gemma4_rope_bench
tuna-prefill-bench: $(BUILD_DIR)/experiments/gemma4_tuna_prefill_bench
sgemm-prefill-bench: $(BUILD_DIR)/experiments/gemma4_sgemm_prefill_bench
sgemm-bf16-prefill-bench: $(BUILD_DIR)/experiments/gemma4_sgemm_bf16_prefill_bench
test-cuda-utils: $(BUILD_DIR)/tests/cuda_utils/test_cuda_utils
	./$<
test-embedding-gather: $(BUILD_DIR)/tests/test_embedding_gather
	./$<
test-rmsnorm: $(BUILD_DIR)/tests/test_rmsnorm
	./$<
test-rope: $(BUILD_DIR)/tests/test_rope
	./$<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/gemma4.o: src/gemma4.cpp src/gemma4.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c src/gemma4.cpp -o $@

$(BUILD_DIR)/gemma4_matmul_kernels.o: src/gemma4_matmul_kernels.cu src/gemma4_matmul_kernels.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c src/gemma4_matmul_kernels.cu -o $@

$(BUILD_DIR)/gemma4_rmsnorm.o: src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c src/gemma4_rmsnorm.cu -o $@

$(BUILD_DIR)/gemma4_rope.o: src/gemma4_rope.cu src/gemma4_rope.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c src/gemma4_rope.cu -o $@

$(BUILD_DIR)/experiments:
	mkdir -p $(BUILD_DIR)/experiments

$(BUILD_DIR)/tests:
	mkdir -p $(BUILD_DIR)/tests

$(BUILD_DIR)/tests/cuda_utils:
	mkdir -p $(BUILD_DIR)/tests/cuda_utils

$(BUILD_DIR)/experiments/gemma4_decode_bench: src/experiments/gemma4_decode_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_matmul_kernels.cu src/gemma4_matmul_kernels.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu -lcublas -lcudnn -o $@

$(BUILD_DIR)/experiments/gemma4_embedding_gather_bench: src/experiments/gemma4_embedding_gather_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_embedding_gather_bench.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/experiments/gemma4_rmsnorm_bench: src/experiments/gemma4_rmsnorm_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(CUDNN_FRONTEND_INCLUDE) src/experiments/gemma4_rmsnorm_bench.cu src/gemma4_rmsnorm.cu -lcudnn -lnvrtc -lcuda -o $@

$(BUILD_DIR)/experiments/gemma4_rmsnorm_hidden_fused_bench: src/experiments/gemma4_rmsnorm_hidden_fused_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_rmsnorm_hidden_fused_bench.cu src/gemma4_rmsnorm.cu -o $@

$(BUILD_DIR)/experiments/gemma4_rope_bench: src/experiments/gemma4_rope_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_rope.cu src/gemma4_rope.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_rope_bench.cu src/gemma4_rope.cu -lcudnn -o $@

$(BUILD_DIR)/experiments/gemma4_tuna_prefill_bench: experiments/tuna/gemma4_prefill_bench.cu src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) experiments/tuna/gemma4_prefill_bench.cu -lcublas -lcublasLt -o $@

$(BUILD_DIR)/experiments/gemma4_sgemm_prefill_bench: experiments/sgemm.cu/gemma4_prefill_bench.cu experiments/sgemm.cu/src/sgemm.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Iexperiments/sgemm.cu/src -Iexperiments/sgemm.cu/common -DGPUCC=86 experiments/sgemm.cu/gemma4_prefill_bench.cu -lcublas -o $@

$(BUILD_DIR)/experiments/gemma4_sgemm_bf16_prefill_bench: experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu -lcublas -lcublasLt -o $@

$(BUILD_DIR)/tests/test_embedding_gather: tests/test_embedding_gather.cu src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_embedding_gather.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/tests/cuda_utils/test_cuda_utils: tests/cuda_utils/test_cuda_utils.cu src/gemma4_cuda_utils.cuh | $(BUILD_DIR)/tests/cuda_utils
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/cuda_utils/test_cuda_utils.cu -o $@

$(BUILD_DIR)/tests/test_rmsnorm: tests/test_rmsnorm.cu src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_rmsnorm.cu src/gemma4_rmsnorm.cu -o $@

$(BUILD_DIR)/tests/test_rope: tests/test_rope.cu src/gemma4_rope.cu src/gemma4_rope.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_rope.cu src/gemma4_rope.cu -o $@

clean:
	rm -rf $(BUILD_DIR)
