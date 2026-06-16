CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Isrc
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -arch=sm_86
CUDNN_FRONTEND_INCLUDE ?= /tmp/cudnn-frontend/include
CUTLASS_INCLUDE ?= /tmp/cutlass/include

BUILD_DIR := build

FLASH_ATTN_SRC_DIR ?= experiments/flash-attention/csrc/flash_attn/src
FLASH_ATTN_CUTLASS_INCLUDE ?= experiments/flash-attention/csrc/cutlass/include
FLASH_ATTN_NVCCFLAGS ?= --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -D_GLIBCXX_USE_CXX11_ABI=1
TORCH_INCLUDE ?= .venv/lib/python3.11/site-packages/torch/include
TORCH_API_INCLUDE ?= .venv/lib/python3.11/site-packages/torch/include/torch/csrc/api/include
TORCH_LIB ?= .venv/lib/python3.11/site-packages/torch/lib
PYTHON_INCLUDE ?= /usr/include/python3.11

.PHONY: all cuda-kernels decode-bench embedding-gather-bench rmsnorm-bench rmsnorm-hidden-fused-bench rope-bench ffn-cudnn-bench ffn-decode-load-bench flash-attn-bench flash-attn-lib flash-attn-reference-lib tuna-prefill-bench sgemm-prefill-bench sgemm-bf16-prefill-bench test-cuda-utils test-embedding-gather test-rmsnorm test-rope test-ffn-decode clean

all: $(BUILD_DIR)/gemma4.o

cuda-kernels: $(BUILD_DIR)/gemma4_matmul_kernels.o $(BUILD_DIR)/gemma4_rmsnorm.o $(BUILD_DIR)/gemma4_rope.o $(BUILD_DIR)/gemma4_flash_attention.o $(BUILD_DIR)/gemma4_ffn_decode.o
decode-bench: $(BUILD_DIR)/experiments/gemma4_decode_bench
embedding-gather-bench: $(BUILD_DIR)/experiments/gemma4_embedding_gather_bench
rmsnorm-bench: $(BUILD_DIR)/experiments/gemma4_rmsnorm_bench
rmsnorm-hidden-fused-bench: $(BUILD_DIR)/experiments/gemma4_rmsnorm_hidden_fused_bench
rope-bench: $(BUILD_DIR)/experiments/gemma4_rope_bench
ffn-cudnn-bench: $(BUILD_DIR)/experiments/gemma4_ffn_cudnn_bench
ffn-decode-load-bench: $(BUILD_DIR)/experiments/gemma4_ffn_decode_load_bench
flash-attn-bench: $(BUILD_DIR)/experiments/gemma4_flash_attention_bench
flash-attn-lib: $(BUILD_DIR)/libgemma4_flash_attention.so
flash-attn-reference-lib: $(BUILD_DIR)/libgemma4_flash_attention_reference.so
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
test-ffn-decode: $(BUILD_DIR)/tests/test_ffn_decode
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

$(BUILD_DIR)/gemma4_flash_attention.o: src/gemma4_flash_attention.cu src/gemma4_flash_attention.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASH_ATTN_NVCCFLAGS) $(CPPFLAGS) -I$(FLASH_ATTN_CUTLASS_INCLUDE) -c src/gemma4_flash_attention.cu -o $@

$(BUILD_DIR)/gemma4_ffn_decode.o: src/gemma4_ffn_decode.cu src/gemma4_ffn_decode.cuh src/gemma4_matmul_device.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c src/gemma4_ffn_decode.cu -o $@

$(BUILD_DIR)/libgemma4_flash_attention.so: src/gemma4_flash_attention.cu src/gemma4_flash_attention.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASH_ATTN_NVCCFLAGS) -Xcompiler -fPIC -shared $(CPPFLAGS) -I$(FLASH_ATTN_CUTLASS_INCLUDE) src/gemma4_flash_attention.cu -o $@

$(BUILD_DIR)/libgemma4_flash_attention_reference.so: src/experiments/gemma4_flash_attention_reference.cu src/gemma4.h $(FLASH_ATTN_SRC_DIR)/flash_fwd_hdim256_bf16_sm80.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FLASH_ATTN_NVCCFLAGS) -Xcompiler -fPIC -shared $(CPPFLAGS) -Isrc/third_party_stubs -I$(FLASH_ATTN_SRC_DIR) -I$(FLASH_ATTN_CUTLASS_INCLUDE) -I$(TORCH_INCLUDE) -I$(TORCH_API_INCLUDE) -I$(PYTHON_INCLUDE) src/experiments/gemma4_flash_attention_reference.cu $(FLASH_ATTN_SRC_DIR)/flash_fwd_hdim256_bf16_sm80.cu -L$(TORCH_LIB) -Xlinker -rpath -Xlinker $(abspath $(TORCH_LIB)) -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -o $@

$(BUILD_DIR)/experiments:
	mkdir -p $(BUILD_DIR)/experiments

$(BUILD_DIR)/tests:
	mkdir -p $(BUILD_DIR)/tests

$(BUILD_DIR)/tests/cuda_utils:
	mkdir -p $(BUILD_DIR)/tests/cuda_utils

$(BUILD_DIR)/experiments/gemma4_decode_bench: src/experiments/gemma4_decode_bench.cu src/experiments/gemma4_bench_utils.cuh src/experiments/gemma4_matmul_device_kernels.cu src/experiments/gemma4_matmul_device_kernels.cuh src/gemma4_matmul_device.cuh src/gemma4_matmul_kernels.cu src/gemma4_matmul_kernels.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu src/experiments/gemma4_matmul_device_kernels.cu -lcublas -lcudnn -o $@

$(BUILD_DIR)/experiments/gemma4_embedding_gather_bench: src/experiments/gemma4_embedding_gather_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_embedding_gather_bench.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/experiments/gemma4_rmsnorm_bench: src/experiments/gemma4_rmsnorm_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(CUDNN_FRONTEND_INCLUDE) src/experiments/gemma4_rmsnorm_bench.cu src/gemma4_rmsnorm.cu -lcudnn -lnvrtc -lcuda -o $@

$(BUILD_DIR)/experiments/gemma4_rmsnorm_hidden_fused_bench: src/experiments/gemma4_rmsnorm_hidden_fused_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_rmsnorm_hidden_fused_bench.cu src/gemma4_rmsnorm.cu -o $@

$(BUILD_DIR)/experiments/gemma4_rope_bench: src/experiments/gemma4_rope_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_rope.cu src/gemma4_rope.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_rope_bench.cu src/gemma4_rope.cu -lcudnn -o $@

$(BUILD_DIR)/experiments/gemma4_ffn_cudnn_bench: src/experiments/gemma4_ffn_cudnn_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_ffn_decode.cu src/gemma4_ffn_decode.cuh src/gemma4_matmul_device.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(CUDNN_FRONTEND_INCLUDE) src/experiments/gemma4_ffn_cudnn_bench.cu src/gemma4_ffn_decode.cu -lcudnn -lnvrtc -lcuda -o $@

$(BUILD_DIR)/experiments/gemma4_ffn_decode_load_bench: src/experiments/gemma4_ffn_decode_load_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_ffn_decode.cu src/gemma4_ffn_decode.cuh src/gemma4_matmul_device.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/experiments/gemma4_ffn_decode_load_bench.cu src/gemma4_ffn_decode.cu -o $@

$(BUILD_DIR)/experiments/gemma4_flash_attention_bench: src/experiments/gemma4_flash_attention_bench.cu src/experiments/gemma4_bench_utils.cuh src/gemma4_flash_attention.cu src/gemma4_flash_attention.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(FLASH_ATTN_NVCCFLAGS) $(CPPFLAGS) -I$(FLASH_ATTN_CUTLASS_INCLUDE) src/experiments/gemma4_flash_attention_bench.cu src/gemma4_flash_attention.cu -o $@

$(BUILD_DIR)/experiments/gemma4_tuna_prefill_bench: experiments/tuna/gemma4_prefill_bench.cu src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) experiments/tuna/gemma4_prefill_bench.cu -lcublas -lcublasLt -o $@

$(BUILD_DIR)/experiments/gemma4_sgemm_prefill_bench: experiments/sgemm.cu/gemma4_prefill_bench.cu experiments/sgemm.cu/src/sgemm.cuh src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Iexperiments/sgemm.cu/src -Iexperiments/sgemm.cu/common -DGPUCC=86 experiments/sgemm.cu/gemma4_prefill_bench.cu -lcublas -o $@

$(BUILD_DIR)/experiments/gemma4_sgemm_bf16_prefill_bench: experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu src/gemma4.h | $(BUILD_DIR)/experiments
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(CUTLASS_INCLUDE) experiments/sgemm.cu/gemma4_bf16_prefill_bench.cu -lcublas -lcublasLt -o $@

$(BUILD_DIR)/tests/test_embedding_gather: tests/test_embedding_gather.cu src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_embedding_gather.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/tests/cuda_utils/test_cuda_utils: tests/cuda_utils/test_cuda_utils.cu src/gemma4_cuda_utils.cuh | $(BUILD_DIR)/tests/cuda_utils
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/cuda_utils/test_cuda_utils.cu -o $@

$(BUILD_DIR)/tests/test_rmsnorm: tests/test_rmsnorm.cu src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_rmsnorm.cu src/gemma4_rmsnorm.cu -o $@

$(BUILD_DIR)/tests/test_rope: tests/test_rope.cu src/gemma4_rope.cu src/gemma4_rope.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_rope.cu src/gemma4_rope.cu -o $@

$(BUILD_DIR)/tests/test_ffn_decode: tests/test_ffn_decode.cu src/gemma4_ffn_decode.cu src/gemma4_ffn_decode.cuh src/gemma4_matmul_device.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_ffn_decode.cu src/gemma4_ffn_decode.cu -o $@

clean:
	rm -rf $(BUILD_DIR)
