CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Isrc
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
NVCCFLAGS ?= -std=c++17 -O3 -arch=sm_86
RDC_NVCCFLAGS ?= -rdc=true
CUDA_LIB ?= $(CUDA_HOME)/targets/x86_64-linux/lib
CUDA_LDFLAGS ?= -L$(CUDA_LIB) -Xlinker -rpath -Xlinker $(CUDA_LIB)
CUDNN_INCLUDE ?= /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include
CUDNN_LIB ?= /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib
CUDNN_LIBS ?= -l:libcudnn.so.9
CUDNN_LDFLAGS ?= -L$(CUDNN_LIB) -Xlinker -rpath -Xlinker $(CUDNN_LIB) $(CUDNN_LIBS)
CUDNN_FRONTEND_INCLUDE ?= .venv/lib/python3.12/site-packages/include
CUTLASS_INCLUDE ?= /tmp/cutlass/include
CUTLASS_DUAL_GEMM_INCLUDE ?= /tmp/cutlass/examples/45_dual_gemm
FFN_CUTLASS_CPPFLAGS ?= -I$(CUTLASS_INCLUDE) -I$(CUTLASS_DUAL_GEMM_INCLUDE) -I$(FLASH_ATTN_CUTLASS_INCLUDE)
FFN_CUTLASS_NVCCFLAGS ?= --expt-relaxed-constexpr

BUILD_DIR := build

FLASH_ATTN_CUTLASS_INCLUDE ?= $(CUTLASS_INCLUDE)
FLASH_ATTN_CUTLASS_CPPFLAGS ?= -I$(FLASH_ATTN_CUTLASS_INCLUDE) -I$(CUTLASS_INCLUDE)
FLASH_ATTN_NVCCFLAGS ?= --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -D_GLIBCXX_USE_CXX11_ABI=1
TORCH_PYTHON ?= python3
TORCH_ROOT ?= $(shell $(TORCH_PYTHON) -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)' 2>/dev/null)
TORCH_INCLUDE ?= $(TORCH_ROOT)/include
TORCH_API_INCLUDE ?= $(TORCH_INCLUDE)/torch/csrc/api/include
TORCH_LIB ?= $(TORCH_ROOT)/lib
TORCH_CPPFLAGS ?= -I$(TORCH_INCLUDE) -I$(TORCH_API_INCLUDE)
TORCH_NVCCFLAGS ?= -D_GLIBCXX_USE_CXX11_ABI=1
TORCH_LDFLAGS ?= -L$(TORCH_LIB) -Xlinker -rpath -Xlinker $(TORCH_LIB)
TORCH_LIBS ?= -ltorch -ltorch_cuda -ltorch_cpu -lc10_cuda -lc10
PYTHON_INCLUDE ?= /usr/include/python3.11

.PHONY: all cuda-kernels prompt benchmarks decode-bench embedding-gather-bench sampling-bench rmsnorm-bench rmsnorm-hidden-fused-bench ffn-libtorch-bench flash-attn-bench kv-cache-bench tokenizer-bench flash-attn-lib test-cuda-utils test-tokenizer test-embedding-gather test-sampling test-rmsnorm test-ffn-decode test-prefill-gemm test-kv-cache test-flash-attention test-flash-attention-cpp test-flash-attention-pytorch test-runtime-state test-decode-megakernel test-prefill-megakernel test-checkpoint-loader clean

all: cuda-kernels

cuda-kernels: $(BUILD_DIR)/gemma4_matmul_kernels.o $(BUILD_DIR)/gemma4_sampling.o $(BUILD_DIR)/gemma4_rmsnorm.o $(BUILD_DIR)/gemma4_rope.o $(BUILD_DIR)/gemma4_flash_attention.o $(BUILD_DIR)/gemma4_ffn.o $(BUILD_DIR)/gemma4_kv_cache.o $(BUILD_DIR)/gemma4_runtime.o $(BUILD_DIR)/gemma4_decode_megakernel.o $(BUILD_DIR)/gemma4_prefill_megakernel.o $(BUILD_DIR)/gemma4_checkpoint.o
prompt: $(BUILD_DIR)/gemma4_prompt
benchmarks: decode-bench embedding-gather-bench sampling-bench rmsnorm-bench rmsnorm-hidden-fused-bench ffn-libtorch-bench flash-attn-bench kv-cache-bench tokenizer-bench
decode-bench: $(BUILD_DIR)/benches/gemma4_decode_bench
embedding-gather-bench: $(BUILD_DIR)/benches/gemma4_embedding_gather_bench
sampling-bench: $(BUILD_DIR)/benches/gemma4_sampling_bench
rmsnorm-bench: $(BUILD_DIR)/benches/gemma4_rmsnorm_bench
rmsnorm-hidden-fused-bench: $(BUILD_DIR)/benches/gemma4_rmsnorm_hidden_fused_bench
ffn-libtorch-bench: $(BUILD_DIR)/benches/gemma4_ffn_libtorch_bench
flash-attn-bench: $(BUILD_DIR)/benches/gemma4_flash_attention_bench
kv-cache-bench: $(BUILD_DIR)/benches/gemma4_kv_cache_bench
tokenizer-bench: $(BUILD_DIR)/benches/gemma4_tokenizer_bench
flash-attn-lib: $(BUILD_DIR)/libgemma4_flash_attention.so
test-cuda-utils: $(BUILD_DIR)/tests/cuda_utils/test_cuda_utils
	./$<
test-tokenizer: $(BUILD_DIR)/tests/test_tokenizer
	./$<
test-embedding-gather: $(BUILD_DIR)/tests/test_embedding_gather
	./$<
test-sampling: $(BUILD_DIR)/tests/test_sampling
	./$<
test-rmsnorm: $(BUILD_DIR)/tests/test_rmsnorm
	./$<
test-ffn-decode: $(BUILD_DIR)/tests/test_ffn_decode
	./$<
test-prefill-gemm: $(BUILD_DIR)/tests/test_prefill_gemm
	./$<
test-kv-cache: $(BUILD_DIR)/tests/test_kv_cache
	./$<
test-flash-attention: test-flash-attention-cpp test-flash-attention-pytorch
test-flash-attention-cpp: $(BUILD_DIR)/tests/test_flash_attention
	./$(BUILD_DIR)/tests/test_flash_attention
test-flash-attention-pytorch: $(BUILD_DIR)/libgemma4_flash_attention.so
	GEMMA4_FLASH_ATTENTION_LIB=$(BUILD_DIR)/libgemma4_flash_attention.so uv run python tests/test_flash_attention_pytorch.py
test-runtime-state: $(BUILD_DIR)/tests/test_runtime_state
	./$<
test-decode-megakernel: $(BUILD_DIR)/tests/test_decode_megakernel
	./$<
test-prefill-megakernel: $(BUILD_DIR)/tests/test_prefill_megakernel
	./$<
test-checkpoint-loader: $(BUILD_DIR)/tests/test_checkpoint_loader
	./$<
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/gemma4_tokenizer.o: src/gemma4_tokenizer.cpp src/gemma4_tokenizer.cuh | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c src/gemma4_tokenizer.cpp -o $@

$(BUILD_DIR)/gemma4_matmul_kernels.o: src/gemma4_matmul_kernels.cu src/gemma4_matmul_kernels.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) -c src/gemma4_matmul_kernels.cu -o $@

$(BUILD_DIR)/gemma4_sampling.o: src/gemma4_sampling.cu src/gemma4_sampling.cuh src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(CPPFLAGS) -c src/gemma4_sampling.cu -o $@

$(BUILD_DIR)/gemma4_rmsnorm.o: src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(CPPFLAGS) -I$(CUTLASS_INCLUDE) -c src/gemma4_rmsnorm.cu -o $@

$(BUILD_DIR)/gemma4_rope.o: src/gemma4_rope.cu src/gemma4_rope.cuh src/gemma4_cuda_utils.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(CPPFLAGS) -c src/gemma4_rope.cu -o $@

$(BUILD_DIR)/gemma4_flash_attention.o: src/gemma4_flash_attention.cu src/gemma4_flash_attention.cuh src/gemma4_decode_megakernel.cuh src/gemma4_ffn.cuh src/gemma4_rope.cuh src/gemma4_kv_cache.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(FLASH_ATTN_NVCCFLAGS) $(CPPFLAGS) $(FLASH_ATTN_CUTLASS_CPPFLAGS) -c src/gemma4_flash_attention.cu -o $@

$(BUILD_DIR)/gemma4_ffn.o: src/gemma4_ffn.cu src/gemma4_ffn.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) -c src/gemma4_ffn.cu -o $@

$(BUILD_DIR)/gemma4_kv_cache.o: src/gemma4_kv_cache.cu src/gemma4_kv_cache.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(FLASH_ATTN_CUTLASS_INCLUDE) -c src/gemma4_kv_cache.cu -o $@

$(BUILD_DIR)/gemma4_runtime.o: src/gemma4_runtime.cu src/gemma4_runtime.cuh src/gemma4_kv_cache.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(FLASH_ATTN_CUTLASS_INCLUDE) -c src/gemma4_runtime.cu -o $@

$(BUILD_DIR)/gemma4_decode_megakernel.o: src/gemma4_decode_megakernel.cu src/gemma4_decode_megakernel.cuh src/gemma4_flash_attention.cuh src/gemma4_kv_cache.cuh src/gemma4_ffn.cuh src/gemma4_sampling.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(CPPFLAGS) $(FLASH_ATTN_CUTLASS_CPPFLAGS) -c src/gemma4_decode_megakernel.cu -o $@

$(BUILD_DIR)/gemma4_prefill_megakernel.o: src/gemma4_prefill_megakernel.cu src/gemma4_decode_megakernel.cuh src/gemma4_checkpoint.cuh src/gemma4_flash_attention.cuh src/gemma4_kv_cache.cuh src/gemma4_ffn.cuh src/gemma4_matmul_kernels.cuh src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(FLASH_ATTN_CUTLASS_INCLUDE) -c src/gemma4_prefill_megakernel.cu -o $@

$(BUILD_DIR)/gemma4_checkpoint.o: src/gemma4_checkpoint.cu src/gemma4_checkpoint.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -c src/gemma4_checkpoint.cu -o $@

$(BUILD_DIR)/libgemma4_flash_attention.so: src/gemma4_flash_attention.cu src/gemma4_flash_attention.cuh src/gemma4_decode_megakernel.cu src/gemma4_decode_megakernel.cuh src/gemma4_runtime.cu src/gemma4_runtime.cuh src/gemma4_sampling.cu src/gemma4_sampling.cuh src/gemma4_ffn.cu src/gemma4_ffn.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_kv_cache.cu src/gemma4_kv_cache.cuh src/gemma4_rope.cu src/gemma4_rope.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(FLASH_ATTN_NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) -Xcompiler -fPIC -shared $(CPPFLAGS) $(FLASH_ATTN_CUTLASS_CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) src/gemma4_flash_attention.cu src/gemma4_decode_megakernel.cu src/gemma4_runtime.cu src/gemma4_sampling.cu src/gemma4_ffn.cu src/gemma4_rmsnorm.cu src/gemma4_kv_cache.cu src/gemma4_rope.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/gemma4_prompt: src/gemma4_prompt.cu src/gemma4_tokenizer.cuh src/gemma4_cuda_utils.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_runtime.cuh src/gemma4_decode_megakernel.cuh $(BUILD_DIR)/gemma4_tokenizer.o $(BUILD_DIR)/gemma4_checkpoint.o $(BUILD_DIR)/gemma4_runtime.o $(BUILD_DIR)/gemma4_prefill_megakernel.o $(BUILD_DIR)/gemma4_decode_megakernel.o $(BUILD_DIR)/gemma4_sampling.o $(BUILD_DIR)/gemma4_matmul_kernels.o $(BUILD_DIR)/gemma4_flash_attention.o $(BUILD_DIR)/gemma4_rope.o $(BUILD_DIR)/gemma4_kv_cache.o $(BUILD_DIR)/gemma4_ffn.o $(BUILD_DIR)/gemma4_rmsnorm.o | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(FLASH_ATTN_NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(FLASH_ATTN_CUTLASS_CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) src/gemma4_prompt.cu src/gemma4_embedding_gather.cu $(BUILD_DIR)/gemma4_tokenizer.o $(BUILD_DIR)/gemma4_checkpoint.o $(BUILD_DIR)/gemma4_runtime.o $(BUILD_DIR)/gemma4_prefill_megakernel.o $(BUILD_DIR)/gemma4_decode_megakernel.o $(BUILD_DIR)/gemma4_sampling.o $(BUILD_DIR)/gemma4_matmul_kernels.o $(BUILD_DIR)/gemma4_flash_attention.o $(BUILD_DIR)/gemma4_rope.o $(BUILD_DIR)/gemma4_kv_cache.o $(BUILD_DIR)/gemma4_ffn.o $(BUILD_DIR)/gemma4_rmsnorm.o $(CUDA_LDFLAGS) -lcublas -o $@

$(BUILD_DIR)/benches:
	mkdir -p $(BUILD_DIR)/benches

$(BUILD_DIR)/tests:
	mkdir -p $(BUILD_DIR)/tests

$(BUILD_DIR)/tests/cuda_utils:
	mkdir -p $(BUILD_DIR)/tests/cuda_utils

$(BUILD_DIR)/benches/gemma4_decode_bench: src/benches/gemma4_decode_bench.cu src/benches/gemma4_bench_utils.cuh src/gemma4_matmul_kernels.cu src/gemma4_matmul_kernels.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/benches
	$(NVCC) $(NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) -I$(CUDNN_INCLUDE) src/benches/gemma4_decode_bench.cu src/gemma4_matmul_kernels.cu $(CUDA_LDFLAGS) -lcublas $(CUDNN_LDFLAGS) -o $@

$(BUILD_DIR)/benches/gemma4_embedding_gather_bench: src/benches/gemma4_embedding_gather_bench.cu src/benches/gemma4_bench_utils.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/benches
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) src/benches/gemma4_embedding_gather_bench.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/benches/gemma4_sampling_bench: src/benches/gemma4_sampling_bench.cu src/benches/gemma4_bench_utils.cuh src/gemma4_sampling.cu src/gemma4_sampling.cuh src/gemma4_matmul_kernels.cu src/gemma4_matmul_kernels.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/benches
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(TORCH_NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(TORCH_CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) src/benches/gemma4_sampling_bench.cu src/gemma4_sampling.cu src/gemma4_matmul_kernels.cu src/gemma4_embedding_gather.cu $(TORCH_LDFLAGS) $(TORCH_LIBS) -o $@

$(BUILD_DIR)/benches/gemma4_rmsnorm_bench: src/benches/gemma4_rmsnorm_bench.cu src/benches/gemma4_bench_utils.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/benches
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(CUTLASS_INCLUDE) -I$(CUDNN_INCLUDE) -I$(CUDNN_FRONTEND_INCLUDE) src/benches/gemma4_rmsnorm_bench.cu src/gemma4_rmsnorm.cu $(CUDA_LDFLAGS) $(CUDNN_LDFLAGS) -lnvrtc -lcuda -o $@

$(BUILD_DIR)/benches/gemma4_rmsnorm_hidden_fused_bench: src/benches/gemma4_rmsnorm_hidden_fused_bench.cu src/benches/gemma4_bench_utils.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/benches
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(CUTLASS_INCLUDE) src/benches/gemma4_rmsnorm_hidden_fused_bench.cu src/gemma4_rmsnorm.cu -o $@

$(BUILD_DIR)/benches/gemma4_ffn_libtorch_bench: src/benches/gemma4_ffn_libtorch_bench.cu src/benches/gemma4_bench_utils.cuh src/gemma4_ffn.cu src/gemma4_ffn.cuh src/gemma4_ffn_tier2.cu src/gemma4_ffn_tier2.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/benches
	$(NVCC) $(NVCCFLAGS) $(TORCH_NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(TORCH_CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) src/benches/gemma4_ffn_libtorch_bench.cu src/gemma4_ffn.cu src/gemma4_ffn_tier2.cu src/gemma4_rmsnorm.cu $(TORCH_LDFLAGS) $(TORCH_LIBS) -o $@

$(BUILD_DIR)/benches/gemma4_flash_attention_bench: src/benches/gemma4_flash_attention_bench.cu src/benches/gemma4_bench_utils.cuh src/gemma4_flash_attention.cu src/gemma4_flash_attention.cuh src/gemma4_decode_megakernel.cu src/gemma4_decode_megakernel.cuh src/gemma4_runtime.cu src/gemma4_runtime.cuh src/gemma4_kv_cache.cu src/gemma4_kv_cache.cuh src/gemma4_sampling.cu src/gemma4_sampling.cuh src/gemma4_ffn.cu src/gemma4_ffn.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_rope.cu src/gemma4_rope.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/benches
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(TORCH_NVCCFLAGS) $(FLASH_ATTN_NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(TORCH_CPPFLAGS) $(FLASH_ATTN_CUTLASS_CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) src/benches/gemma4_flash_attention_bench.cu src/gemma4_flash_attention.cu src/gemma4_decode_megakernel.cu src/gemma4_runtime.cu src/gemma4_kv_cache.cu src/gemma4_sampling.cu src/gemma4_ffn.cu src/gemma4_rope.cu src/gemma4_embedding_gather.cu src/gemma4_rmsnorm.cu $(TORCH_LDFLAGS) $(TORCH_LIBS) -o $@

$(BUILD_DIR)/benches/gemma4_kv_cache_bench: src/benches/gemma4_kv_cache_bench.cu src/benches/gemma4_bench_utils.cuh src/gemma4_kv_cache.cu src/gemma4_kv_cache.cuh src/gemma4_flash_attention.cu src/gemma4_flash_attention.cuh src/gemma4_decode_megakernel.cu src/gemma4_decode_megakernel.cuh src/gemma4_runtime.cu src/gemma4_runtime.cuh src/gemma4_sampling.cu src/gemma4_sampling.cuh src/gemma4_ffn.cu src/gemma4_ffn.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_rope.cu src/gemma4_rope.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/benches
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(TORCH_NVCCFLAGS) $(FLASH_ATTN_NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(TORCH_CPPFLAGS) $(FLASH_ATTN_CUTLASS_CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) src/benches/gemma4_kv_cache_bench.cu src/gemma4_kv_cache.cu src/gemma4_flash_attention.cu src/gemma4_decode_megakernel.cu src/gemma4_runtime.cu src/gemma4_sampling.cu src/gemma4_ffn.cu src/gemma4_rope.cu src/gemma4_embedding_gather.cu src/gemma4_rmsnorm.cu $(TORCH_LDFLAGS) $(TORCH_LIBS) -o $@

$(BUILD_DIR)/benches/gemma4_tokenizer_bench: src/benches/gemma4_tokenizer_bench.cpp $(BUILD_DIR)/gemma4_tokenizer.o | $(BUILD_DIR)/benches
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) src/benches/gemma4_tokenizer_bench.cpp $(BUILD_DIR)/gemma4_tokenizer.o -o $@

$(BUILD_DIR)/tests/test_embedding_gather: tests/test_embedding_gather.cu src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_embedding_gather.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/tests/test_sampling: tests/test_sampling.cu src/gemma4_sampling.cu src/gemma4_sampling.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(CPPFLAGS) tests/test_sampling.cu src/gemma4_sampling.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/tests/cuda_utils/test_cuda_utils: tests/cuda_utils/test_cuda_utils.cu src/gemma4_cuda_utils.cuh | $(BUILD_DIR)/tests/cuda_utils
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/cuda_utils/test_cuda_utils.cu -o $@

$(BUILD_DIR)/tests/test_tokenizer: tests/test_tokenizer.cpp $(BUILD_DIR)/gemma4_tokenizer.o | $(BUILD_DIR)/tests
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) tests/test_tokenizer.cpp $(BUILD_DIR)/gemma4_tokenizer.o -o $@

$(BUILD_DIR)/tests/test_rmsnorm: tests/test_rmsnorm.cu src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(CUTLASS_INCLUDE) tests/test_rmsnorm.cu src/gemma4_rmsnorm.cu -o $@

$(BUILD_DIR)/tests/test_ffn_decode: tests/test_ffn_decode.cu src/gemma4_ffn.cu src/gemma4_ffn.cuh src/gemma4_ffn_tier2.cu src/gemma4_ffn_tier2.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) tests/test_ffn_decode.cu src/gemma4_ffn.cu src/gemma4_ffn_tier2.cu src/gemma4_rmsnorm.cu -o $@

$(BUILD_DIR)/tests/test_prefill_gemm: tests/test_prefill_gemm.cu src/gemma4_matmul_kernels.cu src/gemma4_matmul_kernels.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) tests/test_prefill_gemm.cu src/gemma4_matmul_kernels.cu -o $@

$(BUILD_DIR)/tests/test_kv_cache: tests/test_kv_cache.cu src/gemma4_kv_cache.cu src/gemma4_kv_cache.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(FLASH_ATTN_CUTLASS_INCLUDE) tests/test_kv_cache.cu src/gemma4_kv_cache.cu -o $@

$(BUILD_DIR)/tests/test_flash_attention: tests/test_flash_attention.cu src/gemma4_kv_cache.cu src/gemma4_kv_cache.cuh src/gemma4_flash_attention.cu src/gemma4_flash_attention.cuh src/gemma4_decode_megakernel.cu src/gemma4_decode_megakernel.cuh src/gemma4_runtime.cu src/gemma4_runtime.cuh src/gemma4_sampling.cu src/gemma4_sampling.cuh src/gemma4_ffn.cu src/gemma4_ffn.cuh src/gemma4_rmsnorm.cu src/gemma4_rmsnorm.cuh src/gemma4_rope.cu src/gemma4_rope.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(FLASH_ATTN_NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(FLASH_ATTN_CUTLASS_CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) tests/test_flash_attention.cu src/gemma4_kv_cache.cu src/gemma4_flash_attention.cu src/gemma4_decode_megakernel.cu src/gemma4_runtime.cu src/gemma4_sampling.cu src/gemma4_ffn.cu src/gemma4_rope.cu src/gemma4_embedding_gather.cu src/gemma4_rmsnorm.cu $(CUDA_LDFLAGS) -lcublas -o $@

$(BUILD_DIR)/tests/test_runtime_state: tests/test_runtime_state.cu src/gemma4_runtime.cu src/gemma4_runtime.cuh src/gemma4_kv_cache.cu src/gemma4_kv_cache.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -I$(FLASH_ATTN_CUTLASS_INCLUDE) tests/test_runtime_state.cu src/gemma4_runtime.cu src/gemma4_kv_cache.cu -o $@

$(BUILD_DIR)/tests/test_decode_megakernel: tests/test_decode_megakernel.cu src/gemma4_decode_megakernel.cu src/gemma4_flash_attention.cu src/gemma4_runtime.cu src/gemma4_runtime.cuh src/gemma4_kv_cache.cu src/gemma4_kv_cache.cuh src/gemma4_sampling.cu src/gemma4_ffn.cu src/gemma4_rmsnorm.cu src/gemma4_decode_megakernel.cuh src/gemma4_flash_attention.cuh src/gemma4_rmsnorm.cuh src/gemma4_rope.cu src/gemma4_rope.cuh src/gemma4_ffn.cuh src/gemma4_sampling.cuh src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh src/gemma4_cuda_utils.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(FLASH_ATTN_NVCCFLAGS) $(FFN_CUTLASS_NVCCFLAGS) $(CPPFLAGS) $(FLASH_ATTN_CUTLASS_CPPFLAGS) $(FFN_CUTLASS_CPPFLAGS) tests/test_decode_megakernel.cu src/gemma4_decode_megakernel.cu src/gemma4_flash_attention.cu src/gemma4_runtime.cu src/gemma4_kv_cache.cu src/gemma4_sampling.cu src/gemma4_ffn.cu src/gemma4_rope.cu src/gemma4_rmsnorm.cu src/gemma4_embedding_gather.cu -o $@

$(BUILD_DIR)/tests/test_prefill_megakernel: tests/test_prefill_megakernel.cu src/gemma4_embedding_gather.cu src/gemma4_embedding_gather.cuh $(BUILD_DIR)/gemma4_prefill_megakernel.o $(BUILD_DIR)/gemma4_matmul_kernels.o $(BUILD_DIR)/gemma4_flash_attention.o $(BUILD_DIR)/gemma4_decode_megakernel.o $(BUILD_DIR)/gemma4_runtime.o $(BUILD_DIR)/gemma4_sampling.o $(BUILD_DIR)/gemma4_rope.o $(BUILD_DIR)/gemma4_kv_cache.o $(BUILD_DIR)/gemma4_ffn.o $(BUILD_DIR)/gemma4_rmsnorm.o | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(RDC_NVCCFLAGS) $(CPPFLAGS) -I$(FLASH_ATTN_CUTLASS_INCLUDE) tests/test_prefill_megakernel.cu src/gemma4_embedding_gather.cu $(BUILD_DIR)/gemma4_prefill_megakernel.o $(BUILD_DIR)/gemma4_matmul_kernels.o $(BUILD_DIR)/gemma4_flash_attention.o $(BUILD_DIR)/gemma4_decode_megakernel.o $(BUILD_DIR)/gemma4_runtime.o $(BUILD_DIR)/gemma4_sampling.o $(BUILD_DIR)/gemma4_rope.o $(BUILD_DIR)/gemma4_kv_cache.o $(BUILD_DIR)/gemma4_ffn.o $(BUILD_DIR)/gemma4_rmsnorm.o $(CUDA_LDFLAGS) -lcublas -o $@

$(BUILD_DIR)/tests/test_checkpoint_loader: tests/test_checkpoint_loader.cu src/gemma4_checkpoint.cu src/gemma4_checkpoint.cuh src/gemma4.h | $(BUILD_DIR)/tests
	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) tests/test_checkpoint_loader.cu src/gemma4_checkpoint.cu -o $@

clean:
	rm -rf $(BUILD_DIR)
