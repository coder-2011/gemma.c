CC ?= clang
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Isrc
NVCC ?= nvcc
NVCCFLAGS ?= -std=c++17 -O3 -arch=sm_86

BUILD_DIR := build

.PHONY: all cuda-kernels clean

all: $(BUILD_DIR)/gemma4.o

cuda-kernels: $(BUILD_DIR)/gemma4_matmul_kernels.o

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/gemma4.o: src/gemma4.c src/gemma4.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/gemma4.c -o $@

$(BUILD_DIR)/gemma4_matmul_kernels.o: src/gemma4_matmul_kernels.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c src/gemma4_matmul_kernels.cu -o $@

clean:
	rm -rf $(BUILD_DIR)
