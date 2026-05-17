CC ?= clang
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Iinclude
NVCC ?= nvcc
NVCCFLAGS ?= --use_fast_math -std=c++17 -O3 -Iinclude

BUILD_DIR := build
NVCC_EXISTS := $(shell command -v $(NVCC) 2>/dev/null)
CUDA_OBJECTS :=

ifneq ($(NVCC_EXISTS),)
CUDA_OBJECTS += $(BUILD_DIR)/gemma4_kernels.o
endif

.PHONY: all clean

all: $(BUILD_DIR)/gemma4.o $(CUDA_OBJECTS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/gemma4.o: src/gemma4.c include/gemma4.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/gemma4.c -o $@

$(BUILD_DIR)/gemma4_kernels.o: src/kernels/gemma4_kernels.cu include/gemma4.h | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c src/kernels/gemma4_kernels.cu -o $@

clean:
	rm -rf $(BUILD_DIR)
