CC ?= clang
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic
CPPFLAGS ?= -Isrc

BUILD_DIR := build

.PHONY: all clean

all: $(BUILD_DIR)/gemma4.o

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/gemma4.o: src/gemma4.c src/gemma4.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c src/gemma4.c -o $@

clean:
	rm -rf $(BUILD_DIR)
