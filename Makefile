CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(if $(wildcard $(CUDA_HOME)/bin/nvcc),$(CUDA_HOME)/bin/nvcc,nvcc)
CUDA_ARCH ?= sm_89
BUILD_DIR := .build
BIN_DIR := bin
INCLUDES := -I.
COMMON_FLAGS := -std=c++17 -lineinfo -Xcompiler -Wall,-Wextra -Xcompiler -Wno-unused-parameter
RELEASE_FLAGS := -O3 -DNDEBUG
DEBUG_FLAGS := -O0 -g -G
NVCCFLAGS ?= $(COMMON_FLAGS) $(RELEASE_FLAGS) -arch=$(CUDA_ARCH)
LDFLAGS :=

COMMON_SRCS := \
	src/common/reference.cu

GEMM_SRCS := \
	src/gemm/gemm_kernels.cu \
	src/gemm/gemm_runner.cu

ATTENTION_SRCS := \
	src/attention/attention_kernels.cu \
	src/attention/attention_runner.cu

TEST_GEMM_SRCS := \
	src/common/reference.cu \
	src/gemm/gemm_kernels.cu \
	tests/test_gemm.cu

TEST_ATTENTION_SRCS := \
	src/common/reference.cu \
	src/attention/attention_kernels.cu \
	tests/test_attention.cu

.PHONY: all build test bench clean debug

all: build

build: $(BIN_DIR)/gemm_runner $(BIN_DIR)/attention_runner $(BIN_DIR)/test_gemm $(BIN_DIR)/test_attention

debug: NVCCFLAGS := $(COMMON_FLAGS) $(DEBUG_FLAGS) -arch=$(CUDA_ARCH)
debug: clean build

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR)/gemm_runner: $(COMMON_SRCS) $(GEMM_SRCS) | $(BIN_DIR) $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/attention_runner: $(COMMON_SRCS) $(ATTENTION_SRCS) | $(BIN_DIR) $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/test_gemm: $(TEST_GEMM_SRCS) | $(BIN_DIR) $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

$(BIN_DIR)/test_attention: $(TEST_ATTENTION_SRCS) | $(BIN_DIR) $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

test: build
	./bin/test_gemm
	./bin/test_attention

bench: build
	./scripts/run_benchmarks.sh

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
