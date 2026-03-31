#pragma once

#include "src/common/cuda_check.h"

struct CudaEventTimer {
    cudaEvent_t start {};
    cudaEvent_t stop {};

    CudaEventTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    ~CudaEventTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // CUDA events are GPU resources — copying would cause double-destroy.
    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&&) = delete;
    CudaEventTimer& operator=(CudaEventTimer&&) = delete;

    void begin(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(start, stream));
    }

    float end(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        return elapsed_ms;
    }
};

