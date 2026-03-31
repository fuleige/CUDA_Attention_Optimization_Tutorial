#pragma once

#include "src/common/reference.h"
#include "src/common/util.h"

#include <string>
#include <vector>

enum class GemmKernelKind {
    kNaive,
    kCoalesced,
    kShared,
    kRegisterBlocked,
    kVectorized,
    kDoubleBuffered,
    kAsyncPipeline,
    kWmma,
};

struct GemmShape {
    int m {256};
    int n {256};
    int k {256};
};

std::string gemm_kernel_name(GemmKernelKind kind);
GemmKernelKind parse_gemm_kernel(const std::string& name);

template <typename T>
float launch_gemm(
    GemmKernelKind kind,
    const T* d_a,
    const T* d_b,
    float* d_c,
    const GemmShape& shape
);

template <typename T>
void run_gemm_reference(
    const std::vector<T>& h_a,
    const std::vector<T>& h_b,
    std::vector<float>* h_c,
    const GemmShape& shape
);
