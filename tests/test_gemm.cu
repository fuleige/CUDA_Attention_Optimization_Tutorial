#include "src/common/cli.h"
#include "src/gemm/gemm_kernels.h"

#include <iostream>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
bool run_one(GemmKernelKind kind, const GemmShape& shape) {
    validate_gemm_inputs(kind, std::is_same_v<T, half> ? DataType::kFloat16 : DataType::kFloat32, shape);

    auto h_a = random_vector<T>(shape.m * shape.k, 0.5f, 3);
    auto h_b = random_vector<T>(shape.k * shape.n, 0.5f, 5);
    auto h_ref_b = h_b;
    if constexpr (std::is_same_v<T, half>) {
        if (kind == GemmKernelKind::kWmma) {
            std::vector<T> col_major(shape.k * shape.n);
            for (int row = 0; row < shape.k; ++row) {
                for (int col = 0; col < shape.n; ++col) {
                    col_major[col * shape.k + row] = h_b[row * shape.n + col];
                }
            }
            h_b.swap(col_major);
        }
    }

    T* d_a = nullptr;
    T* d_b = nullptr;
    float* d_c = nullptr;
    std::vector<float> h_out(shape.m * shape.n, 0.0f);
    std::vector<float> h_ref;

    CUDA_CHECK(cudaMalloc(&d_a, sizeof(T) * h_a.size()));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(T) * h_b.size()));
    CUDA_CHECK(cudaMalloc(&d_c, sizeof(float) * h_out.size()));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sizeof(T) * h_a.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), sizeof(T) * h_b.size(), cudaMemcpyHostToDevice));

    launch_gemm(kind, d_a, d_b, d_c, shape);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, sizeof(float) * h_out.size(), cudaMemcpyDeviceToHost));

    run_gemm_reference(h_a, h_ref_b, &h_ref, shape);
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    const bool pass = compare_vectors(h_out, h_ref, 3e-2f, 3e-2f, &max_abs_err, &max_rel_err);
    std::cout << "[gemm] kernel=" << gemm_kernel_name(kind)
              << " dtype=" << (std::is_same_v<T, half> ? "fp16" : "fp32")
              << " pass=" << (pass ? "true" : "false")
              << " max_abs_err=" << max_abs_err
              << " max_rel_err=" << max_rel_err
              << std::endl;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return pass;
}

}  // namespace

int main() {
    print_device_summary();
    bool ok = true;
    const std::vector<GemmKernelKind> fp32_kernels = {
        GemmKernelKind::kNaive,
        GemmKernelKind::kCoalesced,
        GemmKernelKind::kShared,
        GemmKernelKind::kRegisterBlocked,
        GemmKernelKind::kVectorized,
        GemmKernelKind::kDoubleBuffered,
        GemmKernelKind::kAsyncPipeline,
    };
    for (const auto kind : fp32_kernels) {
        ok = run_one<float>(kind, GemmShape {64, 80, 96}) && ok;
    }
    const std::vector<GemmKernelKind> fp16_kernels = {
        GemmKernelKind::kShared,
        GemmKernelKind::kRegisterBlocked,
        GemmKernelKind::kDoubleBuffered,
        GemmKernelKind::kAsyncPipeline,
        GemmKernelKind::kWmma,
    };
    for (const auto kind : fp16_kernels) {
        ok = run_one<half>(kind, GemmShape {64, 64, 64}) && ok;
    }
    return ok ? 0 : 1;
}
