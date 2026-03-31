#include "src/gemm/gemm_kernels.h"

#include "src/common/cli.h"
#include "src/common/cuda_check.h"
#include "src/common/timer.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {

template <typename T>
int run_case(const CliArgs& args, GemmKernelKind kind) {
    const GemmShape shape {
        args.get_int("m", 256),
        args.get_int("n", 256),
        args.get_int("k", 256),
    };
    const int warmup = args.get_int("warmup", 10);
    const int iters = args.get_int("iters", 50);
    const bool check = args.get_bool("check", false);
    const bool csv = args.get_bool("csv", false);
    const bool csv_header = args.get_bool("csv-header", false);

    validate_gemm_inputs(kind, std::is_same_v<T, half> ? DataType::kFloat16 : DataType::kFloat32, shape);

    auto h_a = random_vector<T>(shape.m * shape.k, 0.5f, 7);
    auto h_b = random_vector<T>(shape.k * shape.n, 0.5f, 11);
    if constexpr (std::is_same_v<T, half>) {
        if (kind == GemmKernelKind::kWmma) {
            std::vector<T> transposed(shape.k * shape.n);
            for (int row = 0; row < shape.k; ++row) {
                for (int col = 0; col < shape.n; ++col) {
                    transposed[col * shape.k + row] = h_b[row * shape.n + col];
                }
            }
            h_b.swap(transposed);
        }
    }
    std::vector<float> h_out(shape.m * shape.n, 0.0f);
    std::vector<float> h_ref;

    T* d_a = nullptr;
    T* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, sizeof(T) * h_a.size()));
    CUDA_CHECK(cudaMalloc(&d_b, sizeof(T) * h_b.size()));
    CUDA_CHECK(cudaMalloc(&d_c, sizeof(float) * h_out.size()));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), sizeof(T) * h_a.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), sizeof(T) * h_b.size(), cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
        launch_gemm(kind, d_a, d_b, d_c, shape);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.begin();
    for (int i = 0; i < iters; ++i) {
        launch_gemm(kind, d_a, d_b, d_c, shape);
    }
    const float total_ms = timer.end();
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_c, sizeof(float) * h_out.size(), cudaMemcpyDeviceToHost));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    bool pass = true;
    if (check) {
        auto ref_b = h_b;
        if constexpr (std::is_same_v<T, half>) {
            if (kind == GemmKernelKind::kWmma) {
                std::vector<T> row_major(shape.k * shape.n);
                for (int col = 0; col < shape.n; ++col) {
                    for (int row = 0; row < shape.k; ++row) {
                        row_major[row * shape.n + col] = h_b[col * shape.k + row];
                    }
                }
                ref_b.swap(row_major);
            }
        }
        run_gemm_reference(h_a, ref_b, &h_ref, shape);
        pass = compare_vectors(h_out, h_ref, 2e-2f, 2e-2f, &max_abs_err, &max_rel_err);
    }

    const float avg_ms = total_ms / static_cast<float>(iters);
    const double flops = 2.0 * static_cast<double>(shape.m) * shape.n * shape.k;
    const double tflops = flops / (avg_ms * 1e-3) / 1e12;

    if (csv) {
        const std::vector<std::pair<std::string, std::string>> fields = {
            {"kernel", gemm_kernel_name(kind)},
            {"dtype", dtype_name(std::is_same_v<T, half> ? DataType::kFloat16 : DataType::kFloat32)},
            {"m", std::to_string(shape.m)},
            {"n", std::to_string(shape.n)},
            {"k", std::to_string(shape.k)},
            {"avg_ms", std::to_string(avg_ms)},
            {"tflops_est", std::to_string(tflops)},
            {"max_abs_err", std::to_string(max_abs_err)},
            {"max_rel_err", std::to_string(max_rel_err)},
            {"pass", pass ? "1" : "0"},
        };
        if (csv_header) {
            print_csv_header(fields);
        }
        print_csv_row(fields);
    } else {
        std::cout << "kernel=" << gemm_kernel_name(kind)
                  << " dtype=" << (std::is_same_v<T, half> ? "fp16" : "fp32")
                  << " shape=(" << shape.m << "," << shape.n << "," << shape.k << ")"
                  << " avg_ms=" << std::fixed << std::setprecision(4) << avg_ms
                  << " tflops_est=" << tflops
                  << " pass=" << (pass ? "true" : "false")
                  << " max_abs_err=" << max_abs_err
                  << " max_rel_err=" << max_rel_err
                  << std::endl;
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return pass ? 0 : 1;
}

void print_help() {
    std::cout << "Usage: ./bin/gemm_runner [options]\n"
              << "  --kernel naive|coalesced|shared|register_blocked|vectorized|double_buffered|async_pipeline|wmma\n"
              << "  --dtype fp32|fp16\n"
              << "  --m 256 --n 256 --k 256\n"
              << "  --warmup 10 --iters 50\n"
              << "  --check true\n"
              << "  --csv true\n";
}

}  // namespace

int main(int argc, char** argv) {
    CliArgs args(argc, argv);
    if (args.has("help")) {
        print_help();
        return 0;
    }

    const auto kernel = parse_gemm_kernel(args.get_string("kernel", "shared"));
    const auto dtype = parse_dtype(args.get_string("dtype", "fp32"));

    try {
        if (dtype == DataType::kFloat16) {
            return run_case<half>(args, kernel);
        }
        return run_case<float>(args, kernel);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
}
