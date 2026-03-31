#include "src/attention/attention_kernels.h"

#include "src/common/cli.h"
#include "src/common/cuda_check.h"
#include "src/common/timer.h"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {

AttentionShape parse_shape(const CliArgs& args) {
    const int heads = args.get_int("heads", 8);
    return AttentionShape {
        args.get_int("batch", 1),
        heads,
        args.get_int("kv-heads", heads),
        args.get_int("seq-q", 128),
        args.get_int("seq-kv", 128),
        args.get_int("head-dim", 64),
    };
}

AttentionOptions parse_options(const CliArgs& args, AttentionKernelKind kind) {
    AttentionOptions options;
    options.causal = args.get_bool("causal", false);
    options.window_left = args.get_int("window", kind == AttentionKernelKind::kSlidingForward ? 64 : -1);
    options.block_size = args.get_int("block-size", 16);
    options.block_sparse = args.get_bool("block-sparse", kind == AttentionKernelKind::kBlockSparseForward);
    return options;
}

template <typename T>
int run_forward(const CliArgs& args, AttentionKernelKind kind) {
    const AttentionShape shape = parse_shape(args);
    const AttentionOptions options = parse_options(args, kind);
    const int warmup = args.get_int("warmup", 5);
    const int iters = args.get_int("iters", 20);
    const bool check = args.get_bool("check", false);
    const bool csv = args.get_bool("csv", false);
    const int page_size = args.get_int("page-size", 16);
    validate_attention_inputs(
        kind,
        std::is_same_v<T, half> ? DataType::kFloat16 : DataType::kFloat32,
        shape,
        options,
        page_size
    );

    auto h_q = random_vector<T>(shape.batch_size * shape.num_heads * shape.seq_len_q * shape.head_dim, 0.5f, 13);
    auto h_k = random_vector<T>(shape.batch_size * shape.num_kv_heads * shape.seq_len_kv * shape.head_dim, 0.5f, 17);
    auto h_v = random_vector<T>(shape.batch_size * shape.num_kv_heads * shape.seq_len_kv * shape.head_dim, 0.5f, 19);
    std::vector<T> h_out(shape.batch_size * shape.num_heads * shape.seq_len_q * shape.head_dim);
    std::vector<T> h_ref;

    T* d_q = nullptr;
    T* d_k = nullptr;
    T* d_v = nullptr;
    T* d_out = nullptr;
    int* d_page_table = nullptr;
    std::vector<int> page_table;

    if (kind == AttentionKernelKind::kPagedForward) {
        const int num_pages = (shape.seq_len_kv + page_size - 1) / page_size;
        page_table.resize(num_pages);
        for (int i = 0; i < num_pages; ++i) {
            page_table[i] = num_pages - 1 - i;
        }
        std::vector<T> paged_k(num_pages * shape.batch_size * shape.num_kv_heads * page_size * shape.head_dim);
        std::vector<T> paged_v = paged_k;
        for (int logical = 0; logical < shape.seq_len_kv; ++logical) {
            const int page_slot = logical / page_size;
            const int offset_in_page = logical % page_size;
            const int physical_page = page_table[page_slot];
            for (int b = 0; b < shape.batch_size; ++b) {
                for (int h = 0; h < shape.num_kv_heads; ++h) {
                    for (int d = 0; d < shape.head_dim; ++d) {
                        const int src = (((b * shape.num_kv_heads + h) * shape.seq_len_kv + logical) * shape.head_dim + d);
                        const int dst =
                            ((((physical_page * shape.batch_size + b) * shape.num_kv_heads + h) * page_size + offset_in_page) *
                             shape.head_dim + d);
                        paged_k[dst] = h_k[src];
                        paged_v[dst] = h_v[src];
                    }
                }
            }
        }
        h_k.swap(paged_k);
        h_v.swap(paged_v);
        CUDA_CHECK(cudaMalloc(&d_page_table, sizeof(int) * page_table.size()));
        CUDA_CHECK(cudaMemcpy(d_page_table, page_table.data(), sizeof(int) * page_table.size(), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMalloc(&d_q, sizeof(T) * h_q.size()));
    CUDA_CHECK(cudaMalloc(&d_k, sizeof(T) * h_k.size()));
    CUDA_CHECK(cudaMalloc(&d_v, sizeof(T) * h_v.size()));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(T) * h_out.size()));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), sizeof(T) * h_q.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), sizeof(T) * h_k.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), sizeof(T) * h_v.size(), cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
        launch_attention_forward(kind, d_q, d_k, d_v, d_out, shape, options, d_page_table, page_size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.begin();
    for (int i = 0; i < iters; ++i) {
        launch_attention_forward(kind, d_q, d_k, d_v, d_out, shape, options, d_page_table, page_size);
    }
    const float avg_ms = timer.end() / static_cast<float>(iters);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(T) * h_out.size(), cudaMemcpyDeviceToHost));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    bool pass = true;
    if (check) {
        switch (kind) {
            case AttentionKernelKind::kPagedForward:
                paged_attention_reference(h_q, h_k, h_v, page_table, page_size, shape, options, &h_ref);
                break;
            case AttentionKernelKind::kSlidingForward:
            case AttentionKernelKind::kBlockSparseForward:
            case AttentionKernelKind::kBasicForward:
            case AttentionKernelKind::kGqaForward:
            case AttentionKernelKind::kFlashForward:
                attention_forward_reference(h_q, h_k, h_v, shape, options, &h_ref);
                break;
            default:
                break;
        }
        pass = compare_vectors(h_out, h_ref, 4e-2f, 4e-2f, &max_abs_err, &max_rel_err);
    }

    const double flops =
        2.0 * static_cast<double>(shape.batch_size) * shape.num_heads * shape.seq_len_q * shape.seq_len_kv * shape.head_dim;
    const double tflops = flops / (avg_ms * 1e-3) / 1e12;
    const double bytes = static_cast<double>(h_q.size() + h_k.size() + h_v.size() + h_out.size()) * sizeof(T);
    const double bandwidth = bytes / (avg_ms * 1e-3) / 1e9;

    if (csv) {
        print_csv_row({
            {"kernel", attention_kernel_name(kind)},
            {"dtype", dtype_name(std::is_same_v<T, half> ? DataType::kFloat16 : DataType::kFloat32)},
            {"batch", std::to_string(shape.batch_size)},
            {"heads", std::to_string(shape.num_heads)},
            {"kv_heads", std::to_string(shape.num_kv_heads)},
            {"seq_q", std::to_string(shape.seq_len_q)},
            {"seq_kv", std::to_string(shape.seq_len_kv)},
            {"head_dim", std::to_string(shape.head_dim)},
            {"avg_ms", std::to_string(avg_ms)},
            {"tflops_est", std::to_string(tflops)},
            {"bandwidth_est", std::to_string(bandwidth)},
            {"max_abs_err", std::to_string(max_abs_err)},
            {"max_rel_err", std::to_string(max_rel_err)},
            {"pass", pass ? "1" : "0"},
        });
    } else {
        std::cout << "kernel=" << attention_kernel_name(kind)
                  << " dtype=" << (std::is_same_v<T, half> ? "fp16" : "fp32")
                  << " shape=(" << shape.batch_size << "," << shape.num_heads << "," << shape.num_kv_heads
                  << "," << shape.seq_len_q << "," << shape.seq_len_kv << "," << shape.head_dim << ")"
                  << " avg_ms=" << std::fixed << std::setprecision(4) << avg_ms
                  << " tflops_est=" << tflops
                  << " bandwidth_est=" << bandwidth
                  << " pass=" << (pass ? "true" : "false")
                  << " max_abs_err=" << max_abs_err
                  << " max_rel_err=" << max_rel_err
                  << std::endl;
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
    if (d_page_table != nullptr) {
        CUDA_CHECK(cudaFree(d_page_table));
    }
    return pass ? 0 : 1;
}

int run_backward(const CliArgs& args, AttentionKernelKind kind) {
    const AttentionShape shape = parse_shape(args);
    const AttentionOptions options = parse_options(args, kind);
    const int warmup = args.get_int("warmup", 3);
    const int iters = args.get_int("iters", 10);
    const bool check = args.get_bool("check", false);
    const bool csv = args.get_bool("csv", false);
    validate_attention_inputs(kind, DataType::kFloat32, shape, options, 0);

    auto h_q = random_vector<float>(shape.batch_size * shape.num_heads * shape.seq_len_q * shape.head_dim, 0.5f, 23);
    auto h_k = random_vector<float>(shape.batch_size * shape.num_kv_heads * shape.seq_len_kv * shape.head_dim, 0.5f, 29);
    auto h_v = random_vector<float>(shape.batch_size * shape.num_kv_heads * shape.seq_len_kv * shape.head_dim, 0.5f, 31);
    auto h_grad_out = random_vector<float>(shape.batch_size * shape.num_heads * shape.seq_len_q * shape.head_dim, 0.5f, 37);

    std::vector<float> h_grad_q(h_q.size(), 0.0f);
    std::vector<float> h_grad_k(h_k.size(), 0.0f);
    std::vector<float> h_grad_v(h_v.size(), 0.0f);
    std::vector<float> ref_grad_q;
    std::vector<float> ref_grad_k;
    std::vector<float> ref_grad_v;

    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_grad_out = nullptr;
    float* d_grad_q = nullptr;
    float* d_grad_k = nullptr;
    float* d_grad_v = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float) * h_q.size()));
    CUDA_CHECK(cudaMalloc(&d_k, sizeof(float) * h_k.size()));
    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float) * h_v.size()));
    CUDA_CHECK(cudaMalloc(&d_grad_out, sizeof(float) * h_grad_out.size()));
    CUDA_CHECK(cudaMalloc(&d_grad_q, sizeof(float) * h_grad_q.size()));
    CUDA_CHECK(cudaMalloc(&d_grad_k, sizeof(float) * h_grad_k.size()));
    CUDA_CHECK(cudaMalloc(&d_grad_v, sizeof(float) * h_grad_v.size()));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), sizeof(float) * h_q.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), sizeof(float) * h_k.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), sizeof(float) * h_v.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_out, h_grad_out.data(), sizeof(float) * h_grad_out.size(), cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
        CUDA_CHECK(cudaMemset(d_grad_q, 0, sizeof(float) * h_grad_q.size()));
        CUDA_CHECK(cudaMemset(d_grad_k, 0, sizeof(float) * h_grad_k.size()));
        CUDA_CHECK(cudaMemset(d_grad_v, 0, sizeof(float) * h_grad_v.size()));
        launch_attention_backward(kind, d_q, d_k, d_v, d_grad_out, d_grad_q, d_grad_k, d_grad_v, shape, options);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEventTimer timer;
    timer.begin();
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemset(d_grad_q, 0, sizeof(float) * h_grad_q.size()));
        CUDA_CHECK(cudaMemset(d_grad_k, 0, sizeof(float) * h_grad_k.size()));
        CUDA_CHECK(cudaMemset(d_grad_v, 0, sizeof(float) * h_grad_v.size()));
        launch_attention_backward(kind, d_q, d_k, d_v, d_grad_out, d_grad_q, d_grad_k, d_grad_v, shape, options);
    }
    const float avg_ms = timer.end() / static_cast<float>(iters);

    CUDA_CHECK(cudaMemcpy(h_grad_q.data(), d_grad_q, sizeof(float) * h_grad_q.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_k.data(), d_grad_k, sizeof(float) * h_grad_k.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_grad_v.data(), d_grad_v, sizeof(float) * h_grad_v.size(), cudaMemcpyDeviceToHost));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    bool pass = true;
    if (check) {
        attention_backward_reference(h_q, h_k, h_v, h_grad_out, shape, options, &ref_grad_q, &ref_grad_k, &ref_grad_v);
        float abs_q = 0.0f;
        float rel_q = 0.0f;
        float abs_k = 0.0f;
        float rel_k = 0.0f;
        float abs_v = 0.0f;
        float rel_v = 0.0f;
        pass = compare_vectors(h_grad_q, ref_grad_q, 6e-2f, 6e-2f, &abs_q, &rel_q) && pass;
        pass = compare_vectors(h_grad_k, ref_grad_k, 6e-2f, 6e-2f, &abs_k, &rel_k) && pass;
        pass = compare_vectors(h_grad_v, ref_grad_v, 6e-2f, 6e-2f, &abs_v, &rel_v) && pass;
        max_abs_err = std::max(abs_q, std::max(abs_k, abs_v));
        max_rel_err = std::max(rel_q, std::max(rel_k, rel_v));
    }

    if (csv) {
        print_csv_row({
            {"kernel", attention_kernel_name(kind)},
            {"dtype", "fp32"},
            {"avg_ms", std::to_string(avg_ms)},
            {"max_abs_err", std::to_string(max_abs_err)},
            {"max_rel_err", std::to_string(max_rel_err)},
            {"pass", pass ? "1" : "0"},
        });
    } else {
        std::cout << "kernel=" << attention_kernel_name(kind)
                  << " dtype=fp32"
                  << " avg_ms=" << std::fixed << std::setprecision(4) << avg_ms
                  << " pass=" << (pass ? "true" : "false")
                  << " max_abs_err=" << max_abs_err
                  << " max_rel_err=" << max_rel_err
                  << std::endl;
    }

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_grad_out));
    CUDA_CHECK(cudaFree(d_grad_q));
    CUDA_CHECK(cudaFree(d_grad_k));
    CUDA_CHECK(cudaFree(d_grad_v));
    return pass ? 0 : 1;
}

void print_help() {
    std::cout << "Usage: ./bin/attention_runner [options]\n"
              << "  --kernel basic_fwd|flash_fwd|paged_fwd|gqa_fwd|sliding_fwd|block_sparse_fwd|basic_bwd|flash_bwd\n"
              << "  --dtype fp32|fp16\n"
              << "  --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64\n"
              << "  --causal true --window 64 --block-size 16 --block-sparse true --page-size 16\n"
              << "  Notes:\n"
              << "    basic_fwd/flash_fwd/gqa_fwd support causal masking only\n"
              << "    sliding_fwd uses --window\n"
              << "    block_sparse_fwd uses --block-size\n"
              << "    paged_fwd supports causal/window/block-sparse options plus --page-size\n"
              << "    forward kernels currently require --head-dim <= 256\n"
              << "    flash_* are teaching implementations, not production-faithful FlashAttention kernels\n"
              << "  --warmup 5 --iters 20 --check true --csv true\n";
}

}  // namespace

int main(int argc, char** argv) {
    CliArgs args(argc, argv);
    if (args.has("help")) {
        print_help();
        return 0;
    }

    const auto kernel = parse_attention_kernel(args.get_string("kernel", "basic_fwd"));
    const auto dtype = parse_dtype(args.get_string("dtype", "fp32"));

    try {
        if (kernel == AttentionKernelKind::kBasicBackward || kernel == AttentionKernelKind::kFlashBackward) {
            if (dtype != DataType::kFloat32) {
                std::cerr << "Backward kernels currently require --dtype fp32" << std::endl;
                return 1;
            }
            return run_backward(args, kernel);
        }
        if (dtype == DataType::kFloat16) {
            return run_forward<half>(args, kernel);
        }
        return run_forward<float>(args, kernel);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
}
