#include "src/attention/attention_kernels.h"
#include "src/common/cli.h"
#include "src/common/cuda_check.h"

#include <iostream>
#include <type_traits>

namespace {

template <typename T>
bool run_forward_case(AttentionKernelKind kind, AttentionShape shape, AttentionOptions options, int page_size = 16) {
    auto h_q = random_vector<T>(shape.batch_size * shape.num_heads * shape.seq_len_q * shape.head_dim, 0.5f, 41);
    auto h_k = random_vector<T>(shape.batch_size * shape.num_kv_heads * shape.seq_len_kv * shape.head_dim, 0.5f, 43);
    auto h_v = random_vector<T>(shape.batch_size * shape.num_kv_heads * shape.seq_len_kv * shape.head_dim, 0.5f, 47);
    std::vector<T> h_ref;
    std::vector<T> h_out(shape.batch_size * shape.num_heads * shape.seq_len_q * shape.head_dim);

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
    }

    T* d_q = nullptr;
    T* d_k = nullptr;
    T* d_v = nullptr;
    T* d_out = nullptr;
    int* d_page_table = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(T) * h_q.size()));
    CUDA_CHECK(cudaMalloc(&d_k, sizeof(T) * h_k.size()));
    CUDA_CHECK(cudaMalloc(&d_v, sizeof(T) * h_v.size()));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(T) * h_out.size()));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), sizeof(T) * h_q.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), sizeof(T) * h_k.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), sizeof(T) * h_v.size(), cudaMemcpyHostToDevice));

    if (!page_table.empty()) {
        CUDA_CHECK(cudaMalloc(&d_page_table, sizeof(int) * page_table.size()));
        CUDA_CHECK(cudaMemcpy(d_page_table, page_table.data(), sizeof(int) * page_table.size(), cudaMemcpyHostToDevice));
        paged_attention_reference(h_q, h_k, h_v, page_table, page_size, shape, &h_ref);
    } else {
        attention_forward_reference(h_q, h_k, h_v, shape, options, &h_ref);
    }

    launch_attention_forward(kind, d_q, d_k, d_v, d_out, shape, options, d_page_table, page_size);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sizeof(T) * h_out.size(), cudaMemcpyDeviceToHost));

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    const bool pass = compare_vectors(h_out, h_ref, 5e-2f, 5e-2f, &max_abs_err, &max_rel_err);
    std::cout << "[attention-fwd] kernel=" << attention_kernel_name(kind)
              << " dtype=" << (std::is_same_v<T, half> ? "fp16" : "fp32")
              << " pass=" << (pass ? "true" : "false")
              << " max_abs_err=" << max_abs_err
              << " max_rel_err=" << max_rel_err
              << std::endl;

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
    if (d_page_table != nullptr) {
        CUDA_CHECK(cudaFree(d_page_table));
    }
    return pass;
}

bool run_backward_case(AttentionKernelKind kind) {
    AttentionShape shape {1, 4, 4, 32, 32, 32};
    AttentionOptions options {};
    options.causal = true;

    auto h_q = random_vector<float>(shape.batch_size * shape.num_heads * shape.seq_len_q * shape.head_dim, 0.5f, 53);
    auto h_k = random_vector<float>(shape.batch_size * shape.num_kv_heads * shape.seq_len_kv * shape.head_dim, 0.5f, 59);
    auto h_v = random_vector<float>(shape.batch_size * shape.num_kv_heads * shape.seq_len_kv * shape.head_dim, 0.5f, 61);
    auto h_grad_out = random_vector<float>(shape.batch_size * shape.num_heads * shape.seq_len_q * shape.head_dim, 0.5f, 67);
    std::vector<float> ref_gq;
    std::vector<float> ref_gk;
    std::vector<float> ref_gv;
    attention_backward_reference(h_q, h_k, h_v, h_grad_out, shape, options, &ref_gq, &ref_gk, &ref_gv);

    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_grad_out = nullptr;
    float* d_gq = nullptr;
    float* d_gk = nullptr;
    float* d_gv = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, sizeof(float) * h_q.size()));
    CUDA_CHECK(cudaMalloc(&d_k, sizeof(float) * h_k.size()));
    CUDA_CHECK(cudaMalloc(&d_v, sizeof(float) * h_v.size()));
    CUDA_CHECK(cudaMalloc(&d_grad_out, sizeof(float) * h_grad_out.size()));
    CUDA_CHECK(cudaMalloc(&d_gq, sizeof(float) * ref_gq.size()));
    CUDA_CHECK(cudaMalloc(&d_gk, sizeof(float) * ref_gk.size()));
    CUDA_CHECK(cudaMalloc(&d_gv, sizeof(float) * ref_gv.size()));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), sizeof(float) * h_q.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), sizeof(float) * h_k.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), sizeof(float) * h_v.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_out, h_grad_out.data(), sizeof(float) * h_grad_out.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_gq, 0, sizeof(float) * ref_gq.size()));
    CUDA_CHECK(cudaMemset(d_gk, 0, sizeof(float) * ref_gk.size()));
    CUDA_CHECK(cudaMemset(d_gv, 0, sizeof(float) * ref_gv.size()));

    launch_attention_backward(kind, d_q, d_k, d_v, d_grad_out, d_gq, d_gk, d_gv, shape, options);
    std::vector<float> h_gq(ref_gq.size());
    std::vector<float> h_gk(ref_gk.size());
    std::vector<float> h_gv(ref_gv.size());
    CUDA_CHECK(cudaMemcpy(h_gq.data(), d_gq, sizeof(float) * h_gq.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gk.data(), d_gk, sizeof(float) * h_gk.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_gv.data(), d_gv, sizeof(float) * h_gv.size(), cudaMemcpyDeviceToHost));

    float abs_q = 0.0f;
    float rel_q = 0.0f;
    float abs_k = 0.0f;
    float rel_k = 0.0f;
    float abs_v = 0.0f;
    float rel_v = 0.0f;
    bool pass = compare_vectors(h_gq, ref_gq, 6e-2f, 6e-2f, &abs_q, &rel_q);
    pass = compare_vectors(h_gk, ref_gk, 6e-2f, 6e-2f, &abs_k, &rel_k) && pass;
    pass = compare_vectors(h_gv, ref_gv, 6e-2f, 6e-2f, &abs_v, &rel_v) && pass;

    std::cout << "[attention-bwd] kernel=" << attention_kernel_name(kind)
              << " pass=" << (pass ? "true" : "false")
              << " max_abs_err=" << std::max(abs_q, std::max(abs_k, abs_v))
              << " max_rel_err=" << std::max(rel_q, std::max(rel_k, rel_v))
              << std::endl;

    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_grad_out));
    CUDA_CHECK(cudaFree(d_gq));
    CUDA_CHECK(cudaFree(d_gk));
    CUDA_CHECK(cudaFree(d_gv));
    return pass;
}

}  // namespace

int main() {
    print_device_summary();

    bool ok = true;
    AttentionShape dense_shape {1, 4, 4, 32, 32, 32};
    AttentionOptions dense_opts {};
    dense_opts.causal = true;
    dense_opts.window_left = -1;
    dense_opts.block_size = 8;

    AttentionOptions sliding_opts = dense_opts;
    sliding_opts.window_left = 16;

    ok = run_forward_case<float>(AttentionKernelKind::kBasicForward, dense_shape, dense_opts) && ok;
    ok = run_forward_case<float>(AttentionKernelKind::kFlashForward, dense_shape, dense_opts) && ok;
    ok = run_forward_case<float>(AttentionKernelKind::kGqaForward, AttentionShape {1, 8, 2, 32, 32, 32}, dense_opts) && ok;
    ok = run_forward_case<float>(AttentionKernelKind::kSlidingForward, dense_shape, sliding_opts) && ok;
    AttentionOptions block_sparse_opts = dense_opts;
    block_sparse_opts.block_sparse = true;
    ok = run_forward_case<float>(AttentionKernelKind::kBlockSparseForward, dense_shape, block_sparse_opts) && ok;
    ok = run_forward_case<float>(AttentionKernelKind::kPagedForward, dense_shape, AttentionOptions {}, 8) && ok;
    ok = run_forward_case<half>(AttentionKernelKind::kBasicForward, dense_shape, dense_opts) && ok;
    ok = run_forward_case<half>(AttentionKernelKind::kFlashForward, dense_shape, dense_opts) && ok;
    ok = run_backward_case(AttentionKernelKind::kBasicBackward) && ok;
    ok = run_backward_case(AttentionKernelKind::kFlashBackward) && ok;
    return ok ? 0 : 1;
}
