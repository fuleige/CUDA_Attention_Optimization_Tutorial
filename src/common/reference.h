#pragma once

#include "src/common/util.h"

#include <tuple>
#include <vector>

struct AttentionShape {
    int batch_size {1};
    int num_heads {1};
    int num_kv_heads {1};
    int seq_len_q {1};
    int seq_len_kv {1};
    int head_dim {64};
};

struct AttentionOptions {
    bool causal {false};
    int window_left {-1};
    int block_size {16};
    bool block_sparse {false};
};

template <typename T>
void gemm_reference(
    const std::vector<T>& a,
    const std::vector<T>& b,
    std::vector<T>* c,
    int m,
    int n,
    int k
);

template <typename T>
void attention_forward_reference(
    const std::vector<T>& q,
    const std::vector<T>& k,
    const std::vector<T>& v,
    const AttentionShape& shape,
    const AttentionOptions& options,
    std::vector<T>* out
);

template <typename T>
void attention_backward_reference(
    const std::vector<T>& q,
    const std::vector<T>& k,
    const std::vector<T>& v,
    const std::vector<T>& grad_out,
    const AttentionShape& shape,
    const AttentionOptions& options,
    std::vector<T>* grad_q,
    std::vector<T>* grad_k,
    std::vector<T>* grad_v
);

template <typename T>
void paged_attention_reference(
    const std::vector<T>& q,
    const std::vector<T>& k_pages,
    const std::vector<T>& v_pages,
    const std::vector<int>& page_table,
    int page_size,
    const AttentionShape& shape,
    const AttentionOptions& options,
    std::vector<T>* out
);
