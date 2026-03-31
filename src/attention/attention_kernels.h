#pragma once

#include "src/common/reference.h"

#include <string>
#include <vector>

enum class AttentionKernelKind {
    kBasicForward,
    kFlashForward,
    kPagedForward,
    kGqaForward,
    kSlidingForward,
    kBlockSparseForward,
    // Only one backward kernel is provided.  A production "flash backward"
    // would re-derive softmax probabilities from the saved log-sum-exp
    // rather than storing the full attention matrix, drastically cutting
    // memory usage.  That optimisation is out of scope for this teaching
    // codebase — see the FlashAttention-2 paper for details.
    kBasicBackward,
};

std::string attention_kernel_name(AttentionKernelKind kind);
AttentionKernelKind parse_attention_kernel(const std::string& name);

void validate_attention_inputs(
    AttentionKernelKind kind,
    DataType dtype,
    const AttentionShape& shape,
    const AttentionOptions& options,
    int page_size
);

template <typename T>
void launch_attention_forward(
    AttentionKernelKind kind,
    const T* d_q,
    const T* d_k,
    const T* d_v,
    T* d_out,
    const AttentionShape& shape,
    const AttentionOptions& options,
    const int* d_page_table,
    int page_size
);

void launch_attention_backward(
    AttentionKernelKind kind,
    const float* d_q,
    const float* d_k,
    const float* d_v,
    const float* d_grad_out,
    float* d_grad_q,
    float* d_grad_k,
    float* d_grad_v,
    const AttentionShape& shape,
    const AttentionOptions& options
);
