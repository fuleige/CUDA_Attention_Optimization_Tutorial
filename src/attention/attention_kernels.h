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
    kBasicBackward,
    kFlashBackward,
};

std::string attention_kernel_name(AttentionKernelKind kind);
AttentionKernelKind parse_attention_kernel(const std::string& name);

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

