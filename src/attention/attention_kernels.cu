#include "src/attention/attention_kernels.h"

#include "src/common/cli.h"
#include "src/common/cuda_check.h"

#include <math_constants.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

enum class AttentionMode {
    kDense,
    kSliding,
    kBlockSparse,
};

template <typename T>
__device__ inline float load_attn(T value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float load_attn<half>(half value) {
    return __half2float(value);
}

template <typename T>
__device__ inline T cast_attn(float value) {
    return static_cast<T>(value);
}

template <>
__device__ inline half cast_attn<half>(float value) {
    return __float2half(value);
}

__device__ inline bool allow_dense_mask(
    int q_idx,
    int kv_idx,
    int seq_len_q,
    int seq_len_kv,
    bool causal
) {
    return !causal || kv_idx <= q_idx + (seq_len_kv - seq_len_q);
}

template <AttentionMode mode>
__device__ inline bool allow_attention(
    int q_idx,
    int kv_idx,
    int seq_len_q,
    int seq_len_kv,
    bool causal,
    int window_left,
    int block_size
) {
    if (!allow_dense_mask(q_idx, kv_idx, seq_len_q, seq_len_kv, causal)) {
        return false;
    }
    if constexpr (mode == AttentionMode::kSliding) {
        return window_left < 0 || kv_idx >= q_idx - window_left;
    }
    if constexpr (mode == AttentionMode::kBlockSparse) {
        const int q_block = q_idx / block_size;
        const int kv_block = kv_idx / block_size;
        return abs(q_block - kv_block) <= 1;
    }
    return true;
}

template <typename T, AttentionMode mode>
__global__ void attention_basic_forward_kernel(
    const T* q,
    const T* k,
    const T* v,
    T* out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    bool causal,
    int window_left,
    int block_size
) {
    const int row = blockIdx.x;
    const int q_idx = row % seq_len_q;
    const int tmp = row / seq_len_q;
    const int h = tmp % num_heads;
    const int b = tmp / num_heads;
    const int kv_h = (h * num_kv_heads) / num_heads;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    __shared__ float shared_values[2];
    float* shared_max = &shared_values[0];
    float* shared_weight = &shared_values[1];
    __shared__ float shared_denom;

    if (threadIdx.x == 0) {
        float max_score = -CUDART_INF_F;
        const int q_offset = (((b * num_heads + h) * seq_len_q + q_idx) * head_dim);
        for (int kv_idx = 0; kv_idx < seq_len_kv; ++kv_idx) {
            if (!allow_attention<mode>(q_idx, kv_idx, seq_len_q, seq_len_kv, causal, window_left, block_size)) {
                continue;
            }
            const int k_offset = (((b * num_kv_heads + kv_h) * seq_len_kv + kv_idx) * head_dim);
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += load_attn(q[q_offset + d]) * load_attn(k[k_offset + d]);
            }
            max_score = fmaxf(max_score, dot * scale);
        }
        *shared_max = max_score;
    }
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const int d0 = threadIdx.x;
    const int d1 = threadIdx.x + blockDim.x;
    const int q_offset = (((b * num_heads + h) * seq_len_q + q_idx) * head_dim);
    if (threadIdx.x == 0) {
        shared_denom = 0.0f;
    }
    __syncthreads();
    for (int kv_idx = 0; kv_idx < seq_len_kv; ++kv_idx) {
        if (threadIdx.x == 0) {
            if (!allow_attention<mode>(q_idx, kv_idx, seq_len_q, seq_len_kv, causal, window_left, block_size)) {
                *shared_weight = 0.0f;
            } else {
                const int k_offset = (((b * num_kv_heads + kv_h) * seq_len_kv + kv_idx) * head_dim);
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += load_attn(q[q_offset + d]) * load_attn(k[k_offset + d]);
                }
                *shared_weight = __expf(dot * scale - *shared_max);
                shared_denom += *shared_weight;
            }
        }
        __syncthreads();
        const int v_offset = (((b * num_kv_heads + kv_h) * seq_len_kv + kv_idx) * head_dim);
        if (d0 < head_dim) {
            acc0 += *shared_weight * load_attn(v[v_offset + d0]);
        }
        if (d1 < head_dim) {
            acc1 += *shared_weight * load_attn(v[v_offset + d1]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(shared_denom, 1e-12f);
    if (d0 < head_dim) {
        out[q_offset + d0] = cast_attn<T>(acc0 / denom);
    }
    if (d1 < head_dim) {
        out[q_offset + d1] = cast_attn<T>(acc1 / denom);
    }
}

template <typename T, AttentionMode mode, int TILE_KV>
__global__ void attention_flash_forward_kernel(
    const T* q,
    const T* k,
    const T* v,
    T* out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    bool causal,
    int window_left,
    int block_size
) {
    const int row = blockIdx.x;
    const int q_idx = row % seq_len_q;
    const int tmp = row / seq_len_q;
    const int h = tmp % num_heads;
    const int b = tmp / num_heads;
    const int kv_h = (h * num_kv_heads) / num_heads;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int q_offset = (((b * num_heads + h) * seq_len_q + q_idx) * head_dim);

    __shared__ float tile_scores[TILE_KV];
    __shared__ float shared_scalars[5];
    float& m_prev = shared_scalars[0];
    float& l_prev = shared_scalars[1];
    float& scale_old = shared_scalars[2];
    float& scale_new = shared_scalars[3];
    float& tile_max_shared = shared_scalars[4];

    const int d0 = threadIdx.x;
    const int d1 = threadIdx.x + blockDim.x;
    float out_acc0 = 0.0f;
    float out_acc1 = 0.0f;
    if (threadIdx.x == 0) {
        m_prev = -CUDART_INF_F;
        l_prev = 0.0f;
    }
    __syncthreads();

    for (int tile_start = 0; tile_start < seq_len_kv; tile_start += TILE_KV) {
        if (threadIdx.x == 0) {
            float tile_max = -CUDART_INF_F;
            for (int offset = 0; offset < TILE_KV; ++offset) {
                const int kv_idx = tile_start + offset;
                if (kv_idx >= seq_len_kv ||
                    !allow_attention<mode>(q_idx, kv_idx, seq_len_q, seq_len_kv, causal, window_left, block_size)) {
                    tile_scores[offset] = -CUDART_INF_F;
                    continue;
                }
                const int k_offset = (((b * num_kv_heads + kv_h) * seq_len_kv + kv_idx) * head_dim);
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += load_attn(q[q_offset + d]) * load_attn(k[k_offset + d]);
                }
                tile_scores[offset] = dot * scale;
                tile_max = fmaxf(tile_max, tile_scores[offset]);
            }

            float tile_sum = 0.0f;
            for (int offset = 0; offset < TILE_KV; ++offset) {
                if (tile_scores[offset] == -CUDART_INF_F) {
                    continue;
                }
                tile_sum += __expf(tile_scores[offset] - tile_max);
            }

            const float m_new = fmaxf(m_prev, tile_max);
            const float alpha = (m_prev == -CUDART_INF_F) ? 0.0f : __expf(m_prev - m_new);
            const float beta = (tile_max == -CUDART_INF_F) ? 0.0f : __expf(tile_max - m_new);
            const float l_new = alpha * l_prev + beta * tile_sum;
            scale_old = (l_new > 0.0f) ? (alpha * l_prev / l_new) : 0.0f;
            scale_new = (l_new > 0.0f) ? (beta / l_new) : 0.0f;
            tile_max_shared = tile_max;
            m_prev = m_new;
            l_prev = l_new;
        }
        __syncthreads();

        float tile_acc0 = 0.0f;
        float tile_acc1 = 0.0f;
        for (int offset = 0; offset < TILE_KV; ++offset) {
            const int kv_idx = tile_start + offset;
            if (kv_idx >= seq_len_kv || tile_scores[offset] == -CUDART_INF_F) {
                continue;
            }
            const float weight = __expf(tile_scores[offset] - tile_max_shared);
            const int v_offset = (((b * num_kv_heads + kv_h) * seq_len_kv + kv_idx) * head_dim);
            if (d0 < head_dim) {
                tile_acc0 += weight * load_attn(v[v_offset + d0]);
            }
            if (d1 < head_dim) {
                tile_acc1 += weight * load_attn(v[v_offset + d1]);
            }
        }
        out_acc0 = out_acc0 * scale_old + tile_acc0 * scale_new;
        out_acc1 = out_acc1 * scale_old + tile_acc1 * scale_new;
        __syncthreads();
    }

    if (d0 < head_dim) {
        out[q_offset + d0] = cast_attn<T>(out_acc0);
    }
    if (d1 < head_dim) {
        out[q_offset + d1] = cast_attn<T>(out_acc1);
    }
}

template <typename T>
__global__ void paged_attention_forward_kernel(
    const T* q,
    const T* k_pages,
    const T* v_pages,
    const int* page_table,
    T* out,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    int page_size
) {
    const int row = blockIdx.x;
    const int q_idx = row % seq_len_q;
    const int tmp = row / seq_len_q;
    const int h = tmp % num_heads;
    const int b = tmp / num_heads;
    const int kv_h = (h * num_kv_heads) / num_heads;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int q_offset = (((b * num_heads + h) * seq_len_q + q_idx) * head_dim);

    __shared__ float shared_max;
    __shared__ float shared_weight;
    __shared__ float shared_denom;

    if (threadIdx.x == 0) {
        shared_max = -CUDART_INF_F;
        for (int logical = 0; logical < seq_len_kv; ++logical) {
            const int page_slot = logical / page_size;
            const int offset_in_page = logical % page_size;
            const int physical_page = page_table[page_slot];
            const int k_offset =
                ((((physical_page * batch_size + b) * num_kv_heads + kv_h) * page_size + offset_in_page) * head_dim);
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += load_attn(q[q_offset + d]) * load_attn(k_pages[k_offset + d]);
            }
            shared_max = fmaxf(shared_max, dot * scale);
        }
        shared_denom = 0.0f;
    }
    __syncthreads();

    const int d0 = threadIdx.x;
    const int d1 = threadIdx.x + blockDim.x;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (int logical = 0; logical < seq_len_kv; ++logical) {
        const int page_slot = logical / page_size;
        const int offset_in_page = logical % page_size;
        const int physical_page = page_table[page_slot];
        const int page_offset =
            ((((physical_page * batch_size + b) * num_kv_heads + kv_h) * page_size + offset_in_page) * head_dim);
        if (threadIdx.x == 0) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += load_attn(q[q_offset + d]) * load_attn(k_pages[page_offset + d]);
            }
            shared_weight = __expf(dot * scale - shared_max);
            shared_denom += shared_weight;
        }
        __syncthreads();
        if (d0 < head_dim) {
            acc0 += shared_weight * load_attn(v_pages[page_offset + d0]);
        }
        if (d1 < head_dim) {
            acc1 += shared_weight * load_attn(v_pages[page_offset + d1]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(shared_denom, 1e-12f);
    if (d0 < head_dim) {
        out[q_offset + d0] = cast_attn<T>(acc0 / denom);
    }
    if (d1 < head_dim) {
        out[q_offset + d1] = cast_attn<T>(acc1 / denom);
    }
}

template <AttentionMode mode>
__global__ void attention_backward_kernel(
    const float* q,
    const float* k,
    const float* v,
    const float* grad_out,
    float* grad_q,
    float* grad_k,
    float* grad_v,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    bool causal,
    int window_left,
    int block_size
) {
    const int row = blockIdx.x;
    const int q_idx = row % seq_len_q;
    const int tmp = row / seq_len_q;
    const int h = tmp % num_heads;
    const int b = tmp / num_heads;
    const int kv_h = (h * num_kv_heads) / num_heads;
    const float scale = rsqrtf(static_cast<float>(head_dim));
    const int q_offset = (((b * num_heads + h) * seq_len_q + q_idx) * head_dim);

    extern __shared__ float workspace[];
    float* logits = workspace;
    float* probs = workspace + seq_len_kv;
    float* dp = workspace + seq_len_kv * 2;

    if (threadIdx.x != 0) {
        return;
    }

    float max_score = -CUDART_INF_F;
    for (int kv_idx = 0; kv_idx < seq_len_kv; ++kv_idx) {
        if (!allow_attention<mode>(q_idx, kv_idx, seq_len_q, seq_len_kv, causal, window_left, block_size)) {
            logits[kv_idx] = -CUDART_INF_F;
            probs[kv_idx] = 0.0f;
            dp[kv_idx] = 0.0f;
            continue;
        }
        const int k_offset = (((b * num_kv_heads + kv_h) * seq_len_kv + kv_idx) * head_dim);
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q[q_offset + d] * k[k_offset + d];
        }
        logits[kv_idx] = dot * scale;
        max_score = fmaxf(max_score, logits[kv_idx]);
    }

    float denom = 0.0f;
    for (int kv_idx = 0; kv_idx < seq_len_kv; ++kv_idx) {
        if (logits[kv_idx] == -CUDART_INF_F) {
            continue;
        }
        probs[kv_idx] = expf(logits[kv_idx] - max_score);
        denom += probs[kv_idx];
    }
    denom = fmaxf(denom, 1e-12f);
    for (int kv_idx = 0; kv_idx < seq_len_kv; ++kv_idx) {
        probs[kv_idx] /= denom;
    }

    float weighted_sum = 0.0f;
    for (int kv_idx = 0; kv_idx < seq_len_kv; ++kv_idx) {
        const int v_offset = (((b * num_kv_heads + kv_h) * seq_len_kv + kv_idx) * head_dim);
        float value = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            value += grad_out[q_offset + d] * v[v_offset + d];
            atomicAdd(&grad_v[v_offset + d], probs[kv_idx] * grad_out[q_offset + d]);
        }
        dp[kv_idx] = value;
        weighted_sum += value * probs[kv_idx];
    }

    for (int kv_idx = 0; kv_idx < seq_len_kv; ++kv_idx) {
        if (probs[kv_idx] == 0.0f) {
            continue;
        }
        const float ds = probs[kv_idx] * (dp[kv_idx] - weighted_sum);
        const int k_offset = (((b * num_kv_heads + kv_h) * seq_len_kv + kv_idx) * head_dim);
        for (int d = 0; d < head_dim; ++d) {
            grad_q[q_offset + d] += ds * scale * k[k_offset + d];
            atomicAdd(&grad_k[k_offset + d], ds * scale * q[q_offset + d]);
        }
    }
}

template <typename T>
void launch_dense_like_forward(
    AttentionKernelKind kind,
    const T* d_q,
    const T* d_k,
    const T* d_v,
    T* d_out,
    const AttentionShape& shape,
    const AttentionOptions& options
) {
    const int rows = shape.batch_size * shape.num_heads * shape.seq_len_q;
    const int threads = 128;
    switch (kind) {
        case AttentionKernelKind::kBasicForward:
        case AttentionKernelKind::kGqaForward:
            attention_basic_forward_kernel<T, AttentionMode::kDense><<<rows, threads>>>(
                d_q, d_k, d_v, d_out, shape.batch_size, shape.num_heads, shape.num_kv_heads, shape.seq_len_q,
                shape.seq_len_kv, shape.head_dim, options.causal, options.window_left, options.block_size);
            break;
        case AttentionKernelKind::kSlidingForward:
            attention_basic_forward_kernel<T, AttentionMode::kSliding><<<rows, threads>>>(
                d_q, d_k, d_v, d_out, shape.batch_size, shape.num_heads, shape.num_kv_heads, shape.seq_len_q,
                shape.seq_len_kv, shape.head_dim, options.causal, options.window_left, options.block_size);
            break;
        case AttentionKernelKind::kBlockSparseForward:
            attention_basic_forward_kernel<T, AttentionMode::kBlockSparse><<<rows, threads>>>(
                d_q, d_k, d_v, d_out, shape.batch_size, shape.num_heads, shape.num_kv_heads, shape.seq_len_q,
                shape.seq_len_kv, shape.head_dim, options.causal, options.window_left, options.block_size);
            break;
        case AttentionKernelKind::kFlashForward:
            attention_flash_forward_kernel<T, AttentionMode::kDense, 32><<<rows, threads>>>(
                d_q, d_k, d_v, d_out, shape.batch_size, shape.num_heads, shape.num_kv_heads, shape.seq_len_q,
                shape.seq_len_kv, shape.head_dim, options.causal, options.window_left, options.block_size);
            break;
        default:
            throw std::runtime_error("Unsupported forward kernel kind.");
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

std::string attention_kernel_name(AttentionKernelKind kind) {
    switch (kind) {
        case AttentionKernelKind::kBasicForward:
            return "basic_fwd";
        case AttentionKernelKind::kFlashForward:
            return "flash_fwd";
        case AttentionKernelKind::kPagedForward:
            return "paged_fwd";
        case AttentionKernelKind::kGqaForward:
            return "gqa_fwd";
        case AttentionKernelKind::kSlidingForward:
            return "sliding_fwd";
        case AttentionKernelKind::kBlockSparseForward:
            return "block_sparse_fwd";
        case AttentionKernelKind::kBasicBackward:
            return "basic_bwd";
        case AttentionKernelKind::kFlashBackward:
            return "flash_bwd";
    }
    return "unknown";
}

AttentionKernelKind parse_attention_kernel(const std::string& name) {
    const std::string lowered = CliArgs::lower(name);
    if (lowered == "basic_fwd") {
        return AttentionKernelKind::kBasicForward;
    }
    if (lowered == "flash_fwd") {
        return AttentionKernelKind::kFlashForward;
    }
    if (lowered == "paged_fwd") {
        return AttentionKernelKind::kPagedForward;
    }
    if (lowered == "gqa_fwd") {
        return AttentionKernelKind::kGqaForward;
    }
    if (lowered == "sliding_fwd") {
        return AttentionKernelKind::kSlidingForward;
    }
    if (lowered == "block_sparse_fwd") {
        return AttentionKernelKind::kBlockSparseForward;
    }
    if (lowered == "basic_bwd") {
        return AttentionKernelKind::kBasicBackward;
    }
    if (lowered == "flash_bwd") {
        return AttentionKernelKind::kFlashBackward;
    }
    throw std::runtime_error("Unknown attention kernel: " + name);
}

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
) {
    if (kind == AttentionKernelKind::kPagedForward) {
        const int rows = shape.batch_size * shape.num_heads * shape.seq_len_q;
        paged_attention_forward_kernel<<<rows, 128>>>(
            d_q, d_k, d_v, d_page_table, d_out, shape.batch_size, shape.num_heads, shape.num_kv_heads,
            shape.seq_len_q, shape.seq_len_kv, shape.head_dim, page_size);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    launch_dense_like_forward(kind, d_q, d_k, d_v, d_out, shape, options);
}

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
) {
    const int rows = shape.batch_size * shape.num_heads * shape.seq_len_q;
    const std::size_t shared_bytes = sizeof(float) * shape.seq_len_kv * 3;
    switch (kind) {
        case AttentionKernelKind::kBasicBackward:
            attention_backward_kernel<AttentionMode::kDense><<<rows, 1, shared_bytes>>>(
                d_q, d_k, d_v, d_grad_out, d_grad_q, d_grad_k, d_grad_v, shape.batch_size, shape.num_heads,
                shape.num_kv_heads, shape.seq_len_q, shape.seq_len_kv, shape.head_dim, options.causal,
                options.window_left, options.block_size);
            break;
        case AttentionKernelKind::kFlashBackward:
            attention_backward_kernel<AttentionMode::kDense><<<rows, 1, shared_bytes>>>(
                d_q, d_k, d_v, d_grad_out, d_grad_q, d_grad_k, d_grad_v, shape.batch_size, shape.num_heads,
                shape.num_kv_heads, shape.seq_len_q, shape.seq_len_kv, shape.head_dim, options.causal,
                options.window_left, options.block_size);
            break;
        default:
            throw std::runtime_error("Unsupported backward kernel kind.");
    }
    CUDA_CHECK(cudaGetLastError());
}

template void launch_attention_forward<float>(
    AttentionKernelKind,
    const float*,
    const float*,
    const float*,
    float*,
    const AttentionShape&,
    const AttentionOptions&,
    const int*,
    int
);
template void launch_attention_forward<half>(
    AttentionKernelKind,
    const half*,
    const half*,
    const half*,
    half*,
    const AttentionShape&,
    const AttentionOptions&,
    const int*,
    int
);
