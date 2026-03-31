#include "src/common/reference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace {

inline bool attention_mask_allows(
    int q_index,
    int kv_index,
    int seq_len_q,
    int seq_len_kv,
    const AttentionOptions& options
) {
    if (options.causal && kv_index > q_index + (seq_len_kv - seq_len_q)) {
        return false;
    }
    if (options.window_left >= 0 && kv_index < q_index - options.window_left) {
        return false;
    }
    if (options.block_sparse) {
        const int q_block = q_index / options.block_size;
        const int kv_block = kv_index / options.block_size;
        if (std::abs(q_block - kv_block) > 1) {
            return false;
        }
    }
    return true;
}

}  // namespace

template <typename T>
void gemm_reference(
    const std::vector<T>& a,
    const std::vector<T>& b,
    std::vector<T>* c,
    int m,
    int n,
    int k
) {
    c->assign(m * n, from_float<T>(0.0f));
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                acc += to_float(a[row * k + inner]) * to_float(b[inner * n + col]);
            }
            (*c)[row * n + col] = from_float<T>(acc);
        }
    }
}

template <typename T>
void attention_forward_reference(
    const std::vector<T>& q,
    const std::vector<T>& k,
    const std::vector<T>& v,
    const AttentionShape& shape,
    const AttentionOptions& options,
    std::vector<T>* out
) {
    const int d = shape.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    out->assign(shape.batch_size * shape.num_heads * shape.seq_len_q * d, from_float<T>(0.0f));
    for (int b = 0; b < shape.batch_size; ++b) {
        for (int h = 0; h < shape.num_heads; ++h) {
            const int kv_h = (h * shape.num_kv_heads) / shape.num_heads;
            for (int q_idx = 0; q_idx < shape.seq_len_q; ++q_idx) {
                std::vector<float> scores(shape.seq_len_kv, -std::numeric_limits<float>::infinity());
                float max_score = -std::numeric_limits<float>::infinity();
                for (int kv_idx = 0; kv_idx < shape.seq_len_kv; ++kv_idx) {
                    if (!attention_mask_allows(q_idx, kv_idx, shape.seq_len_q, shape.seq_len_kv, options)) {
                        continue;
                    }
                    float dot = 0.0f;
                    const int q_offset = (((b * shape.num_heads + h) * shape.seq_len_q + q_idx) * d);
                    const int k_offset = (((b * shape.num_kv_heads + kv_h) * shape.seq_len_kv + kv_idx) * d);
                    for (int dim = 0; dim < d; ++dim) {
                        dot += to_float(q[q_offset + dim]) * to_float(k[k_offset + dim]);
                    }
                    scores[kv_idx] = dot * scale;
                    max_score = std::max(max_score, scores[kv_idx]);
                }
                float denom = 0.0f;
                for (float& value : scores) {
                    if (value == -std::numeric_limits<float>::infinity()) {
                        value = 0.0f;
                        continue;
                    }
                    value = std::exp(value - max_score);
                    denom += value;
                }
                const int out_offset = (((b * shape.num_heads + h) * shape.seq_len_q + q_idx) * d);
                for (int kv_idx = 0; kv_idx < shape.seq_len_kv; ++kv_idx) {
                    if (scores[kv_idx] == 0.0f) {
                        continue;
                    }
                    const float prob = scores[kv_idx] / denom;
                    const int v_offset = (((b * shape.num_kv_heads + kv_h) * shape.seq_len_kv + kv_idx) * d);
                    for (int dim = 0; dim < d; ++dim) {
                        const float accum = to_float((*out)[out_offset + dim]) + prob * to_float(v[v_offset + dim]);
                        (*out)[out_offset + dim] = from_float<T>(accum);
                    }
                }
            }
        }
    }
}

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
) {
    const int d = shape.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    grad_q->assign(q.size(), from_float<T>(0.0f));
    grad_k->assign(k.size(), from_float<T>(0.0f));
    grad_v->assign(v.size(), from_float<T>(0.0f));

    for (int b = 0; b < shape.batch_size; ++b) {
        for (int h = 0; h < shape.num_heads; ++h) {
            const int kv_h = (h * shape.num_kv_heads) / shape.num_heads;
            for (int q_idx = 0; q_idx < shape.seq_len_q; ++q_idx) {
                std::vector<float> logits(shape.seq_len_kv, -std::numeric_limits<float>::infinity());
                std::vector<float> probs(shape.seq_len_kv, 0.0f);
                float max_score = -std::numeric_limits<float>::infinity();
                const int q_offset = (((b * shape.num_heads + h) * shape.seq_len_q + q_idx) * d);
                for (int kv_idx = 0; kv_idx < shape.seq_len_kv; ++kv_idx) {
                    if (!attention_mask_allows(q_idx, kv_idx, shape.seq_len_q, shape.seq_len_kv, options)) {
                        continue;
                    }
                    const int k_offset = (((b * shape.num_kv_heads + kv_h) * shape.seq_len_kv + kv_idx) * d);
                    float dot = 0.0f;
                    for (int dim = 0; dim < d; ++dim) {
                        dot += to_float(q[q_offset + dim]) * to_float(k[k_offset + dim]);
                    }
                    logits[kv_idx] = dot * scale;
                    max_score = std::max(max_score, logits[kv_idx]);
                }
                float denom = 0.0f;
                for (int kv_idx = 0; kv_idx < shape.seq_len_kv; ++kv_idx) {
                    if (logits[kv_idx] == -std::numeric_limits<float>::infinity()) {
                        continue;
                    }
                    probs[kv_idx] = std::exp(logits[kv_idx] - max_score);
                    denom += probs[kv_idx];
                }
                for (float& p : probs) {
                    p = denom > 0.0f ? (p / denom) : 0.0f;
                }

                std::vector<float> dp(shape.seq_len_kv, 0.0f);
                std::vector<float> ds(shape.seq_len_kv, 0.0f);
                for (int kv_idx = 0; kv_idx < shape.seq_len_kv; ++kv_idx) {
                    if (probs[kv_idx] == 0.0f) {
                        continue;
                    }
                    const int v_offset = (((b * shape.num_kv_heads + kv_h) * shape.seq_len_kv + kv_idx) * d);
                    const int grad_out_offset = q_offset;
                    float value = 0.0f;
                    for (int dim = 0; dim < d; ++dim) {
                        value += to_float(grad_out[grad_out_offset + dim]) * to_float(v[v_offset + dim]);
                        const float gv = to_float((*grad_v)[v_offset + dim]) +
                            probs[kv_idx] * to_float(grad_out[grad_out_offset + dim]);
                        (*grad_v)[v_offset + dim] = from_float<T>(gv);
                    }
                    dp[kv_idx] = value;
                }
                float weighted_sum = 0.0f;
                for (int kv_idx = 0; kv_idx < shape.seq_len_kv; ++kv_idx) {
                    weighted_sum += dp[kv_idx] * probs[kv_idx];
                }
                for (int kv_idx = 0; kv_idx < shape.seq_len_kv; ++kv_idx) {
                    ds[kv_idx] = probs[kv_idx] * (dp[kv_idx] - weighted_sum);
                }
                for (int kv_idx = 0; kv_idx < shape.seq_len_kv; ++kv_idx) {
                    if (probs[kv_idx] == 0.0f) {
                        continue;
                    }
                    const int k_offset = (((b * shape.num_kv_heads + kv_h) * shape.seq_len_kv + kv_idx) * d);
                    for (int dim = 0; dim < d; ++dim) {
                        const float gq = to_float((*grad_q)[q_offset + dim]) +
                            ds[kv_idx] * scale * to_float(k[k_offset + dim]);
                        const float gk = to_float((*grad_k)[k_offset + dim]) +
                            ds[kv_idx] * scale * to_float(q[q_offset + dim]);
                        (*grad_q)[q_offset + dim] = from_float<T>(gq);
                        (*grad_k)[k_offset + dim] = from_float<T>(gk);
                    }
                }
            }
        }
    }
}

template <typename T>
void paged_attention_reference(
    const std::vector<T>& q,
    const std::vector<T>& k_pages,
    const std::vector<T>& v_pages,
    const std::vector<int>& page_table,
    int page_size,
    const AttentionShape& shape,
    std::vector<T>* out
) {
    const int num_pages = static_cast<int>(page_table.size());
    std::vector<T> k(shape.batch_size * shape.num_kv_heads * shape.seq_len_kv * shape.head_dim);
    std::vector<T> v(shape.batch_size * shape.num_kv_heads * shape.seq_len_kv * shape.head_dim);
    for (int logical = 0; logical < shape.seq_len_kv; ++logical) {
        const int page_slot = logical / page_size;
        const int offset_in_page = logical % page_size;
        const int physical_page = page_table[page_slot];
        for (int b = 0; b < shape.batch_size; ++b) {
            for (int h = 0; h < shape.num_kv_heads; ++h) {
                for (int d = 0; d < shape.head_dim; ++d) {
                    const int dst = (((b * shape.num_kv_heads + h) * shape.seq_len_kv + logical) * shape.head_dim + d);
                    const int src = ((((physical_page * shape.batch_size + b) * shape.num_kv_heads + h) * page_size +
                        offset_in_page) * shape.head_dim + d);
                    k[dst] = k_pages[src];
                    v[dst] = v_pages[src];
                }
            }
        }
    }
    AttentionOptions options {};
    attention_forward_reference(q, k, v, shape, options, out);
}

template void gemm_reference<float>(
    const std::vector<float>&,
    const std::vector<float>&,
    std::vector<float>*,
    int,
    int,
    int
);
template void gemm_reference<half>(
    const std::vector<half>&,
    const std::vector<half>&,
    std::vector<half>*,
    int,
    int,
    int
);

template void attention_forward_reference<float>(
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    const AttentionShape&,
    const AttentionOptions&,
    std::vector<float>*
);
template void attention_forward_reference<half>(
    const std::vector<half>&,
    const std::vector<half>&,
    const std::vector<half>&,
    const AttentionShape&,
    const AttentionOptions&,
    std::vector<half>*
);

template void attention_backward_reference<float>(
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    const AttentionShape&,
    const AttentionOptions&,
    std::vector<float>*,
    std::vector<float>*,
    std::vector<float>*
);
template void attention_backward_reference<half>(
    const std::vector<half>&,
    const std::vector<half>&,
    const std::vector<half>&,
    const std::vector<half>&,
    const AttentionShape&,
    const AttentionOptions&,
    std::vector<half>*,
    std::vector<half>*,
    std::vector<half>*
);

template void paged_attention_reference<float>(
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<float>&,
    const std::vector<int>&,
    int,
    const AttentionShape&,
    std::vector<float>*
);
template void paged_attention_reference<half>(
    const std::vector<half>&,
    const std::vector<half>&,
    const std::vector<half>&,
    const std::vector<int>&,
    int,
    const AttentionShape&,
    std::vector<half>*
);
