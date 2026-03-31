#include "src/gemm/gemm_kernels.h"

#include "src/common/cuda_check.h"

#include <mma.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace {

template <typename T>
__device__ inline float load_as_float(T value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float load_as_float<half>(half value) {
    return __half2float(value);
}

// ── Naive kernel ──────────────────────────────────────────────────────
// threadIdx.x is mapped to *rows* so that adjacent threads write to
// addresses that are `n` elements apart in C (row-major).  This causes
// non-coalesced global-memory writes — the GPU must issue one transaction
// per thread instead of merging them.  Compare with the coalesced kernel
// below to see the performance impact of fixing this one mapping.
template <typename T>
__global__ void gemm_naive_kernel(const T* a, const T* b, float* c, int m, int n, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;   // threadIdx.x → row (BAD for coalescing)
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        acc += load_as_float(a[row * k + inner]) * load_as_float(b[inner * n + col]);
    }
    // Adjacent threads (threadIdx.x, threadIdx.x+1) write to c[row*n+col]
    // and c[(row+1)*n+col] — stride of `n`, not 1 → non-coalesced.
    c[row * n + col] = acc;
}

// ── Coalesced kernel ─────────────────────────────────────────────────
// The only change from the naive kernel: threadIdx.x is now mapped to
// *columns*.  Adjacent threads write c[row*n+col] and c[row*n+col+1] —
// stride of 1 — so the hardware can merge them into a single wide
// transaction.  This alone can give a 2–5× speedup on large matrices.
template <typename T>
__global__ void gemm_coalesced_kernel(const T* a, const T* b, float* c, int m, int n, int k) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;   // threadIdx.x → col (GOOD for coalescing)
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int inner = 0; inner < k; ++inner) {
        acc += load_as_float(a[row * k + inner]) * load_as_float(b[inner * n + col]);
    }
    c[row * n + col] = acc;
}

// ── Shared-memory tiled kernel ───────────────────────────────────────
// Each thread block loads a TILE×TILE sub-matrix of A and B into shared
// memory, then every thread in the block can reuse that data.  This
// converts O(TILE) global loads per element into O(1) shared loads,
// reducing global memory traffic by ~TILE×.
template <typename T, int TILE>
__global__ void gemm_shared_kernel(const T* a, const T* b, float* c, int m, int n, int k) {
    __shared__ T a_tile[TILE][TILE];
    __shared__ T b_tile[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    for (int tile = 0; tile < (k + TILE - 1) / TILE; ++tile) {
        const int a_col = tile * TILE + threadIdx.x;
        const int b_row = tile * TILE + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : from_float<T>(0.0f);
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : from_float<T>(0.0f);
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < TILE; ++inner) {
            acc += load_as_float(a_tile[threadIdx.y][inner]) * load_as_float(b_tile[inner][threadIdx.x]);
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

// ── Register-blocked kernel ──────────────────────────────────────────
// Each thread computes a TM×TN sub-tile of the output, keeping partial
// sums in registers.  This increases the compute-to-load ratio: each
// shared-memory value is reused TM (or TN) times before being evicted.
// Tuning BM/BN/BK/TM/TN is the main lever for hitting peak throughput.
template <typename T, int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_register_blocked_kernel(const T* a, const T* b, float* c, int m, int n, int k) {
    __shared__ T a_tile[BM][BK];
    __shared__ T b_tile[BK][BN];

    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    constexpr int row_stride = BM / TM;
    constexpr int col_stride = BN / TN;
    float accum[TM][TN] {};

    for (int tile = 0; tile < (k + BK - 1) / BK; ++tile) {
        for (int i = local_row; i < BM; i += blockDim.y) {
            for (int j = local_col; j < BK; j += blockDim.x) {
                const int g_row = block_row + i;
                const int g_col = tile * BK + j;
                a_tile[i][j] = (g_row < m && g_col < k) ? a[g_row * k + g_col] : from_float<T>(0.0f);
            }
        }
        for (int i = local_row; i < BK; i += blockDim.y) {
            for (int j = local_col; j < BN; j += blockDim.x) {
                const int g_row = tile * BK + i;
                const int g_col = block_col + j;
                b_tile[i][j] = (g_row < k && g_col < n) ? b[g_row * n + g_col] : from_float<T>(0.0f);
            }
        }
        __syncthreads();
        for (int inner = 0; inner < BK; ++inner) {
            float a_frag[TM];
            float b_frag[TN];
            for (int i = 0; i < TM; ++i) {
                const int row = local_row + i * row_stride;
                a_frag[i] = load_as_float(a_tile[row][inner]);
            }
            for (int j = 0; j < TN; ++j) {
                const int col = local_col + j * col_stride;
                b_frag[j] = load_as_float(b_tile[inner][col]);
            }
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; ++i) {
        const int row = block_row + local_row + i * row_stride;
        if (row >= m) {
            continue;
        }
        for (int j = 0; j < TN; ++j) {
            const int col = block_col + local_col + j * col_stride;
            if (col < n) {
                c[row * n + col] = accum[i][j];
            }
        }
    }
}

// ── Vectorized (unrolled) kernel ──────────────────────────────────────
// Same tiling as the shared-memory kernel, but the inner loop is manually
// unrolled by 4.  This reduces loop overhead and lets the compiler
// schedule independent multiply-adds in parallel.
// IMPORTANT: TILE must be a multiple of 4; otherwise the unrolled loop
// reads past the end of shared memory — a silent, hard-to-debug bug.
template <typename T, int TILE>
__global__ void gemm_vectorized_kernel(const T* a, const T* b, float* c, int m, int n, int k) {
    static_assert(TILE % 4 == 0, "TILE must be a multiple of 4 for the unrolled inner loop");
    __shared__ T a_tile[TILE][TILE];
    __shared__ T b_tile[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int tile = 0; tile < (k + TILE - 1) / TILE; ++tile) {
        const int a_col = tile * TILE + threadIdx.x;
        const int b_row = tile * TILE + threadIdx.y;

        a_tile[threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : from_float<T>(0.0f);
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : from_float<T>(0.0f);
        __syncthreads();

#pragma unroll 4
        for (int inner = 0; inner < TILE; inner += 4) {
            acc += load_as_float(a_tile[threadIdx.y][inner + 0]) * load_as_float(b_tile[inner + 0][threadIdx.x]);
            acc += load_as_float(a_tile[threadIdx.y][inner + 1]) * load_as_float(b_tile[inner + 1][threadIdx.x]);
            acc += load_as_float(a_tile[threadIdx.y][inner + 2]) * load_as_float(b_tile[inner + 2][threadIdx.x]);
            acc += load_as_float(a_tile[threadIdx.y][inner + 3]) * load_as_float(b_tile[inner + 3][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

// ── Double-buffered kernel ───────────────────────────────────────────
// Two sets of shared-memory tiles ("buffers") are used: while one is
// being read for the current tile's computation, the next tile's data is
// loaded into the other buffer.  This overlaps global-memory latency
// with arithmetic, keeping the ALUs busy instead of stalling.
template <typename T, int TILE>
__global__ void gemm_double_buffered_kernel(const T* a, const T* b, float* c, int m, int n, int k) {
    __shared__ T a_tile[2][TILE][TILE];
    __shared__ T b_tile[2][TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    const int tile_count = (k + TILE - 1) / TILE;
    float acc = 0.0f;

    if (tile_count == 0) {
        return;
    }

    auto load_tile = [&](int buffer, int tile) {
        const int a_col = tile * TILE + threadIdx.x;
        const int b_row = tile * TILE + threadIdx.y;
        a_tile[buffer][threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : from_float<T>(0.0f);
        b_tile[buffer][threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : from_float<T>(0.0f);
    };

    load_tile(0, 0);
    __syncthreads();

    for (int tile = 0; tile < tile_count; ++tile) {
        const int read_buffer = tile % 2;
        const int next_tile = tile + 1;
        if (next_tile < tile_count) {
            load_tile((tile + 1) % 2, next_tile);
        }
#pragma unroll
        for (int inner = 0; inner < TILE; ++inner) {
            acc += load_as_float(a_tile[read_buffer][threadIdx.y][inner]) *
                load_as_float(b_tile[read_buffer][inner][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

// ── Async-pipeline kernel (teaching placeholder) ────────────────────
// Structurally identical to the double-buffered kernel above.  On Ampere
// (sm_80+) GPUs you would replace the explicit global→shared copies with
// `cp.async` instructions and `__pipeline_*` fences, letting the hardware
// copy data in the background without occupying register file or ALU.
// This kernel exists as a placeholder so you can benchmark the same loop
// structure and then drop in the real async copies as an exercise.
template <typename T, int TILE>
__global__ void gemm_async_pipeline_kernel(const T* a, const T* b, float* c, int m, int n, int k) {
    __shared__ T a_tile[2][TILE][TILE];
    __shared__ T b_tile[2][TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    const int tile_count = (k + TILE - 1) / TILE;
    float acc = 0.0f;

    auto load_tile = [&](int buffer, int tile) {
        const int a_col = tile * TILE + threadIdx.x;
        const int b_row = tile * TILE + threadIdx.y;
        a_tile[buffer][threadIdx.y][threadIdx.x] =
            (row < m && a_col < k) ? a[row * k + a_col] : from_float<T>(0.0f);
        b_tile[buffer][threadIdx.y][threadIdx.x] =
            (b_row < k && col < n) ? b[b_row * n + col] : from_float<T>(0.0f);
    };

    if (tile_count == 0) {
        return;
    }

    load_tile(0, 0);
    __syncthreads();
    for (int tile = 0; tile < tile_count; ++tile) {
        const int read_buffer = tile % 2;
        if (tile + 1 < tile_count) {
            load_tile((tile + 1) % 2, tile + 1);
        }
#pragma unroll
        for (int inner = 0; inner < TILE; ++inner) {
            acc += load_as_float(a_tile[read_buffer][threadIdx.y][inner]) *
                load_as_float(b_tile[read_buffer][inner][threadIdx.x]);
        }
        __syncthreads();
    }
    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

// ── WMMA (Tensor Core) kernel ────────────────────────────────────────
// Uses the nvcuda::wmma API to offload 16×16×16 matrix-multiply-
// accumulate to Tensor Cores (available on Volta / sm_70+).  B is stored
// in column-major so it can be loaded as wmma::col_major, which avoids
// an in-kernel transpose.  Each warp computes one 16×16 output tile.
__global__ void gemm_wmma_kernel(const half* a, const half* b_col_major, float* c, int m, int n, int k) {
    using namespace nvcuda;
    const int warp_idx = threadIdx.x / warpSize;
    const int warp_m = blockIdx.x * (blockDim.x / warpSize) + warp_idx;
    const int warp_n = blockIdx.y;

    const int row = warp_n * 16;
    const int col = warp_m * 16;
    if (row >= m || col >= n) {
        return;
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int tile = 0; tile < k; tile += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        const half* a_ptr = a + row * k + tile;
        const half* b_ptr = b_col_major + col * k + tile;
        wmma::load_matrix_sync(a_frag, a_ptr, k);
        wmma::load_matrix_sync(b_frag, b_ptr, k);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(c + row * n + col, c_frag, n, wmma::mem_row_major);
}

template <typename T>
float dispatch_gemm(GemmKernelKind kind, const T* d_a, const T* d_b, float* d_c, const GemmShape& shape) {
    constexpr int tile = 16;
    dim3 block(tile, tile);
    // Default grid: blockIdx.x covers columns, blockIdx.y covers rows.
    dim3 grid((shape.n + tile - 1) / tile, (shape.m + tile - 1) / tile);

    switch (kind) {
        case GemmKernelKind::kNaive: {
            // Naive kernel maps threadIdx.x → row, so blockIdx.x covers rows.
            dim3 naive_grid((shape.m + tile - 1) / tile, (shape.n + tile - 1) / tile);
            gemm_naive_kernel<<<naive_grid, block>>>(d_a, d_b, d_c, shape.m, shape.n, shape.k);
            break;
        }
        case GemmKernelKind::kCoalesced:
            gemm_coalesced_kernel<<<grid, block>>>(d_a, d_b, d_c, shape.m, shape.n, shape.k);
            break;
        case GemmKernelKind::kShared:
            gemm_shared_kernel<T, tile><<<grid, block>>>(d_a, d_b, d_c, shape.m, shape.n, shape.k);
            break;
        case GemmKernelKind::kRegisterBlocked:
            gemm_register_blocked_kernel<T, 32, 32, 8, 2, 2><<<dim3((shape.n + 31) / 32, (shape.m + 31) / 32), dim3(16, 16)>>>(
                d_a, d_b, d_c, shape.m, shape.n, shape.k);
            break;
        case GemmKernelKind::kVectorized:
            gemm_vectorized_kernel<T, 16><<<grid, block>>>(d_a, d_b, d_c, shape.m, shape.n, shape.k);
            break;
        case GemmKernelKind::kDoubleBuffered:
            gemm_double_buffered_kernel<T, 16><<<grid, block>>>(d_a, d_b, d_c, shape.m, shape.n, shape.k);
            break;
        case GemmKernelKind::kAsyncPipeline:
            gemm_async_pipeline_kernel<T, 16><<<grid, block>>>(d_a, d_b, d_c, shape.m, shape.n, shape.k);
            break;
        case GemmKernelKind::kWmma:
            throw std::runtime_error("WMMA is only available for fp16 inputs.");
    }
    CUDA_CHECK(cudaGetLastError());
    return 0.0f;
}

}  // namespace

std::string gemm_kernel_name(GemmKernelKind kind) {
    switch (kind) {
        case GemmKernelKind::kNaive:
            return "naive";
        case GemmKernelKind::kCoalesced:
            return "coalesced";
        case GemmKernelKind::kShared:
            return "shared";
        case GemmKernelKind::kRegisterBlocked:
            return "register_blocked";
        case GemmKernelKind::kVectorized:
            return "vectorized";
        case GemmKernelKind::kDoubleBuffered:
            return "double_buffered";
        case GemmKernelKind::kAsyncPipeline:
            return "async_pipeline";
        case GemmKernelKind::kWmma:
            return "wmma";
    }
    return "unknown";
}

GemmKernelKind parse_gemm_kernel(const std::string& name) {
    const std::string lowered = CliArgs::lower(name);
    if (lowered == "naive") {
        return GemmKernelKind::kNaive;
    }
    if (lowered == "coalesced") {
        return GemmKernelKind::kCoalesced;
    }
    if (lowered == "shared") {
        return GemmKernelKind::kShared;
    }
    if (lowered == "register_blocked") {
        return GemmKernelKind::kRegisterBlocked;
    }
    if (lowered == "vectorized") {
        return GemmKernelKind::kVectorized;
    }
    if (lowered == "double_buffered") {
        return GemmKernelKind::kDoubleBuffered;
    }
    if (lowered == "async_pipeline") {
        return GemmKernelKind::kAsyncPipeline;
    }
    if (lowered == "wmma") {
        return GemmKernelKind::kWmma;
    }
    throw std::runtime_error("Unknown GEMM kernel: " + name);
}

void validate_gemm_inputs(GemmKernelKind kind, DataType dtype, const GemmShape& shape) {
    const auto require = [](bool condition, const std::string& message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    };
    require(shape.m > 0 && shape.n > 0 && shape.k > 0, "m, n, k must all be > 0");
    if (kind == GemmKernelKind::kWmma) {
        require(dtype == DataType::kFloat16, "WMMA kernel requires --dtype fp16");
        require((shape.m % 16) == 0 && (shape.n % 16) == 0 && (shape.k % 16) == 0,
                "WMMA requires m, n, k to be multiples of 16");
    }
    // Guard against int32 overflow in row*k+col / row*n+col index math.
    {
        const long long max_idx = std::max({
            static_cast<long long>(shape.m) * shape.k,
            static_cast<long long>(shape.k) * shape.n,
            static_cast<long long>(shape.m) * shape.n,
        });
        constexpr long long limit = static_cast<long long>(std::numeric_limits<int>::max());
        require(max_idx <= limit,
                "Matrix size exceeds int32 range (" + std::to_string(max_idx) +
                " elements).  Use smaller m/n/k.");
    }
}

template <typename T>
float launch_gemm(
    GemmKernelKind kind,
    const T* d_a,
    const T* d_b,
    float* d_c,
    const GemmShape& shape
) {
    return dispatch_gemm(kind, d_a, d_b, d_c, shape);
}

template <>
float launch_gemm<half>(
    GemmKernelKind kind,
    const half* d_a,
    const half* d_b,
    float* d_c,
    const GemmShape& shape
) {
    if (kind == GemmKernelKind::kWmma) {
        dim3 block(128);
        dim3 grid((shape.n + 63) / 64, (shape.m + 15) / 16);
        gemm_wmma_kernel<<<grid, block>>>(d_a, d_b, d_c, shape.m, shape.n, shape.k);
        CUDA_CHECK(cudaGetLastError());
        return 0.0f;
    }
    return dispatch_gemm(kind, d_a, d_b, d_c, shape);
}

template <typename T>
void run_gemm_reference(
    const std::vector<T>& h_a,
    const std::vector<T>& h_b,
    std::vector<float>* h_c,
    const GemmShape& shape
) {
    std::vector<T> tmp;
    gemm_reference(h_a, h_b, &tmp, shape.m, shape.n, shape.k);
    h_c->resize(tmp.size());
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        (*h_c)[i] = to_float(tmp[i]);
    }
}

template float launch_gemm<float>(GemmKernelKind, const float*, const float*, float*, const GemmShape&);
template float launch_gemm<half>(GemmKernelKind, const half*, const half*, float*, const GemmShape&);

template void run_gemm_reference<float>(
    const std::vector<float>&,
    const std::vector<float>&,
    std::vector<float>*,
    const GemmShape&
);
template void run_gemm_reference<half>(
    const std::vector<half>&,
    const std::vector<half>&,
    std::vector<float>*,
    const GemmShape&
);
