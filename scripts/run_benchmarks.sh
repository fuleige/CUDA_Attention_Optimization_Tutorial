#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/.build/bench"
mkdir -p "${OUT_DIR}"

GEMM_OUT="${OUT_DIR}/gemm_bench.csv"
ATTN_FWD_OUT="${OUT_DIR}/attention_fwd_bench.csv"
ATTN_BWD_OUT="${OUT_DIR}/attention_bwd_bench.csv"

# --csv-header on the first invocation prints the column names so that
# the CSV stays in sync with the code (no hand-maintained header line).

"${ROOT_DIR}/bin/gemm_runner" --kernel naive --dtype fp32 --m 256 --n 256 --k 256 --warmup 5 --iters 20 --check true --csv true --csv-header true > "${GEMM_OUT}"
"${ROOT_DIR}/bin/gemm_runner" --kernel shared --dtype fp32 --m 512 --n 512 --k 512 --warmup 5 --iters 30 --check true --csv true >> "${GEMM_OUT}"
"${ROOT_DIR}/bin/gemm_runner" --kernel register_blocked --dtype fp32 --m 1024 --n 1024 --k 1024 --warmup 5 --iters 30 --check true --csv true >> "${GEMM_OUT}"
"${ROOT_DIR}/bin/gemm_runner" --kernel async_pipeline --dtype fp16 --m 1024 --n 1024 --k 1024 --warmup 5 --iters 30 --check true --csv true >> "${GEMM_OUT}"
"${ROOT_DIR}/bin/gemm_runner" --kernel wmma --dtype fp16 --m 1024 --n 1024 --k 1024 --warmup 5 --iters 30 --check true --csv true >> "${GEMM_OUT}"

"${ROOT_DIR}/bin/attention_runner" --kernel basic_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --warmup 3 --iters 10 --check true --csv true --csv-header true > "${ATTN_FWD_OUT}"
"${ROOT_DIR}/bin/attention_runner" --kernel flash_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --warmup 3 --iters 10 --check true --csv true >> "${ATTN_FWD_OUT}"
"${ROOT_DIR}/bin/attention_runner" --kernel gqa_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 2 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --warmup 3 --iters 10 --check true --csv true >> "${ATTN_FWD_OUT}"
"${ROOT_DIR}/bin/attention_runner" --kernel sliding_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --window 32 --warmup 3 --iters 10 --check true --csv true >> "${ATTN_FWD_OUT}"
"${ROOT_DIR}/bin/attention_runner" --kernel block_sparse_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --block-size 16 --warmup 3 --iters 10 --check true --csv true >> "${ATTN_FWD_OUT}"
"${ROOT_DIR}/bin/attention_runner" --kernel paged_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 1 --seq-kv 256 --head-dim 64 --page-size 16 --warmup 3 --iters 10 --check true --csv true >> "${ATTN_FWD_OUT}"
"${ROOT_DIR}/bin/attention_runner" --kernel basic_bwd --dtype fp32 --batch 1 --heads 4 --kv-heads 4 --seq-q 64 --seq-kv 64 --head-dim 32 --causal true --warmup 2 --iters 5 --check true --csv true --csv-header true > "${ATTN_BWD_OUT}"

echo "Wrote:"
echo "  ${GEMM_OUT}"
echo "  ${ATTN_FWD_OUT}"
echo "  ${ATTN_BWD_OUT}"
