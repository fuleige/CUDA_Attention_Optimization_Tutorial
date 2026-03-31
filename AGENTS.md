# Repository Guidelines

## Project Structure & Module Organization

This is a CUDA C++ teaching project covering GEMM and Attention kernel optimizations.

- `src/common/` — shared utilities: CLI parsing, CUDA error checking, timing, data types, CPU reference implementations
- `src/gemm/` — GEMM kernels (naive → coalesced → shared → register-blocked → vectorized → double-buffered → async-pipeline → WMMA/Tensor Core)
- `src/attention/` — Attention kernels (basic, flash, paged, GQA, sliding window, block sparse) with forward and backward
- `tests/` — correctness tests comparing GPU kernels against CPU reference
- `scripts/` — benchmark automation
- `docs/` — learning-oriented documentation
- `bin/` — compiled binaries (gitignored)
- `.build/` — build artifacts and benchmark CSV output (gitignored)

## Build, Test, and Development Commands

- `make build` — compiles `bin/gemm_runner`, `bin/attention_runner`, `bin/test_gemm`, `bin/test_attention`
- `make test` — runs the CUDA correctness suite on the active GPU
- `make bench` — writes CSV benchmark results to `.build/bench/`
- `make debug` — clean rebuild with debug flags (`-O0 -g -G`)
- `make clean` — removes `bin/` and `.build/`

Build configuration overrides:

```bash
make build CUDA_ARCH=sm_80        # target a different GPU architecture
make build CUDA_HOME=/opt/cuda    # use a non-default CUDA installation
```

Default: `CUDA_ARCH=sm_89`, `CUDA_HOME=/usr/local/cuda`.

## Coding Style & Naming Conventions

- 4-space indentation, C++17
- `snake_case` for files, variables, and functions
- `PascalCase` for types and enum values (e.g., `AttentionShape`, `GemmKernelKind::kNaive`)
- All CUDA API calls wrapped in `CUDA_CHECK()` for fail-fast error reporting
- Kernel parameters are explicit (shape, stride, options) — no hidden global state

## Key Constraints

- Attention forward kernels require `head_dim <= 256`
- GQA requires `num_heads % num_kv_heads == 0`
- WMMA requires `m`, `n`, `k` to be multiples of 16 and `dtype=fp16`
- Backward kernel (`basic_bwd`) uses 1 thread per block — correct but slow by design (teaching)
- `flash_fwd` is a teaching FlashAttention-style kernel, not a production implementation
- `async_pipeline` is a structural placeholder for `cp.async` exercises
- Tensor sizes are validated against int32 overflow before kernel launch

## Testing Guidelines

Tests live in `tests/` and compare GPU kernel output against CPU reference implementations in `src/common/reference.cu`. Every kernel variant (including fp16 paths) is covered. Tolerance thresholds are set per-test to account for floating-point precision differences.

## Commit & Pull Request Guidelines

Use clear, imperative commit messages. Keep commits focused.

Pull requests should include:

- a brief summary of the change
- test evidence (`make test` output)
- benchmark comparison if performance-relevant
