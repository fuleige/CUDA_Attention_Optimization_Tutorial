# Example Commands

先记住两个约束：

- attention forward 示例当前要求 `--head-dim <= 256`
- `gqa_fwd` 要求 `--heads` 能被 `--kv-heads` 整除

## GEMM

```bash
./bin/gemm_runner --kernel naive --dtype fp32 --m 256 --n 256 --k 256 --check true
```

```bash
./bin/gemm_runner --kernel shared --dtype fp32 --m 512 --n 512 --k 512 --check true
```

```bash
./bin/gemm_runner --kernel register_blocked --dtype fp32 --m 1024 --n 1024 --k 1024 --check true
```

```bash
./bin/gemm_runner --kernel vectorized --dtype fp32 --m 1024 --n 1024 --k 1024 --check true
```

```bash
./bin/gemm_runner --kernel double_buffered --dtype fp16 --m 1024 --n 1024 --k 1024 --check true
```

```bash
./bin/gemm_runner --kernel async_pipeline --dtype fp16 --m 1024 --n 1024 --k 1024 --check true
```

```bash
./bin/gemm_runner --kernel wmma --dtype fp16 --m 1024 --n 1024 --k 1024 --check true
```

## Attention Forward

```bash
./bin/attention_runner --kernel basic_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --check true
```

```bash
./bin/attention_runner --kernel flash_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --check true
```

```bash
./bin/attention_runner --kernel gqa_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 2 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --check true
```

```bash
./bin/attention_runner --kernel sliding_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --window 32 --check true
```

```bash
./bin/attention_runner --kernel block_sparse_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --block-size 16 --check true
```

```bash
./bin/attention_runner --kernel paged_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 1 --seq-kv 256 --head-dim 64 --page-size 16 --check true
```

```bash
./bin/attention_runner --kernel paged_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --window 32 --block-sparse true --block-size 16 --page-size 16 --check true
```

## Attention Backward

```bash
./bin/attention_runner --kernel basic_bwd --dtype fp32 --batch 1 --heads 4 --kv-heads 4 --seq-q 64 --seq-kv 64 --head-dim 32 --causal true --check true
```
