# CUDA Attention Optimization Examples

这是一个面向学习的 CUDA C++ 工程，主题是“从基础实现出发，逐步理解 attention 和 GEMM 的优化路径”。

它不是一个只展示最终高性能 kernel 的仓库，而是一套可编译、可执行、可测试、可 benchmark 的教程型代码库。

## 这套工程覆盖什么

- GEMM 从 `naive` 到 `WMMA/Tensor Core`
- Basic Attention
- FlashAttention
- PagedAttention
- GQA / MQA 风格的 KV 共享
- Sliding Window Attention
- Block Sparse Attention
- Forward 和部分 Backward 示例
- correctness tests 和 benchmark

## 当前实现边界

- attention forward 示例当前支持 `head_dim <= 256`
- `gqa_fwd` 要求 `num_heads` 能被 `num_kv_heads` 整除
- `flash_fwd` / `flash_bwd` 是教学版 FlashAttention-style 实现
- `async_pipeline` 是教学版双缓冲骨架，不是完整 `cp.async` 版本

默认编译器：

- `/usr/local/cuda/bin/nvcc`

默认目标架构：

- `sm_89`

## 如果你是 CUDA 初学者，先看这些

1. [getting_started.md](/root/codes/deploy_server/docs/getting_started.md)
2. [architecture.md](/root/codes/deploy_server/docs/architecture.md)
3. [examples.md](/root/codes/deploy_server/docs/examples.md)
4. [optimization_playbook.md](/root/codes/deploy_server/docs/optimization_playbook.md)
5. [benchmarking.md](/root/codes/deploy_server/docs/benchmarking.md)
6. [cuda_guidelines.md](/root/codes/deploy_server/docs/cuda_guidelines.md)
7. [troubleshooting.md](/root/codes/deploy_server/docs/troubleshooting.md)

## 最常用命令

### 构建

```bash
make build
```

### 运行正确性测试

```bash
make test
```

### 运行 benchmark

```bash
make bench
```

## 可执行文件

- `bin/gemm_runner`
- `bin/attention_runner`
- `bin/test_gemm`
- `bin/test_attention`

每个 runner 都支持 `--help`。

## 你会在这里学到什么

### GEMM 部分

你会看到一个矩阵乘法如何逐步变快：

- 为什么 naive 版本慢
- 为什么 shared memory 可以复用数据
- 为什么寄存器分块能提升单线程工作量
- 为什么双缓冲可以形成流水线
- 为什么 Tensor Core 需要特定数据类型和布局

### Attention 部分

你会看到几类不同的问题：

- Basic Attention 如何实现
- softmax 为什么要考虑数值稳定性
- FlashAttention 为什么能减少 IO
- PagedAttention 为什么更偏推理态 KV cache 管理
- GQA、sliding window、block sparse 为什么会改变计算图和性能特征

这里要特别注意：

- 仓库中的 `flash_*` 重点在讲解 online softmax 和 tile 化思路
- 它不是对生产级 FlashAttention kernel 的逐指令复刻
- `paged_fwd` 重点在 page table 和逻辑/物理映射，也支持 mask 作用在逻辑 token 顺序上

## 建议的学习路线

1. 先看 `GEMM`
2. 再看 `Basic Attention`
3. 再看 `FlashAttention`
4. 再看 `PagedAttention`
5. 最后回头对照 benchmark 和 profiling

## 快速上手示例

```bash
./bin/gemm_runner --kernel naive --dtype fp32 --m 256 --n 256 --k 256 --check true
```

```bash
./bin/gemm_runner --kernel wmma --dtype fp16 --m 1024 --n 1024 --k 1024 --check true
```

```bash
./bin/attention_runner --kernel basic_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --check true
```

```bash
./bin/attention_runner --kernel flash_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --check true
```

```bash
./bin/attention_runner --kernel paged_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 1 --seq-kv 256 --head-dim 64 --page-size 16 --check true
```

更多命令见 [examples.md](/root/codes/deploy_server/docs/examples.md)。
