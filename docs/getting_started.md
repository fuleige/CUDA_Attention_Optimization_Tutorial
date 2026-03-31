# Getting Started

这份文档面向“刚接触 CUDA”的读者。你不需要先完全理解所有 kernel 细节，先把环境、命令和观察方式跑通更重要。

## 1. 这是什么

这个仓库是一套 CUDA C++ 示例，主题是“attention 和矩阵乘法怎么一步一步优化”。

你会看到两类内容：

- 教学版实现
  - 目标是容易理解
  - 通常会保留比较直接的循环和内存访问方式
- 优化版实现
  - 目标是让 GPU 跑得更快
  - 会逐渐引入 tile、shared memory、寄存器分块、Tensor Core、online softmax 等方法

## 2. 你至少需要知道的 CUDA 概念

### Host 和 Device

- Host: CPU 侧代码
- Device: GPU 侧代码

一般流程是：

1. 在 CPU 上准备数据
2. 把数据拷贝到 GPU
3. 启动 kernel
4. 把结果拷回 CPU
5. 和 reference 结果比较

### Kernel

kernel 就是“在 GPU 上并行执行的函数”。在本工程里，像 GEMM 和 attention 的核心计算都写成了 kernel。

### Thread / Block / Grid

- thread: 最小执行单元
- block: 一组 thread
- grid: 一组 block

你可以先用一个简单心智模型：

- 一个 thread 负责一小块计算
- 一个 block 协作处理一个 tile
- 整个 grid 覆盖完整输出矩阵或完整 attention 输出

### Global Memory / Shared Memory / Register

- global memory: 容量大，但慢
- shared memory: block 内共享，速度快很多，但容量小
- register: 每个线程私有，最快，但非常有限

大部分优化的本质都是：

- 减少 global memory 访问次数
- 提高访存连续性
- 让数据在 shared memory / register 里复用

## 3. 仓库结构怎么读

- [src/common](/root/codes/deploy_server/src/common)
  - 公共工具、reference、计时器、CLI
- [src/gemm](/root/codes/deploy_server/src/gemm)
  - 矩阵乘法的不同版本
- [src/attention](/root/codes/deploy_server/src/attention)
  - 基础 attention、FlashAttention、PagedAttention 和其他变体
- [tests](/root/codes/deploy_server/tests)
  - 正确性测试
- [scripts](/root/codes/deploy_server/scripts)
  - benchmark 脚本
- [docs](/root/codes/deploy_server/docs)
  - 文档

## 4. 第一次运行应该做什么

### 构建

```bash
make build
```

### 跑测试

```bash
make test
```

如果这一步通过，说明：

- 编译器可用
- GPU 可访问
- 主要 kernel 至少能在默认 case 上正确运行

### 跑 benchmark

```bash
make bench
```

输出会写到：

- `.build/bench/gemm_bench.csv`
- `.build/bench/attention_fwd_bench.csv`
- `.build/bench/attention_bwd_bench.csv`

## 5. 第一次建议跑的单个命令

先从最简单的 GEMM 开始：

```bash
./bin/gemm_runner --kernel naive --dtype fp32 --m 256 --n 256 --k 256 --check true
```

然后看 shared memory 版：

```bash
./bin/gemm_runner --kernel shared --dtype fp32 --m 256 --n 256 --k 256 --check true
```

再对比 Tensor Core：

```bash
./bin/gemm_runner --kernel wmma --dtype fp16 --m 1024 --n 1024 --k 1024 --check true
```

attention 建议顺序：

```bash
./bin/attention_runner --kernel basic_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --check true
```

```bash
./bin/attention_runner --kernel flash_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true --check true
```

```bash
./bin/attention_runner --kernel paged_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 1 --seq-kv 256 --head-dim 64 --page-size 16 --check true
```

## 6. 输出怎么看

### correctness 输出

常见字段：

- `pass=true`: 本次输出和 reference 对比通过
- `max_abs_err`: 最大绝对误差
- `max_rel_err`: 最大相对误差

如果是 `fp16`，相对误差会比 `fp32` 大一些，这是正常现象。

### benchmark 输出

- `avg_ms`: 平均耗时，越小越好
- `tflops_est`: 估算吞吐，越大越好
- `bandwidth_est`: 有效带宽估算，越大通常越好

不要只看一个数字。一个 kernel 可能 `TFLOPS` 不高，但访存更稳，或者只适合某些尺寸。

## 7. 初学者常见误区

### 误区 1：能跑就说明写得好

不是。GPU kernel “能跑”只说明没有明显错误，不说明访存高效、不说明 occupancy 合理，也不说明数值稳定。

### 误区 2：线程越多越快

不是。block 配置要和：

- shared memory 占用
- 寄存器占用
- 访存模式
- warp 调度

一起看。

### 误区 3：只看 kernel 时间，不看 correctness

这是最危险的。优化里最常见的问题之一，就是“跑得很快，但结果不对”。

### 误区 4：FlashAttention 只是一个更快的 softmax

不是。FlashAttention 的核心是 IO-aware 设计，它在减少中间张量落地和全局内存访问上收益很大。

## 8. 建议学习顺序

1. 先理解 naive GEMM
2. 理解为什么 shared memory 能加速 GEMM
3. 再看寄存器分块和双缓冲
4. 看 WMMA/Tensor Core 版本
5. 再回到 basic attention
6. 理解 softmax 的数值稳定性
7. 再看 FlashAttention 的 online softmax
8. 最后看 PagedAttention、GQA、sliding window、block sparse

## 9. 如果你准备自己改代码

建议每次只改一件事：

1. 先改 launch 参数
2. 再改 tile 大小
3. 再改 memory layout
4. 每改一次都跑 `make test`
5. 再跑一个固定 benchmark case

不要同时改 4 件事，否则你很难知道性能变化和错误来自哪里。

