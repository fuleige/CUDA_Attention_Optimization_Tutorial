# Benchmarking Notes

这份文档面向第一次接触 CUDA benchmark 的读者，重点不是“怎么得到一个数字”，而是“怎么得到一个可信的数字”。

## 1. 统一入口

- `make bench`
- 输出文件：
  - `.build/bench/gemm_bench.csv`
  - `.build/bench/attention_fwd_bench.csv`
  - `.build/bench/attention_bwd_bench.csv`

## 2. 为什么 benchmark 不能只看一次运行

第一次运行通常会混入很多额外成本，例如：

- 上下文初始化
- 数据首次分配
- cache 尚未热起来

所以 runner 都提供了：

- `--warmup`
- `--iters`

推荐做法是：

- 先 warmup
- 再统计多次平均值

## 3. 推荐观察指标

- `avg_ms`: 平均延迟
- `tflops_est`: 理论估算吞吐
- `bandwidth_est`: 有效带宽估算
- `max_abs_err` / `max_rel_err`: 正确性偏差

### 怎么理解这些指标

#### `avg_ms`

最直观，越小越好。

#### `tflops_est`

反映单位时间完成了多少浮点计算，但它是估算值，不代表真实硬件利用率已经达到极限。

#### `bandwidth_est`

更适合观察 memory-bound kernel。

#### `max_abs_err` / `max_rel_err`

这两个字段是在提醒你：

- 这个 kernel 的速度是否建立在“结果仍然可信”的前提上

## 4. benchmark 时该怎么比较

正确比较方法：

- 同样的 shape
- 同样的数据类型
- 同样的 warmup
- 同样的 iteration 次数

错误比较方法：

- 一个 kernel 跑 `fp16`
- 另一个 kernel 跑 `fp32`
- 然后直接说谁更快

## 5. 建议 profiling 命令

```bash
nsys profile --stats=true ./bin/gemm_runner --kernel shared --dtype fp32 --m 1024 --n 1024 --k 1024
```

```bash
ncu --set full ./bin/attention_runner --kernel flash_fwd --dtype fp32 --batch 1 --heads 8 --kv-heads 8 --seq-q 128 --seq-kv 128 --head-dim 64 --causal true
```

## 6. 建议重点看

- occupancy
- warp stall 原因
- global memory throughput
- shared memory throughput
- Tensor Core 利用率（`wmma` 路径）

### 这些指标分别想说明什么

#### occupancy

说明 GPU 上能并发驻留多少线程块/warp，但它不是唯一目标。

#### warp stall

说明 warp 为什么在等，例如：

- 等内存
- 等依赖
- 等同步

#### global memory throughput

说明全局内存通道是否已经成为主要瓶颈。

#### shared memory throughput

说明 shared memory 是否被高强度使用，也能帮助你判断 shared memory 设计是否有效。

#### Tensor Core utilization

主要看 `wmma` 路径是否真正吃到了 Tensor Core 红利。

## 7. 对初学者的 benchmark 建议

建议你先只做三类对比：

1. `naive` vs `shared`
2. `basic_fwd` vs `flash_fwd`
3. `basic_fwd` vs `sliding_fwd` / `block_sparse_fwd`

这三类最容易建立“为什么会变快”的直觉。

做 `flash_fwd` 对比时要带着一个正确预期：

- 这里的 `flash_fwd` 是教学实现
- 你看到的是“算法思路上的收益趋势”
- 不是生产级 FlashAttention 的最终性能数字

## 8. 结果应该怎么记

建议至少记录：

- 命令
- 日期
- GPU 型号
- CUDA 版本
- 平均时间
- 是否通过 correctness

如果你没有把实验条件记下来，后面很难复现实验。
