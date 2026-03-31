# Troubleshooting

这份文档收集初学者最常见的问题。建议先看报错属于哪一类，再决定要不要改代码。

## 1. `make build` 失败

### 现象

- `nvcc: command not found`
- `nvcc fatal   : Unsupported gpu architecture 'sm_89'`
- 找不到 CUDA 头文件
- 链接阶段失败

### 检查项

1. 确认 `/usr/local/cuda/bin/nvcc --version` 或 `nvcc --version` 至少有一个可执行
2. 确认 [Makefile](../Makefile) 中 `CUDA_HOME` / `NVCC` 配置符合你的机器环境
3. 如果默认路径不对，显式执行 `make build CUDA_HOME=/path/to/cuda` 或 `make build NVCC=/path/to/nvcc`
4. 如果默认架构不适合你的 GPU 或 CUDA Toolkit，显式执行 `make build CUDA_ARCH=sm_80`
5. 确认机器上安装的 CUDA 版本和驱动兼容

### 常见原因

- CUDA Toolkit 没装完整
- 驱动版本太旧
- 本机有多个 CUDA 版本，路径指错
- 默认 `sm_89` 和本机工具链或 GPU 不匹配

## 2. `make test` 失败

### 现象

- 程序直接报 CUDA error
- `pass=false`
- 某些 kernel 失败，另一些成功

### 排查顺序

1. 先看是不是设备不可访问
2. 再看是不是 shape 不满足约束
3. 再看是不是数值误差超过阈值

### 设备不可访问

先运行：

```bash
nvidia-smi
```

如果这一步都失败，问题通常不在仓库代码，而在：

- 驱动
- 容器权限
- GPU 可见性设置

### shape 不满足约束

比如 `wmma` 路径要求：

- `m`
- `n`
- `k`

都是 `16` 的倍数。

如果你随手改了 shape，先确认是不是违反了这类约束。

attention 里常见约束还有：

- forward 示例当前要求 `head_dim <= 256`
- `gqa_fwd` 要求 `num_heads % num_kv_heads == 0`
- `window` 只适用于 `sliding_fwd` 和 `paged_fwd`

### 数值误差超阈值

先判断是不是这几类情况：

- 你把 `fp32` 改成了 `fp16`
- 你改了 softmax 或 backward 路径
- 你改了 tile 或 accumulation 方式

`fp16` 的误差大于 `fp32` 是正常的，但如果突然大很多，就要检查实现。

## 3. `make bench` 能跑，但数字看起来不对

### 情况 1：第一次特别慢

这通常是正常的，常见原因：

- CUDA 上下文初始化
- 第一次分配显存
- cache 未热起来

所以 benchmark 会先 warmup。

### 情况 2：TFLOPS 很低

先不要立刻怀疑代码坏了，先问：

- 这个 shape 是否太小
- 当前 kernel 是否本来就是教学版
- 当前瓶颈是不是内存，不是算力

### 情况 3：两个 kernel 差不多快

可能原因：

- 问题规模太小
- benchmark case 不敏感
- 当前 GPU 对这个 case 的瓶颈不在你优化的点上

## 4. `pass=true`，但 `max_rel_err` 看起来很大

这通常出现在：

- 参考值本身很接近 0
- 相对误差被放大

这种情况下要结合：

- `max_abs_err`
- 数据类型
- 输出量级

一起判断，不要只盯一个 `rel_err`。

## 5. 为什么 `flash_fwd` 没有“快很多”

先确认：

1. 你的 `seq_len` 是否足够大
2. 你比较的是不是同样的 `dtype`
3. 你是不是只看了一个很小的测试 case

FlashAttention 的优势往往在：

- 更长序列
- 更明显的 IO 压力
- 更大的中间张量成本

另外要记住：

- 这个仓库里的 `flash_fwd` 是教学版 FlashAttention-style 实现
- 它更适合帮助你建立算法直觉，而不是拿来代表生产级实现的上限

## 6. 为什么 `paged_fwd` 不一定比 `flash_fwd` 快

因为它们解决的问题不同。

- `flash_fwd` 更偏 prefill / 降低 attention 中间 IO
- `paged_fwd` 更偏 decode / KV cache 的分页组织

不要把它们当成同一维度的替换关系。

## 7. 修改代码后应该先做什么

建议固定流程：

1. `make build`
2. `make test`
3. 跑一个单独 case
4. `make bench`

不要一上来就只跑 benchmark。

## 8. 如何判断是代码错还是 benchmark 设错

如果：

- 小尺寸 case 都不对
- naive 和优化版都不对

更可能是：

- shape 索引逻辑错
- 数据布局理解错
- reference 也被改坏了

如果：

- 只有某个优化版不对
- naive 对，优化版错

更可能是：

- shared memory 索引错
- 边界条件错
- 同步错
- register blocking 写错

## 9. 为什么初学阶段不要同时改太多地方

因为 CUDA bug 往往不容易定位。

一次改很多地方时，你很难判断到底是：

- 线程映射错了
- 访存越界了
- 同步缺失了
- 数值稳定性坏了
- benchmark case 改了

## 10. 如果你要继续深入

建议下一步学习：

1. 用 `nsys` 看整体时间线
2. 用 `ncu` 看单 kernel 指标
3. 对比不同 tile 大小
4. 对比 `fp32` 和 `fp16`
5. 对比短序列和长序列
