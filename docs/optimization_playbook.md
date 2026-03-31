# Optimization Playbook

这份文档不是只列“可以做什么优化”，而是解释“为什么这样优化”和“什么时候应该这样做”。如果你对 CUDA 不熟，建议把这份文档当成分析框架，而不是口诀表。

## 1. 先建立一个总原则

大多数 CUDA 优化都在回答下面四个问题：

1. 你的计算是算力瓶颈，还是访存瓶颈
2. 你是否重复从 global memory 读取了相同数据
3. 线程之间的工作划分是否均衡
4. 你为了更高吞吐付出了多少复杂度和可维护性代价

如果你不先回答这四个问题，就很容易进入“盲调 block size”的状态。

## 2. 优化前先做什么

### 先有 reference

没有 reference，不要开始优化。

理由：

- 你需要知道结果是不是对
- 你需要知道优化有没有引入数值问题

### 先有 baseline

没有 baseline，不要讨论“加速”。

例如：

- naive GEMM 是 baseline
- basic attention 是 baseline

### 先固定 benchmark case

至少固定：

- 数据类型
- shape
- warmup 次数
- 计时 iteration 次数

否则你每次测出来的数字不可比较。

## 3. GEMM 优化策略

### 策略 1：先从 naive 找问题

naive GEMM 的典型问题：

- 每次乘加都访问 global memory
- 同一个 A/B 元素会被反复读取
- 算术强度低

判断依据：

- 时间长
- 带宽利用不高
- 随矩阵变大扩展性差

### 策略 2：用 shared memory 做 tile 复用

适用场景：

- 同一个 block 内会反复使用同一小块 A/B 数据

原理：

- 一次从 global memory 读入 tile
- block 内线程重复使用

收益：

- 减少 global memory 读取
- 提高数据复用

风险：

- shared memory 不够用
- bank conflict
- tile 太大导致 occupancy 下降

### 策略 3：让单线程做更多工作

也就是 register blocking。

原理：

- 一个线程不只算一个输出
- 在寄存器里保留多个累加器

收益：

- 更高的计算密度
- 更好的数据复用

风险：

- 寄存器压力升高
- occupancy 下降

### 策略 4：向量化和展开

原理：

- 让内存访问更连续
- 减少循环控制开销

收益：

- 通常能进一步榨干访存吞吐

风险：

- 对齐要求更严格
- 代码更不直观

### 策略 5：双缓冲

原理：

- 当前 tile 计算时，同时准备下一 tile

收益：

- 减少“等数据”的时间
- 更容易形成流水线

### 策略 6：Tensor Core

适用场景：

- 数据类型和矩阵形状适合 Tensor Core

收益：

- 常常是数量级级别的吞吐提升

限制：

- 不是任意 shape 都适合
- 数据布局要求更严格
- fragment API 学习成本更高

## 4. Attention 优化策略

### 策略 1：先理解 basic attention 慢在哪里

basic attention 的问题通常不是“公式复杂”，而是“中间数据太大”。

核心代价：

- `QK^T` 产生很大的中间分数矩阵
- softmax 还要再次访问这些数据
- 后面乘 `V` 又要再读一遍

所以瓶颈往往是：

- global memory 流量太大
- 中间张量写回和重读代价高

### 策略 2：先处理数值稳定性

softmax 优化之前，先保证它是稳定的。

常见做法：

- 先减去最大值
- 再做 `exp`
- 再归一化

如果不做这一步：

- 容易溢出
- backward 也会变得不可靠

### 策略 3：FlashAttention 的核心不是“算得更巧”，而是“写回更少”

FlashAttention 重点是 IO-aware。

它做的关键事情：

- 分 tile 处理 K/V
- 不物化完整 attention score 矩阵
- 用 online softmax 在 tile 间保持数值正确

收益：

- 少写中间结果
- 少读中间结果
- 更适合长序列

结合本仓库要注意：

- 这里的 `flash_fwd` 是教学版 FlashAttention-style kernel
- 它的价值在于帮助你理解 online softmax 和 tile 化思路
- 不应把它当作生产级 FlashAttention 的性能上界

### 策略 4：把注意力模式本身变稀疏或局部化

如果业务允许，不一定非要算 full attention。

可选路线：

- sliding window
- block sparse
- GQA / MQA

收益：

- 直接减少计算量
- 直接减少显存压力

代价：

- 模型行为会变化
- 不是所有场景都能接受

### 策略 5：PagedAttention 优化的是“推理态 KV cache 管理”

它关注的问题和 FlashAttention 不完全一样。

FlashAttention 更关注：

- prefill
- 大量 attention 计算
- 降低中间张量 IO

PagedAttention 更关注：

- decode
- KV cache 动态增长
- 分页存储和访问映射

在本仓库里，`paged_fwd` 还演示了一点：

- page table 解决的是“数据放在哪里”
- causal / window / block-sparse 解决的是“逻辑上能看哪些 token”
- 这两类问题可以叠加，但不要混为一谈

### 策略 6：backward 优化通常比 forward 更难

因为 backward 往往需要：

- 更多中间量
- 更复杂的数据依赖
- 更多梯度累加

初学阶段建议：

1. 先写对
2. 再验证
3. 最后再优化 backward

## 5. 怎么判断该优化哪里

### 如果 GPU 很忙，但速度还是不快

看：

- Tensor Core 有没有用起来
- 指令吞吐是否受限
- 是否计算本身就太多

### 如果 GPU 看起来不忙

看：

- occupancy 是否太低
- block 配置是否不合理
- 是否大量 stall 在 memory dependency

### 如果 global memory throughput 很高，但性能仍差

看：

- 是否纯粹是带宽瓶颈
- 是否缺少数据复用
- 是否应该引入 shared memory / tiling

### 如果 shared memory 吞吐很高，但收益不明显

看：

- 是否 bank conflict
- 是否 tile 太大
- 是否 shared memory 用了很多，但没有真正提高复用

## 6. 初学者的实战优化流程

建议严格按这个顺序：

1. 先写 reference
2. 写 naive kernel
3. 跑 correctness
4. 跑 benchmark，记录 baseline
5. 只改一处优化
6. 再跑 correctness
7. 再跑 benchmark
8. 记录变化
9. 如果变快了，再继续下一步
10. 如果变慢了，撤回并分析原因

## 7. 你在 benchmark 中应该记录什么

至少记录：

- kernel 名称
- 数据类型
- shape
- 平均耗时
- correctness 是否通过
- 最大绝对误差
- 最大相对误差

更进一步还可以记录：

- occupancy
- dram throughput
- sm throughput
- Tensor Core utilization

## 8. 常见优化误区

### 误区 1：shared memory 一定更快

不一定。

如果：

- tile 很小
- 数据复用不高
- shared memory 带来复杂同步

那它不一定值回票价。

### 误区 2：Occupancy 越高越好

不一定。

高 occupancy 只是说明“能驻留的线程多”，不代表单线程工作有效，也不代表内存系统高效。

### 误区 3：用了 Tensor Core 就一定是最优

不一定。

如果：

- shape 不合适
- layout 不合适
- 数据转换代价高

那整体收益可能不理想。

### 误区 4：FlashAttention 就是所有 attention 的终点

不一定。

对推理 decode 来说，PagedAttention 和 KV cache 管理可能更关键。

### 误区 5：只盯着单个 kernel

真实系统中，瓶颈可能在：

- kernel 之间的数据搬运
- launch 开销
- memory allocation
- 不合理的数据布局

## 9. 结合本仓库的建议练习

### 练习 1

比较：

- `naive`
- `shared`
- `register_blocked`

观察相同 shape 下性能变化。

### 练习 2

比较：

- `basic_fwd`
- `flash_fwd`

逐步增加 `seq_len`，观察差距是否扩大。

### 练习 3

比较：

- `basic_fwd`
- `sliding_fwd`
- `block_sparse_fwd`

理解“改变注意力模式本身”对性能的影响。

### 练习 4

比较：

- `basic_fwd`
- `paged_fwd`

重点理解它们解决的是不同问题，而不是单纯谁更快。
