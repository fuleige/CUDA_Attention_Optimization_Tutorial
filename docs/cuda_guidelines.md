# CUDA Coding Guidelines

这份文档回答两个问题：

1. 这个仓库里的 CUDA 代码遵循什么约定
2. 初学者写 CUDA 时，哪些习惯最容易出问题

## 1. 总体原则

这套示例默认遵循以下编码约定：

- 所有 CUDA API 和 kernel launch 都必须经过统一错误检查
- kernel 参数尽量显式传入 shape / stride / options，不依赖隐藏全局状态
- tile 大小、warp 映射、page size 等关键参数集中声明，避免魔法数字散落
- 在 shared memory 路径里优先写清楚：
  - tile 的逻辑含义
  - bank conflict 风险
  - 对齐和向量化假设
- 教学版 kernel 和优化版 kernel 分开命名，避免一个实现同时承担太多概念
- benchmark 输出必须包含 correctness 校验结果，否则性能数字没有解释力

## 2. 为什么这些规范重要

### 错误检查

CUDA 程序有一个容易让初学者困惑的点：

- 很多操作是异步的
- 你在 A 处犯错，可能在 B 处才看到错误

所以统一错误检查几乎不是“建议”，而是“必需”。

### 显式参数

显式传 shape/stride/options 的好处：

- 调试更直接
- 不容易把不同 layout 混在一起
- 更容易复用 runner 和 benchmark

### 不滥用模板和宏

模板和宏能让代码更通用，但也会让代码更难读。

这个仓库的选择是：

- 需要抽象时才抽象
- 优先让学习者看得懂

## 3. 数值稳定性

- softmax 一律采用 `max-subtract` 路径
- FlashAttention 示例使用 online softmax 更新
- backward 示例采用重算 logits / probs，优先保证清晰和正确

### 初学者要特别记住

“结果看起来差不多”不等于数值稳定。

你应该始终问：

- 有没有 overflow / underflow 风险
- `fp16` 和 `fp32` 的误差是否在可接受范围
- backward 是否比 forward 更容易暴露数值问题

## 4. 内存使用习惯

### Global memory

适合：

- 存大张量
- block 之间共享的数据

问题：

- 慢
- 重复读代价高

### Shared memory

适合：

- block 内高复用数据
- tile 化计算

风险：

- bank conflict
- 用量太大导致 occupancy 下降
- 同步不当导致错误

### Register

适合：

- 每个线程自己的热点中间值
- 累加器

风险：

- 寄存器压力太大时，反而会伤害 occupancy

## 5. kernel 设计习惯

推荐你在写 kernel 前先回答：

1. 一个 thread 负责什么
2. 一个 block 负责什么
3. 数据复用发生在哪里
4. 需要几次同步
5. 边界条件怎么处理

如果这五点说不清，代码通常也不会稳定。

## 6. benchmark 习惯

不要这样做：

- 只跑一次
- 不做 warmup
- 不做 correctness 校验
- 随手换 shape 又直接比较数字

应该这样做：

- 固定 case
- 先 warmup
- 多次迭代取平均
- 保存结果
- 对照 reference

## 7. 当前实现边界

- `flash_bwd` 当前是“可运行的教学版 backward”，不是高度优化的生产实现
- `async_pipeline` 保留了双缓冲的调度结构，用于承接后续 `cp.async` 深化
- `wmma` 版本要求矩阵维度是 `16` 的倍数，并使用 `fp16` 输入

## 8. 对初学者最重要的建议

每次只改一件事。

最推荐的工作节奏是：

1. 改一个 kernel
2. `make test`
3. 跑一个固定 benchmark
4. 记录结果
5. 再决定下一步
