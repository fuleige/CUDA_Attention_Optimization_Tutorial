# Code Architecture Guide

这份文档回答两个问题：

1. 代码应该从哪里开始看
2. 每个文件大概承担什么职责

## 1. 推荐阅读顺序

如果你是初学者，建议按下面顺序看代码：

1. [util.h](/root/codes/deploy_server/src/common/util.h)
2. [reference.h](/root/codes/deploy_server/src/common/reference.h)
3. [reference.cu](/root/codes/deploy_server/src/common/reference.cu)
4. [gemm_kernels.h](/root/codes/deploy_server/src/gemm/gemm_kernels.h)
5. [gemm_kernels.cu](/root/codes/deploy_server/src/gemm/gemm_kernels.cu)
6. [gemm_runner.cu](/root/codes/deploy_server/src/gemm/gemm_runner.cu)
7. [attention_kernels.h](/root/codes/deploy_server/src/attention/attention_kernels.h)
8. [attention_kernels.cu](/root/codes/deploy_server/src/attention/attention_kernels.cu)
9. [attention_runner.cu](/root/codes/deploy_server/src/attention/attention_runner.cu)

## 2. 公共层

### [cuda_check.h](/root/codes/deploy_server/src/common/cuda_check.h)

作用：

- 统一检查 CUDA API 错误
- 统一检查 kernel launch 错误

为什么重要：

- CUDA 的错误常常是“异步暴露”的
- 如果不及时检查，错误会在后面某一步才炸开，很难定位

### [timer.h](/root/codes/deploy_server/src/common/timer.h)

作用：

- 使用 CUDA event 计时

为什么不用普通 CPU 计时：

- kernel 是异步执行的
- 不同步就计时，经常会得到错误结论

### [cli.h](/root/codes/deploy_server/src/common/cli.h)

作用：

- 解析命令行参数
- 打印 GPU 设备摘要

### [util.h](/root/codes/deploy_server/src/common/util.h)

作用：

- 数据类型转换
- 随机初始化
- 向量比较
- CSV 输出

### [reference.h](/root/codes/deploy_server/src/common/reference.h) 和 [reference.cu](/root/codes/deploy_server/src/common/reference.cu)

作用：

- 提供 CPU reference
- 为 correctness test 提供基准结果

你应该把 reference 看成“标准答案生成器”。GPU kernel 改再多，最终都要回到这里做对比。

## 3. GEMM 层

### [gemm_kernels.h](/root/codes/deploy_server/src/gemm/gemm_kernels.h)

作用：

- 定义 GEMM kernel 类型
- 定义 shape
- 暴露统一 launch 接口

### [gemm_kernels.cu](/root/codes/deploy_server/src/gemm/gemm_kernels.cu)

这里是 GEMM 的主体，按“难度递增”组织。

#### naive

特点：

- 一个 thread 算一个输出元素
- 每次乘加都从 global memory 取数据
- 最容易理解
- 也是最好的性能下界参考

#### coalesced

重点：

- 尝试让线程的访存更连续
- 仍然没有真正的数据复用

#### shared

重点：

- 把 A/B 的 tile 搬进 shared memory
- 让同一个 block 内的线程复用输入数据

这是 CUDA 优化里最经典的一步。

#### register_blocked

重点：

- 每个线程不只算一个输出
- 用寄存器缓存更多中间结果

好处：

- 提高 arithmetic intensity
- 降低 shared memory / global memory 压力

#### vectorized

重点：

- 加强访存和计算展开

#### double_buffered

重点：

- 当前 tile 在计算时，下一 tile 的数据准备进入另一个 buffer
- 这是典型的 pipeline 思路

#### async_pipeline

当前实现含义：

- 保留了和真正 `cp.async` 路线一致的结构化调度思路
- 便于继续向更完整的 Ampere async copy 写法演进
- 当前代码仍是教学骨架，不是完整 `cp.async` 实现

#### wmma

重点：

- 使用 Tensor Core
- 输入是 `fp16`
- 累加用 `float`

初学者要注意：

- Tensor Core 很快，但条件多
- shape、内存布局、数据类型、fragment 使用方式都有约束

### [gemm_runner.cu](/root/codes/deploy_server/src/gemm/gemm_runner.cu)

作用：

- 解析命令行
- 分配显存
- 调用 kernel
- 做 warmup、计时、correctness 校验

这个文件很适合用来理解“一个 CUDA 示例从输入到输出”的完整流程。

## 4. Attention 层

### [attention_kernels.h](/root/codes/deploy_server/src/attention/attention_kernels.h)

作用：

- 定义 attention kernel 类型
- 暴露 forward/backward launch 接口

### [attention_kernels.cu](/root/codes/deploy_server/src/attention/attention_kernels.cu)

这里放 attention 相关 kernel。

#### basic_fwd

计算路径：

1. 计算 `QK^T`
2. 做 scale
3. 应用 mask
4. 做 softmax
5. 乘以 `V`

学习重点：

- 为什么 softmax 需要数值稳定处理
- causal mask 和非 causal mask 的区别
- attention 的 global memory 访问有多重

#### flash_fwd

学习重点：

- online softmax
- tile 化处理
- 为什么不显式写出完整的 attention score 矩阵

这是 attention 优化里最重要的一章。

这里要明确：

- 当前仓库的 `flash_fwd` 是 FlashAttention-style 教学实现
- 它保留了 online softmax 和分 tile 思路
- 但没有实现生产级 FlashAttention 那种更完整的 warp/block 并行分工

#### paged_fwd

学习重点：

- 为什么大模型推理要把 KV cache 分页
- 逻辑顺序和物理存储顺序不一致时，如何通过 page table 找回数据

当前实现还额外演示了：

- mask 仍然作用在逻辑 token 顺序上
- page table 只改变物理存储位置，不改变注意力的逻辑可见性规则

#### gqa_fwd

学习重点：

- `num_heads` 和 `num_kv_heads` 不同
- 多个 query heads 共享更少的 KV heads

#### sliding_fwd

学习重点：

- 只看局部窗口
- 计算量随可见范围下降

#### block_sparse_fwd

学习重点：

- 不是每个 query 都看所有 key
- 用块级稀疏规则减少计算

#### basic_bwd / flash_bwd

学习重点：

- backward 怎么从 `grad_out` 反推 `dQ/dK/dV`
- softmax backward 的结构是什么

当前实现更偏教学性，优先保证：

- 结构清晰
- 正确性可验证

而不是极致性能。

特别是：

- `flash_bwd` 当前是 FlashAttention-style 教学 backward
- 它不是生产级 flash backward kernel

### [attention_runner.cu](/root/codes/deploy_server/src/attention/attention_runner.cu)

作用和 `gemm_runner.cu` 类似，只是输入维度更复杂。

建议你重点看：

- shape 参数怎么传进去
- 不同 attention 变体怎么共用 runner
- paged attention 的 page table 怎么构造

## 5. 测试层

### [test_gemm.cu](/root/codes/deploy_server/tests/test_gemm.cu)

作用：

- 批量覆盖 GEMM 版本
- 对比 CPU reference

### [test_attention.cu](/root/codes/deploy_server/tests/test_attention.cu)

作用：

- 覆盖 forward/backward
- 覆盖 dense、flash、paged、GQA、window、block sparse

## 6. 你在看代码时应该追什么线

建议一开始不要逐行啃所有 kernel，而是追下面这些问题：

### 线索 1：输入怎么进来

- CLI 参数
- host vector
- `cudaMalloc`
- `cudaMemcpy`

### 线索 2：kernel 怎么启动

- grid
- block
- shape 到索引的映射

### 线索 3：每个线程负责什么

- 一个元素
- 一个 tile
- 多个输出

### 线索 4：数据复用发生在哪里

- global memory
- shared memory
- register

### 线索 5：结果怎么验证

- reference
- `compare_vectors`
- `pass=true/false`
