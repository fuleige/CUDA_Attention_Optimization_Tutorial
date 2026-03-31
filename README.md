# CUDA Attention Optimization Tutorial

这是一个面向初学者的 CUDA 学习型工程，目标不是只展示几个能跑的 kernel，而是把下面几件事串成一条完整学习路径：

- CUDA 程序怎么编译、运行、测试
- 一个最基础的矩阵乘法和 attention 是怎么写出来的
- 为什么基础实现会慢
- shared memory、寄存器分块、双缓冲、Tensor Core、online softmax、PagedAttention 分别解决什么问题
- 性能测试应该怎么看，正确性又该怎么验证

如果你对 CUDA 还不熟，建议按这个顺序阅读：

1. [overview.md](/root/codes/deploy_server/docs/overview.md)
2. [getting_started.md](/root/codes/deploy_server/docs/getting_started.md)
3. [architecture.md](/root/codes/deploy_server/docs/architecture.md)
4. [examples.md](/root/codes/deploy_server/docs/examples.md)
5. [optimization_playbook.md](/root/codes/deploy_server/docs/optimization_playbook.md)
6. [benchmarking.md](/root/codes/deploy_server/docs/benchmarking.md)
7. [cuda_guidelines.md](/root/codes/deploy_server/docs/cuda_guidelines.md)
8. [troubleshooting.md](/root/codes/deploy_server/docs/troubleshooting.md)

常用命令：

```bash
make build
make test
make bench
```

默认编译器是 `/usr/local/cuda/bin/nvcc`，默认目标架构是 `sm_89`。
