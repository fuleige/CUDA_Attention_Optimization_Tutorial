# CUDA Attention Optimization Tutorial

这是一个面向初学者的 CUDA 学习型工程，目标不是只展示几个能跑的 kernel，而是把下面几件事串成一条完整学习路径：

- CUDA 程序怎么编译、运行、测试
- 一个最基础的矩阵乘法和 attention 是怎么写出来的
- 为什么基础实现会慢
- shared memory、寄存器分块、双缓冲、Tensor Core、online softmax、PagedAttention 分别解决什么问题
- 性能测试应该怎么看，正确性又该怎么验证

如果你对 CUDA 还不熟，建议按这个顺序阅读：

1. [overview.md](docs/overview.md)
2. [getting_started.md](docs/getting_started.md)
3. [architecture.md](docs/architecture.md)
4. [examples.md](docs/examples.md)
5. [optimization_playbook.md](docs/optimization_playbook.md)
6. [benchmarking.md](docs/benchmarking.md)
7. [cuda_guidelines.md](docs/cuda_guidelines.md)
8. [troubleshooting.md](docs/troubleshooting.md)

常用命令：

```bash
make build
make test
make bench
```

构建默认值：

- `Makefile` 会优先使用 `/usr/local/cuda/bin/nvcc`；如果你的 CUDA 安装在别处，可以覆盖 `CUDA_HOME` 或 `NVCC`
- 默认目标架构是 `sm_89`；如果你的 GPU 或 CUDA 版本不同，请显式覆盖 `CUDA_ARCH`

例如：

```bash
make build CUDA_ARCH=sm_80
make build CUDA_HOME=/opt/cuda
```

当前要先知道的几个约束：

- attention forward 示例当前要求 `head_dim <= 256`
- `GQA/MQA` 风格示例要求 `num_heads % num_kv_heads == 0`
- `flash_*` 是 FlashAttention-style 教学实现
- `async_pipeline` 是为后续 `cp.async` 演进保留的教学骨架，不是完整生产实现
