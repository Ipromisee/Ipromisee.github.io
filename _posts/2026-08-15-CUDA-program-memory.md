---
title: "CUDA 编程模型与内存优化：从线程组织到数据流动"
date: 2026-08-15 22:30:00 +0800
categories:
  - AI Infra
  - CUDA
tags:
  - cuda
  - gpu
  - memory-optimization
  - warp
  - shared-memory
  - occupancy
description: "从 Grid、Block、Warp 和 Thread 出发，串联 Global Memory、Shared Memory、Register、Tiling、Bank Conflict 与 Occupancy，建立 CUDA 线程组织和数据流动的统一性能分析框架。"
toc: true
math: true
comments: false
---

刚开始学习 CUDA 时，我们通常会把“编程模型”和“内存模型”分成两部分理解：前者介绍 Grid、Block、Thread、Warp，后者介绍 Global Memory、Shared Memory、Register、Local Memory。

但真正开始写 CUDA Kernel 后会发现，这两部分其实很难分开。

CUDA 编程模型解决的是：

> **谁来计算，以及这些计算线程如何组织。**

CUDA 内存模型解决的则是：

> **数据放在哪里，以及这些线程如何访问数据。**

二者真正的联系可以概括成：

```text
Grid
  ↓
Block  ←→ Shared Memory
  ↓
Warp   ←→ Global Memory / Shared Memory Bank / Shuffle
  ↓
Thread ←→ Register / Local Memory
  ↓
SM     ←→ Register File + Shared Memory/L1
```

其中 Thread、Warp、Block 和 SM 几乎都能找到与之对应的内存层次：Thread 私有数据通常放在 Register 中，Block 内线程通过 Shared Memory 协作，而 Global Memory 的访问效率又往往需要从 Warp 的角度分析。

理解了这一层对应关系之后，CUDA 中很多看似零散的性能优化技巧，其实都可以串成一条完整的因果链。

## 线程与 Warp：线程索引决定的不只是“谁算哪个元素”

最经典的 CUDA Kernel 通常从下面这段代码开始：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float x = data[idx];
```

在学习编程模型时，我们关注的是线程与数据之间的映射关系：

```text
thread 0 → data[0]
thread 1 → data[1]
thread 2 → data[2]
...
```

也就是说，每个线程到底负责哪个数据。

但是到了性能优化阶段，同样一段代码需要换一个角度来看。假设 `data` 是 `float`，那么一个 Warp 中连续的线程实际上会产生如下地址：

```text
lane 0  → data[0]  → address 0
lane 1  → data[1]  → address 4
lane 2  → data[2]  → address 8
...
lane 31 → data[31] → address 124
```

这 32 个线程访问了一段连续的内存，因此 GPU 能够将它们组织成较少的 Global Memory transaction，也就是我们经常说的 **Coalesced Memory Access，合并访问**。

这也是为什么在二维矩阵 Kernel 中，通常会写成（用 x 表示 col，用 y 表示 row）：

```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;

float x = A[row * width + col];
```

C/C++ 中矩阵通常按 row-major 存储，同一行相邻列在内存中连续。而一个 Warp 中连续的线程 ID 又首先沿 `threadIdx.x` 增长，因此：

```text
lane 0 → A[row][0]
lane 1 → A[row][1]
lane 2 → A[row][2]
...
```

自然就形成连续访问。反过来，如果让 `threadIdx.x` 对应 `row`，一个 Warp 的线程可能访问：

```text
A[0][col]
A[1][col]
A[2][col]
...
```

相邻线程地址之间相隔整整一个 `width`，变成典型的 stride access。

所以看 CUDA Kernel 时，一个很实用的问题不是：

> Thread 0 在算什么？

而是进一步问：

> **一个 Warp 中 lane 0～31 在同一条 load/store 指令上究竟访问了哪些地址？**

这也是 Warp 为什么会成为 CUDA 内存优化中最核心的观察尺度之一。

Global Memory 的合并访问要看 Warp；Shared Memory 的 Bank Conflict 要看 Warp；Warp Divergence 要看 Warp；甚至 `__shfl_down_sync` 这样的寄存器数据交换，本身也是 Warp 内操作。

换句话说：

> **Thread 是你写代码时看到的执行主体，而 Warp 往往才是分析硬件性能时真正重要的尺度。**

Grid-stride Loop 也能体现这一点：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

for (int i = idx; i < N; i += stride) {
    process(data[i]);
}
```

对于某一个线程来说，它前后两次访问的数据可能相隔非常大的 `stride`。但同一轮循环中：

```text
lane 0 → data[idx]
lane 1 → data[idx + 1]
lane 2 → data[idx + 2]
...
```

仍然是连续的。

下一轮也仍然是：

```text
lane 0 → data[idx + stride]
lane 1 → data[idx + stride + 1]
lane 2 → data[idx + stride + 2]
...
```

因此 Global Memory 的访问依然可以保持良好的合并特性。

这里需要建立一个非常重要的区别：

> **单个 Thread 前后访问是否连续，和一个 Warp 在同一条内存指令中是否连续访问，是两个不同的问题。**

Global Memory coalescing 更关心后者。

## Block 与 Shared Memory：CUDA 中真正的“线程协作”

如果说 Warp 更像硬件执行单位，那么 Block 更像 CUDA 编程模型中的协作单位。

一个普通的 Shared Memory 对象：

```cpp
__shared__ float tile[256];
```

会为每个 Block 创建一份独立副本。同一个 Block 中的线程都可以访问它，而不同 Block 通常不能直接通过这个 Shared Memory 相互交换数据。

因此：

```text
Block
   │
   ├── Shared Memory
   │
   └── __syncthreads()
```

这三个概念应该放在一起理解。

例如：

```cpp
__shared__ float tile[256];

int idx = blockIdx.x * blockDim.x + threadIdx.x;

tile[threadIdx.x] = input[idx];

__syncthreads();

float x = tile[255 - threadIdx.x];
```

第一步中，不同线程分别从 Global Memory 搬数据：

```text
Global Memory
      ↓
Thread 0 ─→ tile[0]
Thread 1 ─→ tile[1]
Thread 2 ─→ tile[2]
...
```

随后其他线程可能会读取这些数据。

这就产生了一个问题：Thread 0 什么时候才能确定 Thread 200 已经把自己的数据写进 Shared Memory？

答案就是：

```cpp
__syncthreads();
```

它构成了一个 Block 级同步点，使 Block 内线程可以在 Shared Memory 上安全地进行下一阶段协作。

Shared Memory 真正强大的地方还不只是“共享”，而是 **数据复用**。

这就引出了 CUDA 优化中最重要的思想之一：Tiling。

以矩阵乘法为例：

$$
C=A\times B
$$

假设一个 Block 为：

```cpp
dim3 block(16, 16);
```

也就是 256 个线程。可以让整个 Block 负责输出矩阵 `C` 中一个 `16 × 16` 的区域：

```text
                 C

        ┌──────┬──────┬──────┐
        │Block │Block │Block │
        │ 0,0  │ 1,0  │ 2,0  │
        ├──────┼──────┼──────┤
        │Block │Block │Block │
        │ 0,1  │ 1,1  │ 2,1  │
        └──────┴──────┴──────┘
```

Block 内每个线程负责一个：

```text
C[row][col]
```

然后用 Shared Memory 保存当前计算需要的 A、B 子矩阵：

```cpp
__shared__ float As[TILE][TILE];
__shared__ float Bs[TILE][TILE];

float sum = 0.0f;

for (...) {
    As[threadIdx.y][threadIdx.x] = ...;
    Bs[threadIdx.y][threadIdx.x] = ...;

    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        sum += As[threadIdx.y][k]
             * Bs[k][threadIdx.x];
    }

    __syncthreads();
}
```

这段代码同时包含了几乎所有 CUDA 编程模型和内存模型的对应关系：

```text
Grid
│
├── Block
│     └── 负责一个 C Tile
│
├── Shared Memory
│     ├── A Tile
│     └── B Tile
│
├── Warp
│     └── 协作加载连续 Global Memory
│
└── Thread
      ├── 负责一个 C[row][col]
      └── sum 保存在 Register
```

Tiling 为什么快？

最核心的一句话就是：

> **Global Memory 中的数据搬一次到 Shared Memory，然后让 Block 内多个线程重复使用。**

原本可能是：

```text
Global → Thread
Global → Thread
Global → Thread
Global → Thread
...
```

优化之后变成：

```text
             一次搬运
Global Memory ───────→ Shared Memory
                            │
                    ┌───────┼───────┐
                    ↓       ↓       ↓
                 Thread   Thread   Thread
```

也就是典型的：

> **搬一次，用多次。**

这正是 Tiling 的本质。

不过 Shared Memory 并不是“用了就一定快”。

它内部还存在 Bank。

对于常见 CUDA GPU，可以把 Shared Memory 粗略理解成分成 32 个 Bank。对于连续 `float` 数据：

```cpp
shared[threadIdx.x]
```

通常形成：

```text
lane 0  → Bank 0
lane 1  → Bank 1
lane 2  → Bank 2
...
lane31  → Bank31
```

访问可以很好地并行。

但如果写成：

```cpp
shared[threadIdx.x * 2]
```

则可能出现：

```text
lane 0  → Bank 0
lane16  → Bank 0
```

同一个 Warp 的多个线程访问同一 Bank 中的不同地址，需要分批完成，就产生了 **Bank Conflict**。

矩阵转置中经典的：

```cpp
__shared__ float tile[32][33];
```

也是同样的道理。

原本：

```cpp
float tile[32][32];
```

如果按列访问：

```text
tile[0][0]
tile[1][0]
tile[2][0]
...
```

因为每一行相隔 32 个 `float`，地址映射可能反复落到相同 Bank，形成严重的 Bank Conflict。

将第二维改为 33：

```cpp
float tile[32][33];
```

相当于人为增加一个 padding，使下一行的起始位置向后偏移一个 Bank：

```text
row 0 → Bank 0
row 1 → Bank 1
row 2 → Bank 2
...
```

从而消除冲突。

所以 Shared Memory 有两个不同的问题：

```text
谁能够共享 Shared Memory？
        ↓
       Block


Shared Memory 访问得快不快？
        ↓
       Warp
```

这两个层级不能混淆。

## Register、SM 与 Occupancy：高速内存也是有限资源

如果 Shared Memory 对应 Block，那么 Register 最自然的对应对象就是 Thread。

例如：

```cpp
int idx = ...;
float temp;
float sum = 0.0f;
```

这些线程私有的标量，在条件允许时通常会被编译器放到 Register 中。

因此可以理解成：

```text
Thread 0
 ├── idx
 ├── temp
 └── sum
      ↓
   Register

Thread 1
 ├── idx
 ├── temp
 └── sum
      ↓
   Register
```

在前面的 GEMM 中：

```cpp
float sum = 0.0f;
```

实际上非常重要。

每一次：

```cpp
sum += a * b;
```

中间累加结果如果一直保存在 Register 中，就不需要反复写回 Global Memory。

所以高性能 CUDA Kernel 经常呈现这样一条数据流：

```text
Global Memory
      ↓
Shared Memory
      ↓
Register
      ↓
ALU / Tensor Core
```

越接近计算单元的数据，通常使用频率越高、复用次数也越多。

但是 Register 并不是无限的。

这就与编程模型中的 **Occupancy** 联系起来了。

一个 SM 能同时驻留多少 Block 和 Warp，会受到多种资源共同限制：

```text
Block Size
    +
Registers / Thread
    +
Shared Memory / Block
    +
SM 硬件资源上限
    ↓
Resident Blocks
    ↓
Resident Warps
    ↓
Occupancy
```

假设一个 Kernel 每个线程只需要 32 个 Register，一个 SM 也许可以同时容纳多个 Block。

如果一个线程突然需要 100 个 Register：

```text
100 registers/thread
×
256 threads/block
```

那么单个 Block 对 Register File 的占用会明显增加，一个 SM 能同时驻留的 Block 数量就可能下降，进一步减少 Resident Warp 数量。

这也是为什么：

> **Block 越大，不代表性能越高。**

Block 大意味着线程多，但也意味着一个 Block 可能同时消耗更多 Register 和 Shared Memory。如果资源压力过大，一个 SM 最终只能驻留很少几个 Block，那么真正能够参与调度、用于隐藏内存延迟的 Warp 数量反而可能下降。

另一方面，Register 也不能一味压得很低。

如果一个 Kernel 中线程所需的临时变量过多，Register 不够时，编译器可能将部分数据 spill 到 Local Memory：

```text
原本：

Register
   ↓
高速访问


Spill 之后：

Local Memory
     ↓
L1 / L2
     ↓
Device Memory
```

Local Memory 虽然名字叫“Local”，但它只是逻辑上属于单个 Thread，并不意味着物理上像 Register 一样靠近计算单元。Register Spilling 因此可能带来明显的性能损失。

于是 CUDA 优化经常变成一种平衡：

```text
Register 使用较多
        ↓
减少 Spill
        ↓
单线程执行效率高

但是

Register Pressure ↑
        ↓
Resident Warp ↓
        ↓
Occupancy ↓
```

反过来也一样。

这也是为什么 CUDA 性能优化很少存在“某个指标越高越好”这样的简单答案。

Warp Shuffle 又进一步体现了 Register 与 Warp 的联系。

例如做 Reduction：

```cpp
val += __shfl_down_sync(
    0xffffffff,
    val,
    offset
);
```

每个线程的 `val` 原本都保存在自己的 Register 中：

```text
lane 0 register → x0
lane 1 register → x1
lane 2 register → x2
...
```

一种传统的线程间数据交换方式可能是：

```text
Register
   ↓
Shared Memory
   ↓
同步
   ↓
Shared Memory
   ↓
Register
```

而 Warp Shuffle 可以直接实现：

```text
lane 16 Register
        ↓
lane 0 Register
```

因此它不仅是一种 Warp Reduction 算法技巧，同时也是一种减少 Shared Memory 使用和同步开销的优化方法。

例如 Block 中有 256 个线程，也就是 8 个 Warp，可以采用：

```text
Thread
  ↓
Register 中完成局部累加

  ↓

Warp
  ↓
__shfl_down_sync
  ↓
得到 1 个 Warp Sum

  ↓

8 个 Warp Sum
  ↓
Shared Memory

  ↓

第一个 Warp 再做 Reduction

  ↓

Block Sum
```

于是线程层次与内存层次再次形成对应：

```text
Thread Level → Register

Warp Level   → Shuffle

Block Level  → Shared Memory

Grid Level   → Global Memory
```

## 从 Host 到 Tensor Core：把 CUDA 优化理解成一条数据流

如果把视角继续拉远，CUDA 的数据流实际上从 CPU 就已经开始了：

```text
CPU DRAM
   │
   │ Host → Device
   ↓
GPU Global Memory
   │
   │ Kernel
   ↓
Shared Memory
   ↓
Registers
   ↓
ALU / Tensor Core
```

计算完成后，再反向返回：

```text
Registers
   ↓
Global Memory
   │
   │ Device → Host
   ↓
CPU DRAM
```

因此 CUDA 性能优化可以看成两个尺度。

第一个尺度是：

```text
CPU ↔ GPU
```

重点是减少不必要的 Host/Device 数据搬运。

第二个尺度则发生在 GPU 内部：

```text
Global Memory
      ↓
Shared Memory
      ↓
Register
      ↓
Compute
```

重点是让昂贵的数据读取尽可能少，让已经搬进来的数据尽可能多次复用。

最终，可以把 CUDA 编程模型和内存模型合成下面这一张心智图：

```text
                    GPU Kernel / Grid
                          │
                     Global Memory
                          │
             ┌────────────┴────────────┐
             │                         │
            SM                        SM
             │
      ┌──────┴─────────┐
      │                │
Register File     Shared Memory / L1
      │                │
      └────── Warp ────┘
             │
         32 Threads
```

在这张图上，几乎所有常见 CUDA 优化都能找到自己的位置：

```text
Thread
  ↓
Register
  ↓
注意 Register Pressure / Spill


Warp
  ↓
Global Memory → Coalescing
Shared Memory → Bank Conflict
Register      → Shuffle
Branch        → Divergence


Block
  ↓
Shared Memory
Tiling
__syncthreads()
Block 内数据复用


SM
  ↓
Register + Shared Memory 资源预算
  ↓
Resident Blocks / Warps
  ↓
Occupancy


Grid
  ↓
组织整个数据集
  ↓
主要通过 Global Memory 保存和交换结果
```

所以如果只记住一个 CUDA 优化原则，可以记成：

> **让数据沿着 `Global Memory → Shared Memory → Register` 尽可能靠近计算单元，同时让 `Grid → Block → Warp → Thread` 的线程组织方式与数据布局相匹配。**

线程任务怎么划分，会决定一个 Warp 的内存地址模式，从而影响 Global Memory 是否能够合并访问；Block 怎么划分，会决定哪些数据能够通过 Shared Memory 复用；Warp 内线程如何访问 Shared Memory，又会决定是否发生 Bank Conflict；每个线程承担多少计算，则会影响 Register 使用量；每个 Block 的 Tile 多大，又会影响 Shared Memory 占用。

最终：

```text
线程映射
    ↓
内存访问模式
    ↓
数据复用方式
    ↓
Register / Shared Memory 使用量
    ↓
SM Resident Warps
    ↓
Occupancy 与延迟隐藏
    ↓
Kernel 最终性能
```

这条链才是 CUDA 编程模型和内存模型真正连接起来的地方。

后续无论是学习 **Reduction、GEMM、Softmax 还是 FlashAttention**，本质上都还是围绕同样的问题展开：

> 数据应该由哪些线程计算？
> 一个 Warp 应该访问哪些地址？
> 哪些数据值得放进 Shared Memory？
> 哪些中间结果应该保留在 Register？
> 为了增加数据复用，又付出了多少 Register 和 Shared Memory 资源？

当你开始习惯用这几个问题分析 Kernel 时，CUDA 优化就不再是一堆零散技巧，而会逐渐变成一套完整、统一的性能分析方法。
