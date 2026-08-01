---
title: "Tokenization 与词嵌入：从文本到 Transformer 输入"
date: 2026-07-31 22:00:00 +0800
categories:
  - AI Infra
  - Transformer
tags:
  - tokenization
  - embedding
  - tokenizer
  - transformer
  - llm-inference
description: "从字符级、词级和子词级切分出发，解释 Embedding Lookup、词表参数量，以及 Tokenizer 对 Prefill、KV Cache、Decode 和推理吞吐的影响。"
toc: true
math: true
comments: false
---

## 摘要

文本进入 Transformer 前，先由 Tokenizer 切成 token 并映射为 Token ID，再通过可学习的 Embedding 矩阵查表得到稠密向量；位置编码补上顺序信息后，形成送入第一个 Transformer Block 的输入矩阵。

## 1. 核心链路

```text
原始文本
  -> Tokenization
  -> Token IDs: (N,)
  -> Embedding Lookup
  -> Token Embeddings: (N, d_model)
  -> 注入位置信息
  -> Transformer 输入: (N, d_model)
```

这里的两个基本问题分别是：如何把连续文本切成模型可处理的离散单元，以及如何把离散单元编码为神经网络可计算的连续向量。

## 2. Tokenization

### 2.1 三种粒度

- 字符级：词表通常较小；若字符集或字节回退能够覆盖输入，就能避免传统意义上的生词，但序列更长，计算和 KV Cache 开销随 token 数增加。
- 词级：常见词可以用一个 token 表示，但词表容易膨胀，生词和词形变化难处理。
- 子词级：在词表规模、序列长度和生词覆盖之间折中，是现代大模型的主流方案。

现代大模型普遍采用子词或字节级 Tokenization，BPE（Byte Pair Encoding）及其变体是主流方案之一。BPE 从较小单元开始，反复合并高频组合；常见片段可能成为单独 token，低频词则由多个已有片段拼接。切分结果来自训练语料统计，不保证与语言学中的词根、词缀边界一致。除此之外，SentencePiece Unigram、WordPiece 等方案也有广泛应用。

词表大小会形成直接的系统权衡：更大的词表通常能以更少 token 表示常见文本，但会增大 Embedding 和 LM Head 的参数与显存开销；更小的词表节省静态权重，却可能需要更多 token 表示同一段文本，从而增加 Prefill 计算量、Attention 工作量和 KV Cache 占用。

中文没有天然空格边界，实际切分取决于具体 tokenizer、训练语料、归一化方式、词表以及是否采用字节回退，不能简单等同于“一个汉字一个 token”。面向中文优化的 tokenizer 若收录更多常用汉字组合，通常能降低中文文本的 token 数。

## 3. 从 One-Hot 到 Embedding

One-Hot 向量维度等于词表大小 $V$，非常稀疏，而且任意两个不同 token 都彼此正交，编码本身不表达语义相似性。因此 Transformer 使用可学习的 Embedding 矩阵：

$$
W_{emb}\in\mathbb{R}^{V\times d_{model}}
$$

给定 Token ID $i$，输入向量就是矩阵的第 $i$ 行：

$$
e_i=W_{emb}[i,:]
$$

这在数学上等价于 One-Hot 向量乘以权重矩阵，但工程实现是索引查表，无需执行稀疏矩阵乘法。参数量为：

$$
P_{emb}=V\times d_{model}
$$

Embedding 提供的是上下文化之前的初始表示。同一个 token 在输入层总会查到同一行向量；多义词在具体语境中的动态消歧，主要由后续 Self-Attention 和其他层根据上下文完成。例如，`bank` 的输入 Embedding 是一个固定的初始向量，它综合了训练语料中不同使用语境的统计信号；经过 Transformer 后，包含 `money` 和 `river` 的不同上下文会使它形成不同的隐藏状态。因此，“Embedding 中相近的 token 往往具有相关语义”是有用直觉，但不能把输入 Embedding 直接当成最终的上下文语义表示，也不能简单把多义词的输入向量理解成几个词义中心的几何中点。

## 4. 为什么还要位置信息

仅查 Embedding 表不会编码 token 所在的位置。序列中的表示还必须获得位置信息，模型才能区分相同 token 的不同出现位置以及不同词序。对使用绝对位置编码的原始 Transformer，可以用下式概括输入层：

$$
x_i=W_{emb}[token_i]+PE(i)
$$

对采用 RoPE 的 Decoder-only LLM，更准确的说法是：Token Embedding 先成为残差流输入，而 RoPE 通常施加在 Attention 内部的 Query/Key 上，并非直接把一个位置向量加到输入 Embedding。为什么许多现代 LLM 采用 RoPE，而不是原始 Transformer 的绝对位置编码，我将在下一篇笔记中介绍。

## 5. AI Infra 视角

Tokenization 虽然不属于 Transformer Block 内部的核心算子，但它定义了文本与模型之间的离散接口。Tokenizer 的切分方式决定 token 数，词表大小则直接影响 Embedding 和 LM Head 的参数规模；二者会进一步影响 Prefill、KV Cache、Decode 步数、吞吐与服务成本。

- Token 数是系统成本单位之一：Tokenizer 的切分效率会影响上下文长度、Prefill FLOPs、Attention 工作量、KV Cache 显存、吞吐和计费口径。
- $V$ 增大会线性增大 Embedding 参数；若 LM Head 不共享权重，还会再增加约 $V\times d_{model}$ 个参数。
- Embedding 前向主要是内存访问型的 gather，而 LM Head 是从隐藏维到整个词表的投影，二者不能因为形状相关就视作相同算子。
- Weight Tying（权重绑定，即 Embedding 和 LM Head 共享权重）是否存在，必须查看具体模型配置或实现，不能由模型家族或参数规模推断。
- 量化策略也应按模型和推理框架核验；“Embedding 通常不量化”不是普适规则。

## 6. 可复习的七个问题

### 6.1 字符级、词级、子词级 Tokenization 各自如何影响词表大小和序列长度？

三种方式的根本区别，是在“词表大小”和“序列长度”之间做不同取舍。

#### 1. 字符级 Tokenization

字符级分词把文本拆成单个字符。例如：

```text
Tokenization
→ T / o / k / e / n / i / z / a / t / i / o / n
```

特点：词表通常较小，只需要覆盖字符集合；若字符集或字节回退能够覆盖输入，就很少遇到传统意义上的生词；序列通常较长，一个单词可能变成许多 token。词级语义需要跨更多字符位置聚合，增加了序列建模的跨度和计算负担，但这并不意味着模型必然需要更多 Transformer 层。

对推理系统的影响包括：

- Prompt token 数更多；
- Prefill 计算量增加；
- Attention 处理的序列更长；
- 每层需要保存更多位置的 KV Cache；
- Decode 时，如果一个人类可见的词需要多个 token，模型也必须执行多次自回归迭代才能输出它。

#### 2. 词级 Tokenization

词级分词尝试把每个完整单词作为一个 token。

特点：常见文本的序列较短；词表可能非常大，因为不同单词、时态、单复数和拼写形式都可能成为不同条目；容易遇到未登录词，即训练词表中不存在的新词；对中文、日文等没有天然空格分隔的语言，还需要额外的分词算法。

例如，下面这些词可能都要占用独立词表项：

```text
play
plays
played
playing
player
players
```

#### 3. 子词级 Tokenization

子词级分词把常见词或常见片段作为 token，低频词则拆成多个片段。例如：

```text
playing
→ play / ing
```

特点：词表大小通常处于字符级和词级之间；序列长度也通常处于二者之间；高频词可以用一个 token 表示，降低常见文本的序列长度；低频词、新词和复合词可以由已有子词组合；能在词表规模、生词覆盖和序列长度之间取得较好的平衡。

这就是现代大模型通常使用子词或字节级子词 Tokenization 的主要原因。


### 6.2 BPE 的合并规则为什么不等同于语言学分词？

因为 BPE 的目标不是找出语言学意义上的“词根、前缀、后缀或词”，而是根据训练语料中的频率统计，寻找值得合并的符号组合。

BPE 的简化过程是：

1. 从字符、字节或其他较小单元开始。
2. 统计相邻单元对出现的频率。
3. 将最常出现的一对合并成一个新符号。
4. 重新统计。
5. 重复合并，直到达到目标词表规模或其他停止条件。

例如，在训练语料中：

```text
l o w
l o w
l o w
l o w e r
n e w e r
```

如果 l + o 很常见，BPE 可能先合并成 lo；之后又可能把 lo + w 合并成 low。
这些合并只说明某些片段在训练数据中频繁相邻，不说明它们一定是语言学上的最小语义单位。

因此，BPE 可能产生完整单词、词根、词缀、半个词根、跨越语言学边界的字符组合、包含空格或标点的片段，甚至字节级片段。

例如，某个 tokenizer 可能把：

```text
unbelievable → un / believe / able
```

这看起来符合语言学直觉；另一个 tokenizer 却可能切成：

```text
unbelievable → unbel / iev / able
```
后者在语言学上不自然，但如果这些片段在训练语料中统计上更有价值，BPE 仍可能选择这种切分。

所以，BPE 是一种压缩和编码策略，不是严格的自然语言分词器。它优化的是词表覆盖与序列长度等统计目标，而不是语言学正确性。

### 6.3 为什么 One-Hot 编码既浪费空间，又不直接表达 token 间的相似关系？

假设词表大小为 $V$，One-Hot 编码就用一个 $V$ 维向量表示每个 token，其中只有对应位置为 1，其余位置都是 0。

#### 1. 为什么浪费空间？

如果词表有 128,000 个 token，那么每个 One-Hot 向量都有 128,000 维，但只有一个元素非零。这是高度稀疏的表示。

对长度为 $N$ 的序列，如果显式构造 One-Hot 输入矩阵，其形状是：

$$
N\times V
$$

绝大多数元素都是 0，直接存储和计算非常浪费。

实际框架不会真的构造这个 $N\times V$ 的 One-Hot 输入矩阵，而是保留整数 Token ID，并通过 Embedding Lookup 直接读取所需的权重行。One-Hot 在这里主要用于说明数学等价关系。

#### 2. 为什么不表达相似关系？

任意两个不同 One-Hot 向量都相互正交。

例如：

$$
\operatorname{onehot}(\text{猫})\cdot
\operatorname{onehot}(\text{狗})=0
$$

同时：

$$
\operatorname{onehot}(\text{猫})\cdot
\operatorname{onehot}(\text{银行})=0
$$

在 One-Hot 空间中，“猫”和“狗”的关系，与“猫”和“银行”的关系完全一样：它们都是不同的离散标识。

如果使用欧氏距离，任意两个不同 One-Hot 向量之间的距离也相同：

$$
\|e_i-e_j\|_2=\sqrt{2},\qquad i\ne j
$$

因此 One-Hot 只表达：

> 这是哪一个 token。

它不直接表达：

> 这个 token 和哪些 token 语义相近。

### 6.4 为什么 Embedding Lookup 等价于 One-Hot 乘矩阵，却不按矩阵乘法实现？

设某个 token 的 ID 为 $i$，它的 One-Hot 向量为：

$$
o_i\in\mathbb{R}^{V}
$$

Embedding 矩阵为：

$$
W_{emb}\in\mathbb{R}^{V\times d_{model}}
$$

那么：

$$
o_i^TW_{emb}=W_{emb}[i,:]
$$

原因是 $o_i$ 中只有第 $i$ 个元素是 1，其余位置都是 0。矩阵乘法相当于：

$$
0\cdot W_{emb}[0,:]
+\cdots+
1\cdot W_{emb}[i,:]
+\cdots+
0\cdot W_{emb}[V-1,:]
$$

最后只剩下：

$$
W_{emb}[i,:]
$$

所以，从数学上看：

> One-Hot × Embedding 矩阵 = 直接读取 Embedding 矩阵的第 $i$ 行。

为什么工程上不用矩阵乘法？

因为显式构造 One-Hot 向量会做大量无意义工作：

1. 创建长度为 $V$ 的稀疏向量。
2. 保存几乎全为 0 的数据。
3. 执行大量乘 0 操作。
4. 对整个权重矩阵进行不必要的访问。

实际实现只需执行索引操作：

```python
embedding = W_emb[token_id]
```

对于一批 Token ID：

```text
token_ids: (batch_size, seq_len)
```

查表后得到：

```text
embeddings: (batch_size, seq_len, d_model)
```

这类操作通常称为 Embedding Lookup、Gather 或 Index Select。

#### 反向传播时如何训练？

虽然前向是查表，但 Embedding 仍然是可学习参数。
在反向传播中，只有本批次实际访问过的行会直接接收到对应梯度。不同框架可以使用稠密梯度或稀疏梯度表示；在大模型训练中，实际实现还会受到张量并行、词表并行和优化器设计影响。

### 6.5 如何由 $V$ 和 $d_{model}$ 估算 Embedding 与 LM Head 的参数量？

设：

> $V$：词表大小
>
> $d_{model}$：隐藏维度

#### 1. Token Embedding 参数量

Embedding 矩阵形状是：

$$
W_{emb}\in\mathbb{R}^{V\times d_{model}}
$$

所以参数量为：

$$
P_{emb}=Vd_{model}
$$

例如：

$$
V=32{,}000,\qquad d_{model}=4096
$$

那么：

$$
P_{emb}=32{,}000\times4096
=131{,}072{,}000
$$

约为：

$$
131\text{M parameters}
$$

如果使用 FP16 或 BF16，每个参数约占 2 字节，那么仅 Embedding 权重约占：

$$
131{,}072{,}000\times2
\approx262\text{ MB}
$$

这里的 MB 是十进制近似；若用 MiB 表示，数值会略有不同。

#### 2. LM Head 参数量

LM Head 把隐藏状态从 $d_{model}$ 维映射到 $V$ 维：

$$
W_{lm}\in\mathbb{R}^{V\times d_{model}}
$$

不同实现可能在公式中写成转置形式，但参数元素总数相同：

$$
P_{lm}=Vd_{model}
$$

以上述配置为例，也约为 131M 参数。

#### 3. 不共享权重

如果 Embedding 和 LM Head 各自拥有独立权重，则总参数量约为：

$$
P_{emb+lm}=2Vd_{model}
$$

即：

$$
2\times131\text{M}=262\text{M parameters}
$$

在 FP16/BF16 下，静态权重约为 524 MB。

#### 4. 共享权重：Weight Tying

如果 LM Head 复用 Embedding 矩阵，则只保存一份：

$$
P_{emb+lm}=Vd_{model}
$$

这能节省 $Vd_{model}$ 个参数。

但是否共享必须查看具体模型配置或代码。

真实参数量还可能包含：

- LM Head bias；
- 特殊词表填充产生的额外行；
- 张量并行需要的 padding；
- 输入、输出 Embedding 是否共享；
- 多模态模型额外的投影层；
- 量化格式中的 scale、zero point 和元数据。

因此，$Vd_{model}$ 是核心估算公式，而不是所有实现的精确显存占用。

### 6.6 Tokenizer 为什么会影响 Prefill、KV Cache、吞吐和服务成本？

因为模型处理的是 token 序列，而不是人类看到的字符数、单词数或文本字节数。
同一段文本经过不同 Tokenizer 后，可能得到不同的 token 数。

#### 1. 对 Prefill 的影响

Prefill 阶段需要一次性处理整个 Prompt。

对于长度为 $N$ 的序列：

1. 线性层和 FFN 的主体计算量通常随 $N$ 近似线性增长。
2. 传统 Attention 中 $QK^T$ 和加权聚合部分存在与 $N^2$ 相关的计算结构。

实际 FlashAttention 等优化可以显著减少中间张量和 HBM IO，但不会让长序列变成零成本。

因此，Tokenizer 产生的 token 越多：

1. Prefill FLOPs 越高。
2. 首 token 延迟 TTFT 通常越大。
3. 请求占用 GPU 的时间和调度预算越多。

#### 2. 对 KV Cache 的影响

每个已处理 token 通常都需要在每层保存对应的 Key 和 Value。

粗略地说，KV Cache 大小与以下变量近似成正比：

$$
B\times N\times L\times H_{KV}\times d_{head}\times2\times s
$$

其中：

- $B$：批大小或序列数；
- $N$：缓存的 token 数；
- $L$：Transformer 层数；
- $H_{KV}$：KV Head 数；
- $d_{head}$：每个 Head 的维度；
- 乘以 2：分别保存 K 和 V；
- $s$：每个元素的字节数。

因此，在模型架构不变时：

$$
KV\ Cache\propto N
$$

同一段文字如果产生更多 token，就会占用更多 KV Cache。对于变长批次，实际占用更接近各序列已分配 token 数之和；PagedAttention 等实现还会受到 block 大小、尾块内部碎片和预留策略影响。

#### 3. 对 Decode 的影响

自回归生成每次通常只生成一个 token。一个人类可见的词如果被拆成多个 token，就需要多次模型前向。

因此，Tokenizer 会影响：

1. 每秒输出 token 数和每秒输出字符数之间的关系。
2. 用户实际看到的文本生成速度。
3. 单次请求占用的 Decode 步数。
4. 调度器需要执行的迭代次数。

#### 4. 对吞吐的影响

在相同 GPU 和模型下，更多 token 通常意味着：

1. 每个请求占用更多计算资源。
2. 每个请求停留在批次中的时间更长。
3. KV Cache 容量能同时容纳的请求更少。
4. Continuous Batching 的可用并发空间下降。
5. 系统单位时间能完成的请求数可能下降。

因此，只比较“每秒请求数”而不比较请求的输入、输出 token 分布，会产生误导。

#### 5. 对服务成本的影响

很多服务按输入 token 和输出 token 计费。即使不按 token 直接收费，底层资源仍然主要随 token 负载增长：

1. 输入 token 消耗 Prefill 计算。
2. 总上下文 token 消耗 KV Cache。
3. 输出 token 消耗串行 Decode 迭代。
4. 更长的存活时间影响并发和调度。

因此，Tokenizer 不只是文本预处理细节，而是模型服务成本模型的一部分。

一个实用结论是：

> 比较模型或推理服务时，不能只比较原始文本长度；应使用各自 Tokenizer 统计输入与输出 token，并同时报告字符数或词数，避免不同词表造成的指标错觉。

### 6.7 输入 Embedding、上下文化隐藏状态和输出 LM Head 分别承担什么角色？

这三个概念处于语言模型流水线的不同位置，不能混为一谈。

完整链路可以写成：

> 文本 → Tokenizer → Token IDs → Input Embedding → 多层 Transformer → 上下文化隐藏状态 → LM Head → Logits → 采样 → 下一个 Token ID

#### 1. 输入 Embedding：把离散 ID 变成初始连续向量

Tokenizer 输出的是整数 ID，例如：

```text
[3421, 528, 11082, 319]
```

这些数字只是词表索引，本身没有可直接供神经网络处理的连续几何结构。

Embedding 层执行：

$$
e_i=W_{emb}[token_i,:]
$$

得到每个 token 的初始向量。

其职责是：

1. 把离散 Token ID 映射到 $d_{model}$ 维空间。
2. 为后续 Transformer 层提供初始表示。
3. 在训练中学习与 token 使用模式相关的基础结构。

但输入 Embedding 尚未充分结合当前句子的上下文。

例如，假设一个 token 可以表示多义词“苹果”，单纯查表时无论出现在下面哪个句子里，初始向量都相同：

```text
我吃了一个苹果。
苹果发布了新电脑。
```

#### 2. 上下文化隐藏状态：结合上下文表示当前位置

经过多层 Transformer 后，每个位置得到隐藏状态：

$$
H\in\mathbb{R}^{N\times d_{model}}
$$

第 $i$ 个位置的隐藏状态可以记为：

$h_i$ 不再只是该 token 的静态查表结果，而是经过 Self-Attention、FFN、残差连接、Norm 和多层非线性变换之后形成的上下文相关表示。

对于 Decoder-only 模型，由于使用因果掩码，第 $i$ 个位置的隐藏状态只能融合当前位置及之前位置的信息：

$$
h_i=f(x_1,x_2,\ldots,x_i)
$$

因此，前面两个“苹果”在进入 Embedding 层时可能相同，但经过上下文建模后会产生不同的隐藏状态：

> “吃了一个苹果”中的苹果
> → 更接近食物、水果语境
>
> “苹果发布了新电脑”中的苹果
> → 更接近公司、产品语境

这就是上下文化表示与输入 Embedding 的根本区别。

#### 3. LM Head：把隐藏状态映射为词表分数

LM Head 接收隐藏状态，并为词表中的每个候选 token 计算一个 logit。

假设：

$$
h_i\in\mathbb{R}^{d_{model}}
$$

LM Head 权重为：

$$
W_{lm}\in\mathbb{R}^{V\times d_{model}}
$$

则：

$$
z_i=W_{lm}h_i
$$

得到：

$$
z_i\in\mathbb{R}^{V}
$$

其中每一个元素对应一个词表 token 的未归一化分数。

再经过 Softmax：

$$
P(token=j\mid context)
=
\frac{\exp(z_{i,j})}
{\sum_{k=1}^{V}\exp(z_{i,k})}
$$

得到下一个 token 的概率分布。

采样器再根据策略选择 token。

#### 三者的职责对照

| 组件 | 输入 | 输出 | 核心职责 |
|---|---|---|---|
| Input Embedding | Token ID | 初始稠密向量 | 把离散符号映射到模型隐藏空间 |
| Transformer | 隐藏状态 | 上下文化向量 | 根据上下文更新每个位置的表示 |
| LM Head | 隐藏状态 | 全词表 logits | 给下一个 token 的所有候选打分 |

可以用一句更容易记忆的话总结：

> Embedding 负责“把 token 放进隐藏空间”，Transformer 负责“结合上下文理解当前位置”，LM Head 负责“把当前理解转换成下一个 token 的候选分数”。

## 来源

- [AIInfraGuide：3.9 Tokenization 与词嵌入](https://caomaolufei.github.io/AIInfraGuide/guides/%E6%A8%A1%E5%9D%97%E4%B8%80-%E5%89%8D%E7%BD%AE%E7%9F%A5%E8%AF%86/transformer/39-tokenization%E4%B8%8E%E8%AF%8D%E5%B5%8C%E5%85%A5/)
- [Meta LLaMA-2-7B Hugging Face 模型仓库](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- [Llama 2 论文](https://arxiv.org/abs/2307.09288)
- [SentencePiece 官方仓库](https://github.com/google/sentencepiece)
