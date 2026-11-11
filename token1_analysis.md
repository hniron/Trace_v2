# Token 1 逐层访存行为分析报告

分析对象：Qwen3-Next-80B-A3B 推理时 **decode 阶段 token 1**（纯 decode，无 prefill 混入）。

## 一、总体概览

一个 token 从头到尾（layer 0 → 47）产生约 **7408 万条**访存记录，按 kernel 分类：

| 访存类型 | 数据量 | 记录数 | 说明 |
|---|---|---|---|
| gemm（专家权重） | **8.56 GB** | 4656 万 | MoE 专家权重 + 主权重读取，绝对主导 |
| gemv（decode 投影） | 1.47 GB | 2295 万 | attention 的 QKV 投影（batch=1 退化为矩阵-向量乘） |
| elementwise | 0.43 GB | 335 万 | 激活的逐元素操作 |
| moe（路由） | 0.09 GB | 67 万 | topk gating、expert prefix sum |
| linear_attn | 0.09 GB | 22 万 | GDN 层的 delta rule 递推 |
| 其他（norm/fp4/activation） | 0.03 GB | — | 归一化、量化转换 |

**核心结论**：一个 token 的访存 **~80% 是 MoE 专家权重读取**，这是 HBM 压力的主要来源。

## 二、48 层访存数据量规律

每层分为两类，严格按 `3 GDN + 1 全注意力` 循环（每 4 层一组）：

| 层类型 | 层数 | kernel 数 | 记录数 | 读字节 | 写字节 |
|---|---|---|---|---|---|
| GDN 层（线性注意力） | 36 个 | ~62 | ~113 万 | ~195 MB | ~1.8 MB |
| 全注意力层 | 11 个 | ~55 | ~88 万 | ~155 MB | ~0.7 MB |
| layer 47（含输出阶段） | 1 | 93 | 2041 万 | 1406 MB | 3.8 MB |

两个异常点：
- **layer 3（第一个全注意力层）**：记录数 412 万（其他全注意力层仅 88 万），多出 **413 MB 连续写** —— 这是 KV cache 块的初始化填充（`FillFunctor<unsigned char>`，continuity=1.0，连续写）。KV cache 在首个全注意力层惰性初始化。
- **layer 47**：含模型输出阶段（lm_head 大矩阵 + 采样），记录数 2041 万，是其他层的 ~18 倍。

## 三、每层内部的 kernel 时间线

### GDN 层（以 layer 0 为例，按时间顺序）

```
elementwise × ~10  (激活准备, <0.1MB, 连续)   ← 前处理
gemm               (903936条, 158MB, cont=0.0025) ← MoE 专家权重，稀疏离散
gemv               (8256条, 0.5MB)             ← attention QKV 投影
elementwise × ~6   (激活, <0.1MB)              ← 后处理
linear_attn        (5248条, 2.2MB)             ← delta rule 递推（线性注意力）
norm               (160条)                     ← 归一化
fp4_convert        (2064条)                    ← FP16→FP4 量化
gemm               (128条, 0.016MB)            ← 共享专家 GEMM
```

### 全注意力层（以 layer 3 为例）

```
elementwise FillFunctor  (323万条, 413MB, cont=1.0) ← KV cache 初始化（连续写）
gemm (wmma bf16)         (678万条, 118MB, cont=0.0019) ← MoE 专家 GEMM
gemm (GemmUniversal)     (6万条, 15MB)               ← MoE 相关
gemv                     (6.6万条, 4.2MB)            ← attention 投影
moe (mergeExpertPrefixSum)(8242条, 1MB)              ← 专家路由汇总
kernel_mha               (386条)                     ← 多头注意力（KV cache 读写）
fp4_convert              (2064条)
```

**结构规律**：每层 = 「激活准备 → MoE 专家权重读取（主访存）→ attention 投影 → 激活后处理 → 归一化/量化」。

## 四、地址连续性：访问是"离散"还是"连续"？

用 `连续度 = 总访问字节 / 地址跨度` 衡量（接近 1=连续，接近 0=离散）：

| 访问类型 | 连续性 | 含义 |
|---|---|---|
| gemm（MoE 专家） | **0.002** | 极度离散 |
| gemv（投影） | ~0.0001 | 离散 |
| KV cache 初始化 | **1.0** | 完全连续 |
| norm/activation | 0.9~1.0 | 连续 |

**关键发现**：主导访存的 MoE 专家权重读取，连续度仅 **0.002**。即单层读了 188MB 数据，却跨越了 **91GB 的地址空间**。

原因：MoE 有 512 个专家，权重总量巨大（散布在 91GB 地址空间里），但每个 token 只激活 10 个专家（~188MB），所以是"从巨大的地址空间里稀疏地读少量权重"。

这正是文档里说的「MoE 路由导致权重访问模式高度不规则」的量化证据。

## 五、访存行为与模型组件的对应

| 模型组件 | 对应 kernel | 单 token 数据量 | 连续性 | 特征 |
|---|---|---|---|---|
| **MoE 专家权重** | gemm | 8.56 GB | 0.002（离散） | 稀疏读激活专家，HBM 压力主体 |
| **注意力权重(QKV)** | gemv | 1.47 GB | ~0.0001（离散） | batch=1 的矩阵-向量乘 |
| **KV cache** | FillFunctor + kernel_mha | 0.41 GB（layer3 写） | 1.0（连续） | 首次初始化连续写，后续增量读写 |
| **线性注意力状态** | linear_attn | 0.09 GB | 离散 | GDN 层 conv/ssm 递推状态 |
| **激活/归一化** | elementwise/norm | 0.43 GB | 连续 | 中间激活流式读写 |

## 六、对 FPGA 仿真的启示

1. **HBM 压力的核心是"稀疏大跨度读取"**：不是想象中的"大片连续顺序读"，而是从 ~91GB 地址空间里稀疏地读 ~188MB/层。FPGA 侧需要模拟这种"地址离散、跨度巨大"的访问模式。
2. **每层访存高度规律**：GDN 层 ~188MB MoE 读，全注意力层 ~148MB MoE 读 + KV cache 操作，48 层循环。可以用"层模板"来近似整个 token 的行为。
3. **KV cache 是"连续写 + 稀疏读"**：初始化是连续大块写，推理中是分页的增量访问。
4. **写远远少于读**（读写比约 50:1），读是绝对主导。

## 附：数据文件

- `token1_layer_summary.csv` — 48 层汇总（每层 kernel 数、记录数、读写字节、地址跨度、连续度、页数）
- `token1_layer_kernels.csv` — 2929 个 kernel 的逐 kernel 统计（按时间顺序）
