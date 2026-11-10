# Trace_v2 —— 带层号的 HBM 访存 trace 拆分结果

本仓库是 **Qwen3-Next-80B-A3B-Instruct** 推理时 HBM 访存 trace 的 **v2 版本**统计结果。

v2 相比 v1 的关键改进：通过 marker kernel + 层 forward hook 机制，在 trace 记录中**加入了 `layer_id`（模型层号）字段**，从而可以精确地把访存 trace 按"token × 层"拆分。

## 一、采集方法

| 项目 | 说明 |
|---|---|
| 模型 | Qwen3-Next-80B-A3B-Instruct（NVFP4 量化，MoE，48 层混合线性注意力+全注意力） |
| GPU | NVIDIA RTX 6000D（Blackwell，sm_120） |
| 采集工具 | NVBit 1.8.1 + 自研 `hbm_trace_v2`（记录含 layer_id） |
| 层号标记 | load_inline 编译 marker kernel + `register_forward_pre_hook`，每层 forward 前 launch marker |
| 采集范围 | decode 阶段 3 个 token |

## 二、trace 记录格式（原始 .bin，32 字节/条）

| 字段 | 类型 | 含义 |
|---|---|---|
| grid_launch_id | u64 | kernel 启动序号 |
| cta_id_x | u32 | block 在 x 维度的编号 |
| warp_id | u16 | warp 在 block 内的编号 |
| opcode_id | u16 | 访存指令类型编号 |
| addr | u64 | warp 首地址（GPU 虚拟地址） |
| size | u32 | 每线程访存字节数 |
| layer_id | u32 | **模型层号（0~47）** |

## 三、文件清单

### 3.1 split_summary.csv —— 拆分统计汇总

145 个拆分文件（3 token × 48 层 + 1 残留）的逐文件统计。

| 列名 | 含义 |
|---|---|
| token | token 编号（0/1/2，`preamble` 为开头残留） |
| layer | 层号（0~47） |
| records | 该文件访存记录数 |
| read_count | 读（Load）次数 |
| write_count | 写（Store）次数 |
| atomic_count | 原子操作次数 |
| file_name | 对应拆分文件名 |
| file_size_bytes | 文件字节数 |

### 3.2 代表性层样本文件（本仓库包含）

145 个原始 CSV 共约 9.8GB，超 GitHub 限制，本仓库只放了**有代表性的 6 个层**（覆盖两种关键层类型 × 3 个 token）：

| 文件 | 层类型 | 覆盖的访存行为 |
|---|---|---|
| token0_layer00.csv | GDN 层（线性注意力） | conv_state/ssm_state 递推状态流式访问 |
| token0_layer03.csv | 全注意力层 | KV cache 分页访问 + QKV 权重 |
| token1_layer00.csv | GDN 层 | 同上 |
| token1_layer07.csv | 全注意力层 | 同上（layer03 超 100MB，用 layer07 代替） |
| token2_layer00.csv | GDN 层 | 同上 |
| token2_layer03.csv | 全注意力层 | 同上 |

样本文件的列格式（CSV）：

```
grid_launch_id,cta_id_x,warp_id,opcode,rw,addr,size,layer_id
28,2,0,LDG.E.U16,R,0x00007959dfdc1e00,2,0
```

- `opcode`：SASS 指令名（`LDG.*`=全局读，`STG.*`=全局写，`LD`/`LDL`=局部，`ATOMG`/`REDG`=原子）
- `rw`：`R`=读 / `W`=写 / `A`=原子
- `addr`：16 进制地址
- `size`：每线程字节数

## 四、数据特征（已知，供分析参考）

1. **layer 47 异常大**：每个 token 的 layer 47 约 1.1GB（2000 万条），因为层号标记只在每层 forward **前**插入，第 47 层之后的**模型输出阶段**（lm_head + 采样）都被归到 layer 47。
2. **token 0 略大**：混入了 prefill 阶段（decode 前处理 prompt）的 kernel。
3. **全注意力层 vs GDN 层**：全注意力层（layer 3/7/11/...，每 4 层一个）有 KV cache 访问，GDN 层（其余层）是线性注意力状态访问，两者访存模式不同。

## 五、完整数据获取

145 个完整拆分文件（9.8GB）及原始 trace 因体积超 GitHub 限制未上传，保存在本地 `/home/workstation1/nvbit_trace_v2/`。
