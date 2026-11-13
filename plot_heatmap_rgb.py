#!/usr/bin/env python
"""方案 A：RGB 分色热度图。

红 R = MoE 专家权重读取（gemm + moe）
绿 G = attention / KV cache（gemv + attention + linear_attn）
蓝 B = 其他（norm / elementwise / activation 等）

颜色混合：黄=MoE+attention 重叠，白=三类都强，黑=无访存。
结合像素放大（纵轴分桶减到 300，每个桶覆盖更大地址范围）。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REC_DTYPE = np.dtype([
    ('grid', '<u8'), ('cta', '<u4'), ('warp', '<u2'), ('opcode', '<u2'),
    ('addr', '<u8'), ('size', '<u4'), ('layer', '<u4'),
])


def load_kernel_map(meta_path):
    km = {}
    with open(meta_path, errors="replace") as f:
        for line in f:
            parts = line.split(None, 2)
            if parts and parts[0] == "KERNEL":
                km[int(parts[1])] = parts[2].strip() if len(parts) > 2 else "?"
    return km


def classify_rgb(name):
    """返回 'R'(MoE) / 'G'(attention/KV) / 'B'(其他)"""
    n = name.lower()
    if ("expert" in n or "topkgating" in n or "fused_moe" in n or
            "moe" in n or "gemm" in n or "cutlass" in n or "wmma" in n):
        return "R"  # MoE 专家权重 + 权重 GEMM
    if ("gemvx" in n or "attention" in n or "xqa" in n or "flashinfer" in n
            or "rope" in n or "delta_rule" in n or "gdn" in n or
            "mamba" in n or "kv_block" in n or "apply_write" in n or
            "kernel_mha" in n):
        return "G"  # attention / KV
    return "B"  # norm / elementwise / activation / fp4 / 其他


def main():
    prefix = "hbm_trace_v2_out.99818"
    data = np.fromfile(prefix + ".bin", dtype=REC_DTYPE)
    kernel_map = load_kernel_map(prefix + ".meta")

    layer = data["layer"].astype(np.int32)
    grid = data["grid"].astype(np.int64)
    addr = data["addr"].astype(np.int64)
    n_grids = int(grid.max()) + 1

    grid_layer = np.full(n_grids, -1, dtype=np.int32)
    seen = np.zeros(n_grids, dtype=bool)
    for i in range(len(grid)):
        g = int(grid[i])
        if not seen[g]:
            seen[g] = True
            grid_layer[g] = layer[i]
            if seen.all():
                break
    boundaries = [i for i in range(1, n_grids)
                  if grid_layer[i - 1] == 47 and grid_layer[i] == 0]
    k = len(boundaries)
    tok = np.full(n_grids, -1, dtype=np.int32)
    for i in range(k - 1):
        tok[boundaries[i]:boundaries[i + 1]] = i
    tok[boundaries[k - 1]:] = k - 1
    token_arr = tok[grid]
    m = token_arr == 1
    grid_t = grid[m]
    addr_t = addr[m]

    # kernel 类型（RGB 分类），按 grid 映射
    cls = np.array([classify_rgb(kernel_map.get(int(g), "?"))
                    for g in np.unique(grid_t)])

    # kernel 序号映射
    grids = np.unique(grid_t)
    grid_to_idx = {int(g): i for i, g in enumerate(sorted(grids))}
    kidx = np.array([grid_to_idx[int(g)] for g in grid_t])
    # 每条记录的 RGB 类别
    gcls = np.array([classify_rgb(kernel_map.get(int(g), "?"))
                     for g in grid_t])

    BASE = int(addr_t.min())
    offset = addr_t - BASE
    span = int(addr_t.max() - addr_t.min())
    ymax = span / 1e9
    off = offset.astype(np.float64) / 1e9

    nb_x = 600   # 像素放大：横轴 2929 kernel 合并到 600 桶
    nb_y = 80    # 像素放大：纵轴 91.7GB 分 80 桶（每桶约 1.1GB）

    def hist2d(cls_label):
        mcls = gcls == cls_label
        H, _, _ = np.histogram2d(kidx[mcls], off[mcls],
                                 bins=[nb_x, nb_y])
        return H

    Hr = hist2d("R")
    Hg = hist2d("G")
    Hb = hist2d("B")

    def norm(H):
        H = np.log1p(H)
        mx = H.max()
        return H / mx if mx > 0 else H

    R = norm(Hr)
    G = norm(Hg)
    B = norm(Hb)
    img = np.stack([R, G, B], axis=-1)

    fig, ax = plt.subplots(figsize=(18, 9))
    ax.imshow(np.transpose(img, (1, 0, 2)), origin="lower", aspect="auto",
              extent=[0, len(grids), 0, ymax], interpolation="nearest")
    ax.set_xlabel("Kernel 序号（时间顺序）", fontsize=13)
    ax.set_ylabel("地址偏移（GB，基址 0x%x）" % BASE, fontsize=12)
    ax.set_title("Token1 访存热度图（RGB 分色：红=MoE, 绿=attention/KV, 蓝=其他）",
                 fontsize=14)

    # 黑背景下坐标轴用白色
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

    # layer 边界
    layer_starts = {}
    for i, g in enumerate(sorted(grids)):
        l = int(grid_layer[int(g)])
        if l >= 0 and l not in layer_starts:
            layer_starts[l] = i
    for l in range(0, 48, 4):
        if l in layer_starts:
            x = layer_starts[l]
            ax.axvline(x, color="white", linewidth=0.4, alpha=0.4)
            ax.text(x, ymax * 1.01, f"L{l}", color="white", fontsize=7,
                    ha="center")

    # 图例
    import matplotlib.patches as mpatches
    legend = [mpatches.Patch(color="red", label="MoE 专家权重 (gemm/moe)"),
              mpatches.Patch(color="lime", label="attention / KV (gemv/attn)"),
              mpatches.Patch(color="blue", label="其他 (norm/elementwise)")]
    ax.legend(handles=legend, loc="upper right", fontsize=9,
              facecolor="black", labelcolor="white")

    fig.tight_layout()
    fig.savefig("token1_heatmap_rgb.png", dpi=140, facecolor="black")
    plt.close(fig)

    # 统计各类总量
    print("各类访存记录数:")
    print(f"  MoE (R): {int((gcls=='R').sum()):,} 条")
    print(f"  attention/KV (G): {int((gcls=='G').sum()):,} 条")
    print(f"  其他 (B): {int((gcls=='B').sum()):,} 条")
    print(f"图片已保存: token1_heatmap_rgb.png")


if __name__ == "__main__":
    main()
