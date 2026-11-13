#!/usr/bin/env python
"""按"地址里存的数据类型"分类的热度图。

地址分区（由该地址被哪类 kernel 访问主导决定）：
  红 = 权重区（MoE专家权重 gemm + QKV投影 gemv + 量化 fp4 + 路由 moe）
  绿 = KV cache 区（kernel_mha + KV 管理）
  蓝 = 其他区（DeltaNet 状态 + 激活 elementwise + norm 等）
深浅 = 该地址在该时刻的访问频率（log 归一化，高=深、低=浅、无=白）。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches

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


def kernel_category(name):
    """返回 'weight' / 'kv' / 'other'"""
    n = name.lower()
    # KV cache 读写（真正的 KV）
    if "kernel_mha" in n or "kv_block" in n or "apply_write" in n:
        return "kv"
    # 权重：MoE专家 + QKV投影 + 量化 + 路由
    if ("expert" in n or "topkgating" in n or "fused_moe" in n or "moe" in n
            or "gemm" in n or "gemvx" in n or "cutlass" in n or "wmma" in n
            or "fp4" in n or "cvt" in n or "quant" in n):
        return "weight"
    # 其他：DeltaNet状态 + 激活 + norm
    return "other"


def main():
    prefix = "hbm_trace_v2_out.99818"
    data = np.fromfile(prefix + ".bin", dtype=REC_DTYPE)
    kmap = load_kernel_map(prefix + ".meta")

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
    m = tok[grid] == 1
    grid_t = grid[m]
    addr_t = addr[m]

    # kernel 类别
    grids = np.unique(grid_t)
    grid_to_idx = {int(g): i for i, g in enumerate(sorted(grids))}
    kidx = np.array([grid_to_idx[int(g)] for g in grid_t])
    gcat = np.array([kernel_category(kmap.get(int(g), "?")) for g in grid_t])

    BASE = int(addr_t.min())
    offset = addr_t - BASE
    span = int(addr_t.max() - addr_t.min())
    ymax = span / 1e9
    off = offset.astype(np.float64)

    nb_x = 800          # 横轴：2929 kernel 合并到 800 桶（可压缩，不会混色）
    nb_y = 400          # 纵轴：91.7GB 分 400 个地址桶（每桶约 230MB）
    bucket_size = span / nb_y
    abucket = (off // bucket_size).astype(np.int64)
    abucket = np.clip(abucket, 0, nb_y - 1)

    # 1. 每个地址桶的主导类型
    #    分别统计三类记录数
    bucket_type = np.full(nb_y, "empty", dtype=object)
    type_names = ["weight", "kv", "other"]
    cnt_by_type = {t: np.zeros(nb_y, dtype=np.int64) for t in type_names}
    for t in type_names:
        mt = gcat == t
        bc = np.bincount(abucket[mt], minlength=nb_y)
        cnt_by_type[t] = bc
    total_bucket = sum(cnt_by_type[t] for t in type_names)
    for b in range(nb_y):
        if total_bucket[b] == 0:
            continue
        counts = {t: cnt_by_type[t][b] for t in type_names}
        bucket_type[b] = max(counts, key=counts.get)

    # 2. 2D 直方图（kernel桶 × 地址桶）
    kbin = (kidx * nb_x // len(grids)).astype(np.int64)
    kbin = np.clip(kbin, 0, nb_x - 1)
    H, _, _ = np.histogram2d(kbin, abucket, bins=[nb_x, nb_y])
    Ht = H.T  # (nb_y, nb_x)  行=地址桶, 列=kernel桶

    # 3. 上色：按地址桶类型选色系，深浅=频率
    Hlog = np.log1p(Ht)
    mx = Hlog.max()
    Hnorm = Hlog / mx if mx > 0 else Hlog

    cmaps = {"weight": cm.Reds, "kv": cm.Greens, "other": cm.Blues}
    img = np.ones((nb_y, nb_x, 3))  # 默认白底
    for b in range(nb_y):
        t = bucket_type[b]
        if t == "empty":
            continue
        cmap = cmaps[t]
        vals = Hnorm[b, :]
        img[b, :, :] = cmap(vals)[:, :3]

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(img, origin="lower", aspect="auto",
              extent=[0, len(grids), 0, ymax], interpolation="nearest")
    ax.set_xlabel("Kernel 序号（时间顺序）", fontsize=13)
    ax.set_ylabel("地址偏移（GB，基址 0x%x）" % BASE, fontsize=12)
    ax.set_title("Token1 访存热度图（按地址数据类型分色，深浅=访问频率）",
                 fontsize=15)
    ax.tick_params(labelsize=10)

    # layer 边界
    layer_starts = {}
    for i, g in enumerate(sorted(grids)):
        l = int(grid_layer[int(g)])
        if l >= 0 and l not in layer_starts:
            layer_starts[l] = i
    for l in range(0, 48, 4):
        if l in layer_starts:
            x = layer_starts[l]
            ax.axvline(x, color="black", linewidth=0.5, alpha=0.4)
            ax.text(x, ymax * 1.005, f"L{l}", color="black", fontsize=8,
                    ha="center")

    # 图例
    legend = [mpatches.Patch(color=cm.Reds(0.8),
                             label="权重区（MoE专家+QKV投影+量化）"),
              mpatches.Patch(color=cm.Greens(0.8), label="KV cache 区"),
              mpatches.Patch(color=cm.Blues(0.8), label="其他区（状态+激活+norm）")]
    ax.legend(handles=legend, loc="upper right", fontsize=10,
              framealpha=0.95)

    fig.tight_layout()
    fig.savefig("token1_heatmap_region.png", dpi=140, facecolor="white")
    plt.close(fig)

    # 输出地址分区表
    print("地址分区（按主导访问类型）:")
    prev = None
    seg_start = 0
    for b in range(nb_y + 1):
        cur = bucket_type[b] if b < nb_y else None
        if cur != prev:
            if prev is not None and prev != "empty":
                print(f"  0x{ int(BASE + seg_start*bucket_size):x} ~ "
                      f"0x{ int(BASE + b*bucket_size):x} "
                      f"({seg_start*bucket_size/1e9:.1f}-{b*bucket_size/1e9:.1f}GB): {prev}")
            seg_start = b
            prev = cur
    print("\n图片已保存: token1_heatmap_region.png")
    print(f"  权重区记录: {int(cnt_by_type['weight'].sum()):,}")
    print(f"  KV区记录:   {int(cnt_by_type['kv'].sum()):,}")
    print(f"  其他区记录: {int(cnt_by_type['other'].sum()):,}")


if __name__ == "__main__":
    main()
