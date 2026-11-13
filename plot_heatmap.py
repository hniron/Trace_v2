#!/usr/bin/env python
"""基于 token 1 的访存 trace 绘制热度图 + 时序图。

热度图：x=Kernel 序号（时间），y=地址偏移（GB），亮度=访存频率
时序图：几个热点地址区域的访存频率随 kernel 序号的变化
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from collections import Counter

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Micro Hei",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 自定义配色：浅色底（白）→ 深色高（深红），增强对比度
CMAP = LinearSegmentedColormap.from_list(
    "white_to_darkred",
    ["#ffffff", "#fde0dd", "#f4a582", "#d6604d", "#b2182b", "#67001f"])

REC_DTYPE = np.dtype([
    ('grid', '<u8'), ('cta', '<u4'), ('warp', '<u2'), ('opcode', '<u2'),
    ('addr', '<u8'), ('size', '<u4'), ('layer', '<u4'),
])


def load_token1(prefix):
    data = np.fromfile(prefix + ".bin", dtype=REC_DTYPE)
    layer = data["layer"].astype(np.int32)
    grid = data["grid"].astype(np.int64)
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
    return data[m], grid_layer, boundaries


def main():
    prefix = "hbm_trace_v2_out.99818"
    d, grid_layer, boundaries = load_token1(prefix)

    grid = d["grid"].astype(np.int64)
    addr = d["addr"].astype(np.int64)
    layer = d["layer"].astype(np.int32)

    # kernel 序号映射（grid -> 连续 0..N-1）
    grids = np.unique(grid)
    grid_to_idx = {int(g): i for i, g in enumerate(sorted(grids))}
    kidx = np.array([grid_to_idx[int(g)] for g in grid])

    # 地址偏移（GB），动态基址 = 该 token 最小地址
    BASE = int(addr.min())
    offset = addr - BASE
    span = int(addr.max() - addr.min())
    ymax = span / 1e9
    off = offset.astype(np.float64) / 1e9
    n_valid = len(grid)

    # ===== 热度图 =====
    nb_x = len(grids)
    nb_y = 1200
    H, _, _ = np.histogram2d(kidx, off, bins=[nb_x, nb_y])

    fig, ax = plt.subplots(figsize=(16, 9))
    Ht = H.T + 1
    im = ax.imshow(Ht, origin="lower", aspect="auto",
                   extent=[0, nb_x, 0, ymax], cmap=CMAP,
                   norm=LogNorm(vmin=1, vmax=Ht.max()))
    ax.set_xlabel("Kernel 序号（时间顺序）", fontsize=12)
    ax.set_ylabel("地址偏移（GB，基址 0x%x）" % BASE, fontsize=11)
    ax.set_title("Token1 访存热度图（x=Kernel 顺序, y=地址, 亮度=访存频率）",
                 fontsize=14)
    ax.set_facecolor("#ffffff")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("访存记录数（log 尺度）", fontsize=11)

    # 标注 layer 边界（每层第一个 kernel 的位置）
    layer_starts = {}
    for i, g in enumerate(sorted(grids)):
        l = int(grid_layer[int(g)])
        if l >= 0 and l not in layer_starts:
            layer_starts[l] = i
    for l in range(0, 48, 4):
        if l in layer_starts:
            x = layer_starts[l]
            ax.axvline(x, color="cyan", linewidth=0.6, alpha=0.5)
            ax.text(x, ymax + ymax * 0.02, f"L{l}", color="cyan", fontsize=8,
                    ha="center")

    fig.tight_layout()
    fig.savefig("token1_heatmap.png", dpi=130)
    plt.close(fig)
    print(f"热度图已保存 (记录 {n_valid:,}, 地址跨度 {ymax:.1f} GB)")

    # ===== 时序图：热点区域的频率变化 =====
    region = (offset // (512 * 1024 * 1024)).astype(np.int64)
    region_cnt = Counter(region.tolist())
    top_regions = [r for r, _ in region_cnt.most_common(5)]
    top_regions.sort()

    fig, axes = plt.subplots(len(top_regions), 1, figsize=(16, 12),
                             sharex=True)
    for idx, r in enumerate(top_regions):
        m = region == r
        Hk, _ = np.histogram(kidx[m], bins=nb_x, range=(0, nb_x))
        base_addr = BASE + r * 512 * 1024 * 1024
        ax = axes[idx] if len(top_regions) > 1 else axes
        ax.plot(np.arange(nb_x), Hk, linewidth=0.5, color="tab:blue")
        ax.set_ylabel(f"区域 0x{base_addr:x}\n访存记录数", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Kernel 序号（时间顺序）", fontsize=12)
    fig.suptitle("Token1 各热点地址区域的访存频率随时间变化", fontsize=14)
    fig.tight_layout()
    fig.savefig("token1_timeseries.png", dpi=130)
    plt.close(fig)
    print("时序图已保存")

    print("\n热点区域（512MB 桶）:")
    for r in top_regions:
        base_addr = BASE + r * 512 * 1024 * 1024
        print(f"  偏移 {r*512/1024:.0f}GB (0x{base_addr:x}): {region_cnt[r]:,} 条")


if __name__ == "__main__":
    main()
