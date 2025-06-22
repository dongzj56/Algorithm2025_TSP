#!/usr/bin/env python3
# plot_att48.py —— 绘制 att48 最优巡回（蓝点+红线）

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------- 文件路径 ----------
base_dir = Path(r"C:\Users\dongz\Desktop\Algorithm_TSP_2025\data\tsp")
tsp_file  = base_dir / "att48.tsp"
tour_file = base_dir / "att48.opt.tour"

# ---------- 读取 .tsp -> 坐标 ----------
with open(tsp_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
idx = lines.index('NODE_COORD_SECTION\n')
coords = []
for ln in lines[idx+1:]:
    parts = ln.strip().split()
    if parts[0].upper() == 'EOF':
        break
    coords.append([float(parts[1]), float(parts[2])])
coords = np.array(coords, dtype=np.float32)   # [48,2]

# ---------- 读取 .tour -> 顺序 ----------
with open(tour_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
idx = lines.index('TOUR_SECTION\n')
order = []
for ln in lines[idx+1:]:
    n = int(ln.strip())
    if n == -1:         # -1 结束标志
        break
    order.append(n-1)   # tsplib 索引从 1 开始，减 1 转为 0 基
order = np.array(order, dtype=np.int32)       # [48]

# ---------- 组合巡回 ----------
tour_coords = coords[order]
tour_cycle  = np.vstack([tour_coords, tour_coords[0]])

# ---------- 绘图 ----------
plt.figure(figsize=(6,6))
plt.scatter(tour_cycle[:,0], tour_cycle[:,1], c='blue', s=15, zorder=2)
plt.plot(tour_cycle[:,0], tour_cycle[:,1], c='red', lw=0.8, zorder=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks([]); plt.yticks([])
plt.title('att48 最优巡回')
plt.tight_layout()

# ---------- 保存 ----------
os.makedirs('output', exist_ok=True)
out_path = Path('output/att48_route.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"路线图已保存至 {out_path}")

plt.show()
