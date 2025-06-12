import tsplib95
import matplotlib.pyplot as plt
from pathlib import Path


def plot_instance(tsp_path, show_index=False, tour=None, ax=None):
    """绘制 TSPLIB 实例散点图，若提供 tour 则画路线"""
    problem = tsplib95.load(tsp_path)
    coords = problem.node_coords  # {id: (x,y)}

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # 1. 画城市散点
    xs, ys = zip(*coords.values())
    ax.scatter(xs, ys, s=18, zorder=3)

    # 2. 标注索引（可选）
    if show_index:
        for idx, (x, y) in coords.items():
            ax.text(x, y, str(idx), fontsize=7, ha='right', va='bottom')

    # 3. 画巡回路线（可选）
    if tour is None and problem.tours:
        tour = problem.tours[0]  # TSPLIB 内置最优/已知解
    if tour:
        tour_xy = [coords[i] for i in tour]  # 转为坐标序列
        xs_t, ys_t = zip(*tour_xy)
        ax.plot(xs_t, ys_t, lw=0.8, zorder=2)

    ax.set_title(f'{problem.name}  (n={problem.dimension})')
    ax.set_aspect('equal')
    ax.axis('off')


# ---------- 批量可视化 ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
plot_instance('st70.tsp', show_index=True, ax=axes[0])
plot_instance('kroA100.tsp', show_index=False, ax=axes[1])
plt.tight_layout()
plt.show()
