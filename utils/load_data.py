import matplotlib.pyplot as plt
from pathlib import Path
import re, os

# ---------- 工具函数 ----------
def load_tsp_coords(file_path: Path) -> dict[int, tuple[float, float]]:
    coords, reading = {}, False
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading = True
                continue
            if line == "EOF" or not reading:
                if line == "EOF":
                    break
                continue
            idx, x, y = re.split(r"\s+", line)[:3]
            coords[int(idx)] = (float(x), float(y))
    return coords

def load_opt_tour(file_path: Path) -> list[int]:
    tour = []
    with open(file_path, "r") as f:
        for line in f:
            token = line.strip()
            if token.isdigit():
                tour.append(int(token))
            elif token in ("-1", "EOF"):
                break
    return tour

# ---------- 路径 ----------
tsp_path  = Path("/data/coding/Algorithm_TSP_2025/data/tsp/pr2392.tsp")
tour_path = Path("/data/coding/Algorithm_TSP_2025/data/tsp/pr2392.opt.tour")

# ---------- 数据 ----------
coords = load_tsp_coords(tsp_path)
n      = len(coords)
xs, ys = zip(*[coords[i] for i in range(1, n + 1)])

tour_nodes = load_opt_tour(tour_path)
tour_nodes.append(tour_nodes[0])                 # 闭合
tx, ty = zip(*[coords[k] for k in tour_nodes])

# ---------- 绘图 ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# 左：散点 + 标签
ax1.scatter(xs, ys, color="orange")
for idx, (x, y) in coords.items():
    ax1.text(x, y, str(idx), fontsize=8, ha="right", va="bottom")
ax1.set_title(f"berlin52  (n={n})")
ax1.set_aspect("equal")
ax1.axis("off")

# 右：最优巡回
ax2.plot(tx, ty, "-o", color="steelblue", markersize=4)
for idx, (x, y) in coords.items():
    ax2.text(x, y, str(idx), fontsize=8, ha="right", va="bottom")
ax2.set_title("berlin52 optimal tour")
ax2.set_aspect("equal")
ax2.axis("off")

plt.tight_layout()

# ---------- 保存 ----------
save_dir = Path("/data/coding/Algorithm_TSP_2025/img/result")
save_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(save_dir / "berlin52_scatter_and_tour.png", dpi=300)

# 如仍想在屏幕上查看，可保留显示
# plt.show()
