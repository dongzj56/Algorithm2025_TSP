#!/usr/bin/env python3
# dp_plot_save.py —— 动态规划求 TSP 并保存路线图

import math, os, time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ---------------- DP 类 ----------------
class DP(object):
    def __init__(self, num_city, data):
        self.num_city = num_city
        self.location = data
        self.dis_mat  = self._dist_mat()

    def _dist_mat(self):
        n = self.num_city
        loc = self.location
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                d[i, j] = np.inf if i == j else np.linalg.norm(loc[i] - loc[j])
        return d

    def _path_len(self, path, goback=True):
        d = self.dis_mat
        total = d[path[-1], path[0]] if goback else 0.0
        for i in range(len(path) - 1):
            total += d[path[i], path[i + 1]]
        return total

    # 最近插入启发式
    def run(self):
        rest = list(range(1, self.num_city))
        path = [0]
        length = 0.0
        while rest:
            c = rest.pop(0)
            if len(path) == 1:
                path.append(c)
                length = self._path_len(path)
                continue
            best_pos, best_len = 0, math.inf
            for i in range(len(path)):
                a = path[i - 1] if i else path[-1]
                b = path[i]
                new_len = length + self.dis_mat[c, a] + self.dis_mat[c, b] - self.dis_mat[a, b]
                if new_len < best_len:
                    best_len, best_pos = new_len, i
            path.insert(best_pos, c)
            length = best_len
        return self.location[path], length

# -------------- 读取 TSPLIB --------------
def read_tsp(path):
    lines = Path(path).read_text().splitlines()
    idx = lines.index('NODE_COORD_SECTION')
    coords = []
    for ln in lines[idx + 1:]:
        s = ln.strip().split()
        if s[0].upper() == 'EOF':
            break
        coords.append([float(s[1]), float(s[2])])
    return np.array(coords, dtype=np.float32)

# -------------- 主程序 -------------------
if __name__ == "__main__":
    tsp_file = rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025\data\tsp\att532.tsp"
    coords = read_tsp(tsp_file)

    model = DP(num_city=coords.shape[0], data=coords.copy())
    best_path, best_len = model.run()
    print(f"规划的路径长度: {best_len:.2f}")

    # 绘图
    cycle = np.vstack([best_path, best_path[0]])
    plt.figure(figsize=(6, 6))
    plt.scatter(cycle[:, 0], cycle[:, 1], c='blue', s=10, zorder=2)
    plt.plot(cycle[:, 0], cycle[:, 1], c='red', lw=0.8, zorder=1)
    plt.title(Path(tsp_file).stem + " 动态规划结果")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()

    # ---- 保存到 output 目录 ----
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (Path(tsp_file).stem + "_dp_route.png")
    plt.savefig(out_path, dpi=300)
    print(f"路线图已保存至 {out_path}")

    plt.show()
