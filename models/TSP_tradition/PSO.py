# -*- coding: utf-8 -*-
# pso_tsp.py —— PSO 求解 TSP，并分别保存「路径图」和「收敛曲线」

import random, math, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


# ---------------- PSO 类（与之前一致） ----------------
class PSO(object):
    def __init__(self, num_city, data):
        self.iter_max = 500
        self.num      = 200
        self.num_city = num_city
        self.location = data

        self.dis_mat   = self.compute_dis_mat(num_city, self.location)
        self.particals = self.greedy_init(self.dis_mat, self.num, num_city)
        self.lenths    = self.compute_paths(self.particals)

        init_len  = min(self.lenths)
        init_idx  = self.lenths.index(init_len)
        self.global_best     = self.particals[init_idx]
        self.global_best_len = init_len
        self.local_best      = self.particals.copy()
        self.local_best_len  = self.lenths.copy()
        self.best_l, self.best_path = init_len, self.global_best

        self.iter_x = [0]
        self.iter_y = [init_len]

    # ---------- 初始化 ----------
    def greedy_init(self, dmat, num_total, n):
        res = []; start = 0
        for _ in range(num_total):
            rest = list(range(n))
            if start >= n:
                start = np.random.randint(0, n)
                res.append(res[start].copy()); continue
            cur = start; rest.remove(cur); path=[cur]
            while rest:
                nxt = min(rest, key=lambda x: dmat[cur][x])
                path.append(nxt); rest.remove(nxt); cur = nxt
            res.append(path); start += 1
        return res

    # ---------- 距离与适应度 ----------
    @staticmethod
    def compute_dis_mat(n, loc):
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                d[i, j] = np.inf if i == j else np.linalg.norm(loc[i] - loc[j])
        return d

    def compute_pathlen(self, path, dmat):
        return dmat[path[-1]][path[0]] + \
               sum(dmat[path[i]][path[i + 1]] for i in range(len(path) - 1))

    def compute_paths(self, paths):
        return [self.compute_pathlen(p, self.dis_mat) for p in paths]

    # ---------- 交叉 / 变异 ----------
    def cross(self, cur, best):
        x, y = sorted(random.sample(range(self.num_city), 2))
        seg = best[x:y]; rest = [p for p in cur if p not in seg]
        cand1, cand2 = rest + seg, seg + rest
        l1 = self.compute_pathlen(cand1, self.dis_mat)
        l2 = self.compute_pathlen(cand2, self.dis_mat)
        return (cand1, l1) if l1 < l2 else (cand2, l2)

    def mutate(self, path):
        x, y = sorted(random.sample(range(self.num_city), 2))
        path = path.copy(); path[x], path[y] = path[y], path[x]
        return path, self.compute_pathlen(path, self.dis_mat)

    # ---------- 主 PSO 循环 ----------
    def eval_particals(self):
        min_len = min(self.lenths); idx = self.lenths.index(min_len)
        if min_len < self.global_best_len:
            self.global_best_len, self.global_best = min_len, self.particals[idx]

        for i, l in enumerate(self.lenths):
            if l < self.local_best_len[i]:
                self.local_best_len[i], self.local_best[i] = l, self.particals[i]

    def pso(self):
        for it in range(1, self.iter_max):
            for i, path in enumerate(self.particals):
                cur_len = self.lenths[i]
                for target in (self.local_best[i], self.global_best):
                    new_p, new_l = self.cross(path, target)
                    if new_l < cur_len or random.random() < 0.1:
                        path, cur_len = new_p, new_l
                path, cur_len = self.mutate(path)
                self.particals[i], self.lenths[i] = path, cur_len

            self.eval_particals()
            if self.global_best_len < self.best_l:
                self.best_l, self.best_path = self.global_best_len, self.global_best

            self.iter_x.append(it)
            self.iter_y.append(self.best_l)
            print(it, self.best_l)
        return self.best_l, self.best_path

    def run(self):
        best_len, best_path = self.pso()
        return self.location[best_path], best_len


# ---------------- 读取 TSPLIB ----------------
def read_tsp(path):
    lines = Path(path).read_text().splitlines()
    idx   = lines.index('NODE_COORD_SECTION')
    coords = []
    for ln in lines[idx+1:]:
        s = ln.strip().split()
        if s[0] == 'EOF': break
        coords.append([float(s[1]), float(s[2])])
    return np.array(coords, dtype=np.float32)


# ---------------- 主程序 ----------------
if __name__ == '__main__':
    tsp_path = rf'C:\Users\dongz\Desktop\Algorithm_TSP_2025\data\tsp\nrw1379.tsp'
    coords   = read_tsp(tsp_path)

    model = PSO(num_city=coords.shape[0], data=coords.copy())
    best_coords, best_len = model.run()
    print(f"最优路径长度: {best_len:.2f}")

    # ---------- 路径图 ----------
    cycle = np.vstack([best_coords, best_coords[0]])

    fig_path = plt.figure(figsize=(6, 6))
    ax1 = fig_path.add_subplot(111)
    ax1.scatter(cycle[:, 0], cycle[:, 1], c='blue', s=10, zorder=2)
    ax1.plot(cycle[:, 0], cycle[:, 1], c='red', lw=0.8, zorder=1)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_title('规划结果')
    fig_path.tight_layout()

    # ---------- 收敛曲线 ----------
    fig_curve = plt.figure(figsize=(10, 6))
    ax2 = fig_curve.add_subplot(111)
    ax2.plot(model.iter_x, model.iter_y, c='black')
    ax2.set_title('收敛曲线')
    ax2.set_xlabel('Iteration'); ax2.set_ylabel('Best Length')
    fig_curve.tight_layout()

    # ---------- 保存 ----------
    os.makedirs('output', exist_ok=True)
    fig_path.savefig('output/route.png', dpi=300, bbox_inches='tight')
    fig_curve.savefig('output/convergence.png', dpi=300, bbox_inches='tight')
    print('route.png 与 convergence.png 已保存到 output/')

    plt.show()   # 如不想弹窗，可注释掉
