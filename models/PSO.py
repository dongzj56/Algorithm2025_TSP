#!/usr/bin/env python3
# pso_batch.py —— Particle Swarm Optimization 批量求解 TSP（前 10 文件）

import random, math, time
from pathlib import Path
from typing import List, Tuple
import numpy as np


# ------------------------------- PSO 类 ------------------------------- #
class PSO:
    def __init__(self, num_city: int, coords: np.ndarray,
                 swarm_size: int = 200, iter_max: int = 500):
        self.num_city   = num_city
        self.coords     = coords
        self.num        = swarm_size
        self.iter_max   = iter_max

        self.dis_mat = self._dist_mat(coords)

        greedy_pop  = self._greedy_init(self.num)
        random_pop  = self._random_init(existing=len(greedy_pop))
        self.swarm  = greedy_pop + random_pop          # 粒子群

        self.lengths = [self._path_len(p) for p in self.swarm]

        idx_best             = int(np.argmin(self.lengths))
        self.global_best     = self.swarm[idx_best]
        self.global_best_len = self.lengths[idx_best]

        self.local_best      = self.swarm.copy()
        self.local_best_len  = self.lengths.copy()

    # ----------- 距离与路径长度 ----------- #
    def _dist_mat(self, coords):
        diff = coords[:, None, :] - coords[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(d, np.inf)
        return d

    def _path_len(self, path: List[int]) -> float:
        d = self.dis_mat
        return d[path[-1], path[0]] + sum(d[path[i], path[i+1]]
                                          for i in range(len(path)-1))

    # ----------- 初始化 ----------- #
    def _greedy_init(self, k: int):
        pop = []
        for start in range(min(k, self.num_city)):
            rest, cur = list(range(self.num_city)), start
            path = [cur]; rest.remove(cur)
            while rest:
                cur = min(rest, key=lambda x: self.dis_mat[cur, x])
                path.append(cur); rest.remove(cur)
            pop.append(path)
        return pop

    def _random_init(self, existing: int):
        base, pop = list(range(self.num_city)), []
        while existing + len(pop) < self.num:
            random.shuffle(base)
            pop.append(base.copy())
        return pop

    # ----------- 交叉 + 变异 ----------- #
    def _cross(self, cur, best):
        n = self.num_city
        x, y = sorted(random.sample(range(n), 2))
        seg  = best[x:y]
        rest = [g for g in cur if g not in seg]
        cand1 = rest + seg
        cand2 = seg + rest
        return (cand1, self._path_len(cand1)) if self._path_len(cand1) < self._path_len(cand2) \
               else (cand2, self._path_len(cand2))

    def _mutate(self, path):
        a, b = sorted(random.sample(range(self.num_city), 2))
        path[a], path[b] = path[b], path[a]
        return path, self._path_len(path)

    # ---------------- PSO 迭代 ---------------- #
    def run(self) -> Tuple[float, List[int]]:
        best_len, best_path = self.global_best_len, self.global_best

        for _ in range(1, self.iter_max):
            for i, particle in enumerate(self.swarm):
                length = self.lengths[i]

                # 与个体最优交叉
                new_p, new_l = self._cross(particle, self.local_best[i])
                if new_l < length or random.random() < .1:
                    particle, length = new_p, new_l

                # 与全局最优交叉
                new_p, new_l = self._cross(particle, self.global_best)
                if new_l < length or random.random() < .1:
                    particle, length = new_p, new_l

                # 变异
                particle, length = self._mutate(particle)

                # 更新粒子
                self.swarm[i]  = particle
                self.lengths[i] = length

                # 更新个体最优
                if length < self.local_best_len[i]:
                    self.local_best_len[i] = length
                    self.local_best[i]    = particle

            # 更新全局最优
            idx = int(np.argmin(self.lengths))
            if self.lengths[idx] < best_len:
                best_len, best_path = self.lengths[idx], self.swarm[idx]
                self.global_best_len, self.global_best = best_len, best_path

        return best_len, best_path


# ----------------- 读取 TSPLIB ----------------- #
def read_tsp(path: str | Path) -> np.ndarray:
    lines = Path(path).read_text().splitlines()
    idx   = next(i for i,l in enumerate(lines) if l.strip().upper()=="NODE_COORD_SECTION")
    coords=[]
    for line in lines[idx+1:]:
        line=line.strip()
        if not line or line.upper()=="EOF": break
        coords.append([float(x) for x in line.split()[1:]])
    return np.array(coords)


# ----------------- 批量评测 ----------------- #
def batch_pso(dir_path: str | Path,
              swarm_size=200, iter_max=500):
    dir_path = Path(dir_path)
    files    = sorted(dir_path.glob("*.tsp"))[:3]   # 取前 100
    if not files:
        print("目录下未找到 tsp 文件"); return

    tot_len, tot_time = 0.0, 0.0
    for f in files:
        coords = read_tsp(f)
        solver = PSO(len(coords), coords, swarm_size, iter_max)

        t0 = time.process_time()
        best_len, _ = solver.run()
        cpu_ms = (time.process_time() - t0)*1000

        tot_len  += best_len
        tot_time += cpu_ms
        print(f"{f.name:<25} | length {best_len:.2f} | CPU {cpu_ms:.2f} ms")

    n = len(files)
    print("\n----------- Summary (first 10 files) -----------")
    print(f"平均最优长度 : {tot_len / n:.2f}")
    print(f"平均CPU时间 : {tot_time / n:.2f} ms")


# ------------------------- main ------------------------- #
if __name__ == "__main__":
    batch_pso(
        dir_path   = rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025\data\generate\n5000",   # 目标目录
        swarm_size = 50,
        iter_max   = 100
    )
