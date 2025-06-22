#!/usr/bin/env python3
# ga_batch.py  ——  Genetic Algorithm 批量求解 TSP（前 10 个文件）

import random, math, time
from pathlib import Path
from typing import List, Tuple
import numpy as np


# --------------------------- GA 类 --------------------------- #
class GA:
    def __init__(self, num_city: int, num_total: int,
                 iteration: int, data: np.ndarray):
        self.num_city   = num_city
        self.num_total  = num_total
        self.iteration  = iteration
        self.location   = data

        self.ga_choose_ratio = 0.2
        self.mutate_ratio    = 0.05

        self.dis_mat = self._dist_mat(data)

        # ---------- 修正：先贪婪，再随机补足 ----------
        greedy_pop  = self._greedy_init()
        random_pop  = self._random_init(existing_len=len(greedy_pop))
        self.fruits = greedy_pop + random_pop
        # ---------------------------------------------

    # -------- 辅助函数 -------- #
    def _dist_mat(self, coords):
        diff = coords[:, None, :] - coords[None, :, :]
        d    = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(d, np.inf)
        return d

    def _path_len(self, route: List[int]) -> float:
        d = self.dis_mat
        return d[route[-1], route[0]] + sum(d[route[i], route[i+1]]
                                            for i in range(len(route)-1))

    def _fitness(self, route: List[int]) -> float:
        return 1.0 / self._path_len(route)

    def _greedy_init(self):
        init = []
        for start in range(min(self.num_total, self.num_city)):
            rest = list(range(self.num_city))
            cur  = start
            route= [cur]; rest.remove(cur)
            while rest:
                nxt = min(rest, key=lambda x: self.dis_mat[cur, x])
                route.append(nxt); rest.remove(nxt); cur = nxt
            init.append(route)
        return init

    # ---------- 修改: 传入已存在个体数 ----------
    def _random_init(self, existing_len: int):
        base = list(range(self.num_city))
        rand_pop = []
        while existing_len + len(rand_pop) < self.num_total:
            random.shuffle(base)
            rand_pop.append(base.copy())
        return rand_pop
    # --------------------------------------------------

    # ---- 遗传算子（选择、交叉、变异） ---- #
    def _select_parents(self, fitness):
        top = max(2, int(self.ga_choose_ratio * self.num_total))
        idx = np.argsort(-fitness)[:top]
        parents = [self.fruits[i] for i in idx]
        probs   = fitness[idx] / fitness[idx].sum()
        p1, p2  = np.random.choice(len(parents), 2, p=probs, replace=False)
        return parents[p1][:], parents[p2][:]

    def _crossover(self, x, y):
        n = len(x)
        a, b = sorted(random.sample(range(n), 2))
        seg_x, seg_y = x[a:b], y[a:b]

        def fill(parent, seg):
            rest = [g for g in parent if g not in seg]
            return rest[:a] + seg + rest[a:]
        return fill(x, seg_y), fill(y, seg_x)

    def _mutate(self, gene):
        if random.random() < self.mutate_ratio:
            a, b = sorted(random.sample(range(len(gene)), 2))
            gene[a:b] = reversed(gene[a:b])

    # ---------------- 主过程 ---------------- #
    def run(self) -> Tuple[float, List[int]]:
        best_len, best_route = math.inf, None

        for _ in range(self.iteration):
            fitness = np.array([self._fitness(r) for r in self.fruits])
            elite_idx = int(np.argmax(fitness))
            cur_len   = 1.0 / fitness[elite_idx]
            if cur_len < best_len:
                best_len, best_route = cur_len, self.fruits[elite_idx]

            new_pop = [self.fruits[elite_idx]]
            while len(new_pop) < self.num_total:
                x, y   = self._select_parents(fitness)
                cx, cy = self._crossover(x, y)
                self._mutate(cx); self._mutate(cy)
                new_pop.extend([cx, cy])
            self.fruits = new_pop[:self.num_total]

        return best_len, best_route


# ------------------- TSPLIB 读取 ------------------- #
def read_tsp(path: str | Path) -> np.ndarray:
    lines = Path(path).read_text().splitlines()
    idx   = next(i for i,l in enumerate(lines)
                 if l.strip().upper() == "NODE_COORD_SECTION")
    coords=[]
    for line in lines[idx+1:]:
        line=line.strip()
        if not line or line.upper()=="EOF": break
        coords.append([float(x) for x in line.split()[1:]])
    return np.array(coords)


# ------------------- 批量评测 ------------------- #
def batch_ga(dir_path: str | Path,
             num_total=50, iteration=500):
    dir_path = Path(dir_path)
    files    = sorted(dir_path.glob("*.tsp"))[:2]   # 取前 100 个
    if not files:
        print("目录下未找到 tsp 文件"); return

    total_len, total_time = 0.0, 0.0
    for f in files:
        coords = read_tsp(f)
        ga     = GA(len(coords), num_total, iteration, coords)

        t0 = time.process_time()
        best_len, _ = ga.run()
        cpu_ms = (time.process_time() - t0) * 1000

        total_len  += best_len
        total_time += cpu_ms
        print(f"{f.name:<25} | length {best_len:.2f} | CPU {cpu_ms:.2f} ms")

    n = len(files)
    print("\n----------- Summary (first 10 files) -----------")
    print(f"平均最优长度 : {total_len / n:.2f}")
    print(f"平均CPU时间 : {total_time / n:.2f} ms")


# ------------------------ main ------------------------ #
if __name__ == "__main__":
    batch_ga(
        dir_path  = rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025\data\generate\n10000",  # 修改为目标目录
        num_total = 50,
        iteration = 100
    )
