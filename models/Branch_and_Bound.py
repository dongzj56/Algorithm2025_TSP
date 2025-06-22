import math, time, csv
from queue import PriorityQueue
from pathlib import Path
import numpy as np

# ------------------------ Branch & Bound ------------------------
class Node:
    def __init__(self, level=None, path=None, bound=None):
        self.level = level
        self.path  = path
        self.bound = bound
    def __lt__(self, other):      # 让 PriorityQueue 按 bound 升序
        return self.bound < other.bound

class BB_TSP:
    """分支界限法求 TSP（最小插入下界）"""
    def __init__(self, coords: np.ndarray):
        self.location = coords
        self.n        = len(coords)
        self.dis_mat  = self._dist_mat(coords)

    @staticmethod
    def _dist_mat(coords):
        n = len(coords)
        dm = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    dm[i, j] = np.linalg.norm(coords[i] - coords[j])
                else:
                    dm[i, j] = np.inf
        return dm

    # ---------- 公共辅助 ----------
    def _tour_len(self, tour):
        dm = self.dis_mat
        return sum(dm[tour[i], tour[i+1]] for i in range(len(tour)-1)) + dm[tour[-1], tour[0]]

    def _bound(self, path):
        n, dm = self.n, self.dis_mat
        last  = path[-1]
        rest  = [i for i in range(n) if i not in path]
        bnd   = sum(dm[path[i], path[i+1]] for i in range(len(path)-1))
        if rest:
            bnd += min(dm[last, r]   for r in rest)              # outgoing
            for r in rest:
                others = [path[0]] + [x for x in rest if x != r]
                bnd   += min(dm[r, k] for k in others)           # minimal outgoing from rest
        return bnd

    # ---------- 主算法 ----------
    def solve(self, start=0):
        best_len = math.inf
        best_tour = []

        root = Node(level=0, path=[start], bound=0)
        root.bound = self._bound(root.path)

        pq = PriorityQueue()
        pq.put(root)

        while not pq.empty():
            v = pq.get()
            if v.bound >= best_len:   # 剪枝
                continue
            if v.level == self.n - 1: # 已构成完整回路
                tour = v.path + [start]
                length = self._tour_len(tour[:-1])
                if length < best_len:
                    best_len  = length
                    best_tour = tour[:-1]
                continue

            for city in range(self.n):
                if city in v.path:
                    continue
                new_path  = v.path + [city]
                new_node  = Node(level=v.level+1, path=new_path)
                new_node.bound = self._bound(new_path)
                if new_node.bound < best_len:
                    pq.put(new_node)

        return best_tour, best_len

# --------------------- TSPLIB 解析 ---------------------
def read_tsp(file_path: str | Path) -> np.ndarray:
    lines = Path(file_path).read_text().splitlines()
    idx   = next((i for i,l in enumerate(lines) if l.strip().upper()=="NODE_COORD_SECTION"), None)
    if idx is None:
        raise ValueError(f"{file_path} 缺少 NODE_COORD_SECTION")
    coords = []
    for line in lines[idx+1:]:
        line = line.strip()
        if line.upper() == "EOF" or not line:
            break
        parts = [float(p) for p in line.split() if p]
        coords.append(parts[1:])      # 丢弃节点编号
    return np.array(coords, dtype=float)

# ------------------- 批量处理并统计 -------------------
def batch_solve_bb(tsp_dir: str|Path, csv_out: str|Path):
    tsp_dir  = Path(tsp_dir)
    files    = sorted(tsp_dir.glob("*.tsp"))
    if not files:
        print("目录下未找到 tsp 文件"); return

    results, sum_time, sum_len = [], 0.0, 0.0

    for f in files:
        coords    = read_tsp(f)
        solver    = BB_TSP(coords)
        t0        = time.perf_counter()
        tour, lg  = solver.solve()
        cost_ms   = (time.perf_counter() - t0) * 1000
        results.append((f.name, f"{cost_ms:.2f}", f"{lg:.2f}", " ".join(map(str, tour))))
        sum_time += cost_ms
        sum_len  += lg
        print(f"{f.name:<20} | 路径长 {lg:.2f} | 时间 {cost_ms:.2f} ms")

    # 写出 CSV
    csv_out = Path(csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["name", "time_ms", "path_len", "route"])
        writer.writerows(results)
    print(f"\n结果已保存到: {csv_out}")

    # 平均值
    avg_time = sum_time / len(results)
    avg_len  = sum_len  / len(results)
    print(f"\n平均路径长度: {avg_len:.2f}    平均耗时: {avg_time:.2f} ms")

# ---------------------- 运行 ----------------------
if __name__ == "__main__":
    batch_solve_bb(
        tsp_dir = "/data/coding/Algorithm_TSP_2025/data/generate/n20",        # tsp 目录
        csv_out = "/data/coding/Algorithm_TSP_2025/output/bb/n20_results.csv" # 结果 CSV
    )
