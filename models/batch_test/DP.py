import math, time, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt   # 若只批量评测可注释掉

# ---------------- DP 类（保持不变） ----------------
class DP(object):
    def __init__(self, num_city, num_total, iteration, data):
        self.num_city = num_city
        self.location = data
        self.dis_mat = self.compute_dis_mat(num_city, data)

    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a, b = location[i], location[j]
                dis_mat[i][j] = np.sqrt(np.sum((a - b) ** 2))
        return dis_mat

    def compute_pathlen(self, path):
        dis = 0.0
        for i in range(len(path) - 1):
            dis += self.dis_mat[path[i]][path[i + 1]]
        dis += self.dis_mat[path[-1]][path[0]]  # 回到起点
        return dis

    def run(self):
        rest = list(range(1, self.num_city))
        path = [0]
        while rest:
            c = rest.pop(0)
            if len(path) == 1:
                path.append(c)
                continue
            best_idx, best_inc = 0, math.inf
            for i in range(len(path)):
                a = path[i - 1]
                b = path[i]
                inc = self.dis_mat[a][c] + self.dis_mat[c][b] - self.dis_mat[a][b]
                if inc < best_inc:
                    best_inc, best_idx = inc, i
            path.insert(best_idx, c)
        best_len = self.compute_pathlen(path)
        return path, best_len


# ---------------- robust 读取 TSPLIB ----------------
def read_tsp(path):
    lines = Path(path).read_text().splitlines()
    idx = next((i for i,l in enumerate(lines) if l.strip().upper()=="NODE_COORD_SECTION"), None)
    if idx is None:
        raise ValueError(f"{path} 缺少 NODE_COORD_SECTION")
    coords = []
    for line in lines[idx+1:]:
        line=line.strip()
        if not line or line.upper()=="EOF": break
        parts=[float(p) for p in line.split() if p]
        coords.append(parts[1:])           # 丢弃序号
    return np.array(coords)


# ---------------- 批量评测 ----------------
# ---------------- 批量评测 ----------------
def batch_solve_tsp(tsp_dir: str | Path, csv_out: str | Path):
    tsp_dir  = Path(tsp_dir)
    results  = []
    tot_time = 0.0
    tot_len  = 0.0
    
    # 仅处理目录中排序后的前 100 个 .tsp 文件
    files = sorted(tsp_dir.glob("*.tsp"))[:2]
    if not files:
        print("目录下未找到 tsp 文件"); return

    for f in files:
        inst_start = time.perf_counter()

        coords = read_tsp(f)
        model  = DP(num_city=len(coords), num_total=0, iteration=0, data=coords)
        route, plen = model.run()

        inst_ms = (time.perf_counter() - inst_start) * 1000
        results.append((f.name, f"{inst_ms:.2f}", f"{plen:.2f}", " ".join(map(str, route))))
        tot_time += inst_ms
        tot_len  += plen
        print(f"{f.name:<15} | path={plen:.2f} | time={inst_ms:.2f} ms")

    # 其余写 CSV、打印平均值部分不变 …


    # 写入 CSV
    csv_out = Path(csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(["name", "time_ms", "path_len", "route"])
        writer.writerows(results)
    print(f"\n结果已保存: {csv_out}")

    # 平均统计
    avg_t = tot_time / len(results)
    avg_l = tot_len  / len(results)
    print(f"\n平均用时: {avg_t:.2f} ms   平均路径长度: {avg_l:.2f}")


# ---------------- 运行 ----------------
if __name__ == "__main__":
    batch_solve_tsp(
        tsp_dir = rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025\data\generate\n10000",   # ← tsp 文件目录
        csv_out = rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025\output\dp\n10000_results.csv"
    )
