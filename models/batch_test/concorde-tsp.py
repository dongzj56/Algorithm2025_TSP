"""
Concorde TSP 批量求解脚本  
依赖：pip install pyconcorde
"""

import time, csv
from pathlib import Path
import numpy as np
from concorde.tsp import TSPSolver


# ------------------- TSPLIB 坐标读取 -------------------
def read_tsp(file_path: str | Path) -> np.ndarray:
    """返回形状 (n,2) 的坐标数组"""
    lines = Path(file_path).read_text().splitlines()
    idx   = next((i for i, l in enumerate(lines)
                  if l.strip().upper() == "NODE_COORD_SECTION"), None)
    if idx is None:
        raise ValueError(f"{file_path} 缺少 NODE_COORD_SECTION")
    coords = []
    for line in lines[idx + 1:]:
        line = line.strip()
        if not line or line.upper() == "EOF":
            break
        parts = [float(p) for p in line.split() if p]
        coords.append(parts[1:])          # 丢弃节点编号
    return np.array(coords, dtype=float)


# ------------------- 批量调用 Concorde -------------------
def batch_solve_concorde(tsp_dir: str | Path,
                         csv_out: str | Path,
                         norm: str = "EUC_2D"):
    """
    tsp_dir : 存放 *.tsp 的目录
    csv_out : 结果 CSV 路径
    norm    : 距离度量，同 Concorde；常用 EUC_2D / GEO / ATT 等
    """
    tsp_dir = Path(tsp_dir)
    files   = sorted(tsp_dir.glob("*.tsp"))
    if not files:
        print("目录下未找到 tsp 文件"); return

    rows, sum_time, sum_len = [], 0.0, 0.0

    for f in files:
        coords = read_tsp(f)
        xs, ys = coords[:, 0], coords[:, 1]

        t0 = time.perf_counter()
        solver   = TSPSolver.from_data(xs, ys, norm=norm)
        solution = solver.solve()
        t_ms     = (time.perf_counter() - t0) * 1000

        tour     = solution.tour  # 巡回顺序（从 0 开始）
        length   = solution.optimal_value

        rows.append((f.name, f"{t_ms:.2f}", f"{length:.2f}",
                     " ".join(map(str, tour))))
        sum_time += t_ms
        sum_len  += length

        print(f"{f.name:<20} | 路径长 {length:.2f} | 时间 {t_ms:.2f} ms")

    # 写 CSV
    csv_out = Path(csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["name", "time_ms", "path_len", "route"])
        writer.writerows(rows)
    print(f"\n结果已保存到: {csv_out}")

    # 输出平均值
    avg_t = sum_time / len(rows)
    avg_l = sum_len  / len(rows)
    print(f"\n平均路径长度: {avg_l:.2f}    平均耗时: {avg_t:.2f} ms")


# ---------------------- 运行示例 ----------------------
if __name__ == "__main__":
    batch_solve_concorde(
        tsp_dir = "/data/coding/Algorithm_TSP_2025/data/generate/n20",       # tsp 目录
        csv_out = "/data/coding/Algorithm_TSP_2025/output/concorde/n20_results.csv"
    )
