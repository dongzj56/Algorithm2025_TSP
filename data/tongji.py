#!/usr/bin/env python3
# tsp_size_scanner.py  ——  一键统计目录下所有 .tsp 文件的城市数量

"""
直接运行即可：
    python tsp_size_scanner.py

脚本会扫描 `SRC_DIR` 目录（含子目录可自行改为 **glob** 递归），
读取每个 .tsp 的城市点数量，按规模升序排序后写入 `OUT_CSV`。
"""

import csv
from pathlib import Path

# ----------- 一键配置：修改这两行即可 ----------- #
SRC_DIR = Path(rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025\data\tsp")        # 存放 .tsp 的目录
OUT_CSV = Path(rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025/output/tsp_sizes.csv")
# ---------------------------------------------- #


def count_city(file_path: Path) -> int:
    """优先解析 DIMENSION 行；若缺失则数 NODE_COORD_SECTION 中的节点行"""
    n = None
    with file_path.open() as fp:
        in_coord = False
        for line in fp:
            up = line.strip().upper()
            if up.startswith("DIMENSION"):
                # DIMENSION or DIMENSION : 280
                try:
                    n = int(line.split(":")[1])
                except Exception:
                    pass
                break
            if up == "NODE_COORD_SECTION":
                in_coord = True
                n = 0
                continue
            if in_coord:
                if up == "EOF":
                    break
                if line.strip():
                    n += 1
    if n is None:
        raise ValueError(f"{file_path.name}: 无法解析城市数量")
    return n


def scan_and_save(src_dir: Path, csv_out: Path):
    tsp_files = sorted(src_dir.glob("*.tsp"))      # 仅顶层；若需递归改为 rglob
    if not tsp_files:
        print(f"[WARN] {src_dir} 下未找到 .tsp 文件")
        return

    results = []
    for f in tsp_files:
        try:
            dim = count_city(f)
            results.append((f.name, dim))
        except Exception as e:
            print(f"[跳过] {f.name}: {e}")

    if not results:
        print("没有成功解析的文件"); return

    results.sort(key=lambda x: x[1])               # 按规模升序
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["file", "dimension"])
        writer.writerows(results)

    print(f"已统计 {len(results)} 个实例，结果保存至 {csv_out}")


if __name__ == "__main__":
    scan_and_save(SRC_DIR, OUT_CSV)
