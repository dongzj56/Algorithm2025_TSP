#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量把 tap/ 目录下的 EXPLICIT-Matrix TSP 转成 EUC_2D 坐标格式 (.tsp)。
生成文件写入 newtsp/，文件名加后缀 _coord.tsp。
支持 FULL_MATRIX, LOWER/UPPER_ROW, LOWER/UPPER_DIAG_ROW 等常见矩阵格式。

用法：
    python matrix2coord_batch.py
"""
import re, os, sys
import numpy as np
from pathlib import Path
from sklearn.manifold import MDS          # classical metric MDS

SRC_DIR = Path(rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025\data\tsp")                     # 源目录
DST_DIR = Path(rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025\data\tsp-new")                  # 目标目录
DST_DIR.mkdir(exist_ok=True)

# ---------- 工具函数 ----------
def read_header(lines, key):
    """大小写无关地提取头部字段 KEY 的第一个 token，兼容 'KEY: v' / 'KEY v'。"""
    key_u = key.upper()
    for ln in lines:
        ln_u = ln.upper().lstrip()
        if not ln_u.startswith(key_u):
            continue
        # 去掉 KEY 后的剩余串，再剥离前导 : 或空格
        rest = ln_u[len(key_u):].lstrip(" :\t")
        if rest:
            return rest.split()[0]
    return None

def collect_numbers(lines):
    """把若干行里的纯数字 token 全部抓出来为 float 列表。"""
    num_pat = re.compile(r"-?\d+(?:\.\d+)?")
    return [float(tok) for ln in lines for tok in num_pat.findall(ln)]

def build_matrix(nums, n, fmt):
    """根据 EDGE_WEIGHT_FORMAT 把数字列表还原成 n×n 对称矩阵。"""
    M = np.zeros((n, n))
    if fmt == "FULL_MATRIX":
        return np.array(nums).reshape((n, n))
    k = 0
    if fmt in {"LOWER_DIAG_ROW", "LOWER_ROW"}:
        for i in range(n):
            rng = range(i + 1) if "DIAG" in fmt else range(i)
            for j in rng:
                M[i, j] = M[j, i] = nums[k]; k += 1
    elif fmt in {"UPPER_DIAG_ROW", "UPPER_ROW"}:
        for i in range(n):
            rng = range(i, n) if "DIAG" in fmt else range(i + 1, n)
            for j in rng:
                M[i, j] = M[j, i] = nums[k]; k += 1
    else:
        raise NotImplementedError(f"暂不支持格式 {fmt}")
    np.fill_diagonal(M, 0)
    return M

def classical_mds(D, dim=2):
    """经典 MDS：把距离矩阵嵌入 dim 维欧氏空间。"""
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (D ** 2) @ H
    w, v = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:dim]
    L = np.diag(np.sqrt(np.clip(w[idx], 0, None)))
    return v[:, idx] @ L

def scale_and_round(X, bound=1000):
    """线性缩放到 [0,bound] 后取整，保持相对几何形状。"""
    X -= X.min(axis=0)
    X *= bound / (X.ptp(axis=0).max() + 1e-9)
    return np.rint(X).astype(int)

def write_euc2d(name, coords, out_path):
    n = coords.shape[0]
    lines = [
        f"NAME : {name}_coord",
        f"COMMENT : derived from {name} by classical MDS",
        "TYPE : TSP",
        f"DIMENSION : {n}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    lines += [f"{i+1} {x} {y}" for i, (x, y) in enumerate(coords)]
    lines.append("EOF")
    out_path.write_text("\n".join(lines), encoding="utf-8")

# ---------- 主流程 ----------
total, done = 0, 0
for tsp in SRC_DIR.rglob("*.tsp"):
    total += 1
    text = tsp.read_text(encoding="utf-8", errors="ignore").splitlines()
    if "EXPLICIT" not in "".join(text).upper():
        continue                       # 跳过坐标类或函数类
    n_str = read_header(text, "DIMENSION")
    if not n_str:
        print(f"[跳过] {tsp.name}: 缺少 DIMENSION")
        continue
    n   = int(n_str)
    fmt = read_header(text, "EDGE_WEIGHT_FORMAT")
    if not fmt:
        fmt = "FULL_MATRIX"            # 部分早期实例没写格式
    fmt = fmt.upper()

    # 定位数据段
    try:
        start = next(i for i, l in enumerate(text) if "EDGE_WEIGHT_SECTION" in l.upper()) + 1
    except StopIteration:
        print(f"[跳过] {tsp.name}: 找不到 EDGE_WEIGHT_SECTION"); continue
    try:
        end = next(i for i, l in enumerate(text) if l.strip().upper() == "EOF")
    except StopIteration:
        end = len(text)
    nums = collect_numbers(text[start:end])
    try:
        D = build_matrix(nums, n, fmt)
    except Exception as e:
        print(f"[跳过] {tsp.name}: {e}"); continue

    # 嵌入坐标 + 写文件
    X = classical_mds(D, dim=2)
    coords = scale_and_round(X, bound=1000)
    dst_file = DST_DIR / f"{tsp.stem}_coord.tsp"
    write_euc2d(tsp.stem, coords, dst_file)
    done += 1
    print(f"✓ {tsp.name} → {dst_file.name}")

print(f"\n转换完成：成功 {done} / 共 {total} 个 .tsp")
