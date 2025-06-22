import random, os
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------- 1. 生成随机 TSP ----------------
def generate_random_tsp(name: str,
                        n: int = 30,
                        low: int = 0,
                        high: int = 100,
                        save_dir: str = "/data/coding/Algorithm_TSP_2025/data/generate"):
    """
    生成随机欧氏 TSP 实例并保存为 .tsp。
    返回值：
        coords  -> [(x1,y1), …] 方便后续可视化
        tsp_path -> 写入的 .tsp 文件路径
    """
    coords = [(random.uniform(low, high), random.uniform(low, high)) for _ in range(n)]
    lines = [
        f"NAME : {name}",
        "TYPE : TSP",
        "COMMENT : random generated instance",
        f"DIMENSION : {n}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION"
    ] + [f"{i} {x:.3f} {y:.3f}" for i, (x, y) in enumerate(coords, 1)] + ["EOF"]

    os.makedirs(save_dir, exist_ok=True)
    tsp_path = Path(save_dir) / f"{name}.tsp"
    tsp_path.write_text("\n".join(lines))
    return coords, tsp_path

# ---------------- 2. 可视化并保存 ----------------
def visualize_tsp(coords,
                  name: str,
                  save_dir: str = "/data/coding/Algorithm_TSP_2025/img/generate"):
    """
    根据坐标可视化并把 PNG 保存到指定目录。
    """
    os.makedirs(save_dir, exist_ok=True)
    xs, ys = zip(*coords)
    
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, c='blue', s=40)
    for i, (x, y) in enumerate(coords, 1):
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=8)
    plt.title(f"Random TSP: {name} (n={len(coords)})")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.grid(True); plt.tight_layout()
    
    img_path = Path(save_dir) / f"{name}.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    return img_path

# ---------------- 3. 示例 ----------------
coords, tsp_path = generate_random_tsp("rand40_demo", n=40)
img_path = visualize_tsp(coords, "rand40_demo")

print("生成文件：", tsp_path)
print("可视化图片：", img_path)
