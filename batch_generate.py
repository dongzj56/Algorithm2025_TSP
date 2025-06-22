import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable

# ---------- 生成单个实例 ----------
def generate_random_tsp(name: str,
                        n: int,
                        rng: np.random.Generator,
                        base_side: float = 100,
                        n_ref: int = 100,
                        save_dir: str | Path = rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025/data/generate"):
    """
    生成随机欧氏 TSP (.tsp) 并返回 (coords, tsp_path, side)。
    画布边长 side = base_side * sqrt(n / n_ref)
    """
    side = base_side * np.sqrt(n / n_ref)
    coords = rng.uniform(0, side, size=(n, 2)).tolist()

    lines = [
        f"NAME : {name}", "TYPE : TSP",
        "COMMENT : random generated instance",
        f"DIMENSION : {n}", "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        *[f"{i} {x:.3f} {y:.3f}" for i, (x, y) in enumerate(coords, 1)],
        "EOF",
    ]
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    tsp_path = save_dir / f"{name}.tsp"
    tsp_path.write_text("\n".join(lines))
    return coords, tsp_path, side


# ---------- 可视化 ----------
def visualize_tsp(coords,
                  name: str,
                  side: float,
                  save_dir: str | Path = rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025/img/generate"):
    """根据坐标绘图并保存 .png，返回 img_path。"""
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    xs, ys = zip(*coords)
    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, c='tab:blue', s=10)
    # 微偏移避免遮挡
    for i, (x, y) in enumerate(coords, 1):
        plt.text(x + .02 * side, y + .02 * side, str(i), fontsize=7)
    plt.title(f"Random TSP: {name} (n={len(coords)})")
    plt.xlim(0, side); plt.ylim(0, side)
    plt.axis('equal'); plt.axis('off'); plt.tight_layout()
    img_path = save_dir / f"{name}.png"
    plt.savefig(img_path, dpi=300); plt.close()
    return img_path


# ---------- 批量生成 ----------
def batch_generate(dataset_name: str,
                   sizes: Iterable[int],
                   instances_per_size: int = 3,
                   base_side: float = 100,
                   n_ref: int = 100,
                   data_root: str | Path = rf"C:\Users\dongz\Desktop\Algorithm_TSP_2025",
                   base_seed: int | None = 42):
    """
    批量生成随机 TSP 数据集，支持自适应画布大小。
    base_seed 用于复现；设 None 则每次随机。
    """
    rng_global = np.random.default_rng(base_seed)
    meta = []

    for n in sizes:
        for idx in range(instances_per_size):
            rng = np.random.default_rng(rng_global.integers(0, 2**32))
            tag = f"{dataset_name}_n{n}_{idx}"

            coords, tsp_path, side = generate_random_tsp(
                tag, n, rng, base_side, n_ref,
                save_dir=Path(data_root) / "data" / "generate")

            img_path = visualize_tsp(
                coords, tag, side,
                save_dir=Path(data_root) / "img" / "generate")

            meta.append((tag, n, side, tsp_path, img_path))

    # 保存索引
    catalog = Path(data_root) / "data" / "generate" / f"{dataset_name}_index.csv"
    catalog.write_text(
        "name,n,side,tsp_path,img_path\n" +
        "\n".join(f"{t},{n},{side:.1f},{p},{i}" for t, n, side, p, i in meta))
    print(f"[OK] 共生成 {len(meta)} 个实例，索引已保存：{catalog}")


# ----------------- 使用示例 -----------------
if __name__ == "__main__":
    # batch_generate(
    #     dataset_name       = "TSP_Generate",
    #     sizes              = [20, 50, 100, 500],
    #     instances_per_size = 200,
    #     base_side          = 100,   # n_ref=100 时画布边长=100
    #     n_ref              = 100,
    #     base_seed          = 3407
    # )
    # batch_generate(
    #     dataset_name       = "TSP_Generate",
    #     sizes              = [1000],
    #     instances_per_size = 100,
    #     base_side          = 100,   # n_ref=100 时画布边长=100
    #     n_ref              = 100,
    #     base_seed          = 3407
    # )
    batch_generate(
        dataset_name       = "TSP_Generate",
        sizes              = [5000],
        instances_per_size = 10,
        base_side          = 100,   # n_ref=100 时画布边长=100
        n_ref              = 100,
        base_seed          = 3407
    )
    batch_generate(
        dataset_name       = "TSP_Generate",
        sizes              = [10000],
        instances_per_size = 5,
        base_side          = 100,   # n_ref=100 时画布边长=100
        n_ref              = 100,
        base_seed          = 3407
    )
