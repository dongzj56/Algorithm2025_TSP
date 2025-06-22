import pandas as pd
import matplotlib.pyplot as plt
import math, os

csv_path = '/data/coding/Algorithm_TSP_2025/data/tsp_instances_dataset.csv'
save_dir = '/data/coding/Algorithm_TSP_2025/img/csv_result'
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

def extract_coords(row):
    """从一行中提取所有城市坐标"""
    n = int(row['Num_Cities'])
    xs, ys = [], []
    for i in range(1, n + 1):
        x_col, y_col = f'City_{i}_X', f'City_{i}_Y'
        if (x_col in row) and (y_col in row) and not (
            math.isnan(row[x_col]) or math.isnan(row[y_col])
        ):
            xs.append(row[x_col])
            ys.append(row[y_col])
    return xs, ys

# 可视化并保存前 5 个实例
for idx in range(min(5, len(df))):
    xs, ys = extract_coords(df.iloc[idx])

    plt.figure(figsize=(6, 5))
    plt.scatter(xs, ys, c='blue', s=40)
    for i, (x, y) in enumerate(zip(xs, ys)):
        plt.text(x + 0.4, y + 0.4, str(i), fontsize=8)

    plt.title(f"TSP Instance {idx} (n={len(xs)})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    out_path = os.path.join(save_dir, f'instance_{idx}.png')
    plt.savefig(out_path, dpi=300)   # 可根据需要调整分辨率
    plt.close()

print(f"已保存到: {os.path.abspath(save_dir)}")
