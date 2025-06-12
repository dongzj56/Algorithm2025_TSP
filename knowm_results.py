# visualize_tsp.py
# pip install tsplib95 matplotlib

import tsplib95
import matplotlib.pyplot as plt
from pathlib import Path


def load_problem_and_route(tsp_path: Path):
    """返回 (Problem, route 或 None)。"""
    prob = tsplib95.load(tsp_path)

    # ① 尝试直接获取内嵌巡回
    route = prob.tours[0] if prob.tours else None

    # ② 用户没有 .opt.tour，则 route 可能为空
    if route and route[0] != route[-1]:
        route = route + [route[0]]

    return prob, route


def visualize(prob, route,
              point_color='#FF9800', line_color='#37474F',
              save_png: Path = None):
    coords = prob.node_coords
    xs, ys = zip(*coords.values())
    fig, ax = plt.subplots(figsize=(6, 6))

    # 城市散点
    ax.scatter(xs, ys, c=point_color, s=22, zorder=3)

    # 若有巡回则绘制并计算长度
    if route:
        path_xy = [coords[i] for i in route]
        ax.plot(*zip(*path_xy), lw=1, c=line_color, zorder=2)
        length = prob.trace_tours(route)[0]
        title  = f'{prob.name} | n={prob.dimension} | L={length:.1f}'
        print('最优巡回长度 L =', length)
        print('路线序列:', route)
    else:
        title = f'{prob.name} | n={prob.dimension} | 无最优巡回'
        print(f'⚠ {prob.name} 未找到 .opt.tour 或内嵌巡回，只绘制散点')

    # 统一美化
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    if save_png:
        fig.savefig(save_png, dpi=200, bbox_inches='tight')
        print('✅ 图已保存:', save_png.relative_to(Path.cwd()))
    plt.close(fig)


if __name__ == '__main__':
    ROOT = Path(__file__).parent
    DATA_DIR = ROOT / 'data' / 'tsplib'      # 存放 .tsp 的目录
    SAVE_DIR = ROOT / 'img' / 'result'       # 结果目录
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # 遍历目录下所有 .tsp 文件
    for tsp_file in DATA_DIR.glob('*.tsp'):
        prob, route = load_problem_and_route(tsp_file)
        out_png = SAVE_DIR / f'{tsp_file.stem}.png'
        visualize(prob, route, save_png=out_png)
