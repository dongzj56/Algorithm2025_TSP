# -*- coding: utf-8 -*-
# 批量绘制 TSPLIB 实例：有坐标直接画，无坐标→spring_layout 或 MDS
# 依赖：pip install tsplib95 matplotlib networkx scikit-learn

import tsplib95
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, Literal, Dict, Tuple, Union

import networkx as nx                # spring_layout
from sklearn.manifold import MDS      # 可选 MDS 降维


CoordDict = Dict[int, Tuple[float, float]]


def _coords_from_graph(prob: tsplib95.models.Problem,
                       method: Literal['spring', 'mds'] = 'spring') -> CoordDict:
    """若 TSP 无坐标，用图布局或 MDS 生成二维坐标。"""
    G = prob.get_graph()

    if method == 'spring':
        pos = nx.spring_layout(G, seed=42)              # 字典 {node: (x, y)}
    elif method == 'mds':
        dist_mat = [[prob.get_weight(i, j) for j in prob.get_nodes()]
                    for i in prob.get_nodes()]
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        xy = mds.fit_transform(dist_mat)
        pos = {node: tuple(xy[k]) for k, node in enumerate(prob.get_nodes())}
    else:
        raise ValueError('method 必须是 "spring" 或 "mds"')

    return {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}


def plot_tsp(tsp_file: Union[str, Path],
             show_index: bool = True,
             show_opt_tour: bool = True,
             figsize: tuple = (6, 6),
             point_color: str = 'tab:orange',
             line_color: str = '#455A64',
             save_dir: Union[str, Path] = None,
             fallback: Literal['spring', 'mds', 'skip'] = 'spring'):
    """
    单文件绘图。
    fallback: 当没有坐标时的处理方式
        - 'spring' : NetworkX spring_layout
        - 'mds'    : MDS 降维
        - 'skip'   : 打印提示并跳过
    """
    tsp_path = Path(tsp_file)
    prob = tsplib95.load(tsp_path)
    coords = prob.node_coords

    # -- 若无坐标则生成 --
    if not coords:
        if fallback == 'skip':
            print(f'[跳过] {tsp_path.name} 无坐标数据')
            return
        print(f'[生成] {tsp_path.name} 用 {fallback} 方式生成坐标')
        coords = _coords_from_graph(prob, method=fallback)

    # -- 绘图 --
    xs, ys = zip(*coords.values())
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xs, ys, c=point_color, s=20, zorder=3)

    if show_index:
        for cid, (x, y) in coords.items():
            ax.text(x, y, str(cid), fontsize=7,
                    ha='right', va='bottom', zorder=4)

    if show_opt_tour and prob.tours:
        path = [coords[i] for i in prob.tours[0]]
        ax.plot(*zip(*path), lw=0.8, c=line_color, zorder=2)

    ax.set_title(f'{prob.name}  (n={prob.dimension})')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()

    # -- 保存 --
    save_dir = Path(save_dir or Path(__file__).parent / 'img' / 'original')
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f'{tsp_path.stem}.png'
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'✔ 图已保存: {out_path.relative_to(Path.cwd())}')
    plt.close(fig)


def batch_plot_tsp(dir_path: Union[str, Path],
                   tsp_suffix: Iterable[str] = ('.tsp', '.TSP'),
                   **plot_kwargs):
    """批量绘图，自动处理无坐标实例。"""
    dir_path = Path(dir_path)
    assert dir_path.is_dir(), f'{dir_path} 不是有效目录'

    files = [p for p in dir_path.glob('*') if p.suffix in tsp_suffix]
    if not files:
        print('⚠ 目录下没有 *.tsp 文件')
        return

    for tsp_file in files:
        try:
            plot_tsp(tsp_file, **plot_kwargs)
        except Exception as e:
            print(f'[错误] {tsp_file.name}: {e}')


if __name__ == '__main__':
    ROOT = Path(__file__).parent
    DATA_DIR = ROOT / 'data' / 'tsplib'

    # 批处理：无坐标文件用 spring_layout 生成
    batch_plot_tsp(
        DATA_DIR,
        point_color='#FF9800',
        line_color='#37474F',
        fallback='spring'      # 'mds' 或 'skip' 亦可
    )
