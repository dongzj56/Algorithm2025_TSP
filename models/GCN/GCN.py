#!/usr/bin/env python3
# GCN.py —— 固定路径推理；自动补齐配置（最终版）

import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

class DotDict(defaultdict):
    __getattr__ = defaultdict.__getitem__
    __setattr__ = defaultdict.__setitem__
    __delattr__ = defaultdict.__delitem__

# ---------------- 本地硬编码 ----------------
DATA_DIR  = Path(r"C:\Users\dongz\Desktop\Algorithm_TSP_2025\data\generate\n20")
CKPT_PATH = Path(r"C:\Users\dongz\Desktop\Algorithm_TSP_2025\graph-convnet-tsp-master\logs\tsp20\best_val_checkpoint.tar")
NUM_NODES = 20
FALLBACK_CFG = dict(
    num_nodes = NUM_NODES, node_dim = 2, edge_dim = 1,
    hidden_dim = 128, num_layers = 12, readout = "mean",
    mlp_layers = 2,  mlp_hidden_dim = 128,
    voc_nodes_in = 0, voc_edges_in = 0, voc_nodes_out = 0, voc_edges_out = 0)
# ------------------------------------------------

def read_tsp(fp: Path) -> np.ndarray:
    lines = fp.read_text().splitlines()
    idx   = lines.index("NODE_COORD_SECTION")
    coords=[]
    for ln in lines[idx+1:]:
        p = ln.strip().split()
        if p[0].upper()=="EOF": break
        coords.append([float(p[1]), float(p[2])])
    return np.asarray(coords, dtype=np.float32)

def tour_length(tour, coords):
    return sum(np.linalg.norm(coords[tour[i]]-coords[tour[(i+1)%len(tour)]])
               for i in range(len(tour)))

def build_model():
    from models.gcn_model import ResidualGatedGCNModel
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg_dict = ckpt.get("config") or ckpt.get("cfg") or ckpt.get("args") or {}
    for k,v in FALLBACK_CFG.items():
        cfg_dict.setdefault(k, v)
    cfg = DotDict(lambda: None, **cfg_dict)
    model = ResidualGatedGCNModel(cfg, torch.float32, torch.long)
    model.load_state_dict(ckpt.get("model_state") or ckpt.get("state_dict") or ckpt,
                          strict=False)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model

# ---------- 关键改动：edge 加一个维度 -> [1,V,V,1] ----------
def coords_to_graph(coords: np.ndarray):
    node = torch.from_numpy(coords).unsqueeze(0)              # [1, V, 2]

    # 欧氏距离矩阵 [1, V, V, 1]
    edge = torch.cdist(node, node).unsqueeze(-1)              # [1, V, V, 1]

    n = coords.shape[0]
    send, recv = zip(*[(i, j) for i in range(n) for j in range(n) if i != j])
    send = torch.tensor(send).unsqueeze(0)                    # [1, E]
    recv = torch.tensor(recv).unsqueeze(0)                    # [1, E]

    y_edges = torch.zeros_like(send)                          # [1, E]
    edge_cw = torch.ones_like(send, dtype=torch.float32)      # [1, E]

    if torch.cuda.is_available():
        node, edge, send, recv, y_edges, edge_cw = [
            x.cuda() for x in (node, edge, send, recv, y_edges, edge_cw)
        ]

    # forward顺序：x_edges_coord, send_idx, recv_idx, x_nodes_coord, y_edges, edge_cw
    return edge, send, recv, node, y_edges, edge_cw

@torch.no_grad()
def evaluate(model):
    files=sorted(DATA_DIR.glob("*.tsp"))
    if not files:
        raise FileNotFoundError(f"{DATA_DIR} 下没有 .tsp 文件")
    lens,times=[],[]
    for fp in files:
        coords=read_tsp(fp)
        assert coords.shape[0]==NUM_NODES,f"{fp} 节点数≠{NUM_NODES}"
        graph=coords_to_graph(coords)
        t0=time.time()
        logits = model(*graph)                # ★若需 beam-search，请替换此行
        tour   = logits.argmax(-1).tolist()   # 仅占位
        dur_ms=(time.time()-t0)*1000
        L=tour_length(tour,coords)
        print(f"{fp.name:<25} | length {L:8.2f} | {dur_ms:7.2f} ms")
        lens.append(L);times.append(dur_ms)
    print(f"\n平均路径长度 : {np.mean(lens):.2f}")
    print(f"平均推理时间 : {np.mean(times):.2f} ms")

if __name__=="__main__":
    model=build_model()
    evaluate(model)
