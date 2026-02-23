# src/ufb/features/graph_feats.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class UndirectedEdges:
    src: np.ndarray  # int32
    dst: np.ndarray  # int32
    deg: np.ndarray  # float32 degree per node

def build_undirected_edges(edge_index: pd.DataFrame, n_nodes: int) -> UndirectedEdges:
    a = edge_index["from_node"].to_numpy(np.int32, copy=False)
    b = edge_index["to_node"].to_numpy(np.int32, copy=False)

    # undirected: add both directions
    src = np.concatenate([a, b]).astype(np.int32, copy=False)
    dst = np.concatenate([b, a]).astype(np.int32, copy=False)

    deg = np.bincount(src, minlength=n_nodes).astype(np.float32, copy=False)
    return UndirectedEdges(src=src, dst=dst, deg=deg)

def neighbor_mean(values: np.ndarray, edges: UndirectedEdges) -> np.ndarray:
    """
    values: shape (n_nodes,), float32
    returns: shape (n_nodes,), float32 mean of neighbor values (0 if deg=0)
    """
    src, dst = edges.src, edges.dst
    sums = np.bincount(src, weights=values[dst], minlength=values.shape[0]).astype(np.float32, copy=False)
    out = np.zeros_like(sums, dtype=np.float32)
    mask = edges.deg > 0
    out[mask] = sums[mask] / edges.deg[mask]
    return out