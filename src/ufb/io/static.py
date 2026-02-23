from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ufb.features.graph_feats import UndirectedEdges, build_undirected_edges


@dataclass
class ModelStatic:
    model_id: int
    root: Path  # Models/Model_{id}
    split: str  # "train" or "test"

    # Raw static tables
    nodes_1d: pd.DataFrame
    nodes_2d: pd.DataFrame
    edges_1d: pd.DataFrame
    edges_2d: pd.DataFrame
    edge_index_1d: pd.DataFrame
    edge_index_2d: pd.DataFrame
    conn_1d2d: pd.DataFrame

    # Added graph structures
    adj_1d: UndirectedEdges
    adj_2d: UndirectedEdges

    # 1D -> 2D mapping (len = num_1d_nodes), -1 if none
    conn1d_to_2d: np.ndarray  # int32


def load_model_static(models_root: Path, model_id: int, split: str) -> ModelStatic:
    model_root = models_root / f"Model_{model_id}" / split
    req = {
        "edge_index_1d": model_root / "1d_edge_index.csv",
        "edge_index_2d": model_root / "2d_edge_index.csv",
        "nodes_1d": model_root / "1d_nodes_static.csv",
        "nodes_2d": model_root / "2d_nodes_static.csv",
        "edges_1d": model_root / "1d_edges_static.csv",
        "edges_2d": model_root / "2d_edges_static.csv",
        "conn": model_root / "1d2d_connections.csv",
    }
    for k, p in req.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k}: {p}")

    edge_index_1d = pd.read_csv(req["edge_index_1d"])
    edge_index_2d = pd.read_csv(req["edge_index_2d"])
    nodes_1d = pd.read_csv(req["nodes_1d"])
    nodes_2d = pd.read_csv(req["nodes_2d"])
    edges_1d = pd.read_csv(req["edges_1d"])
    edges_2d = pd.read_csv(req["edges_2d"])
    conn_1d2d = pd.read_csv(req["conn"])

    n1 = int(nodes_1d["node_idx"].max()) + 1
    n2 = int(nodes_2d["node_idx"].max()) + 1

    adj_1d = build_undirected_edges(edge_index_1d, n_nodes=n1)
    adj_2d = build_undirected_edges(edge_index_2d, n_nodes=n2)

    # add degree as a static node feature
    nodes_1d = nodes_1d.copy()
    nodes_2d = nodes_2d.copy()
    nodes_1d["deg"] = adj_1d.deg
    nodes_2d["deg"] = adj_2d.deg

    # mapping array 1D -> 2D
    conn1d_to_2d = np.full((n1,), -1, dtype=np.int32)
    a = conn_1d2d["node_1d"].to_numpy(np.int32, copy=False)
    b = conn_1d2d["node_2d"].to_numpy(np.int32, copy=False)
    conn1d_to_2d[a] = b

    return ModelStatic(
        model_id=model_id,
        root=models_root / f"Model_{model_id}",
        split=split,
        edge_index_1d=edge_index_1d,
        edge_index_2d=edge_index_2d,
        nodes_1d=nodes_1d,
        nodes_2d=nodes_2d,
        edges_1d=edges_1d,
        edges_2d=edges_2d,
        conn_1d2d=conn_1d2d,
        adj_1d=adj_1d,
        adj_2d=adj_2d,
        conn1d_to_2d=conn1d_to_2d,
    )