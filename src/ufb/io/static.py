from dataclasses import dataclass
from pathlib import Path

import pandas as pd

@dataclass
class ModelStatic:
    model_id: int
    root: Path  # Models/Model_{id}
    split: str  # "train" or "test" (static files are in split folder in your layout)
    # Static dataframes:
    nodes_1d: pd.DataFrame
    nodes_2d: pd.DataFrame
    edges_1d: pd.DataFrame
    edges_2d: pd.DataFrame
    edge_index_1d: pd.DataFrame
    edge_index_2d: pd.DataFrame
    conn_1d2d: pd.DataFrame


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

    return ModelStatic(
        model_id=model_id,
        root=models_root / f"Model_{model_id}",
        split=split,
        edge_index_1d=pd.read_csv(req["edge_index_1d"]),
        edge_index_2d=pd.read_csv(req["edge_index_2d"]),
        nodes_1d=pd.read_csv(req["nodes_1d"]),
        nodes_2d=pd.read_csv(req["nodes_2d"]),
        edges_1d=pd.read_csv(req["edges_1d"]),
        edges_2d=pd.read_csv(req["edges_2d"]),
        conn_1d2d=pd.read_csv(req["conn"]),
    )
