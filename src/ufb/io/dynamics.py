from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Column name constants — verify against your actual CSVs if loading fails.
# Run:  pd.read_csv(event_dir / "1d_nodes_dynamic_all.csv", nrows=2).columns.tolist()
# and:  pd.read_csv(event_dir / "1d_edges_dynamic_all.csv", nrows=2).columns.tolist()
# ---------------------------------------------------------------------------
_1D_NODE_FLOW_COL  = "inlet_flow"   # exchange flow between 2D surface and 1D node
_1D_EDGE_FLOW_COL  = "flow"         # pipe flow through each 1D edge
# If the actual column names differ, update the two constants above only.


@dataclass
class EventDynamics:
    model_id: int
    split: str
    event_id: int
    event_dir: Path
    timesteps: pd.DataFrame
    nodes_1d_dyn: pd.DataFrame   # columns: timestep, node_idx, water_level[, inlet_flow]
    nodes_2d_dyn: pd.DataFrame   # columns: timestep, node_idx, rainfall, water_level
    edges_1d_dyn: Optional[pd.DataFrame] = None  # columns: timestep, edge_idx, flow


def load_event_dynamics(
    event_dir: Path,
    model_id: int,
    event_id: int,
    split: str,
    load_flow: bool = True,   # set False to fall back to old behaviour
) -> EventDynamics:
    """
    Load all dynamic data for one event.

    Parameters
    ----------
    load_flow : bool
        If True (default), attempt to load inlet_flow from 1d_nodes_dynamic_all.csv
        and edge flow from 1d_edges_dynamic_all.csv.
        If a column is absent the field is silently omitted (no crash).
    """
    p_ts  = event_dir / "timesteps.csv"
    p_1d  = event_dir / "1d_nodes_dynamic_all.csv"
    p_2d  = event_dir / "2d_nodes_dynamic_all.csv"
    p_1de = event_dir / "1d_edges_dynamic_all.csv"

    for p in (p_ts, p_1d, p_2d):
        if not p.exists():
            raise FileNotFoundError(f"Missing dynamic file: {p}")

    ts = pd.read_csv(p_ts)

    # --- 2D nodes: unchanged ---
    n2 = pd.read_csv(p_2d, usecols=["timestep", "node_idx", "rainfall", "water_level"])

    # --- 1D nodes: add inlet_flow if available ---
    if load_flow:
        # Peek at available columns without loading full file
        header_1d = pd.read_csv(p_1d, nrows=0).columns.tolist()
        cols_1d = ["timestep", "node_idx", "water_level"]
        if _1D_NODE_FLOW_COL in header_1d:
            cols_1d.append(_1D_NODE_FLOW_COL)
        n1 = pd.read_csv(p_1d, usecols=cols_1d)
    else:
        n1 = pd.read_csv(p_1d, usecols=["timestep", "node_idx", "water_level"])

    # --- 1D edges: load edge flow if file exists ---
    edges_1d = None
    if load_flow and p_1de.exists():
        header_1de = pd.read_csv(p_1de, nrows=0).columns.tolist()
        cols_1de = ["timestep", "edge_idx"]
        if _1D_EDGE_FLOW_COL in header_1de:
            cols_1de.append(_1D_EDGE_FLOW_COL)
        if len(cols_1de) > 2:   # only load if the flow column actually exists
            edges_1d = pd.read_csv(p_1de, usecols=cols_1de)

    return EventDynamics(
        model_id=model_id,
        split=split,
        event_id=event_id,
        event_dir=event_dir,
        timesteps=ts,
        nodes_1d_dyn=n1,
        nodes_2d_dyn=n2,
        edges_1d_dyn=edges_1d,
    )
