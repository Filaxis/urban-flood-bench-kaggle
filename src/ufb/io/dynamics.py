from dataclasses import dataclass
from pathlib import Path

import pandas as pd

@dataclass
class EventDynamics:
    model_id: int
    split: str
    event_id: int
    event_dir: Path
    timesteps: pd.DataFrame
    nodes_1d_dyn: pd.DataFrame
    nodes_2d_dyn: pd.DataFrame


def load_event_dynamics(event_dir: Path, model_id: int, event_id: int, split: str) -> EventDynamics:
    p_ts = event_dir / "timesteps.csv"
    p_1d = event_dir / "1d_nodes_dynamic_all.csv"
    p_2d = event_dir / "2d_nodes_dynamic_all.csv"
    for p in (p_ts, p_1d, p_2d):
        if not p.exists():
            raise FileNotFoundError(f"Missing dynamic file: {p}")

    ts = pd.read_csv(p_ts)  # small
    n1 = pd.read_csv(p_1d, usecols=["timestep", "node_idx", "water_level"])
    n2 = pd.read_csv(p_2d, usecols=["timestep", "node_idx", "rainfall", "water_level"])

    return EventDynamics(
        model_id=model_id, split=split, event_id=event_id, event_dir=event_dir,
        timesteps=ts, nodes_1d_dyn=n1, nodes_2d_dyn=n2
    )
