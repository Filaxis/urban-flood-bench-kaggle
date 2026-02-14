from pathlib import Path
import numpy as np
import pandas as pd

def summarize_event_rainfall(event_dir: Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    p = event_dir / "2d_nodes_dynamic_all.csv"
    # We aggregate per timestep: max rainfall across nodes and sum rainfall across nodes
    max_by_t = {}
    sum_by_t = {}

    for chunk in pd.read_csv(p, usecols=["timestep", "rainfall"], chunksize=chunksize):
        # group within chunk
        gmax = chunk.groupby("timestep")["rainfall"].max()
        gsum = chunk.groupby("timestep")["rainfall"].sum()
        for t, v in gmax.items():
            max_by_t[t] = max(v, max_by_t.get(t, 0.0))
        for t, v in gsum.items():
            sum_by_t[t] = float(v) + sum_by_t.get(t, 0.0)

    timesteps = sorted(max_by_t.keys())
    max_series = np.array([max_by_t[t] for t in timesteps], dtype=np.float64)
    sum_series = np.array([sum_by_t[t] for t in timesteps], dtype=np.float64)

    return pd.DataFrame({
        "timestep": timesteps,
        "rain_max_over_nodes": max_series,
        "rain_sum_over_nodes": sum_series,
        "is_wet": max_series > 0,
    })
