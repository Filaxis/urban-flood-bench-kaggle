from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ufb.features.graph_feats import UndirectedEdges, neighbor_mean

@dataclass(frozen=True)
class SampleConfig:
    warmup_steps: int = 10
    n_lags: int = 6
    min_t: int = 9               # set to 9 if you want test-like training
    dry_keep_prob: float = 0.15  # fraction of dry timesteps kept
    max_timesteps_per_event: Optional[int] = 120
    seed: int = 42
    dtype: str = "float32"


def _dense_reshape(df: pd.DataFrame, value_cols: List[str], n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    df = df.sort_values(["timestep", "node_idx"], kind="stable")
    ts = df["timestep"].to_numpy()
    unique_ts = np.unique(ts)
    T = unique_ts.size

    expected_rows = T * n_nodes
    if len(df) != expected_rows:
        raise ValueError(f"Dynamics not dense: got {len(df)} rows, expected {expected_rows} (=T*n_nodes). "
                         f"T={T}, n_nodes={n_nodes}")

    vals = df[value_cols].to_numpy()
    vals = vals.reshape(T, n_nodes, len(value_cols))
    return unique_ts, vals

def _add_recession_buffer(selected: np.ndarray, T: int, buffer: int) -> np.ndarray:
    if buffer <= 0 or selected.size == 0:
        return selected
    extra = []
    for t in selected:
        for k in range(1, buffer + 1):
            if t + k < T:
                extra.append(t + k)
    return np.unique(np.concatenate([selected, np.array(extra, dtype=int)]))

def _select_timesteps_for_training(rain_2d: np.ndarray, cfg: SampleConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    T = rain_2d.shape[0]

    t_min = max(cfg.min_t, cfg.n_lags - 1)  # n_lags=6 => max(9, 5) = 9
    t_max = T - 2  # need t+1 for target
    if t_max < t_min:
        return np.array([], dtype=int)

    rain_max = rain_2d.max(axis=1)
    wet = np.where(rain_max > 0)[0]
    wet = _add_recession_buffer(wet, T=T, buffer=10)
    wet = wet[(wet >= t_min) & (wet <= t_max)]

    dry = np.array([t for t in range(t_min, t_max + 1) if rain_max[t] <= 0], dtype=int)
    keep_dry = rng.random(dry.size) < cfg.dry_keep_prob
    dry = dry[keep_dry]

    selected = np.unique(np.concatenate([wet, dry]))
    if cfg.max_timesteps_per_event is not None and selected.size > cfg.max_timesteps_per_event:
        selected = rng.choice(selected, size=cfg.max_timesteps_per_event, replace=False)
        selected.sort()

    return selected


def build_event_training_samples(
    *,
    model_id: int,
    nodes_1d_static: pd.DataFrame,
    nodes_2d_static: pd.DataFrame,
    nodes_1d_dyn: pd.DataFrame,
    nodes_2d_dyn: pd.DataFrame,
    cfg: SampleConfig,
    adj_1d: Optional[UndirectedEdges] = None,
    adj_2d: Optional[UndirectedEdges] = None,
    conn1d_to_2d: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Returns one training DataFrame for ONE event, containing both 1D and 2D rows.

    NOTE: target column is wl_{t+1} - wl_t  (TRUE DELTA).
    This is intentional and must match the GNN training and predictor logic.
    """
    n1 = len(nodes_1d_static)
    n2 = len(nodes_2d_static)

    if set(nodes_1d_static["node_idx"]) != set(range(n1)):
        raise ValueError("1D node_idx are not 0..N-1; need an explicit index mapping.")
    if set(nodes_2d_static["node_idx"]) != set(range(n2)):
        raise ValueError("2D node_idx are not 0..N-1; need an explicit index mapping.")

    _, v1 = _dense_reshape(nodes_1d_dyn, ["water_level"], n1)           # (T, n1, 1)
    _, v2 = _dense_reshape(nodes_2d_dyn, ["rainfall", "water_level"], n2)  # (T, n2, 2)

    wl1 = v1[:, :, 0].astype(cfg.dtype, copy=False)   # (T, n1)
    rain2 = v2[:, :, 0].astype(cfg.dtype, copy=False)  # (T, n2)
    wl2 = v2[:, :, 1].astype(cfg.dtype, copy=False)    # (T, n2)

    T = wl2.shape[0]
    rain_sum_4_all = np.zeros((T, n2), dtype=cfg.dtype)
    rain_sum_4_all[3:, :] = (rain2[3:, :] + rain2[2:-1, :] + rain2[1:-2, :] + rain2[:-3, :]).astype(cfg.dtype, copy=False)

    t_sel = _select_timesteps_for_training(rain2, cfg)
    if t_sel.size == 0:
        return pd.DataFrame()

    use_graph_2d = adj_2d is not None
    use_graph_1d = adj_1d is not None
    use_conn = conn1d_to_2d is not None

    if use_graph_2d:
        nbr_wl2_t   = np.stack([neighbor_mean(wl2[t, :],            adj_2d) for t in t_sel], axis=0)  # (S, n2)
        nbr_wl2_tm1 = np.stack([neighbor_mean(wl2[t - 1, :],        adj_2d) for t in t_sel], axis=0)
        nbr_rsum4   = np.stack([neighbor_mean(rain_sum_4_all[t, :],  adj_2d) for t in t_sel], axis=0)

    if use_graph_1d:
        nbr_wl1_t   = np.stack([neighbor_mean(wl1[t, :],     adj_1d) for t in t_sel], axis=0)  # (S, n1)
        nbr_wl1_tm1 = np.stack([neighbor_mean(wl1[t - 1, :], adj_1d) for t in t_sel], axis=0)

    if use_conn:
        conn = conn1d_to_2d.astype(np.int32, copy=False)
        conn_safe = conn.copy()
        missing = conn_safe < 0
        if missing.any():
            conn_safe[missing] = 0

        conn2d_wl_t  = wl2[t_sel[:, None], conn_safe[None, :]]              # (S, n1)
        conn2d_rsum4 = rain_sum_4_all[t_sel[:, None], conn_safe[None, :]]   # (S, n1)

        if missing.any():
            conn2d_wl_t[:, missing] = 0
            conn2d_rsum4[:, missing] = 0

    # ---- 2D samples ----
    X2 = {
        "model_id":  np.full((t_sel.size * n2,), model_id, dtype=np.int16),
        "node_type": np.full((t_sel.size * n2,), 2, dtype=np.int8),
        "node_id":   np.tile(np.arange(n2, dtype=np.int32), t_sel.size),
        "t":         np.repeat(t_sel.astype(np.int32), n2),

        "wl_t":   wl2[t_sel, :].reshape(-1),
        "wl_tm1": wl2[t_sel - 1, :].reshape(-1),
        "wl_tm2": wl2[t_sel - 2, :].reshape(-1),
        "wl_tm3": wl2[t_sel - 3, :].reshape(-1),
        "wl_tm4": wl2[t_sel - 4, :].reshape(-1),
        "wl_tm5": wl2[t_sel - 5, :].reshape(-1),

        "rain_t":   rain2[t_sel, :].reshape(-1),
        "rain_tm1": rain2[t_sel - 1, :].reshape(-1),
        "rain_tm2": rain2[t_sel - 2, :].reshape(-1),
        "rain_tm3": rain2[t_sel - 3, :].reshape(-1),
    }
    rain_sum_4_2d = (
        rain2[t_sel, :] + rain2[t_sel - 1, :] +
        rain2[t_sel - 2, :] + rain2[t_sel - 3, :]
    ).reshape(-1)
    X2["rain_sum_4"] = rain_sum_4_2d.astype(cfg.dtype, copy=False)
    X2["d_wl_t"] = (wl2[t_sel, :] - wl2[t_sel - 1, :]).reshape(-1)

    if use_graph_2d:
        X2["nbr_wl_mean_t"]   = nbr_wl2_t.reshape(-1).astype(cfg.dtype, copy=False)
        X2["nbr_wl_mean_tm1"] = nbr_wl2_tm1.reshape(-1).astype(cfg.dtype, copy=False)
        X2["nbr_rain_sum_4"]  = nbr_rsum4.reshape(-1).astype(cfg.dtype, copy=False)
    else:
        X2["nbr_wl_mean_t"]   = np.zeros((t_sel.size * n2,), dtype=cfg.dtype)
        X2["nbr_wl_mean_tm1"] = np.zeros((t_sel.size * n2,), dtype=cfg.dtype)
        X2["nbr_rain_sum_4"]  = np.zeros((t_sel.size * n2,), dtype=cfg.dtype)

    # TARGET = TRUE DELTA  wl_{t+1} - wl_t
    X2["target"] = (wl2[t_sel + 1, :] - wl2[t_sel, :]).reshape(-1)

    df2 = pd.DataFrame(X2)
    df2 = df2.merge(nodes_2d_static, left_on="node_id", right_on="node_idx", how="left")
    df2.drop(columns=["node_idx"], inplace=True)

    # ---- 1D samples ----
    X1 = {
        "model_id":  np.full((t_sel.size * n1,), model_id, dtype=np.int16),
        "node_type": np.full((t_sel.size * n1,), 1, dtype=np.int8),
        "node_id":   np.tile(np.arange(n1, dtype=np.int32), t_sel.size),
        "t":         np.repeat(t_sel.astype(np.int32), n1),

        "wl_t":   wl1[t_sel, :].reshape(-1),
        "wl_tm1": wl1[t_sel - 1, :].reshape(-1),
        "wl_tm2": wl1[t_sel - 2, :].reshape(-1),
        "wl_tm3": wl1[t_sel - 3, :].reshape(-1),
        "wl_tm4": wl1[t_sel - 4, :].reshape(-1),
        "wl_tm5": wl1[t_sel - 5, :].reshape(-1),

        "rain_t":     np.zeros((t_sel.size * n1,), dtype=cfg.dtype),
        "rain_tm1":   np.zeros((t_sel.size * n1,), dtype=cfg.dtype),
        "rain_tm2":   np.zeros((t_sel.size * n1,), dtype=cfg.dtype),
        "rain_tm3":   np.zeros((t_sel.size * n1,), dtype=cfg.dtype),
        "rain_sum_4": np.zeros((t_sel.size * n1,), dtype=cfg.dtype),
    }
    X1["d_wl_t"] = (wl1[t_sel, :] - wl1[t_sel - 1, :]).reshape(-1)

    if use_graph_1d:
        X1["nbr_wl_mean_t"]   = nbr_wl1_t.reshape(-1).astype(cfg.dtype, copy=False)
        X1["nbr_wl_mean_tm1"] = nbr_wl1_tm1.reshape(-1).astype(cfg.dtype, copy=False)
    else:
        X1["nbr_wl_mean_t"]   = np.zeros((t_sel.size * n1,), dtype=cfg.dtype)
        X1["nbr_wl_mean_tm1"] = np.zeros((t_sel.size * n1,), dtype=cfg.dtype)

    if use_conn:
        X1["conn2d_wl_t"]      = conn2d_wl_t.reshape(-1).astype(cfg.dtype, copy=False)
        X1["conn2d_rain_sum_4"] = conn2d_rsum4.reshape(-1).astype(cfg.dtype, copy=False)
    else:
        X1["conn2d_wl_t"]      = np.zeros((t_sel.size * n1,), dtype=cfg.dtype)
        X1["conn2d_rain_sum_4"] = np.zeros((t_sel.size * n1,), dtype=cfg.dtype)

    # TARGET = TRUE DELTA  wl_{t+1} - wl_t
    X1["target"] = (wl1[t_sel + 1, :] - wl1[t_sel, :]).reshape(-1)

    df1 = pd.DataFrame(X1)
    df1 = df1.merge(nodes_1d_static, left_on="node_id", right_on="node_idx", how="left")
    df1.drop(columns=["node_idx"], inplace=True)

    df = pd.concat([df1, df2], axis=0, ignore_index=True, copy=False)
    return df
