from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass(frozen=True)
class RolloutConfig:
    warmup_steps: int = 10
    dtype: str = "float32"


def _dense_reshape(df: pd.DataFrame, value_cols: list[str], n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    df = df.sort_values(["timestep", "node_idx"], kind="stable")
    ts = df["timestep"].to_numpy()
    unique_ts = np.unique(ts)
    T = unique_ts.size

    expected_rows = T * n_nodes
    if len(df) != expected_rows:
        raise ValueError(f"Dynamics not dense: got {len(df)} rows, expected {expected_rows}. T={T} n_nodes={n_nodes}")

    vals = df[value_cols].to_numpy().reshape(T, n_nodes, len(value_cols))
    return unique_ts, vals


def rollout_event_model1(
    *,
    model: XGBRegressor,
    feature_cols: list[str],
    model_id: int,
    nodes_1d_static: pd.DataFrame,
    nodes_2d_static: pd.DataFrame,
    nodes_1d_dyn: pd.DataFrame,  # timestep,node_idx,water_level (water_level exists for 0..9 in test)
    nodes_2d_dyn: pd.DataFrame,  # timestep,node_idx,rainfall,water_level (wl exists for 0..9; rainfall all)
    H: int,
    cfg: RolloutConfig,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Returns dict:
      key: (node_type, node_id)
      value: predicted water levels of shape (H,) for that node
    """

    n1 = len(nodes_1d_static)
    n2 = len(nodes_2d_static)

    # Dense rainfall for all timesteps; dense water levels for warmup
    # 2D:
    _, v2 = _dense_reshape(nodes_2d_dyn, ["rainfall", "water_level"], n2)  # (T,n2,2)
    rain2 = v2[:, :, 0].astype(cfg.dtype, copy=False)  # (T,n2)
    wl2_all = v2[:, :, 1]  # may be NaN beyond warmup

    # 1D:
    _, v1 = _dense_reshape(nodes_1d_dyn, ["water_level"], n1)  # (T,n1,1)
    wl1_all = v1[:, :, 0]  # may be NaN beyond warmup

    T = rain2.shape[0]
    # Horizon sanity: usually H == T - warmup_steps
    if cfg.warmup_steps + H > T:
        raise ValueError(f"H too large: warmup={cfg.warmup_steps}, H={H}, T={T}")

    # Initialize lag states from warmup (t=6, t=7, t=8 and t=9)
    # We predict wl[t+1] given features at timestep t (wl_t, wl_tm1, wl_tm2, wl_tm3, rain_t, rain_tm1, rain_tm2 and rain_sum_3).
    # 2D wl lags at warmup end
    wl2_tm3 = wl2_all[cfg.warmup_steps - 4, :].astype(cfg.dtype)
    wl2_tm2 = wl2_all[cfg.warmup_steps - 3, :].astype(cfg.dtype)
    wl2_tm1 = wl2_all[cfg.warmup_steps - 2, :].astype(cfg.dtype)
    wl2_t   = wl2_all[cfg.warmup_steps - 1, :].astype(cfg.dtype)

    # 1D wl lags
    wl1_tm3 = wl1_all[cfg.warmup_steps - 4, :].astype(cfg.dtype)
    wl1_tm2 = wl1_all[cfg.warmup_steps - 3, :].astype(cfg.dtype)
    wl1_tm1 = wl1_all[cfg.warmup_steps - 2, :].astype(cfg.dtype)
    wl1_t   = wl1_all[cfg.warmup_steps - 1, :].astype(cfg.dtype)

    # Static feature frames keyed by node_id (node_idx) – keep as DataFrame for easy column alignment
    s1 = nodes_1d_static.copy()
    s2 = nodes_2d_static.copy()

    # We’ll assemble features as DataFrames (readable, safe). With your scale, it’s fine.
    # If you need more speed later, we can switch to pure NumPy matrices.

    # Output arrays: (H, N)
    pred1 = np.empty((H, n1), dtype=cfg.dtype)
    pred2 = np.empty((H, n2), dtype=cfg.dtype)

    for k in range(H):
        # current timestep t used for features
        t = cfg.warmup_steps - 1 + k  # starts at 9 for k=0
        # rain at timestep t for 2D nodes
        rain_t = rain2[t, :].astype(cfg.dtype, copy=False)
        rain_tm1 = rain2[t - 1, :].astype(cfg.dtype, copy=False)
        rain_tm2 = rain2[t - 2, :].astype(cfg.dtype, copy=False)
        rain_sum_3 = rain_t + rain_tm1 + rain_tm2

        # --- 2D batch ---
        df2 = pd.DataFrame({
            "model_id": np.full(n2, model_id, dtype=np.int16),
            "node_type": np.full(n2, 2, dtype=np.int8),
            "node_id": np.arange(n2, dtype=np.int32),
            "t": np.full(n2, t, dtype=np.int32),

            "wl_t": wl2_t,
            "wl_tm1": wl2_tm1,
            "wl_tm2": wl2_tm2,
            "wl_tm3": wl2_tm3,

            "rain_t": rain_t,
            "rain_tm1": rain_tm1,
            "rain_tm2": rain_tm2,
            "rain_sum_3": rain_sum_3,
        })
        df2 = df2.merge(s2, left_on="node_id", right_on="node_idx", how="left")
        df2.drop(columns=["node_idx"], inplace=True)

        X2 = df2.reindex(columns=feature_cols)
        y2 = model.predict(X2).astype(cfg.dtype, copy=False)
        pred2[k, :] = y2

        # --- 1D batch ---
        df1 = pd.DataFrame({
            "model_id": np.full(n1, model_id, dtype=np.int16),
            "node_type": np.full(n1, 1, dtype=np.int8),
            "node_id": np.arange(n1, dtype=np.int32),
            "t": np.full(n1, t, dtype=np.int32),

            "wl_t": wl1_t,
            "wl_tm1": wl1_tm1,
            "wl_tm2": wl1_tm2,
            "wl_tm3": wl1_tm3,

            "rain_t": np.zeros(n1, dtype=cfg.dtype),
            "rain_tm1": np.zeros(n1, dtype=cfg.dtype),
            "rain_tm2": np.zeros(n1, dtype=cfg.dtype),
            "rain_sum_3": np.zeros(n1, dtype=cfg.dtype),
        })
        df1 = df1.merge(s1, left_on="node_id", right_on="node_idx", how="left")
        df1.drop(columns=["node_idx"], inplace=True)

        X1 = df1.reindex(columns=feature_cols)
        y1 = model.predict(X1).astype(cfg.dtype, copy=False)
        pred1[k, :] = y1

        # shift lags
        wl2_tm3, wl2_tm2, wl2_tm1, wl2_t = wl2_tm2, wl2_tm1, wl2_t, y2
        wl1_tm3, wl1_tm2, wl1_tm1, wl1_t = wl1_tm2, wl1_tm1, wl1_t, y1

    # Convert to dict keyed by (node_type,node_id)
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for nid in range(n1):
        out[(1, nid)] = pred1[:, nid]
    for nid in range(n2):
        out[(2, nid)] = pred2[:, nid]
    return out

def rollout_event_two_models(
    *,
    model_1d: XGBRegressor,
    feature_cols_1d: list[str],
    model_2d: XGBRegressor,
    feature_cols_2d: list[str],
    model_id: int,
    nodes_1d_static: pd.DataFrame,
    nodes_2d_static: pd.DataFrame,
    nodes_1d_dyn: pd.DataFrame,  # timestep,node_idx,water_level (water_level exists for 0..9 in test)
    nodes_2d_dyn: pd.DataFrame,  # timestep,node_idx,rainfall,water_level (wl exists for 0..9; rainfall all)
    H: int,
    cfg: RolloutConfig,
    alpha_1d: float = 1.0,                 # 1.0 = no damping; <1.0 adds inertia toward wl1_t
    clip_1d: tuple[float, float] | None = None,  # e.g. (-5.0, 50.0); None = no clip
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Two-model rollout:
      - model_2d predicts 2D node water levels
      - model_1d predicts 1D node water levels

    Returns dict:
      key: (node_type, node_id)
      value: predicted water levels of shape (H,) for that node
    """

    if not (0.0 < alpha_1d <= 1.0):
        raise ValueError(f"alpha_1d must be in (0,1], got {alpha_1d}")

    n1 = len(nodes_1d_static)
    n2 = len(nodes_2d_static)

    # 2D: rainfall + water_level (water_level is only reliable for warmup; rainfall for all timesteps)
    _, v2 = _dense_reshape(nodes_2d_dyn, ["rainfall", "water_level"], n2)  # (T,n2,2)
    rain2 = v2[:, :, 0].astype(cfg.dtype, copy=False)  # (T,n2)
    wl2_all = v2[:, :, 1]  # may be NaN beyond warmup

    # 1D: water_level (reliable for warmup; may be NaN beyond)
    _, v1 = _dense_reshape(nodes_1d_dyn, ["water_level"], n1)  # (T,n1,1)
    wl1_all = v1[:, :, 0]  # may be NaN beyond warmup

    T = rain2.shape[0]
    if cfg.warmup_steps + H > T:
        raise ValueError(f"H too large: warmup={cfg.warmup_steps}, H={H}, T={T}")

    # Initialize lag states from warmup (t=6, t=7, t=8 and t=9)
    wl2_tm3 = wl2_all[cfg.warmup_steps - 4, :].astype(cfg.dtype)
    wl2_tm2 = wl2_all[cfg.warmup_steps - 3, :].astype(cfg.dtype)
    wl2_tm1 = wl2_all[cfg.warmup_steps - 2, :].astype(cfg.dtype)
    wl2_t   = wl2_all[cfg.warmup_steps - 1, :].astype(cfg.dtype)

    wl1_tm3 = wl1_all[cfg.warmup_steps - 4, :].astype(cfg.dtype)
    wl1_tm2 = wl1_all[cfg.warmup_steps - 3, :].astype(cfg.dtype)
    wl1_tm1 = wl1_all[cfg.warmup_steps - 2, :].astype(cfg.dtype)
    wl1_t   = wl1_all[cfg.warmup_steps - 1, :].astype(cfg.dtype)

    s1 = nodes_1d_static.copy()
    s2 = nodes_2d_static.copy()

    pred1 = np.empty((H, n1), dtype=cfg.dtype)
    pred2 = np.empty((H, n2), dtype=cfg.dtype)

    for k in range(H):
        t = cfg.warmup_steps - 1 + k  # starts at warmup-1 (e.g. 9)
        rain_t = rain2[t, :].astype(cfg.dtype, copy=False)
        rain_tm1 = rain2[t - 1, :].astype(cfg.dtype, copy=False)
        rain_tm2 = rain2[t - 2, :].astype(cfg.dtype, copy=False)
        rain_sum_3 = rain_t + rain_tm1 + rain_tm2

        # --- 2D batch ---
        df2 = pd.DataFrame({
            "model_id": np.full(n2, model_id, dtype=np.int16),
            "node_type": np.full(n2, 2, dtype=np.int8),
            "node_id": np.arange(n2, dtype=np.int32),
            "t": np.full(n2, t, dtype=np.int32),

            "wl_t": wl2_t,
            "wl_tm1": wl2_tm1,
            "wl_tm2": wl2_tm2,
            "wl_tm3": wl2_tm3,

            "rain_t": rain_t,
            "rain_tm1": rain_tm1,
            "rain_tm2": rain_tm2,
            "rain_sum_3": rain_sum_3,
        })
        df2 = df2.merge(s2, left_on="node_id", right_on="node_idx", how="left")
        df2.drop(columns=["node_idx"], inplace=True)

        X2 = df2.reindex(columns=feature_cols_2d)
        y2 = model_2d.predict(X2).astype(cfg.dtype, copy=False)
        pred2[k, :] = y2

        # --- 1D batch ---
        df1 = pd.DataFrame({
            "model_id": np.full(n1, model_id, dtype=np.int16),
            "node_type": np.full(n1, 1, dtype=np.int8),
            "node_id": np.arange(n1, dtype=np.int32),
            "t": np.full(n1, t, dtype=np.int32),
            
            "wl_t": wl1_t,
            "wl_tm1": wl1_tm1,
            "wl_tm2": wl1_tm2,
            "wl_tm3": wl1_tm3,
            
            "rain_t": np.zeros(n1, dtype=cfg.dtype),
            "rain_tm1": np.zeros(n1, dtype=cfg.dtype),
            "rain_tm2": np.zeros(n1, dtype=cfg.dtype),
            "rain_sum_3": np.zeros(n1, dtype=cfg.dtype),
        })
        df1 = df1.merge(s1, left_on="node_id", right_on="node_idx", how="left")
        df1.drop(columns=["node_idx"], inplace=True)

        X1 = df1.reindex(columns=feature_cols_1d)
        y1 = model_1d.predict(X1).astype(cfg.dtype, copy=False)

        # Optional damping toward previous level (inertia)
        if alpha_1d < 1.0:
            y1 = alpha_1d * y1 + (1.0 - alpha_1d) * wl1_t

        # Optional clipping
        if clip_1d is not None:
            y1 = np.clip(y1, clip_1d[0], clip_1d[1])

        pred1[k, :] = y1

        # shift lags
        wl2_tm3, wl2_tm2, wl2_tm1, wl2_t = wl2_tm2, wl2_tm1, wl2_t, y2
        wl1_tm3, wl1_tm2, wl1_tm1, wl1_t = wl1_tm2, wl1_tm1, wl1_t, y1

    out: Dict[Tuple[int, int], np.ndarray] = {}
    for nid in range(n1):
        out[(1, nid)] = pred1[:, nid]
    for nid in range(n2):
        out[(2, nid)] = pred2[:, nid]
    return out
