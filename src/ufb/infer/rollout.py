from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ufb.features.graph_feats import UndirectedEdges, neighbor_mean
from ufb.infer.predictor_base import Predictor


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


def _precompute_rainfall_context(
    rain2: np.ndarray,  # (T, n2)
    dtype: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precompute three scalar rainfall context features for all T timesteps.
    Returns three (T,) arrays — broadcast to nodes inside the rollout loop.
      rain_frac_remaining   -- fraction of total event rain still to fall
      rain_steps_since_peak -- steps elapsed since rainfall peak (0 before/at peak)
      rain_intensity_trend  -- mean(last 5 steps) - mean(prior 5 steps)
    """
    T = rain2.shape[0]
    rain_ts = rain2.max(axis=1).astype(np.float64)  # (T,)

    total = rain_ts.sum()
    cum   = np.cumsum(rain_ts)
    frac_remaining = np.clip(1.0 - cum / total, 0.0, 1.0) if total > 0 else np.zeros(T)

    peak_t = int(np.argmax(rain_ts))
    steps_since_peak = np.clip(np.arange(T, dtype=np.float64) - peak_t, 0.0, None)

    trend = np.zeros(T, dtype=np.float64)
    for ti in range(9, T):
        trend[ti] = rain_ts[ti - 4: ti + 1].mean() - rain_ts[ti - 9: ti - 4].mean()

    return (
        frac_remaining.astype(dtype),
        steps_since_peak.astype(dtype),
        trend.astype(dtype),
    )


def rollout_event_model1(
    *,
    predictor: Predictor,
    feature_cols_2d: list[str],
    feature_cols_1d: list[str],
    model_id: int,
    nodes_1d_static: pd.DataFrame,
    nodes_2d_static: pd.DataFrame,
    nodes_1d_dyn: pd.DataFrame,
    nodes_2d_dyn: pd.DataFrame,
    H: int,
    cfg: RolloutConfig,
    adj_1d: UndirectedEdges,
    adj_2d: UndirectedEdges,
    conn1d_to_2d: np.ndarray,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Autoregressive rollout for Model_1 or Model_2 (GNN predictor).
    Returns dict  (node_type, node_id) -> np.ndarray of shape (H,).
    """
    n1 = len(nodes_1d_static)
    n2 = len(nodes_2d_static)

    _, v2 = _dense_reshape(nodes_2d_dyn, ["rainfall", "water_level"], n2)
    rain2   = v2[:, :, 0].astype(cfg.dtype, copy=False)
    wl2_all = v2[:, :, 1]

    _, v1 = _dense_reshape(nodes_1d_dyn, ["water_level"], n1)
    wl1_all = v1[:, :, 0]

    T = rain2.shape[0]
    if cfg.warmup_steps + H > T:
        raise ValueError(f"H too large: warmup={cfg.warmup_steps}, H={H}, T={T}")

    # Precompute rainfall context for all timesteps — O(T) cost, done once
    rain_frac_all, rain_peak_all, rain_trend_all = _precompute_rainfall_context(rain2, cfg.dtype)

    # Initialise lag states from warmup
    wl2_tm5 = wl2_all[cfg.warmup_steps - 6, :].astype(cfg.dtype)
    wl2_tm4 = wl2_all[cfg.warmup_steps - 5, :].astype(cfg.dtype)
    wl2_tm3 = wl2_all[cfg.warmup_steps - 4, :].astype(cfg.dtype)
    wl2_tm2 = wl2_all[cfg.warmup_steps - 3, :].astype(cfg.dtype)
    wl2_tm1 = wl2_all[cfg.warmup_steps - 2, :].astype(cfg.dtype)
    wl2_t   = wl2_all[cfg.warmup_steps - 1, :].astype(cfg.dtype)

    wl1_tm5 = wl1_all[cfg.warmup_steps - 6, :].astype(cfg.dtype)
    wl1_tm4 = wl1_all[cfg.warmup_steps - 5, :].astype(cfg.dtype)
    wl1_tm3 = wl1_all[cfg.warmup_steps - 4, :].astype(cfg.dtype)
    wl1_tm2 = wl1_all[cfg.warmup_steps - 3, :].astype(cfg.dtype)
    wl1_tm1 = wl1_all[cfg.warmup_steps - 2, :].astype(cfg.dtype)
    wl1_t   = wl1_all[cfg.warmup_steps - 1, :].astype(cfg.dtype)

    s1 = nodes_1d_static.copy()
    s2 = nodes_2d_static.copy()

    pred1 = np.empty((H, n1), dtype=cfg.dtype)
    pred2 = np.empty((H, n2), dtype=cfg.dtype)

    def _ensure_cols(df: pd.DataFrame, cols: list[str], fill: float = 0.0) -> pd.DataFrame:
        for c in cols:
            if c not in df.columns:
                df[c] = fill
        return df

    for k in range(H):
        t = cfg.warmup_steps - 1 + k

        rain_t   = rain2[t,     :].astype(cfg.dtype, copy=False)
        rain_tm1 = rain2[t - 1, :].astype(cfg.dtype, copy=False)
        rain_tm2 = rain2[t - 2, :].astype(cfg.dtype, copy=False)
        rain_tm3 = rain2[t - 3, :].astype(cfg.dtype, copy=False)
        rain_sum_4 = rain_t + rain_tm1 + rain_tm2 + rain_tm3

        d_wl2_t = wl2_t - wl2_tm1
        d_wl1_t = wl1_t - wl1_tm1

        # 2D graph features
        nbr_wl2_t   = neighbor_mean(wl2_t.astype(np.float32,   copy=False), adj_2d).astype(cfg.dtype, copy=False)
        nbr_wl2_tm1 = neighbor_mean(wl2_tm1.astype(np.float32, copy=False), adj_2d).astype(cfg.dtype, copy=False)
        nbr_rsum4   = neighbor_mean(rain_sum_4.astype(np.float32, copy=False), adj_2d).astype(cfg.dtype, copy=False)

        # 1D graph features
        nbr_wl1_t   = neighbor_mean(wl1_t.astype(np.float32,   copy=False), adj_1d).astype(cfg.dtype, copy=False)
        nbr_wl1_tm1 = neighbor_mean(wl1_tm1.astype(np.float32, copy=False), adj_1d).astype(cfg.dtype, copy=False)

        # 1D <- 2D connection features
        conn      = conn1d_to_2d.astype(np.int32, copy=False)
        conn_safe = conn.copy()
        missing   = conn_safe < 0
        if missing.any():
            conn_safe[missing] = 0
        conn2d_wl_t  = wl2_t[conn_safe].astype(cfg.dtype, copy=False)
        conn2d_rsum4 = rain_sum_4[conn_safe].astype(cfg.dtype, copy=False)
        if missing.any():
            conn2d_wl_t[missing]  = 0.0
            conn2d_rsum4[missing] = 0.0

        # Scalar rainfall context values at timestep t
        frac_t  = float(rain_frac_all[t])
        peak_t_val = float(rain_peak_all[t])
        trend_t = float(rain_trend_all[t])

        # ---- Build X2 ----
        df2 = pd.DataFrame({
            "model_id":   np.full(n2, model_id, dtype=np.int16),
            "node_type":  np.full(n2, 2,        dtype=np.int8),
            "node_id":    np.arange(n2,          dtype=np.int32),
            "t":          np.full(n2, t,          dtype=np.int32),

            "wl_t":   wl2_t,   "wl_tm1": wl2_tm1, "wl_tm2": wl2_tm2,
            "wl_tm3": wl2_tm3, "wl_tm4": wl2_tm4, "wl_tm5": wl2_tm5,

            "rain_t":     rain_t,   "rain_tm1": rain_tm1,
            "rain_tm2":   rain_tm2, "rain_tm3": rain_tm3,
            "rain_sum_4": rain_sum_4,
            "d_wl_t":     d_wl2_t,

            "nbr_wl_mean_t":   nbr_wl2_t,
            "nbr_wl_mean_tm1": nbr_wl2_tm1,
            "nbr_rain_sum_4":  nbr_rsum4,

            "rain_frac_remaining":   np.full(n2, frac_t,      dtype=cfg.dtype),
            "rain_steps_since_peak": np.full(n2, peak_t_val,  dtype=cfg.dtype),
            "rain_intensity_trend":  np.full(n2, trend_t,     dtype=cfg.dtype),
        })
        df2 = df2.merge(s2, left_on="node_id", right_on="node_idx", how="left")
        df2.drop(columns=["node_idx"], inplace=True, errors="ignore")
        df2 = _ensure_cols(df2, feature_cols_2d)
        df2[feature_cols_2d] = df2[feature_cols_2d].fillna(0.0)
        X2 = df2.reindex(columns=feature_cols_2d).to_numpy(dtype=np.float32, copy=False)

        # ---- Build X1 ----
        df1 = pd.DataFrame({
            "model_id":   np.full(n1, model_id, dtype=np.int16),
            "node_type":  np.full(n1, 1,        dtype=np.int8),
            "node_id":    np.arange(n1,          dtype=np.int32),
            "t":          np.full(n1, t,          dtype=np.int32),

            "wl_t":   wl1_t,   "wl_tm1": wl1_tm1, "wl_tm2": wl1_tm2,
            "wl_tm3": wl1_tm3, "wl_tm4": wl1_tm4, "wl_tm5": wl1_tm5,

            "rain_t":     np.zeros(n1, dtype=cfg.dtype),
            "rain_tm1":   np.zeros(n1, dtype=cfg.dtype),
            "rain_tm2":   np.zeros(n1, dtype=cfg.dtype),
            "rain_tm3":   np.zeros(n1, dtype=cfg.dtype),
            "rain_sum_4": np.zeros(n1, dtype=cfg.dtype),
            "d_wl_t":     d_wl1_t,

            "nbr_wl_mean_t":    nbr_wl1_t,
            "nbr_wl_mean_tm1":  nbr_wl1_tm1,
            "conn2d_wl_t":      conn2d_wl_t,
            "conn2d_rain_sum_4": conn2d_rsum4,

            "rain_frac_remaining":   np.full(n1, frac_t,      dtype=cfg.dtype),
            "rain_steps_since_peak": np.full(n1, peak_t_val,  dtype=cfg.dtype),
            "rain_intensity_trend":  np.full(n1, trend_t,     dtype=cfg.dtype),
        })
        df1 = df1.merge(s1, left_on="node_id", right_on="node_idx", how="left")
        df1.drop(columns=["node_idx"], inplace=True, errors="ignore")
        df1 = _ensure_cols(df1, feature_cols_1d)
        df1[feature_cols_1d] = df1[feature_cols_1d].fillna(0.0)
        X1 = df1.reindex(columns=feature_cols_1d).to_numpy(dtype=np.float32, copy=False)

        # ---- Predict ----
        if hasattr(predictor, "predict_both"):
            y2, y1 = predictor.predict_both(X2, wl2_t, X1, wl1_t)
        else:
            y2 = predictor.predict_2d(X2, wl2_t).astype(cfg.dtype, copy=False)
            y1 = predictor.predict_1d(X1, wl1_t).astype(cfg.dtype, copy=False)

        # Guardrails
        y2 = np.where(np.isfinite(y2), y2, wl2_t)
        y1 = np.where(np.isfinite(y1), y1, wl1_t)
        y2 = np.clip(y2, -1000.0, 1000.0)
        y1 = np.clip(y1, -1000.0, 1000.0)

        pred2[k, :] = y2
        pred1[k, :] = y1

        # Shift lags
        wl2_tm5, wl2_tm4, wl2_tm3, wl2_tm2, wl2_tm1, wl2_t = \
            wl2_tm4, wl2_tm3, wl2_tm2, wl2_tm1, wl2_t, y2
        wl1_tm5, wl1_tm4, wl1_tm3, wl1_tm2, wl1_tm1, wl1_t = \
            wl1_tm4, wl1_tm3, wl1_tm2, wl1_tm1, wl1_t, y1

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
    nodes_1d_dyn: pd.DataFrame,
    nodes_2d_dyn: pd.DataFrame,
    H: int,
    cfg: RolloutConfig,
    adj_1d: UndirectedEdges,
    adj_2d: UndirectedEdges,
    conn1d_to_2d: np.ndarray,
    alpha_1d: float = 1.0,
    clip_1d: tuple[float, float] | None = None,
) -> Dict[Tuple[int, int], np.ndarray]:
    """XGB two-model rollout — unchanged from working original."""
    if not (0.0 < alpha_1d <= 1.0):
        raise ValueError(f"alpha_1d must be in (0,1], got {alpha_1d}")

    n1 = len(nodes_1d_static)
    n2 = len(nodes_2d_static)

    _, v2 = _dense_reshape(nodes_2d_dyn, ["rainfall", "water_level"], n2)
    rain2   = v2[:, :, 0].astype(cfg.dtype, copy=False)
    wl2_all = v2[:, :, 1]

    _, v1 = _dense_reshape(nodes_1d_dyn, ["water_level"], n1)
    wl1_all = v1[:, :, 0]

    T = rain2.shape[0]
    if cfg.warmup_steps + H > T:
        raise ValueError(f"H too large: warmup={cfg.warmup_steps}, H={H}, T={T}")

    wl2_tm5 = wl2_all[cfg.warmup_steps - 6, :].astype(cfg.dtype)
    wl2_tm4 = wl2_all[cfg.warmup_steps - 5, :].astype(cfg.dtype)
    wl2_tm3 = wl2_all[cfg.warmup_steps - 4, :].astype(cfg.dtype)
    wl2_tm2 = wl2_all[cfg.warmup_steps - 3, :].astype(cfg.dtype)
    wl2_tm1 = wl2_all[cfg.warmup_steps - 2, :].astype(cfg.dtype)
    wl2_t   = wl2_all[cfg.warmup_steps - 1, :].astype(cfg.dtype)

    wl1_tm5 = wl1_all[cfg.warmup_steps - 6, :].astype(cfg.dtype)
    wl1_tm4 = wl1_all[cfg.warmup_steps - 5, :].astype(cfg.dtype)
    wl1_tm3 = wl1_all[cfg.warmup_steps - 4, :].astype(cfg.dtype)
    wl1_tm2 = wl1_all[cfg.warmup_steps - 3, :].astype(cfg.dtype)
    wl1_tm1 = wl1_all[cfg.warmup_steps - 2, :].astype(cfg.dtype)
    wl1_t   = wl1_all[cfg.warmup_steps - 1, :].astype(cfg.dtype)

    s1 = nodes_1d_static.copy()
    s2 = nodes_2d_static.copy()

    pred1 = np.empty((H, n1), dtype=cfg.dtype)
    pred2 = np.empty((H, n2), dtype=cfg.dtype)

    for k in range(H):
        t = cfg.warmup_steps - 1 + k

        rain_t   = rain2[t,     :].astype(cfg.dtype, copy=False)
        rain_tm1 = rain2[t - 1, :].astype(cfg.dtype, copy=False)
        rain_tm2 = rain2[t - 2, :].astype(cfg.dtype, copy=False)
        rain_tm3 = rain2[t - 3, :].astype(cfg.dtype, copy=False)
        rain_sum_4 = rain_t + rain_tm1 + rain_tm2 + rain_tm3

        d_wl2_t = wl2_t - wl2_tm1
        d_wl1_t = wl1_t - wl1_tm1

        nbr_wl2_t   = neighbor_mean(wl2_t.astype(np.float32,   copy=False), adj_2d).astype(cfg.dtype, copy=False)
        nbr_wl2_tm1 = neighbor_mean(wl2_tm1.astype(np.float32, copy=False), adj_2d).astype(cfg.dtype, copy=False)
        nbr_rsum4   = neighbor_mean(rain_sum_4.astype(np.float32, copy=False), adj_2d).astype(cfg.dtype, copy=False)

        nbr_wl1_t   = neighbor_mean(wl1_t.astype(np.float32,   copy=False), adj_1d).astype(cfg.dtype, copy=False)
        nbr_wl1_tm1 = neighbor_mean(wl1_tm1.astype(np.float32, copy=False), adj_1d).astype(cfg.dtype, copy=False)

        conn      = conn1d_to_2d.astype(np.int32, copy=False)
        conn_safe = conn.copy()
        missing   = conn_safe < 0
        if missing.any():
            conn_safe[missing] = 0
        conn2d_wl_t  = wl2_t[conn_safe].astype(cfg.dtype, copy=False)
        conn2d_rsum4 = rain_sum_4[conn_safe].astype(cfg.dtype, copy=False)
        if missing.any():
            conn2d_wl_t[missing]  = 0.0
            conn2d_rsum4[missing] = 0.0

        df2 = pd.DataFrame({
            "model_id":   np.full(n2, model_id, dtype=np.int16),
            "node_type":  np.full(n2, 2,        dtype=np.int8),
            "node_id":    np.arange(n2,          dtype=np.int32),
            "t":          np.full(n2, t,          dtype=np.int32),
            "wl_t":   wl2_t,   "wl_tm1": wl2_tm1, "wl_tm2": wl2_tm2,
            "wl_tm3": wl2_tm3, "wl_tm4": wl2_tm4, "wl_tm5": wl2_tm5,
            "rain_t":     rain_t,   "rain_tm1": rain_tm1,
            "rain_tm2":   rain_tm2, "rain_tm3": rain_tm3,
            "rain_sum_4": rain_sum_4, "d_wl_t": d_wl2_t,
            "nbr_wl_mean_t": nbr_wl2_t, "nbr_wl_mean_tm1": nbr_wl2_tm1,
            "nbr_rain_sum_4": nbr_rsum4,
        })
        df2 = df2.merge(s2, left_on="node_id", right_on="node_idx", how="left")
        df2.drop(columns=["node_idx"], inplace=True, errors="ignore")
        X2 = df2.reindex(columns=feature_cols_2d).to_numpy(dtype=np.float32, copy=False)
        y2 = model_2d.predict(X2).astype(cfg.dtype, copy=False)
        pred2[k, :] = y2

        df1 = pd.DataFrame({
            "model_id":   np.full(n1, model_id, dtype=np.int16),
            "node_type":  np.full(n1, 1,        dtype=np.int8),
            "node_id":    np.arange(n1,          dtype=np.int32),
            "t":          np.full(n1, t,          dtype=np.int32),
            "wl_t":   wl1_t,   "wl_tm1": wl1_tm1, "wl_tm2": wl1_tm2,
            "wl_tm3": wl1_tm3, "wl_tm4": wl1_tm4, "wl_tm5": wl1_tm5,
            "rain_t":     np.zeros(n1, dtype=cfg.dtype),
            "rain_tm1":   np.zeros(n1, dtype=cfg.dtype),
            "rain_tm2":   np.zeros(n1, dtype=cfg.dtype),
            "rain_tm3":   np.zeros(n1, dtype=cfg.dtype),
            "rain_sum_4": np.zeros(n1, dtype=cfg.dtype),
            "d_wl_t":     d_wl1_t,
            "nbr_wl_mean_t":    nbr_wl1_t,
            "nbr_wl_mean_tm1":  nbr_wl1_tm1,
            "conn2d_wl_t":      conn2d_wl_t,
            "conn2d_rain_sum_4": conn2d_rsum4,
        })
        df1 = df1.merge(s1, left_on="node_id", right_on="node_idx", how="left")
        df1.drop(columns=["node_idx"], inplace=True, errors="ignore")
        X1 = df1.reindex(columns=feature_cols_1d).to_numpy(dtype=np.float32, copy=False)
        y1 = model_1d.predict(X1).astype(cfg.dtype, copy=False)

        if alpha_1d < 1.0:
            y1 = alpha_1d * y1 + (1.0 - alpha_1d) * wl1_t
        if clip_1d is not None:
            y1 = np.clip(y1, clip_1d[0], clip_1d[1])

        pred1[k, :] = y1

        wl2_tm5, wl2_tm4, wl2_tm3, wl2_tm2, wl2_tm1, wl2_t = \
            wl2_tm4, wl2_tm3, wl2_tm2, wl2_tm1, wl2_t, y2
        wl1_tm5, wl1_tm4, wl1_tm3, wl1_tm2, wl1_tm1, wl1_t = \
            wl1_tm4, wl1_tm3, wl1_tm2, wl1_tm1, wl1_t, y1

    out: Dict[Tuple[int, int], np.ndarray] = {}
    for nid in range(n1):
        out[(1, nid)] = pred1[:, nid]
    for nid in range(n2):
        out[(2, nid)] = pred2[:, nid]
    return out
