"""
train_gnn_model2_rollout.py  —  TBPTT training for four-output Model_1/Model_2.

Outputs: delta_wl_2d, delta_wl_1d, inlet_flow (abs), edge_flow (abs)

At each rollout step:
  - WL lags are updated with model predictions (TBPTT, gradient detached beyond K)
  - inlet_flow lags are updated with model's predicted inlet_flow
  - edge_flow lags are updated with model's predicted edge_flow
  - Loss = RMSE(wl2) + RMSE(wl1) + λ_inlet*RMSE(inlet) + λ_edge*RMSE(edge)

Usage (Kaggle):
  # Smoke test:
  MAX_EVENTS=3 GNN_EPOCHS=3 GNN_K=6 python scripts/train_gnn_model2_rollout.py

  # Full run:
  GNN_EPOCHS=25 GNN_K=6 python scripts/train_gnn_model2_rollout.py
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ufb.models.gnn_py import Model1Net
from ufb.features.graph_feats import UndirectedEdges, neighbor_mean


# ---------------------------------------------------------------------------
# Feature columns — must match rollout.py exactly
# ---------------------------------------------------------------------------
FEATURE_COLS_2D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t",
    "nbr_wl_mean_t", "nbr_wl_mean_tm1", "nbr_rain_sum_4",
    "position_x", "position_y",
    "area", "roughness", "min_elevation", "elevation",
    "aspect", "curvature", "flow_accumulation",
    # rainfall context
    "rain_frac_remaining", "rain_steps_since_peak", "rain_intensity_trend",
]

FEATURE_COLS_1D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t",
    "nbr_wl_mean_t", "nbr_wl_mean_tm1",
    "conn2d_wl_t", "conn2d_rain_sum_4",
    "position_x", "position_y",
    "depth", "invert_elevation", "surface_elevation", "base_area",
    # inlet_flow lags
    "inlet_flow_t", "inlet_flow_tm1", "inlet_flow_tm2",
    # rainfall context
    "rain_frac_remaining", "rain_steps_since_peak", "rain_intensity_trend",
]

FEATURE_COLS_EDGE = [
    "flow_t", "flow_tm1", "flow_tm2",
    "relative_position_x", "relative_position_y",
    "length", "diameter", "shape", "roughness", "slope",
]

# Dynamic feature names (change each rollout step)
_DYN_2D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t", "nbr_wl_mean_t", "nbr_wl_mean_tm1", "nbr_rain_sum_4",
    "rain_frac_remaining", "rain_steps_since_peak", "rain_intensity_trend",
]
_DYN_1D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t", "nbr_wl_mean_t", "nbr_wl_mean_tm1",
    "conn2d_wl_t", "conn2d_rain_sum_4",
    "inlet_flow_t", "inlet_flow_tm1", "inlet_flow_tm2",
    "rain_frac_remaining", "rain_steps_since_peak", "rain_intensity_trend",
]
_DYN_EDGE = ["flow_t", "flow_tm1", "flow_tm2"]

IDX_DYN_2D  = [FEATURE_COLS_2D.index(c)  for c in _DYN_2D]
IDX_DYN_1D  = [FEATURE_COLS_1D.index(c)  for c in _DYN_1D]
IDX_DYN_E   = [FEATURE_COLS_EDGE.index(c) for c in _DYN_EDGE]
IDX_STAT_2D = [i for i in range(len(FEATURE_COLS_2D))  if i not in IDX_DYN_2D]
IDX_STAT_1D = [i for i in range(len(FEATURE_COLS_1D))  if i not in IDX_DYN_1D]
IDX_STAT_E  = [i for i in range(len(FEATURE_COLS_EDGE)) if i not in IDX_DYN_E]

F2D  = len(FEATURE_COLS_2D)
F1D  = len(FEATURE_COLS_1D)
FEDG = len(FEATURE_COLS_EDGE)

LAMBDA_INLET = 0.3
LAMBDA_EDGE  = 0.3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_edge_index(adj: UndirectedEdges) -> torch.Tensor:
    src = torch.as_tensor(adj.src, dtype=torch.long)
    dst = torch.as_tensor(adj.dst, dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def compute_stats(
    parquet_paths: list[Path],
    edge_paths: list[Path],
) -> dict:
    """Welford single-pass stats across all node and edge parquets."""
    sum1d = sum2d = sum1d2 = sum2d2 = None
    n1d = n2d = 0
    t_sum = t_sum2 = 0.0
    t_n = 0
    # inlet_flow target stats
    ti_sum = ti_sum2 = 0.0
    ti_n = 0
    # edge flow target and feature stats
    se = se2 = None
    ne = 0
    te_sum = te_sum2 = 0.0
    te_n = 0

    for p in parquet_paths:
        df = pd.read_parquet(p).dropna(subset=["target"])
        df1 = df[df["node_type"] == 1]
        df2 = df[df["node_type"] == 2]

        for c in FEATURE_COLS_1D:
            if c not in df1.columns: df1 = df1.copy(); df1[c] = 0.0
        for c in FEATURE_COLS_2D:
            if c not in df2.columns: df2 = df2.copy(); df2[c] = 0.0

        x1 = df1[FEATURE_COLS_1D].fillna(0.0).to_numpy(np.float64)
        x2 = df2[FEATURE_COLS_2D].fillna(0.0).to_numpy(np.float64)
        y  = df["target"].to_numpy(np.float64)

        if sum1d is None:
            sum1d  = np.zeros(F1D,  np.float64)
            sum2d  = np.zeros(F2D,  np.float64)
            sum1d2 = np.zeros(F1D,  np.float64)
            sum2d2 = np.zeros(F2D,  np.float64)

        sum1d  += x1.sum(0);  sum1d2 += (x1**2).sum(0);  n1d += len(x1)
        sum2d  += x2.sum(0);  sum2d2 += (x2**2).sum(0);  n2d += len(x2)
        t_sum  += y.sum();    t_sum2 += (y**2).sum();     t_n += len(y)

        if "target_inlet_flow" in df1.columns:
            yi = df1["target_inlet_flow"].to_numpy(np.float64)
            ti_sum += yi.sum(); ti_sum2 += (yi**2).sum(); ti_n += len(yi)

    for ep in edge_paths:
        if not ep.exists(): continue
        dfe = pd.read_parquet(ep)
        for c in FEATURE_COLS_EDGE:
            if c not in dfe.columns: dfe[c] = 0.0
        xe = dfe[FEATURE_COLS_EDGE].fillna(0.0).to_numpy(np.float64)
        if se is None:
            se  = np.zeros(FEDG, np.float64)
            se2 = np.zeros(FEDG, np.float64)
        se  += xe.sum(0); se2 += (xe**2).sum(0); ne += len(xe)
        if "target_edge_flow" in dfe.columns:
            ye = dfe["target_edge_flow"].to_numpy(np.float64)
            te_sum += ye.sum(); te_sum2 += (ye**2).sum(); te_n += len(ye)

    def _std(s2, s, n):
        return float(np.sqrt(max(s2/n - (s/n)**2, 0.0)))

    mean1d = (sum1d/n1d).astype(np.float32)
    mean2d = (sum2d/n2d).astype(np.float32)
    std1d  = np.sqrt(np.maximum(sum1d2/n1d - (sum1d/n1d)**2, 0)).astype(np.float32)
    std2d  = np.sqrt(np.maximum(sum2d2/n2d - (sum2d/n2d)**2, 0)).astype(np.float32)
    t_mean = float(t_sum/t_n)
    t_std  = _std(t_sum2, t_sum, t_n)

    ti_mean = float(ti_sum/ti_n)  if ti_n > 0 else 0.0
    ti_std  = _std(ti_sum2, ti_sum, ti_n) if ti_n > 0 else 1.0

    mean_e  = (se/ne).astype(np.float32)  if ne > 0 else np.zeros(FEDG, np.float32)
    std_e   = np.sqrt(np.maximum(se2/ne - (se/ne)**2, 0)).astype(np.float32) if ne > 0 \
              else np.ones(FEDG, np.float32)
    te_mean = float(te_sum/te_n) if te_n > 0 else 0.0
    te_std  = _std(te_sum2, te_sum, te_n) if te_n > 0 else 1.0

    std1d = np.where(std1d < 1e-6, 1.0, std1d).astype(np.float32)
    std2d = np.where(std2d < 1e-6, 1.0, std2d).astype(np.float32)
    std_e = np.where(std_e < 1e-6, 1.0, std_e).astype(np.float32)

    return dict(
        feature_mean_1d=mean1d, feature_std_1d=std1d,
        feature_mean_2d=mean2d, feature_std_2d=std2d,
        feature_mean_edge=mean_e, feature_std_edge=std_e,
        target_mean=t_mean, target_std=max(t_std, 1e-6),
        target_inlet_mean=ti_mean, target_inlet_std=max(ti_std, 1e-6),
        target_edge_mean=te_mean, target_edge_std=max(te_std, 1e-6),
    )


def _precompute_rainfall_context(rain2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute three scalar rainfall context features for all T timesteps."""
    T = rain2.shape[0]
    rain_ts = rain2.max(axis=1).astype(np.float64)
    total = rain_ts.sum()
    cum = np.cumsum(rain_ts)
    frac = np.clip(1.0 - cum/total, 0.0, 1.0) if total > 0 else np.zeros(T)
    peak_t = int(np.argmax(rain_ts))
    steps_since_peak = np.clip(np.arange(T, dtype=np.float64) - peak_t, 0.0, None)
    trend = np.zeros(T, np.float64)
    for ti in range(9, T):
        trend[ti] = rain_ts[ti-4:ti+1].mean() - rain_ts[ti-9:ti-4].mean()
    return frac.astype(np.float32), steps_since_peak.astype(np.float32), trend.astype(np.float32)


# ---------------------------------------------------------------------------
# EventSequence
# ---------------------------------------------------------------------------

class EventSequence:
    def __init__(
        self,
        parquet_path: Path,
        edge_parquet_path: Path | None,
        static_2d: pd.DataFrame,
        static_1d: pd.DataFrame,
        edges_1d_static: pd.DataFrame,
        adj_2d: UndirectedEdges,
        adj_1d: UndirectedEdges,
        conn1d_to_2d: np.ndarray,
        stats: dict,
        n2: int,
        n1: int,
        n_edges: int,
        max_seq_len: int | None = None,
    ):
        df = pd.read_parquet(parquet_path).dropna(subset=["target"])

        self.adj_2d = adj_2d
        self.adj_1d = adj_1d
        self.conn1d_to_2d = conn1d_to_2d.astype(np.int32)
        self.n2 = n2; self.n1 = n1; self.n_edges = n_edges

        self.fmean2 = stats["feature_mean_2d"].astype(np.float32)
        self.fstd2  = stats["feature_std_2d"].astype(np.float32)
        self.fmean1 = stats["feature_mean_1d"].astype(np.float32)
        self.fstd1  = stats["feature_std_1d"].astype(np.float32)
        self.fmean_e = stats["feature_mean_edge"].astype(np.float32)
        self.fstd_e  = stats["feature_std_edge"].astype(np.float32)
        self.tmean  = float(stats["target_mean"])
        self.tstd   = float(stats["target_std"])
        self.ti_mean = float(stats["target_inlet_mean"])
        self.ti_std  = float(stats["target_inlet_std"])
        self.te_mean = float(stats["target_edge_mean"])
        self.te_std  = float(stats["target_edge_std"])

        df2 = df[df["node_type"] == 2].copy()
        df1 = df[df["node_type"] == 1].copy()
        t_common = sorted(set(df2["t"].unique()) & set(df1["t"].unique()))
        if not t_common:
            self.valid = False; return
        self.valid = True

        df2_by_t = {t: df2[df2["t"] == t].sort_values("node_id") for t in t_common}
        df1_by_t = {t: df1[df1["t"] == t].sort_values("node_id") for t in t_common}

        T = len(t_common)
        wl2    = np.zeros((T, n2),      np.float32)
        wl1    = np.zeros((T, n1),      np.float32)
        rain2  = np.zeros((T, n2),      np.float32)
        infl1  = np.zeros((T, n1),      np.float32)  # inlet_flow ground truth
        gt_wl2 = np.zeros((T, n2),      np.float32)
        gt_wl1 = np.zeros((T, n1),      np.float32)
        gt_infl= np.zeros((T, n1),      np.float32)

        for i, t in enumerate(t_common):
            r2 = df2_by_t[t]; r1 = df1_by_t[t]
            n2ids = r2["node_id"].to_numpy(np.int32)
            n1ids = r1["node_id"].to_numpy(np.int32)
            wl2[i, n2ids]  = r2["wl_t"].to_numpy(np.float32)
            wl1[i, n1ids]  = r1["wl_t"].to_numpy(np.float32)
            rain2[i, n2ids] = r2["rain_t"].to_numpy(np.float32) if "rain_t" in r2.columns else 0.0
            if "inlet_flow_t" in r1.columns:
                infl1[i, n1ids] = r1["inlet_flow_t"].to_numpy(np.float32)
            gt_wl2[i, n2ids] = (r2["wl_t"] + r2["target"]).to_numpy(np.float32)
            gt_wl1[i, n1ids] = (r1["wl_t"] + r1["target"]).to_numpy(np.float32)
            if "target_inlet_flow" in r1.columns:
                gt_infl[i, n1ids] = r1["target_inlet_flow"].to_numpy(np.float32)

        self.t_common   = np.array(t_common, np.int32)
        self.wl2        = wl2
        self.wl1        = wl1
        self.rain2      = rain2
        self.infl1      = infl1
        self.gt_wl2     = gt_wl2
        self.gt_wl1     = gt_wl1
        self.gt_infl    = gt_infl

        # Precompute rainfall context for all T timesteps
        self.rain_frac, self.rain_peak, self.rain_trend = _precompute_rainfall_context(rain2)

        # Edge flow ground truth
        eflow  = np.zeros((T, n_edges), np.float32)
        gt_ef  = np.zeros((T, n_edges), np.float32)
        if edge_parquet_path is not None and edge_parquet_path.exists():
            dfe = pd.read_parquet(edge_parquet_path)
            dfe_by_t = {t: dfe[dfe["t"] == t] for t in t_common if t in dfe["t"].values}
            for i, t in enumerate(t_common):
                if t not in dfe_by_t: continue
                re = dfe_by_t[t]
                eids = re["edge_id"].to_numpy(np.int32)
                valid_e = (eids >= 0) & (eids < n_edges)
                eids = eids[valid_e]
                if "flow_t" in re.columns:
                    eflow[i, eids] = re["flow_t"].to_numpy(np.float32)[valid_e]
                if "target_edge_flow" in re.columns:
                    gt_ef[i, eids] = re["target_edge_flow"].to_numpy(np.float32)[valid_e]
        self.eflow   = eflow
        self.gt_eflow = gt_ef

        # Find consecutive runs of length >= 2
        self.runs = []
        i = 0
        while i < T:
            j = i
            while j + 1 < T and self.t_common[j+1] == self.t_common[j] + 1:
                j += 1
            if j - i + 1 >= 2:
                self.runs.append((i, j - i + 1))
            i = j + 1

        self.max_seq_len = max_seq_len

        # Precompute normalised static features for nodes
        stat2_cols = [FEATURE_COLS_2D[i]  for i in IDX_STAT_2D]
        stat1_cols = [FEATURE_COLS_1D[i]  for i in IDX_STAT_1D]
        stat_e_cols= [FEATURE_COLS_EDGE[i] for i in IDX_STAT_E]

        s2 = static_2d.sort_values("node_idx").reset_index(drop=True)
        s1 = static_1d.sort_values("node_idx").reset_index(drop=True)
        se = edges_1d_static.sort_values("edge_idx").reset_index(drop=True)

        for c in stat2_cols:
            if c not in s2.columns: s2[c] = 0.0
        for c in stat1_cols:
            if c not in s1.columns: s1[c] = 0.0
        for c in stat_e_cols:
            if c not in se.columns: se[c] = 0.0

        raw2 = s2[stat2_cols].fillna(0.0).to_numpy(np.float32)
        raw1 = s1[stat1_cols].fillna(0.0).to_numpy(np.float32)
        rawe = se[stat_e_cols].fillna(0.0).to_numpy(np.float32)

        self.stat2_norm = ((raw2 - self.fmean2[IDX_STAT_2D]) / self.fstd2[IDX_STAT_2D]).astype(np.float32)
        self.stat1_norm = ((raw1 - self.fmean1[IDX_STAT_1D]) / self.fstd1[IDX_STAT_1D]).astype(np.float32)
        self.stat_e_norm= ((rawe - self.fmean_e[IDX_STAT_E]) / self.fstd_e[IDX_STAT_E]).astype(np.float32)
        np.nan_to_num(self.stat2_norm, copy=False)
        np.nan_to_num(self.stat1_norm, copy=False)
        np.nan_to_num(self.stat_e_norm, copy=False)

    def sample_start(self, K: int, rng: np.random.Generator) -> int | None:
        valid = [(s, l) for s, l in self.runs if l >= K + 1]
        if not valid: return None
        s, l = valid[rng.integers(len(valid))]
        max_s = s + l - K - 1
        return int(rng.integers(s, max_s + 1))

    def build_x2(self, wl2_t, wl2_tm1, wl2_tm2, wl2_tm3, wl2_tm4, wl2_tm5,
                  rain_t, rain_tm1, rain_tm2, rain_tm3, ti: int) -> np.ndarray:
        rain_sum_4 = rain_t + rain_tm1 + rain_tm2 + rain_tm3
        d_wl = wl2_t - wl2_tm1
        nbr_wl_t   = neighbor_mean(wl2_t,      self.adj_2d)
        nbr_wl_tm1 = neighbor_mean(wl2_tm1,    self.adj_2d)
        nbr_rsum4  = neighbor_mean(rain_sum_4,  self.adj_2d)
        dyn = np.stack([
            wl2_t, wl2_tm1, wl2_tm2, wl2_tm3, wl2_tm4, wl2_tm5,
            rain_t, rain_tm1, rain_tm2, rain_tm3, rain_sum_4, d_wl,
            nbr_wl_t, nbr_wl_tm1, nbr_rsum4,
            np.full(self.n2, self.rain_frac[ti],   np.float32),
            np.full(self.n2, self.rain_peak[ti],   np.float32),
            np.full(self.n2, self.rain_trend[ti],  np.float32),
        ], axis=-1)
        X2 = np.empty((self.n2, F2D), np.float32)
        X2[:, IDX_DYN_2D]  = dyn
        X2[:, IDX_STAT_2D] = self.stat2_norm
        X2 = (X2 - self.fmean2) / self.fstd2
        np.nan_to_num(X2, copy=False)
        return X2

    def build_x1(self, wl1_t, wl1_tm1, wl1_tm2, wl1_tm3, wl1_tm4, wl1_tm5,
                  wl2_t, rain_sum_4_2d,
                  infl_t, infl_tm1, infl_tm2, ti: int) -> np.ndarray:
        d_wl = wl1_t - wl1_tm1
        nbr_wl_t   = neighbor_mean(wl1_t,   self.adj_1d)
        nbr_wl_tm1 = neighbor_mean(wl1_tm1, self.adj_1d)
        conn_safe = self.conn1d_to_2d.copy()
        missing = conn_safe < 0
        conn_safe[missing] = 0
        conn2d_wl   = wl2_t[conn_safe];       conn2d_wl[missing]   = 0.0
        conn2d_rsum = rain_sum_4_2d[conn_safe]; conn2d_rsum[missing] = 0.0
        zeros = np.zeros(self.n1, np.float32)
        dyn = np.stack([
            wl1_t, wl1_tm1, wl1_tm2, wl1_tm3, wl1_tm4, wl1_tm5,
            zeros, zeros, zeros, zeros, zeros, d_wl,
            nbr_wl_t, nbr_wl_tm1,
            conn2d_wl, conn2d_rsum,
            infl_t, infl_tm1, infl_tm2,
            np.full(self.n1, self.rain_frac[ti],  np.float32),
            np.full(self.n1, self.rain_peak[ti],  np.float32),
            np.full(self.n1, self.rain_trend[ti], np.float32),
        ], axis=-1)
        X1 = np.empty((self.n1, F1D), np.float32)
        X1[:, IDX_DYN_1D]  = dyn
        X1[:, IDX_STAT_1D] = self.stat1_norm
        X1 = (X1 - self.fmean1) / self.fstd1
        np.nan_to_num(X1, copy=False)
        return X1

    def build_x_edge(self, eflow_t, eflow_tm1, eflow_tm2) -> np.ndarray:
        dyn = np.stack([eflow_t, eflow_tm1, eflow_tm2], axis=-1)  # (n_edges, 3)
        Xe = np.empty((self.n_edges, FEDG), np.float32)
        Xe[:, IDX_DYN_E]  = dyn
        Xe[:, IDX_STAT_E] = self.stat_e_norm
        Xe = (Xe - self.fmean_e) / self.fstd_e
        np.nan_to_num(Xe, copy=False)
        return Xe


# ---------------------------------------------------------------------------
# One-event rollout loss
# ---------------------------------------------------------------------------

def rollout_loss_one_event(
    seq: EventSequence,
    model: torch.nn.Module,
    edge_index_2d: torch.Tensor,
    edge_index_1d: torch.Tensor,
    device: torch.device,
    K: int,
    rng: np.random.Generator,
) -> torch.Tensor | None:

    start = seq.sample_start(K, rng)
    if start is None:
        return None

    def _gt2(i): return seq.wl2[max(i,0)].copy()
    def _gt1(i): return seq.wl1[max(i,0)].copy()
    def _gti(i): return seq.infl1[max(i,0)].copy()
    def _gte(i): return seq.eflow[max(i,0)].copy()

    # Initialise lag states from ground truth
    wl2_t,   wl2_tm1, wl2_tm2 = _gt2(start),   _gt2(start-1), _gt2(start-2)
    wl2_tm3, wl2_tm4, wl2_tm5 = _gt2(start-3),  _gt2(start-4), _gt2(start-5)
    wl1_t,   wl1_tm1, wl1_tm2 = _gt1(start),   _gt1(start-1), _gt1(start-2)
    wl1_tm3, wl1_tm4, wl1_tm5 = _gt1(start-3),  _gt1(start-4), _gt1(start-5)
    infl_t,  infl_tm1, infl_tm2 = _gti(start), _gti(start-1), _gti(start-2)
    eflow_t, eflow_tm1, eflow_tm2 = _gte(start), _gte(start-1), _gte(start-2)

    total_loss = torch.tensor(0.0, device=device)

    for k in range(K):
        ti = start + k
        rain_t   = seq.rain2[ti]
        rain_tm1 = seq.rain2[max(ti-1, 0)]
        rain_tm2 = seq.rain2[max(ti-2, 0)]
        rain_tm3 = seq.rain2[max(ti-3, 0)]
        rain_sum_4 = rain_t + rain_tm1 + rain_tm2 + rain_tm3

        X2 = seq.build_x2(wl2_t, wl2_tm1, wl2_tm2, wl2_tm3, wl2_tm4, wl2_tm5,
                           rain_t, rain_tm1, rain_tm2, rain_tm3, ti)
        X1 = seq.build_x1(wl1_t, wl1_tm1, wl1_tm2, wl1_tm3, wl1_tm4, wl1_tm5,
                           wl2_t, rain_sum_4,
                           infl_t, infl_tm1, infl_tm2, ti)
        Xe = seq.build_x_edge(eflow_t, eflow_tm1, eflow_tm2)

        x2_t = torch.as_tensor(X2, dtype=torch.float32, device=device)
        x1_t = torch.as_tensor(X1, dtype=torch.float32, device=device)
        xe_t = torch.as_tensor(Xe, dtype=torch.float32, device=device)

        d2_norm, d1_norm, inlet_norm, edge_norm = model(
            x2_t, edge_index_2d, x1_t, edge_index_1d, xe_t
        )

        # Denormalise and apply delta for WL (differentiable)
        wl2_next = (torch.as_tensor(wl2_t, dtype=torch.float32, device=device)
                    + d2_norm * seq.tstd + seq.tmean)
        wl1_next = (torch.as_tensor(wl1_t, dtype=torch.float32, device=device)
                    + d1_norm * seq.tstd + seq.tmean)

        # Flow outputs are absolute predictions (not deltas)
        inlet_next = inlet_norm * seq.ti_std + seq.ti_mean
        edge_next  = edge_norm  * seq.te_std + seq.te_mean

        # Ground truth targets
        gt2  = torch.as_tensor(seq.gt_wl2[ti],   dtype=torch.float32, device=device)
        gt1  = torch.as_tensor(seq.gt_wl1[ti],   dtype=torch.float32, device=device)
        gti  = torch.as_tensor(seq.gt_infl[ti],  dtype=torch.float32, device=device)
        gte  = torch.as_tensor(seq.gt_eflow[ti], dtype=torch.float32, device=device)

        loss2 = torch.sqrt(F.mse_loss(wl2_next, gt2) + 1e-8)
        loss1 = torch.sqrt(F.mse_loss(wl1_next, gt1) + 1e-8)
        lossi = torch.sqrt(F.mse_loss(inlet_next, gti) + 1e-8)
        losse = torch.sqrt(F.mse_loss(edge_next,  gte) + 1e-8)
        total_loss = total_loss + loss2 + loss1 + LAMBDA_INLET*lossi + LAMBDA_EDGE*losse

        # Update lag states with model predictions (TBPTT: detach from graph)
        wl2_new   = wl2_next.detach().cpu().numpy()
        wl1_new   = wl1_next.detach().cpu().numpy()
        infl_new  = inlet_next.detach().cpu().numpy()
        eflow_new = edge_next.detach().cpu().numpy()

        wl2_tm5, wl2_tm4, wl2_tm3, wl2_tm2, wl2_tm1, wl2_t = \
            wl2_tm4, wl2_tm3, wl2_tm2, wl2_tm1, wl2_t, wl2_new
        wl1_tm5, wl1_tm4, wl1_tm3, wl1_tm2, wl1_tm1, wl1_t = \
            wl1_tm4, wl1_tm3, wl1_tm2, wl1_tm1, wl1_t, wl1_new
        infl_tm2, infl_tm1, infl_t   = infl_tm1,  infl_t,  infl_new
        eflow_tm2, eflow_tm1, eflow_t = eflow_tm1, eflow_t, eflow_new

    return total_loss / K


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    PARQUET_DIR = Path(os.environ.get(
        "UFB_PARQUET_DIR",
        "/kaggle/input/urbanfloodbench-model2-parquet/model2_train_samples_parquet",
    ))
    EDGE_DIR    = Path(os.environ.get(
        "UFB_EDGE_DIR",
        "/kaggle/input/urbanfloodbench-model2-edges/model2_train_edges_parquet",
    ))
    STATIC_ROOT = Path(os.environ.get(
        "UFB_STATIC_ROOT",
        "/kaggle/input/datasets/radovanvranik/urbanfloodbench-static/urbanfloodbench-static",
    ))
    OUT_DIR    = Path(os.environ.get("UFB_OUT_DIR",   "/kaggle/working/gnn_model2_rollout"))
    MODEL_ID   = int(os.environ.get("UFB_MODEL_ID",   "2"))
    MAX_EVENTS = int(os.environ.get("MAX_EVENTS",     "999"))
    EPOCHS     = int(os.environ.get("GNN_EPOCHS",     "25"))
    K          = int(os.environ.get("GNN_K",          "6"))
    MAX_SEQ    = int(os.environ.get("GNN_MAX_SEQ",    "150"))
    HIDDEN_DIM = int(os.environ.get("GNN_HIDDEN",     "128"))
    LR         = float(os.environ.get("GNN_LR",       "1e-3"))
    RESUME     = os.environ.get("GNN_RESUME",         "")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ufb.io.static import load_model_static
    static = load_model_static(STATIC_ROOT, model_id=MODEL_ID, split="train")
    n2 = len(static.nodes_2d)
    n1 = len(static.nodes_1d)
    n_edges = len(static.edges_1d)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  n2={n2}  n1={n1}  n_edges={n_edges}")

    edge_index_2d = build_edge_index(static.adj_2d).to(device)
    edge_index_1d = build_edge_index(static.adj_1d).to(device)

    parquet_paths = sorted(PARQUET_DIR.glob(f"model{MODEL_ID}_train_event*.parquet"))[:MAX_EVENTS]
    edge_paths    = [EDGE_DIR / f"model{MODEL_ID}_train_event{p.stem.split('event')[1]}_edges.parquet"
                     for p in parquet_paths]
    if not parquet_paths:
        raise FileNotFoundError(f"No parquets in {PARQUET_DIR}")
    print(f"Events: {len(parquet_paths)}  K={K}  Epochs={EPOCHS}  LR={LR}")

    print("Computing normalisation stats...")
    stats = compute_stats(parquet_paths, edge_paths)
    print(f"  target_mean={stats['target_mean']:.6f}  target_std={stats['target_std']:.6f}")
    if abs(stats["target_mean"]) > 2.0:
        print("  WARNING: target_mean is large — check parquets.")
    else:
        print("  OK: target_mean near zero, delta conversion confirmed.")
    print(f"  inlet_mean={stats['target_inlet_mean']:.6f}  inlet_std={stats['target_inlet_std']:.6f}")
    print(f"  edge_mean={stats['target_edge_mean']:.6f}   edge_std={stats['target_edge_std']:.6f}")

    print("Loading event sequences...")
    sequences = []
    for p, ep in zip(parquet_paths, edge_paths):
        try:
            seq = EventSequence(
                parquet_path=p,
                edge_parquet_path=ep if ep.exists() else None,
                static_2d=static.nodes_2d,
                static_1d=static.nodes_1d,
                edges_1d_static=static.edges_1d,
                adj_2d=static.adj_2d,
                adj_1d=static.adj_1d,
                conn1d_to_2d=static.conn1d_to_2d,
                stats=stats,
                n2=n2, n1=n1, n_edges=n_edges,
                max_seq_len=MAX_SEQ,
            )
            if seq.valid and seq.runs:
                sequences.append(seq)
        except Exception as e:
            print(f"  [WARN] {p.name}: {e}")

    print(f"  {len(sequences)} valid sequences loaded.")
    if not sequences:
        raise RuntimeError("No valid sequences — check parquet paths.")

    model = Model1Net(
        in_dim_2d=F2D,
        in_dim_1d=F1D,
        in_dim_edge=FEDG,
        hidden_dim=HIDDEN_DIM,
        gnn_layers_2d=3,
        gnn_layers_1d=2,
        dropout=0.1,
    ).to(device)

    if RESUME:
        print(f"Resuming from {RESUME}")
        model.load_state_dict(torch.load(RESUME, map_location=device))

    opt       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.05)
    rng       = np.random.default_rng(42)

    best_loss  = float("inf")
    best_epoch = -1

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_steps    = 0
        random.shuffle(sequences)

        for seq in sequences:
            n_windows = min(MAX_SEQ, max(1, len(seq.runs) * 3))
            for _ in range(n_windows):
                loss = rollout_loss_one_event(
                    seq, model, edge_index_2d, edge_index_1d, device, K, rng
                )
                if loss is None:
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += float(loss.detach())
                n_steps    += 1

        scheduler.step()
        avg = total_loss / max(n_steps, 1)
        lr  = scheduler.get_last_lr()[0]
        print(f"[EPOCH {epoch:02d}] avg_loss={avg:.6f}  lr={lr:.2e}  steps={n_steps}")

        ckpt = OUT_DIR / f"model{MODEL_ID}_rollout_epoch{epoch:02d}.pt"
        torch.save(model.state_dict(), str(ckpt))
        print(f"  checkpoint → {ckpt}")

        if avg < best_loss:
            best_loss  = avg
            best_epoch = epoch
            torch.save(model.state_dict(), str(OUT_DIR / f"model{MODEL_ID}_state_dict.pt"))
            print(f"  *** new best (epoch {epoch}) → model{MODEL_ID}_state_dict.pt")

    print(f"\nTraining complete. Best epoch: {best_epoch}  loss: {best_loss:.6f}")

    meta = dict(
        model_id=MODEL_ID,
        feature_cols_1d=FEATURE_COLS_1D,
        feature_cols_2d=FEATURE_COLS_2D,
        feature_cols_edge=FEATURE_COLS_EDGE,
        feature_mean_1d=stats["feature_mean_1d"].tolist(),
        feature_std_1d =stats["feature_std_1d"].tolist(),
        feature_mean_2d=stats["feature_mean_2d"].tolist(),
        feature_std_2d =stats["feature_std_2d"].tolist(),
        feature_mean_edge=stats["feature_mean_edge"].tolist(),
        feature_std_edge =stats["feature_std_edge"].tolist(),
        target_mean=stats["target_mean"],
        target_std =stats["target_std"],
        target_inlet_mean=stats["target_inlet_mean"],
        target_inlet_std =stats["target_inlet_std"],
        target_edge_mean =stats["target_edge_mean"],
        target_edge_std  =stats["target_edge_std"],
        hidden_dim=HIDDEN_DIM,
        gnn_layers_2d=3,
        gnn_layers_1d=2,
        n_edges=n_edges,
    )
    meta_path = OUT_DIR / f"model{MODEL_ID}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta → {meta_path}")


if __name__ == "__main__":
    main()
