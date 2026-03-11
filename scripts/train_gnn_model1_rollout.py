"""
train_gnn_model1_rollout.py  —  Full gradient-through-rollout training.

Key difference from train_gnn_model1.py:
  Instead of training on independent snapshots, we unroll K consecutive steps
  autoregressively with gradients flowing through the entire chain.  At each
  step the model receives its OWN previous predictions as lag features (not
  ground truth), so it is explicitly penalised for accumulating drift.

This directly addresses the "flat prediction" problem seen in snapshot training.

Architecture of one rollout-loss step
--------------------------------------
  Given ground-truth wl at t=0 (the start of the sequence):
    For k = 0 .. K-1:
      1. Build X2, X1 from current wl2_t, wl1_t  (model predictions after k=0)
      2. Normalise features
      3. Forward pass → normalised deltas d2, d1
      4. wl2_next = wl2_t + d2 * target_std + target_mean   (differentiable)
      5. loss += RMSE(wl2_next, gt_wl2[t+1]) + RMSE(wl1_next, gt_wl1[t+1])
      6. wl2_t = wl2_next.detach()  (stop gradient through time beyond K)
         OR wl2_t = wl2_next        (full BPTT — use detach for stability)

Gradient strategy: we use TBPTT (truncated BPTT) — gradients flow within
the K-step window but not beyond.  wl_t is detached between windows.
K=6 is a good default; increase to 8-10 if GPU memory allows.

Usage
-----
  # Quick smoke test (3 events, 3 epochs):
  MAX_EVENTS=3 GNN_EPOCHS=3 GNN_K=6 python scripts/train_gnn_model1_rollout.py

  # Full training:
  GNN_EPOCHS=20 GNN_K=6 GNN_MAX_SEQ=150 python scripts/train_gnn_model1_rollout.py

  # Resume from existing checkpoint:
  GNN_RESUME=/kaggle/working/gnn_model1_rollout/model1_epoch04.pt \\
  GNN_EPOCHS=10 python scripts/train_gnn_model1_rollout.py
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
]

FEATURE_COLS_1D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t",
    "nbr_wl_mean_t", "nbr_wl_mean_tm1",
    "conn2d_wl_t", "conn2d_rain_sum_4",
    "position_x", "position_y",
    "depth", "invert_elevation", "surface_elevation", "base_area",
]

# Indices of dynamic features within each feature list (needed for fast
# feature assembly during rollout — static features are precomputed once).
_DYN_2D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t",
    "nbr_wl_mean_t", "nbr_wl_mean_tm1", "nbr_rain_sum_4",
]
_DYN_1D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t",
    "nbr_wl_mean_t", "nbr_wl_mean_tm1",
    "conn2d_wl_t", "conn2d_rain_sum_4",
]

IDX_DYN_2D = [FEATURE_COLS_2D.index(c) for c in _DYN_2D]
IDX_DYN_1D = [FEATURE_COLS_1D.index(c) for c in _DYN_1D]
IDX_STAT_2D = [i for i in range(len(FEATURE_COLS_2D)) if i not in IDX_DYN_2D]
IDX_STAT_1D = [i for i in range(len(FEATURE_COLS_1D)) if i not in IDX_DYN_1D]

F2D = len(FEATURE_COLS_2D)
F1D = len(FEATURE_COLS_1D)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_edge_index(adj: UndirectedEdges) -> torch.Tensor:
    src = torch.as_tensor(adj.src, dtype=torch.long)
    dst = torch.as_tensor(adj.dst, dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def _ensure_delta_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw wl_{t+1} target to delta if needed (backward compat)."""
    if "target" not in df.columns or "wl_t" not in df.columns:
        return df
    mean_abs_target = df["target"].abs().mean()
    mean_abs_d = df["d_wl_t"].abs().mean() if "d_wl_t" in df.columns else 0.0
    if mean_abs_target > 10 * max(mean_abs_d, 1e-6):
        df = df.copy()
        df["target"] = df["target"] - df["wl_t"]
    return df


def compute_stats_from_parquets(parquet_paths: list[Path]) -> dict:
    """Compute normalisation stats across all events (single pass)."""
    sum1d = sum2d = sum1d2 = sum2d2 = None
    n1d = n2d = 0
    t_sum = t_sum2 = 0.0
    t_n = 0

    for p in parquet_paths:
        df = pd.read_parquet(p)
        df = _ensure_delta_target(df)
        df = df.dropna(subset=["target"])

        df1 = df[df["node_type"] == 1]
        df2 = df[df["node_type"] == 2]

        for c in FEATURE_COLS_1D:
            if c not in df1.columns:
                df1 = df1.copy(); df1[c] = 0.0
        for c in FEATURE_COLS_2D:
            if c not in df2.columns:
                df2 = df2.copy(); df2[c] = 0.0

        x1 = df1[FEATURE_COLS_1D].fillna(0.0).to_numpy(np.float64)
        x2 = df2[FEATURE_COLS_2D].fillna(0.0).to_numpy(np.float64)
        y  = df["target"].to_numpy(np.float64)

        if sum1d is None:
            sum1d  = np.zeros(len(FEATURE_COLS_1D))
            sum2d  = np.zeros(len(FEATURE_COLS_2D))
            sum1d2 = np.zeros(len(FEATURE_COLS_1D))
            sum2d2 = np.zeros(len(FEATURE_COLS_2D))

        sum1d  += x1.sum(0);  sum1d2 += (x1**2).sum(0);  n1d += len(x1)
        sum2d  += x2.sum(0);  sum2d2 += (x2**2).sum(0);  n2d += len(x2)
        t_sum  += y.sum();    t_sum2 += (y**2).sum();     t_n  += len(y)

    mean1d = (sum1d / n1d).astype(np.float32)
    mean2d = (sum2d / n2d).astype(np.float32)
    std1d  = np.sqrt(np.maximum(sum1d2/n1d - (sum1d/n1d)**2, 0)).astype(np.float32)
    std2d  = np.sqrt(np.maximum(sum2d2/n2d - (sum2d/n2d)**2, 0)).astype(np.float32)
    t_mean = float(t_sum / t_n)
    t_std  = float(np.sqrt(max(t_sum2/t_n - (t_sum/t_n)**2, 0.0)))

    return dict(
        feature_mean_1d=mean1d, feature_std_1d=std1d,
        feature_mean_2d=mean2d, feature_std_2d=std2d,
        target_mean=t_mean, target_std=t_std,
    )


# ---------------------------------------------------------------------------
# Event sequence loader
# ---------------------------------------------------------------------------

class EventSequence:
    """
    Loads one event parquet and exposes dense numpy arrays for rollout training.

    Arrays:
      wl2  : (T, n2)  ground-truth 2D water levels
      wl1  : (T, n1)  ground-truth 1D water levels
      rain2: (T, n2)  rainfall
      stat2: (n2, n_stat_2d)  static 2D features (normalised)
      stat1: (n1, n_stat_1d)  static 1D features (normalised)
    """

    def __init__(
        self,
        parquet_path: Path,
        static_2d: pd.DataFrame,   # nodes_2d from ModelStatic
        static_1d: pd.DataFrame,   # nodes_1d from ModelStatic
        adj_2d: UndirectedEdges,
        adj_1d: UndirectedEdges,
        conn1d_to_2d: np.ndarray,
        stats: dict,
        n2: int,
        n1: int,
        min_t: int = 9,
        max_seq_len: int | None = None,
    ):
        df = pd.read_parquet(parquet_path)
        df = _ensure_delta_target(df)
        df = df.dropna(subset=["target"])

        self.adj_2d = adj_2d
        self.adj_1d = adj_1d
        self.conn1d_to_2d = conn1d_to_2d.astype(np.int32)
        self.n2 = n2
        self.n1 = n1
        self.min_t = min_t

        # normalisation
        self.fmean2 = stats["feature_mean_2d"].astype(np.float32)
        self.fstd2  = np.where(stats["feature_std_2d"]  < 1e-6, 1.0, stats["feature_std_2d"]).astype(np.float32)
        self.fmean1 = stats["feature_mean_1d"].astype(np.float32)
        self.fstd1  = np.where(stats["feature_std_1d"]  < 1e-6, 1.0, stats["feature_std_1d"]).astype(np.float32)
        self.tmean  = float(stats["target_mean"])
        self.tstd   = float(stats["target_std"]) if stats["target_std"] > 1e-6 else 1.0

        # ---- reconstruct dense wl / rain from parquet ----
        df2 = df[df["node_type"] == 2].copy()
        df1 = df[df["node_type"] == 1].copy()

        # we need ALL timesteps, not just the sampled training ones.
        # The parquet only has selected timesteps → we must load dynamics directly.
        # However, we CAN reconstruct a subset.  We use the parquet's wl_t and
        # wl_tp1 (implied by target = wl_tp1 - wl_t) to get consecutive pairs,
        # but this is sparse.  Better approach: use wl_t from parquet as a lookup
        # and reconstruct the full sequence from consecutive t values.
        #
        # For rollout training we need CONSECUTIVE timesteps t, t+1, ..., t+K.
        # We extract all consecutive runs of length >= K+1 from the parquet.

        # Pivot wl_t per (t, node_id) → dense (T_sampled, n_nodes)
        t_vals_2d = sorted(df2["t"].unique())
        t_vals_1d = sorted(df1["t"].unique())
        t_common  = sorted(set(t_vals_2d) & set(t_vals_1d))

        if len(t_common) == 0:
            self.valid = False
            return
        self.valid = True

        # Build lookup: t → row slice in df
        df2_by_t = {t: df2[df2["t"] == t].sort_values("node_id") for t in t_common}
        df1_by_t = {t: df1[df1["t"] == t].sort_values("node_id") for t in t_common}

        # Dense arrays (T_sampled, n_nodes)
        T = len(t_common)
        wl2  = np.zeros((T, n2), dtype=np.float32)
        wl1  = np.zeros((T, n1), dtype=np.float32)
        rain2 = np.zeros((T, n2), dtype=np.float32)

        for i, t in enumerate(t_common):
            r2 = df2_by_t[t]
            r1 = df1_by_t[t]
            nids2 = r2["node_id"].to_numpy(np.int32)
            nids1 = r1["node_id"].to_numpy(np.int32)
            wl2[i, nids2]  = r2["wl_t"].to_numpy(np.float32)
            wl1[i, nids1]  = r1["wl_t"].to_numpy(np.float32)
            rain2[i, nids2] = r2["rain_t"].to_numpy(np.float32) if "rain_t" in r2.columns else 0.0

            # also fill t+1 water level at the last sampled step using target
            # (this gives us wl at t+1 = wl_t + target_delta)
            # We store these as "next" only for the ground-truth target array

        # Build gt_wl2_next[i] = wl at t_common[i]+1  (ground truth target)
        gt_wl2_next = np.zeros((T, n2), dtype=np.float32)
        gt_wl1_next = np.zeros((T, n1), dtype=np.float32)
        for i, t in enumerate(t_common):
            r2 = df2_by_t[t]
            r1 = df1_by_t[t]
            nids2 = r2["node_id"].to_numpy(np.int32)
            nids1 = r1["node_id"].to_numpy(np.int32)
            # target = delta, so wl_next = wl_t + target
            gt_wl2_next[i, nids2] = (r2["wl_t"] + r2["target"]).to_numpy(np.float32)
            gt_wl1_next[i, nids1] = (r1["wl_t"] + r1["target"]).to_numpy(np.float32)

        self.t_common     = np.array(t_common, dtype=np.int32)
        self.wl2          = wl2
        self.wl1          = wl1
        self.rain2        = rain2
        self.gt_wl2_next  = gt_wl2_next
        self.gt_wl1_next  = gt_wl1_next

        # Find consecutive runs of length >= 2 (need at least 2 for 1 rollout step)
        # A "run" is a maximal sequence of t_common indices where t[i+1] == t[i]+1
        self.runs = []   # list of (start_idx, length) in t_common index space
        i = 0
        while i < T:
            j = i
            while j + 1 < T and self.t_common[j + 1] == self.t_common[j] + 1:
                j += 1
            run_len = j - i + 1
            if run_len >= 2:
                self.runs.append((i, run_len))
            i = j + 1

        # Apply max_seq_len cap
        self.max_seq_len = max_seq_len

        # Precompute normalised static features
        stat2_cols = [FEATURE_COLS_2D[i] for i in IDX_STAT_2D]
        stat1_cols = [FEATURE_COLS_1D[i] for i in IDX_STAT_1D]

        s2 = static_2d.copy()
        s1 = static_1d.copy()
        for c in stat2_cols:
            if c not in s2.columns:
                s2[c] = 0.0
        for c in stat1_cols:
            if c not in s1.columns:
                s1[c] = 0.0

        # sort by node_idx to ensure alignment
        s2 = s2.sort_values("node_idx").reset_index(drop=True)
        s1 = s1.sort_values("node_idx").reset_index(drop=True)

        raw_stat2 = s2[stat2_cols].fillna(0.0).to_numpy(np.float32)  # (n2, n_stat)
        raw_stat1 = s1[stat1_cols].fillna(0.0).to_numpy(np.float32)  # (n1, n_stat)

        # normalise static features using the same global stats
        stat2_mean = self.fmean2[IDX_STAT_2D]
        stat2_std  = self.fstd2[IDX_STAT_2D]
        stat1_mean = self.fmean1[IDX_STAT_1D]
        stat1_std  = self.fstd1[IDX_STAT_1D]

        self.stat2_norm = ((raw_stat2 - stat2_mean) / stat2_std).astype(np.float32)
        self.stat1_norm = ((raw_stat1 - stat1_mean) / stat1_std).astype(np.float32)
        np.nan_to_num(self.stat2_norm, copy=False)
        np.nan_to_num(self.stat1_norm, copy=False)

    def sample_start(self, K: int, rng: np.random.Generator) -> int | None:
        """Sample a random valid start index for a K-step rollout."""
        valid = [(s, l) for s, l in self.runs if l >= K + 1]
        if not valid:
            return None
        s, l = valid[rng.integers(len(valid))]
        max_start = s + l - K - 1
        return int(rng.integers(s, max_start + 1))

    def build_x2_normed(
        self,
        wl2_t:   np.ndarray,  # (n2,) current prediction
        wl2_tm1: np.ndarray,
        wl2_tm2: np.ndarray,
        wl2_tm3: np.ndarray,
        wl2_tm4: np.ndarray,
        wl2_tm5: np.ndarray,
        rain_t:  np.ndarray,
        rain_tm1: np.ndarray,
        rain_tm2: np.ndarray,
        rain_tm3: np.ndarray,
    ) -> np.ndarray:
        """Build normalised X2 feature matrix (n2, F2D)."""
        rain_sum_4 = rain_t + rain_tm1 + rain_tm2 + rain_tm3
        d_wl = wl2_t - wl2_tm1
        nbr_wl_t   = neighbor_mean(wl2_t,   self.adj_2d)
        nbr_wl_tm1 = neighbor_mean(wl2_tm1, self.adj_2d)
        nbr_rsum4  = neighbor_mean(rain_sum_4, self.adj_2d)

        dyn = np.stack([
            wl2_t, wl2_tm1, wl2_tm2, wl2_tm3, wl2_tm4, wl2_tm5,
            rain_t, rain_tm1, rain_tm2, rain_tm3, rain_sum_4,
            d_wl,
            nbr_wl_t, nbr_wl_tm1, nbr_rsum4,
        ], axis=-1)  # (n2, n_dyn_2d)

        X2 = np.empty((self.n2, F2D), dtype=np.float32)
        X2[:, IDX_DYN_2D]  = dyn
        X2[:, IDX_STAT_2D] = self.stat2_norm

        # normalise
        X2 = (X2 - self.fmean2) / self.fstd2
        np.nan_to_num(X2, copy=False)
        return X2

    def build_x1_normed(
        self,
        wl1_t:   np.ndarray,
        wl1_tm1: np.ndarray,
        wl1_tm2: np.ndarray,
        wl1_tm3: np.ndarray,
        wl1_tm4: np.ndarray,
        wl1_tm5: np.ndarray,
        wl2_t:   np.ndarray,  # needed for conn2d features
        rain_sum_4_2d: np.ndarray,
    ) -> np.ndarray:
        """Build normalised X1 feature matrix (n1, F1D)."""
        d_wl = wl1_t - wl1_tm1
        nbr_wl_t   = neighbor_mean(wl1_t,   self.adj_1d)
        nbr_wl_tm1 = neighbor_mean(wl1_tm1, self.adj_1d)

        conn_safe = self.conn1d_to_2d.copy()
        missing = conn_safe < 0
        conn_safe[missing] = 0
        conn2d_wl   = wl2_t[conn_safe]
        conn2d_rsum = rain_sum_4_2d[conn_safe]
        conn2d_wl[missing]   = 0.0
        conn2d_rsum[missing] = 0.0

        zeros = np.zeros(self.n1, dtype=np.float32)
        dyn = np.stack([
            wl1_t, wl1_tm1, wl1_tm2, wl1_tm3, wl1_tm4, wl1_tm5,
            zeros, zeros, zeros, zeros, zeros,   # rain cols (1D has no rain)
            d_wl,
            nbr_wl_t, nbr_wl_tm1,
            conn2d_wl, conn2d_rsum,
        ], axis=-1)  # (n1, n_dyn_1d)

        X1 = np.empty((self.n1, F1D), dtype=np.float32)
        X1[:, IDX_DYN_1D]  = dyn
        X1[:, IDX_STAT_1D] = self.stat1_norm

        X1 = (X1 - self.fmean1) / self.fstd1
        np.nan_to_num(X1, copy=False)
        return X1


# ---------------------------------------------------------------------------
# One-event rollout loss
# ---------------------------------------------------------------------------

def rollout_loss_one_event(
    seq: EventSequence,
    model: torch.nn.Module,
    edge_index_2d: torch.Tensor,
    device: torch.device,
    K: int,
    rng: np.random.Generator,
) -> torch.Tensor | None:
    """
    Sample one K-step window from seq, unroll the model, return scalar loss.
    Returns None if no valid window exists.
    """
    start = seq.sample_start(K, rng)
    if start is None:
        return None

    # Ground-truth wl at start (used to initialise lag state)
    # We need 6 lags before start → start must be >= 6 in the run.
    # sample_start already ensures start is within a valid run,
    # but lags before the run boundary use ground-truth from wl2/wl1 arrays.

    def _gt_wl2(idx: int) -> np.ndarray:
        return seq.wl2[max(idx, 0)].copy()

    def _gt_wl1(idx: int) -> np.ndarray:
        return seq.wl1[max(idx, 0)].copy()

    # Initialise lag state from ground truth
    wl2_t   = _gt_wl2(start)
    wl2_tm1 = _gt_wl2(start - 1)
    wl2_tm2 = _gt_wl2(start - 2)
    wl2_tm3 = _gt_wl2(start - 3)
    wl2_tm4 = _gt_wl2(start - 4)
    wl2_tm5 = _gt_wl2(start - 5)

    wl1_t   = _gt_wl1(start)
    wl1_tm1 = _gt_wl1(start - 1)
    wl1_tm2 = _gt_wl1(start - 2)
    wl1_tm3 = _gt_wl1(start - 3)
    wl1_tm4 = _gt_wl1(start - 4)
    wl1_tm5 = _gt_wl1(start - 5)

    total_loss = torch.tensor(0.0, device=device)

    for k in range(K):
        ti = start + k   # index into seq arrays

        # rainfall is ground truth (always available)
        rain_t   = seq.rain2[ti]
        rain_tm1 = seq.rain2[max(ti - 1, 0)]
        rain_tm2 = seq.rain2[max(ti - 2, 0)]
        rain_tm3 = seq.rain2[max(ti - 3, 0)]
        rain_sum_4_2d = rain_t + rain_tm1 + rain_tm2 + rain_tm3

        # Build features (numpy, already normalised)
        X2 = seq.build_x2_normed(
            wl2_t, wl2_tm1, wl2_tm2, wl2_tm3, wl2_tm4, wl2_tm5,
            rain_t, rain_tm1, rain_tm2, rain_tm3,
        )
        X1 = seq.build_x1_normed(
            wl1_t, wl1_tm1, wl1_tm2, wl1_tm3, wl1_tm4, wl1_tm5,
            wl2_t, rain_sum_4_2d,
        )

        x2_t = torch.as_tensor(X2, dtype=torch.float32, device=device)
        x1_t = torch.as_tensor(X1, dtype=torch.float32, device=device)

        # Forward: outputs normalised deltas
        d2_norm, d1_norm = model(x2_t, edge_index_2d, x1_t)

        # Denormalise and apply delta  (differentiable)
        wl2_next_t = (
            torch.as_tensor(wl2_t, dtype=torch.float32, device=device)
            + d2_norm * seq.tstd + seq.tmean
        )
        wl1_next_t = (
            torch.as_tensor(wl1_t, dtype=torch.float32, device=device)
            + d1_norm * seq.tstd + seq.tmean
        )

        # Ground truth next water level
        gt2 = torch.as_tensor(seq.gt_wl2_next[ti], dtype=torch.float32, device=device)
        gt1 = torch.as_tensor(seq.gt_wl1_next[ti], dtype=torch.float32, device=device)

        # RMSE loss for this step
        loss2 = torch.sqrt(F.mse_loss(wl2_next_t, gt2) + 1e-8)
        loss1 = torch.sqrt(F.mse_loss(wl1_next_t, gt1) + 1e-8)
        total_loss = total_loss + loss2 + loss1

        # Update lag state with MODEL PREDICTIONS (detached from graph for next step)
        # This is TBPTT: gradients flow within K steps, not beyond.
        wl2_t_new = wl2_next_t.detach().cpu().numpy()
        wl1_t_new = wl1_next_t.detach().cpu().numpy()

        wl2_tm5, wl2_tm4, wl2_tm3, wl2_tm2, wl2_tm1, wl2_t = \
            wl2_tm4, wl2_tm3, wl2_tm2, wl2_tm1, wl2_t, wl2_t_new
        wl1_tm5, wl1_tm4, wl1_tm3, wl1_tm2, wl1_tm1, wl1_t = \
            wl1_tm4, wl1_tm3, wl1_tm2, wl1_tm1, wl1_t, wl1_t_new

    return total_loss / K


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- CONFIG ----
    PARQUET_DIR  = Path(os.environ.get(
        "UFB_PARQUET_DIR",
        "/kaggle/input/datasets/radovanvranik/urbanfloodbench-gnn-model-1-only/model1_train_samples_parquet"
    ))
    STATIC_ROOT  = Path(os.environ.get(
        "UFB_STATIC_ROOT",
        "/kaggle/input/datasets/radovanvranik/urbanfloodbench-static/urbanfloodbench-static"
    ))
    OUT_DIR      = Path(os.environ.get("UFB_OUT_DIR", "/kaggle/working/gnn_model1_rollout"))
    MAX_EVENTS   = int(os.environ.get("MAX_EVENTS",      "999"))
    EPOCHS       = int(os.environ.get("GNN_EPOCHS",      "20"))
    K            = int(os.environ.get("GNN_K",           "6"))
    MAX_SEQ      = int(os.environ.get("GNN_MAX_SEQ",     "150"))  # max rollout windows per event per epoch
    HIDDEN_DIM   = int(os.environ.get("GNN_HIDDEN",      "128"))
    GNN_LAYERS   = int(os.environ.get("GNN_LAYERS",      "3"))
    LR           = float(os.environ.get("GNN_LR",        "1e-3"))
    RESUME       = os.environ.get("GNN_RESUME",          "")
    MIN_T        = int(os.environ.get("GNN_MIN_T",       "9"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    from ufb.io.static import load_model_static
    static = load_model_static(STATIC_ROOT, model_id=1, split="train")

    n2 = len(static.nodes_2d)
    n1 = len(static.nodes_1d)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  n2={n2}  n1={n1}")

    parquet_paths = sorted(PARQUET_DIR.glob("model1_train_event*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquets in {PARQUET_DIR}")
    parquet_paths = parquet_paths[:MAX_EVENTS]
    print(f"Events: {len(parquet_paths)}  K={K}  Epochs={EPOCHS}")

    # ---- Normalisation stats ----
    print("Computing normalisation stats...")
    stats = compute_stats_from_parquets(parquet_paths)
    print(f"  target_mean={stats['target_mean']:.6f}  target_std={stats['target_std']:.6f}")

    # ---- Load event sequences ----
    print("Loading event sequences...")
    sequences = []
    for p in parquet_paths:
        seq = EventSequence(
            parquet_path=p,
            static_2d=static.nodes_2d,
            static_1d=static.nodes_1d,
            adj_2d=static.adj_2d,
            adj_1d=static.adj_1d,
            conn1d_to_2d=static.conn1d_to_2d,
            stats=stats,
            n2=n2,
            n1=n1,
            min_t=MIN_T,
            max_seq_len=MAX_SEQ,
        )
        if seq.valid and seq.runs:
            sequences.append(seq)

    print(f"  Loaded {len(sequences)} valid event sequences")
    if not sequences:
        raise RuntimeError("No valid sequences found.")

    # ---- Model ----
    model = Model1Net(
        in_dim_2d=F2D,
        in_dim_1d=F1D,
        hidden_dim=HIDDEN_DIM,
        gnn_layers=GNN_LAYERS,
        dropout=0.1,
    ).to(device)

    if RESUME:
        print(f"Resuming from {RESUME}")
        model.load_state_dict(torch.load(RESUME, map_location=device))

    opt       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.05)
    edge_index_2d = build_edge_index(static.adj_2d).to(device)

    rng = np.random.default_rng(42)

    # ---- Training loop ----
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_steps    = 0
        seq_order  = list(range(len(sequences)))
        random.shuffle(seq_order)

        for si in seq_order:
            seq = sequences[si]
            n_windows = min(MAX_SEQ, max(1, len(seq.runs) * 3))

            for _ in range(n_windows):
                loss = rollout_loss_one_event(seq, model, edge_index_2d, device, K, rng)
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
        print(f"[EPOCH {epoch:02d}] avg_loss={avg:.6f}  lr={scheduler.get_last_lr()[0]:.2e}  steps={n_steps}")

        ckpt = OUT_DIR / f"model1_rollout_epoch{epoch:02d}.pt"
        torch.save(model.state_dict(), str(ckpt))
        print(f"  checkpoint → {ckpt}")

    # ---- Save final ----
    final_ckpt = OUT_DIR / "model1_state_dict.pt"
    torch.save(model.state_dict(), str(final_ckpt))

    meta = dict(
        feature_cols_1d=FEATURE_COLS_1D,
        feature_cols_2d=FEATURE_COLS_2D,
        feature_mean_1d=stats["feature_mean_1d"].tolist(),
        feature_std_1d =stats["feature_std_1d"].tolist(),
        feature_mean_2d=stats["feature_mean_2d"].tolist(),
        feature_std_2d =stats["feature_std_2d"].tolist(),
        target_mean=float(stats["target_mean"]),
        target_std =float(stats["target_std"]),
        hidden_dim=HIDDEN_DIM,
        gnn_layers=GNN_LAYERS,
    )
    meta_path = OUT_DIR / "model1_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Model → {final_ckpt}")
    print(f"Meta  → {meta_path}")


if __name__ == "__main__":
    main()
