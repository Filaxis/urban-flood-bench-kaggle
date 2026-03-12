"""
train_gnn_model2_rollout.py  —  Full gradient-through-rollout training for Model_2.

Key difference from Model_1: parquets store ABSOLUTE targets (wl_{t+1}).
This script converts to deltas on-the-fly: delta = target - wl_t.

Usage (Kaggle):
  Default paths assume parquets uploaded as dataset
  "urbanfloodbench-model2-parquet" (adjust UFB_PARQUET_DIR if different).

  Smoke test  (3 events, 3 epochs):
    MAX_EVENTS=3 GNN_EPOCHS=3 python train_gnn_model2_rollout.py

  Overnight run (20 epochs):
    GNN_EPOCHS=20 python train_gnn_model2_rollout.py

  Resume from checkpoint:
    GNN_RESUME=/kaggle/working/gnn_model2_rollout/model2_rollout_epoch09.pt \
    GNN_EPOCHS=10 python train_gnn_model2_rollout.py
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
]  # 24 features

FEATURE_COLS_1D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t",
    "nbr_wl_mean_t", "nbr_wl_mean_tm1",
    "conn2d_wl_t", "conn2d_rain_sum_4",
    "position_x", "position_y",
    "depth", "invert_elevation", "surface_elevation", "base_area",
]  # 22 features

# Which indices in feature vectors are dynamic (change each timestep)?
_DYN_2D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t", "nbr_wl_mean_t", "nbr_wl_mean_tm1", "nbr_rain_sum_4",
]
_DYN_1D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t", "nbr_wl_mean_t", "nbr_wl_mean_tm1",
    "conn2d_wl_t", "conn2d_rain_sum_4",
]

IDX_DYN_2D  = [FEATURE_COLS_2D.index(c) for c in _DYN_2D]
IDX_DYN_1D  = [FEATURE_COLS_1D.index(c) for c in _DYN_1D]
IDX_STAT_2D = [i for i in range(len(FEATURE_COLS_2D)) if i not in IDX_DYN_2D]
IDX_STAT_1D = [i for i in range(len(FEATURE_COLS_1D)) if i not in IDX_DYN_1D]

F2D = len(FEATURE_COLS_2D)
F1D = len(FEATURE_COLS_1D)


# ---------------------------------------------------------------------------
# Delta conversion
# ---------------------------------------------------------------------------

def to_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parquets for Model_2 store absolute wl_{t+1} as 'target'.
    Convert to delta: target = wl_{t+1} - wl_t.
    Guard: if mean(|target|) is near zero it's already a delta, skip.
    """
    mean_abs = df["target"].abs().mean()
    if mean_abs > 5.0:          # absolute water level (~30-55 m for Model_2)
        df = df.copy()
        df["target"] = df["target"] - df["wl_t"]
    return df


# ---------------------------------------------------------------------------
# Normalisation stats
# ---------------------------------------------------------------------------

def compute_stats(parquet_paths: list[Path]) -> dict:
    sum1d = sum2d = sum1d2 = sum2d2 = None
    n1d = n2d = 0
    t_sum = t_sum2 = 0.0
    t_n = 0

    for p in parquet_paths:
        df = pd.read_parquet(p)
        df = to_delta(df)
        df = df.dropna(subset=["target"])

        df1 = df[df["node_type"] == 1]
        df2 = df[df["node_type"] == 2]

        x1 = df1[FEATURE_COLS_1D].fillna(0.0).to_numpy(np.float64)
        x2 = df2[FEATURE_COLS_2D].fillna(0.0).to_numpy(np.float64)
        y  = df["target"].to_numpy(np.float64)

        if sum1d is None:
            sum1d  = np.zeros(F1D, np.float64)
            sum2d  = np.zeros(F2D, np.float64)
            sum1d2 = np.zeros(F1D, np.float64)
            sum2d2 = np.zeros(F2D, np.float64)

        sum1d  += x1.sum(0);  sum1d2 += (x1**2).sum(0);  n1d += len(x1)
        sum2d  += x2.sum(0);  sum2d2 += (x2**2).sum(0);  n2d += len(x2)
        t_sum  += y.sum();    t_sum2 += (y**2).sum();     t_n += len(y)

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
# EventSequence
# ---------------------------------------------------------------------------

class EventSequence:
    def __init__(
        self,
        parquet_path: Path,
        static_2d: pd.DataFrame,
        static_1d: pd.DataFrame,
        adj_2d: UndirectedEdges,
        adj_1d: UndirectedEdges,
        conn1d_to_2d: np.ndarray,
        stats: dict,
        n2: int,
        n1: int,
    ):
        df = pd.read_parquet(parquet_path)
        df = to_delta(df)
        df = df.dropna(subset=["target"])

        self.adj_2d = adj_2d
        self.adj_1d = adj_1d
        self.conn1d_to_2d = conn1d_to_2d.astype(np.int32)
        self.n2 = n2
        self.n1 = n1

        self.fmean2 = stats["feature_mean_2d"].astype(np.float32)
        self.fstd2  = np.where(stats["feature_std_2d"]  < 1e-6, 1.0, stats["feature_std_2d"]).astype(np.float32)
        self.fmean1 = stats["feature_mean_1d"].astype(np.float32)
        self.fstd1  = np.where(stats["feature_std_1d"]  < 1e-6, 1.0, stats["feature_std_1d"]).astype(np.float32)
        self.tmean  = float(stats["target_mean"])
        self.tstd   = float(stats["target_std"]) if stats["target_std"] > 1e-6 else 1.0

        df2 = df[df["node_type"] == 2].copy()
        df1 = df[df["node_type"] == 1].copy()

        t_common = sorted(set(df2["t"].unique()) & set(df1["t"].unique()))
        if not t_common:
            self.valid = False
            return
        self.valid = True

        df2_by_t = {t: df2[df2["t"] == t].sort_values("node_id") for t in t_common}
        df1_by_t = {t: df1[df1["t"] == t].sort_values("node_id") for t in t_common}

        T = len(t_common)
        wl2   = np.zeros((T, n2), np.float32)
        wl1   = np.zeros((T, n1), np.float32)
        rain2 = np.zeros((T, n2), np.float32)
        gt_wl2_next = np.zeros((T, n2), np.float32)
        gt_wl1_next = np.zeros((T, n1), np.float32)

        for i, t in enumerate(t_common):
            r2 = df2_by_t[t]; r1 = df1_by_t[t]
            n2ids = r2["node_id"].to_numpy(np.int32)
            n1ids = r1["node_id"].to_numpy(np.int32)
            wl2[i, n2ids]        = r2["wl_t"].to_numpy(np.float32)
            wl1[i, n1ids]        = r1["wl_t"].to_numpy(np.float32)
            rain2[i, n2ids]      = r2["rain_t"].to_numpy(np.float32) if "rain_t" in r2.columns else 0.0
            # gt_wl_next = wl_t + delta  (delta is already correct after to_delta)
            gt_wl2_next[i, n2ids] = (r2["wl_t"] + r2["target"]).to_numpy(np.float32)
            gt_wl1_next[i, n1ids] = (r1["wl_t"] + r1["target"]).to_numpy(np.float32)

        self.t_common    = np.array(t_common, np.int32)
        self.wl2         = wl2
        self.wl1         = wl1
        self.rain2       = rain2
        self.gt_wl2_next = gt_wl2_next
        self.gt_wl1_next = gt_wl1_next

        # Identify contiguous runs of timesteps
        self.runs = []
        i = 0
        while i < T:
            j = i
            while j + 1 < T and self.t_common[j+1] == self.t_common[j] + 1:
                j += 1
            if (j - i + 1) >= 2:
                self.runs.append((i, j - i + 1))
            i = j + 1

        # Pre-compute normalised static features
        stat2_cols = [FEATURE_COLS_2D[i] for i in IDX_STAT_2D]
        stat1_cols = [FEATURE_COLS_1D[i] for i in IDX_STAT_1D]

        s2 = static_2d.sort_values("node_idx").reset_index(drop=True)
        s1 = static_1d.sort_values("node_idx").reset_index(drop=True)
        for c in stat2_cols:
            if c not in s2.columns: s2[c] = 0.0
        for c in stat1_cols:
            if c not in s1.columns: s1[c] = 0.0

        raw2 = s2[stat2_cols].fillna(0.0).to_numpy(np.float32)
        raw1 = s1[stat1_cols].fillna(0.0).to_numpy(np.float32)
        self.stat2_norm = np.nan_to_num((raw2 - self.fmean2[IDX_STAT_2D]) / self.fstd2[IDX_STAT_2D])
        self.stat1_norm = np.nan_to_num((raw1 - self.fmean1[IDX_STAT_1D]) / self.fstd1[IDX_STAT_1D])

    def sample_start(self, K: int, rng: np.random.Generator) -> int | None:
        valid = [(s, l) for s, l in self.runs if l >= K + 1]
        if not valid:
            return None
        s, l = valid[rng.integers(len(valid))]
        return int(rng.integers(s, s + l - K))

    def build_x2(self, wl2_t, wl2_tm1, wl2_tm2, wl2_tm3, wl2_tm4, wl2_tm5,
                  rain_t, rain_tm1, rain_tm2, rain_tm3) -> np.ndarray:
        rain_sum_4 = rain_t + rain_tm1 + rain_tm2 + rain_tm3
        d_wl       = wl2_t - wl2_tm1
        nbr_wl_t   = neighbor_mean(wl2_t,     self.adj_2d)
        nbr_wl_tm1 = neighbor_mean(wl2_tm1,   self.adj_2d)
        nbr_rsum4  = neighbor_mean(rain_sum_4, self.adj_2d)
        dyn = np.stack([
            wl2_t, wl2_tm1, wl2_tm2, wl2_tm3, wl2_tm4, wl2_tm5,
            rain_t, rain_tm1, rain_tm2, rain_tm3, rain_sum_4,
            d_wl, nbr_wl_t, nbr_wl_tm1, nbr_rsum4,
        ], axis=-1)
        X2 = np.empty((self.n2, F2D), np.float32)
        X2[:, IDX_DYN_2D]  = dyn
        X2[:, IDX_STAT_2D] = self.stat2_norm
        return np.nan_to_num((X2 - self.fmean2) / self.fstd2)

    def build_x1(self, wl1_t, wl1_tm1, wl1_tm2, wl1_tm3, wl1_tm4, wl1_tm5,
                  wl2_t, rain_sum_4_2d) -> np.ndarray:
        d_wl       = wl1_t - wl1_tm1
        nbr_wl_t   = neighbor_mean(wl1_t,   self.adj_1d)
        nbr_wl_tm1 = neighbor_mean(wl1_tm1, self.adj_1d)
        conn = self.conn1d_to_2d.copy()
        miss = conn < 0; conn[miss] = 0
        conn2d_wl   = wl2_t[conn];        conn2d_wl[miss]   = 0.0
        conn2d_rsum = rain_sum_4_2d[conn]; conn2d_rsum[miss] = 0.0
        zeros = np.zeros(self.n1, np.float32)
        dyn = np.stack([
            wl1_t, wl1_tm1, wl1_tm2, wl1_tm3, wl1_tm4, wl1_tm5,
            zeros, zeros, zeros, zeros, zeros,
            d_wl, nbr_wl_t, nbr_wl_tm1,
            conn2d_wl, conn2d_rsum,
        ], axis=-1)
        X1 = np.empty((self.n1, F1D), np.float32)
        X1[:, IDX_DYN_1D]  = dyn
        X1[:, IDX_STAT_1D] = self.stat1_norm
        return np.nan_to_num((X1 - self.fmean1) / self.fstd1)


# ---------------------------------------------------------------------------
# Rollout loss (TBPTT, K steps)
# ---------------------------------------------------------------------------

def rollout_loss(seq: EventSequence, model, edge_index_2d, device, K: int,
                 rng: np.random.Generator) -> torch.Tensor | None:
    start = seq.sample_start(K, rng)
    if start is None:
        return None

    def _wl2(i): return seq.wl2[max(i, 0)].copy()
    def _wl1(i): return seq.wl1[max(i, 0)].copy()

    wl2_t, wl2_tm1, wl2_tm2 = _wl2(start), _wl2(start-1), _wl2(start-2)
    wl2_tm3, wl2_tm4, wl2_tm5 = _wl2(start-3), _wl2(start-4), _wl2(start-5)
    wl1_t, wl1_tm1, wl1_tm2 = _wl1(start), _wl1(start-1), _wl1(start-2)
    wl1_tm3, wl1_tm4, wl1_tm5 = _wl1(start-3), _wl1(start-4), _wl1(start-5)

    total_loss = torch.tensor(0.0, device=device)

    for k in range(K):
        ti = start + k
        rain_t   = seq.rain2[ti]
        rain_tm1 = seq.rain2[max(ti-1, 0)]
        rain_tm2 = seq.rain2[max(ti-2, 0)]
        rain_tm3 = seq.rain2[max(ti-3, 0)]
        rain_sum_4 = rain_t + rain_tm1 + rain_tm2 + rain_tm3

        X2 = seq.build_x2(wl2_t, wl2_tm1, wl2_tm2, wl2_tm3, wl2_tm4, wl2_tm5,
                           rain_t, rain_tm1, rain_tm2, rain_tm3)
        X1 = seq.build_x1(wl1_t, wl1_tm1, wl1_tm2, wl1_tm3, wl1_tm4, wl1_tm5,
                           wl2_t, rain_sum_4)

        x2_t = torch.as_tensor(X2, dtype=torch.float32, device=device)
        x1_t = torch.as_tensor(X1, dtype=torch.float32, device=device)

        d2_norm, d1_norm = model(x2_t, edge_index_2d, x1_t)

        wl2_next_t = (torch.as_tensor(wl2_t, dtype=torch.float32, device=device)
                      + d2_norm * seq.tstd + seq.tmean)
        wl1_next_t = (torch.as_tensor(wl1_t, dtype=torch.float32, device=device)
                      + d1_norm * seq.tstd + seq.tmean)

        gt2 = torch.as_tensor(seq.gt_wl2_next[ti], dtype=torch.float32, device=device)
        gt1 = torch.as_tensor(seq.gt_wl1_next[ti], dtype=torch.float32, device=device)

        loss2 = torch.sqrt(F.mse_loss(wl2_next_t, gt2) + 1e-8)
        loss1 = torch.sqrt(F.mse_loss(wl1_next_t, gt1) + 1e-8)
        total_loss = total_loss + loss2 + loss1

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
    # ---- Paths (override with env vars on Kaggle) ----
    PARQUET_DIR = Path(os.environ.get(
        "UFB_PARQUET_DIR",
        "/kaggle/input/urbanfloodbench-model2-parquet/model2_train_samples_parquet",
    ))
    STATIC_ROOT = Path(os.environ.get(
        "UFB_STATIC_ROOT",
        "/kaggle/input/urbanfloodbench-static/urbanfloodbench-static",
    ))
    OUT_DIR    = Path(os.environ.get("UFB_OUT_DIR",   "/kaggle/working/gnn_model2_rollout"))
    MAX_EVENTS = int(os.environ.get("MAX_EVENTS",     "999"))
    EPOCHS     = int(os.environ.get("GNN_EPOCHS",     "20"))
    K          = int(os.environ.get("GNN_K",          "6"))
    HIDDEN_DIM = int(os.environ.get("GNN_HIDDEN",     "128"))
    GNN_LAYERS = int(os.environ.get("GNN_LAYERS",     "3"))
    LR         = float(os.environ.get("GNN_LR",       "1e-3"))
    RESUME     = os.environ.get("GNN_RESUME",         "")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Static graph ----
    from ufb.io.static import load_model_static
    static = load_model_static(STATIC_ROOT, model_id=2, split="train")
    n2 = len(static.nodes_2d)
    n1 = len(static.nodes_1d)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  n2={n2}  n1={n1}")

    # ---- Parquets ----
    parquet_paths = sorted(PARQUET_DIR.glob("model2_train_event*.parquet"))[:MAX_EVENTS]
    if not parquet_paths:
        raise FileNotFoundError(f"No parquets found in {PARQUET_DIR}")
    print(f"Events: {len(parquet_paths)}  K={K}  Epochs={EPOCHS}  LR={LR}")

    # ---- Normalisation stats ----
    print("Computing normalisation stats...")
    stats = compute_stats(parquet_paths)
    print(f"  target_mean={stats['target_mean']:.6f}  target_std={stats['target_std']:.6f}")
    # Sanity check: after delta conversion target_mean should be near zero
    if abs(stats["target_mean"]) > 2.0:
        print("  WARNING: target_mean is large — delta conversion may have failed. Check parquets.")
    else:
        print("  OK: target_mean near zero, delta conversion confirmed.")

    # ---- Load sequences ----
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
            n2=n2, n1=n1,
        )
        if seq.valid and seq.runs:
            sequences.append(seq)
    print(f"  {len(sequences)} valid sequences loaded.")
    if not sequences:
        raise RuntimeError("No valid sequences — check parquet contents.")

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
    edge_index_2d = torch.stack([
        torch.as_tensor(static.adj_2d.src, dtype=torch.long),
        torch.as_tensor(static.adj_2d.dst, dtype=torch.long),
    ], dim=0).to(device)

    rng = np.random.default_rng(42)

    # ---- Training loop ----
    best_loss  = float("inf")
    best_epoch = -1

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_steps    = 0
        random.shuffle(sequences)

        for seq in sequences:
            n_windows = max(1, len(seq.runs) * 3)
            for _ in range(n_windows):
                loss = rollout_loss(seq, model, edge_index_2d, device, K, rng)
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

        ckpt = OUT_DIR / f"model2_rollout_epoch{epoch:02d}.pt"
        torch.save(model.state_dict(), str(ckpt))
        print(f"  checkpoint → {ckpt}")

        if avg < best_loss:
            best_loss  = avg
            best_epoch = epoch
            best_ckpt  = OUT_DIR / "model2_state_dict.pt"
            torch.save(model.state_dict(), str(best_ckpt))
            print(f"  *** new best (epoch {epoch}) → {best_ckpt}")

    print(f"\nTraining complete. Best epoch: {best_epoch}  loss: {best_loss:.6f}")

    # ---- Save meta ----
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
    meta_path = OUT_DIR / "model2_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta → {meta_path}")


if __name__ == "__main__":
    main()
