"""
train_gnn_model1.py  —  corrected, production-ready version.

Key fixes vs. original:
  - Trains on ALL event parquets (streams one at a time to keep RAM low)
  - Removes phantom 'deg' feature (never existed in parquets)
  - Removes 1D-only features from feature_cols_2d (conn2d_* columns)
  - Computes delta target on-the-fly if parquet still has raw wl_{t+1}
    (backward-compatible: if target is already delta it stays delta)
  - Saves a checkpoint after each epoch so Kaggle freezes don't lose work
  - MAX_EVENTS env-var for quick smoke-tests (e.g. MAX_EVENTS=5)
  - Feature stats computed across ALL events (single pass) before training

Usage on Kaggle:
    MAX_EVENTS=5 python scripts/train_gnn_model1.py    # quick test
    python scripts/train_gnn_model1.py                 # full training
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ufb.models.gnn_py import Model1Net
from ufb.training.gnn_dataset import Model1SnapshotDataset


# ---------------------------------------------------------------------------
# Feature columns — MUST match rollout.py exactly
# ---------------------------------------------------------------------------
FEATURE_COLS_2D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t",
    "nbr_wl_mean_t", "nbr_wl_mean_tm1", "nbr_rain_sum_4",
    # static 2D node features
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
    # static 1D node features
    "position_x", "position_y",
    "depth", "invert_elevation", "surface_elevation", "base_area",
]


def build_edge_index(adj) -> torch.Tensor:
    src = torch.as_tensor(adj.src, dtype=torch.long)
    dst = torch.as_tensor(adj.dst, dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def _ensure_delta_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the parquet was built with the OLD samples.py (target = raw wl_{t+1}),
    convert it to delta in-place.  We detect this by checking whether the mean
    absolute target is much larger than the mean absolute wl_t change.
    If target is already a delta (new samples.py) this is a no-op.
    """
    if "target" not in df.columns or "wl_t" not in df.columns:
        return df
    # Heuristic: |mean(target)| >> |mean(d_wl_t)| → target is raw wl, not delta
    mean_abs_target = df["target"].abs().mean()
    mean_abs_d = df["d_wl_t"].abs().mean() if "d_wl_t" in df.columns else 0.0
    if mean_abs_target > 10 * max(mean_abs_d, 1e-6):
        # OLD format: convert
        df = df.copy()
        df["target"] = df["target"] - df["wl_t"]
    return df


def compute_stats_from_parquets(parquet_paths: list[Path]) -> dict:
    """Single-pass Welford online mean/std across all events."""
    sum1d  = None
    sum2d  = None
    sum1d2 = None
    sum2d2 = None
    n1d = 0
    n2d = 0
    t_sum = 0.0
    t_sum2 = 0.0
    t_n = 0

    for p in parquet_paths:
        df = pd.read_parquet(p)
        df = _ensure_delta_target(df)
        df = df.dropna(subset=["target"])

        df1 = df[df["node_type"] == 1]
        df2 = df[df["node_type"] == 2]

        # fill missing feature columns with 0
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

        sum1d  += x1.sum(axis=0); sum1d2 += (x1 ** 2).sum(axis=0); n1d += len(x1)
        sum2d  += x2.sum(axis=0); sum2d2 += (x2 ** 2).sum(axis=0); n2d += len(x2)
        t_sum  += y.sum();  t_sum2 += (y ** 2).sum(); t_n += len(y)

    mean1d = (sum1d / n1d).astype(np.float32)
    mean2d = (sum2d / n2d).astype(np.float32)
    std1d  = np.sqrt(np.maximum(sum1d2 / n1d - (sum1d / n1d) ** 2, 0)).astype(np.float32)
    std2d  = np.sqrt(np.maximum(sum2d2 / n2d - (sum2d / n2d) ** 2, 0)).astype(np.float32)
    t_mean = float(t_sum / t_n)
    t_std  = float(np.sqrt(max(t_sum2 / t_n - (t_sum / t_n) ** 2, 0.0)))

    return dict(
        feature_mean_1d=mean1d,
        feature_std_1d=std1d,
        feature_mean_2d=mean2d,
        feature_std_2d=std2d,
        target_mean=t_mean,
        target_std=t_std,
    )


def train_on_parquet(
    parquet_path: Path,
    model: torch.nn.Module,
    edge_index_2d: torch.Tensor,
    opt: torch.optim.Optimizer,
    stats: dict,
    node_ids_1d: np.ndarray,
    node_ids_2d: np.ndarray,
    device: torch.device,
    K: int = 4,
    max_starts: int = 200,
    event_id: int = 0,
) -> float:
    """Train on a single event parquet, return average loss."""
    df = pd.read_parquet(parquet_path)
    df = _ensure_delta_target(df)
    df = df.dropna(subset=["target"])
    df["event_id"] = event_id  # ensure consistent column name

    dataset = Model1SnapshotDataset(
        df=df,
        node_ids_1d=node_ids_1d,
        node_ids_2d=node_ids_2d,
        feature_cols_1d=FEATURE_COLS_1D,
        feature_cols_2d=FEATURE_COLS_2D,
        feature_mean_1d=stats["feature_mean_1d"],
        feature_std_1d=stats["feature_std_1d"],
        feature_mean_2d=stats["feature_mean_2d"],
        feature_std_2d=stats["feature_std_2d"],
        target_mean=stats["target_mean"],
        target_std=stats["target_std"],
        group_event_col="event_id",
        group_time_col="t",
        node_id_col="node_id",
        node_type_col="node_type",
        target_col="target",
    )

    if len(dataset) < K:
        return 0.0

    valid_starts = list(range(len(dataset) - K + 1))
    random.shuffle(valid_starts)
    starts = valid_starts[:min(max_starts, len(valid_starts))]

    total_loss = 0.0
    n = 0
    model.train()

    for s in starts:
        loss = torch.tensor(0.0, device=device)
        for k in range(K):
            snap = dataset[s + k]
            x2 = snap.x2d.to(device)
            y2 = snap.y2d.to(device)
            x1 = snap.x1d.to(device)
            y1 = snap.y1d.to(device)

            d2, d1 = model(x2, edge_index_2d, x1)
            loss = loss + torch.sqrt(torch.mean((d2 - y2) ** 2)) \
                        + torch.sqrt(torch.mean((d1 - y1) ** 2))

        loss = loss / K
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += float(loss.detach())
        n += 1

    return total_loss / max(n, 1)


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
    OUT_DIR      = Path(os.environ.get("UFB_OUT_DIR", "/kaggle/working/gnn_model1"))
    MAX_EVENTS   = int(os.environ.get("MAX_EVENTS", "999"))   # set to e.g. 5 for quick test
    EPOCHS       = int(os.environ.get("GNN_EPOCHS", "5"))
    K            = int(os.environ.get("GNN_K", "4"))
    MAX_STARTS   = int(os.environ.get("GNN_MAX_STARTS", "200"))
    HIDDEN_DIM   = int(os.environ.get("GNN_HIDDEN", "128"))
    GNN_LAYERS   = int(os.environ.get("GNN_LAYERS", "3"))
    LR           = float(os.environ.get("GNN_LR", "2e-3"))
    RESUME       = os.environ.get("GNN_RESUME", "")  # path to existing .pt to resume from

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load static ----
    from ufb.io.static import load_model_static
    static = load_model_static(STATIC_ROOT, model_id=1, split="train")

    node_ids_1d = static.nodes_1d["node_idx"].to_numpy()
    node_ids_2d = static.nodes_2d["node_idx"].to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Discover parquets ----
    parquet_paths = sorted(PARQUET_DIR.glob("model1_train_event*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquets found in {PARQUET_DIR}")
    parquet_paths = parquet_paths[:MAX_EVENTS]
    print(f"Training on {len(parquet_paths)} event parquets  (MAX_EVENTS={MAX_EVENTS})")

    # ---- Compute stats across ALL selected events ----
    print("Computing normalisation stats...")
    stats = compute_stats_from_parquets(parquet_paths)
    print(f"  target_mean={stats['target_mean']:.6f}  target_std={stats['target_std']:.6f}")

    # ---- Build model ----
    model = Model1Net(
        in_dim_2d=len(FEATURE_COLS_2D),
        in_dim_1d=len(FEATURE_COLS_1D),
        hidden_dim=HIDDEN_DIM,
        gnn_layers=GNN_LAYERS,
        dropout=0.1,
    ).to(device)

    if RESUME:
        print(f"Resuming from {RESUME}")
        model.load_state_dict(torch.load(RESUME, map_location=device))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.1)

    edge_index_2d = build_edge_index(static.adj_2d).to(device)

    # ---- Training loop ----
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        epoch_events = 0
        shuffled = parquet_paths.copy()
        random.shuffle(shuffled)

        for i, p in enumerate(shuffled):
            ev_loss = train_on_parquet(
                parquet_path=p,
                model=model,
                edge_index_2d=edge_index_2d,
                opt=opt,
                stats=stats,
                node_ids_1d=node_ids_1d,
                node_ids_2d=node_ids_2d,
                device=device,
                K=K,
                max_starts=MAX_STARTS,
                event_id=i,
            )
            epoch_loss += ev_loss
            epoch_events += 1

            if (i + 1) % 10 == 0:
                print(f"  epoch={epoch}  event={i+1}/{len(shuffled)}  "
                      f"avg_loss_so_far={epoch_loss/epoch_events:.6f}")

        scheduler.step()
        avg = epoch_loss / max(epoch_events, 1)
        print(f"[EPOCH {epoch}] avg_loss={avg:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint after every epoch
        ckpt = OUT_DIR / f"model1_epoch{epoch:02d}.pt"
        torch.save(model.state_dict(), str(ckpt))
        print(f"  checkpoint saved → {ckpt}")

    # ---- Save final model + meta ----
    final_ckpt = OUT_DIR / "model1_state_dict.pt"
    torch.save(model.state_dict(), str(final_ckpt))

    meta = dict(
        feature_cols_1d=FEATURE_COLS_1D,
        feature_cols_2d=FEATURE_COLS_2D,
        feature_mean_1d=stats["feature_mean_1d"].tolist(),
        feature_std_1d=stats["feature_std_1d"].tolist(),
        feature_mean_2d=stats["feature_mean_2d"].tolist(),
        feature_std_2d=stats["feature_std_2d"].tolist(),
        target_mean=float(stats["target_mean"]),
        target_std=float(stats["target_std"]),
        hidden_dim=HIDDEN_DIM,
        gnn_layers=GNN_LAYERS,
    )
    meta_path = OUT_DIR / "model1_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Model saved to {final_ckpt}")
    print(f"Meta saved to {meta_path}")


if __name__ == "__main__":
    main()
