# scripts/train_gnn_model1.py
from __future__ import annotations

import os
import json
import pandas as pd
import numpy as np
import random
from pathlib import Path
import torch
#from torch.utils.data import DataLoader

from ufb.models.gnn_py import Model1Net
from ufb.training.gnn_dataset import Model1SnapshotDataset


def build_edge_index_from_adj(adj) -> torch.Tensor:
    # adj is your UndirectedEdges (has src, dst arrays)
    src = torch.as_tensor(adj.src, dtype=torch.long)
    dst = torch.as_tensor(adj.dst, dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def main():
    # ---- CONFIG (edit these) ----
    parquet_path = os.environ.get("UFB_TRAIN_PARQUET", "/kaggle/input/datasets/radovanvranik/urbanfloodbench-gnn-model-1-only/model1_train_samples_parquet/model1_train_event001.parquet")
    out_dir = os.environ.get("UFB_OUT_DIR", "/kaggle/working/gnn_model1")
    os.makedirs(out_dir, exist_ok=True)

    # You will load these from your existing static loader in Kaggle
    # For skeleton: assume you can import and call it.
    from ufb.io.static import load_model_static  # adjust if your path differs
    STATIC_ROOT = Path("/kaggle/input/datasets/radovanvranik/urbanfloodbench-static/urbanfloodbench-static")
    static = load_model_static(STATIC_ROOT, model_id=1, split="train")

    # Feature columns (reuse your existing CSVs or hardcode for MVP)
    # IMPORTANT: for MVP, ensure these match parquet columns exactly.
    feature_cols_2d = [
        "wl_t","wl_tm1","wl_tm2","wl_tm3","wl_tm4","wl_tm5",
        "rain_t","rain_tm1","rain_tm2","rain_tm3","rain_sum_4",
        "d_wl_t","nbr_wl_mean_t","nbr_wl_mean_tm1",
        "conn2d_wl_t","conn2d_rain_sum_4",
        "position_x","position_y",
        "area","roughness","min_elevation","elevation",
        "aspect","curvature","flow_accumulation","deg",
        "nbr_rain_sum_4",
    ]
    feature_cols_1d = [
        "wl_t","wl_tm1","wl_tm2","wl_tm3","wl_tm4","wl_tm5",
        "rain_t","rain_tm1","rain_tm2","rain_tm3","rain_sum_4",
        "d_wl_t","nbr_wl_mean_t","nbr_wl_mean_tm1",
        "conn2d_wl_t","conn2d_rain_sum_4",
        "position_x","position_y","depth","invert_elevation",
        "surface_elevation","base_area","deg",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_ids_1d = static.nodes_1d["node_idx"].to_numpy()
    node_ids_2d = static.nodes_2d["node_idx"].to_numpy()

    df = pd.read_parquet(parquet_path)
    df["event_id"] = 0

    missing_1d = [c for c in feature_cols_1d if c not in df.columns]
    missing_2d = [c for c in feature_cols_2d if c not in df.columns]

    print("missing_1d:", missing_1d)
    print("missing_2d:", missing_2d)

    if missing_1d or missing_2d:
        raise ValueError("Feature column mismatch with parquet schema")

    DEBUG_MAX_EVENTS = 3
    K = 4
    MAX_STARTS = 100
    EPOCHS = 2
    
    # if DEBUG_MAX_EVENTS is not None:
    #     if "event_id" not in df.columns:
    #         raise ValueError("event_id column required for DEBUG_MAX_EVENTS subsetting")
    #     keep_events = sorted(df["event_id"].dropna().unique())[:DEBUG_MAX_EVENTS]
    #     df = df[df["event_id"].isin(keep_events)].copy()

    tmp_path = "/kaggle/working/tmp_debug_subset.parquet"
    df.to_parquet(tmp_path)

    feature_mean_1d = df[feature_cols_1d].mean().to_numpy(np.float32)
    feature_std_1d = df[feature_cols_1d].std().to_numpy(np.float32)
    feature_mean_2d = df[feature_cols_2d].mean().to_numpy(np.float32)
    feature_std_2d = df[feature_cols_2d].std().to_numpy(np.float32)

    feature_mean_1d = np.nan_to_num(feature_mean_1d, nan=0.0, posinf=0.0, neginf=0.0)
    feature_mean_2d = np.nan_to_num(feature_mean_2d, nan=0.0, posinf=0.0, neginf=0.0)
    feature_std_1d = np.nan_to_num(feature_std_1d, nan=1.0, posinf=1.0, neginf=1.0)
    feature_std_2d = np.nan_to_num(feature_std_2d, nan=1.0, posinf=1.0, neginf=1.0)

    feature_std_1d = np.where(feature_std_1d < 1e-6, 1.0, feature_std_1d)
    feature_std_2d = np.where(feature_std_2d < 1e-6, 1.0, feature_std_2d)

    target_mean = float(df["target"].mean())
    target_std = float(df["target"].std())

    if not np.isfinite(target_mean):
        target_mean = 0.0
    if not np.isfinite(target_std) or target_std < 1e-6:
        target_std = 1.0

    dataset = Model1SnapshotDataset(
        parquet_path=tmp_path,
        node_ids_1d=node_ids_1d,
        node_ids_2d=node_ids_2d,
        feature_cols_1d=feature_cols_1d,
        feature_cols_2d=feature_cols_2d,
        feature_mean_1d=feature_mean_1d,
        feature_std_1d=feature_std_1d,
        feature_mean_2d=feature_mean_2d,
        feature_std_2d=feature_std_2d,
        target_mean=target_mean,
        target_std=target_std,
        node_type_1d_value=1,
        node_type_2d_value=2,
        group_event_col="event_id",   # adjust if needed
        group_time_col="t",
        node_id_col="node_id",
        node_type_col="node_type",    # if absent, dataset falls back to node_id sets
        target_col="target",        # preferred (delta target)
    )

    # loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    edge_index_2d = build_edge_index_from_adj(static.adj_2d).to(device)

    model = Model1Net(
        in_dim_2d=len(feature_cols_2d),
        in_dim_1d=len(feature_cols_1d),
        hidden_dim=128,
        gnn_layers=3,
        dropout=0.1,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    def rmse(pred, y):
        return torch.sqrt(torch.mean((pred - y) ** 2))

    # ---- TRAIN: chunked consecutive-snapshot training ----
    model.train()

    if len(dataset) < K:
        raise ValueError(f"Dataset too short for K={K}. len(dataset)={len(dataset)}")

    valid_starts = list(range(len(dataset) - K + 1))

    for epoch in range(EPOCHS):
        random.shuffle(valid_starts)
        starts = valid_starts[:min(MAX_STARTS, len(valid_starts))]

        total = 0.0
        n = 0

        for step, s in enumerate(starts):
            loss = 0.0

            for k in range(K):
                snap = dataset[s + k]

                x2 = snap.x2d.to(device)
                y2 = snap.y2d.to(device)
                x1 = snap.x1d.to(device)
                y1 = snap.y1d.to(device)

                d2, d1 = model(x2, edge_index_2d, x1)
                loss = loss + rmse(d2, y2) + rmse(d1, y1)

            loss = loss / K

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.detach().cpu())
            n += 1

            if step % 10 == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"loss={float(loss.detach().cpu()):.6f} "
                    f"pred2_mean={float(d2.mean().detach().cpu()):.6f} "
                    f"y2_mean={float(y2.mean().detach().cpu()):.6f} "
                    f"pred2_std={float(d2.std().detach().cpu()):.6f} "
                    f"y2_std={float(y2.std().detach().cpu()):.6f}"
                )

        print(f"epoch={epoch} avg_loss={total/max(n,1):.6f}")

    # ---- SAVE ----
    ckpt_path = os.path.join(out_dir, "model1_state_dict.pt")
    torch.save(model.state_dict(), ckpt_path)

    meta = dict(
        feature_cols_1d=feature_cols_1d,
        feature_cols_2d=feature_cols_2d,
        feature_mean_1d=feature_mean_1d.tolist(),
        feature_std_1d=feature_std_1d.tolist(),
        feature_mean_2d=feature_mean_2d.tolist(),
        feature_std_2d=feature_std_2d.tolist(),
        target_mean=float(target_mean),
        target_std=float(target_std),
    )
    with open(os.path.join(out_dir, "model1_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print("saved:", ckpt_path)


if __name__ == "__main__":
    main()