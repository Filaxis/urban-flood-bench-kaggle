# scripts/train_gnn_model1.py
from __future__ import annotations

import os
import json
import torch
from torch.utils.data import DataLoader

from ufb.models.gnn import Model1Net
from ufb.training.gnn_dataset import Model1SnapshotDataset


def build_edge_index_from_adj(adj) -> torch.Tensor:
    # adj is your UndirectedEdges (has src, dst arrays)
    src = torch.as_tensor(adj.src, dtype=torch.long)
    dst = torch.as_tensor(adj.dst, dtype=torch.long)
    return torch.stack([src, dst], dim=0)


def main():
    # ---- CONFIG (edit these) ----
    parquet_path = os.environ.get("UFB_TRAIN_PARQUET", "/kaggle/input/your-dataset/model1_train.parquet")
    out_dir = os.environ.get("UFB_OUT_DIR", "/kaggle/working/gnn_model1")
    os.makedirs(out_dir, exist_ok=True)

    # You will load these from your existing static loader in Kaggle
    # For skeleton: assume you can import and call it.
    from ufb.io.static import load_model_static  # adjust if your path differs
    static = load_model_static(models_root="/kaggle/input/your-static-root", model_id=1, split="train")

    # Feature columns (reuse your existing CSVs or hardcode for MVP)
    # IMPORTANT: for MVP, ensure these match parquet columns exactly.
    feature_cols_2d = [
        "wl_t","wl_tm1","wl_tm2","wl_tm3","wl_tm4","wl_tm5",
        "rain_t","rain_tm1","rain_tm2","rain_tm3","rain_sum_4",
        "d_wl_t","nbr_wl_mean_t","nbr_wl_mean_tm1",
        "position_x","position_y","depth","invert_elevation","surface_elevation",
        "base_area","deg","area","roughness","min_elevation","elevation",
        "aspect","curvature","flow_accumulation",
    ]
    feature_cols_1d = [
        "wl_t","wl_tm1","wl_tm2","wl_tm3","wl_tm4","wl_tm5",
        "rain_t","rain_tm1","rain_tm2","rain_tm3","rain_sum_4",
        "d_wl_t","nbr_wl_mean_t","nbr_wl_mean_tm1",
        "conn2d_wl_t","conn2d_rain_sum_4",
        "position_x","position_y","depth","invert",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    node_ids_1d = static.nodes_1d["node_id"].to_numpy()
    node_ids_2d = static.nodes_2d["node_id"].to_numpy()

    dataset = Model1SnapshotDataset(
        parquet_path=parquet_path,
        node_ids_1d=node_ids_1d,
        node_ids_2d=node_ids_2d,
        feature_cols_1d=feature_cols_1d,
        feature_cols_2d=feature_cols_2d,
        group_event_col="event_id",   # adjust if needed
        group_time_col="t",
        node_id_col="node_id",
        node_type_col="node_type",    # if absent, dataset falls back to node_id sets
        target_col="d_wl_tp1",        # preferred (delta target)
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    edge_index_2d = build_edge_index_from_adj(static.adj_2d)

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

    # ---- TRAIN (MVP) ----
    model.train()
    for epoch in range(3):  # MVP; increase later
        total = 0.0
        n = 0
        for batch in loader:
            # batch_size=1 => tensors have leading dim 1
            x2 = batch.x2d.squeeze(0).to(device)
            y2 = batch.y2d.squeeze(0).to(device)
            x1 = batch.x1d.squeeze(0).to(device)
            y1 = batch.y1d.squeeze(0).to(device)

            d2, d1 = model(x2, edge_index_2d.to(device), x1)

            loss = rmse(d2, y2) + rmse(d1, y1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total += float(loss.detach().cpu())
            n += 1

        print(f"epoch={epoch} loss={total/max(n,1):.6f}")

    # ---- SAVE ----
    ckpt_path = os.path.join(out_dir, "model1_state_dict.pt")
    torch.save(model.state_dict(), ckpt_path)

    meta = dict(
        feature_cols_1d=feature_cols_1d,
        feature_cols_2d=feature_cols_2d,
    )
    with open(os.path.join(out_dir, "model1_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print("saved:", ckpt_path)


if __name__ == "__main__":
    main()