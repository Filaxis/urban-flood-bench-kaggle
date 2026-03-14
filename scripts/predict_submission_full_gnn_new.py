# scripts/predict_submission_full_gnn.py
# Both Model_1 and Model_2 use GNNPredictor + rollout_event_model1
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ufb.io.write_plan import build_write_plan
from ufb.io.events import index_event_folders
from ufb.io.static import load_model_static
from ufb.io.dynamics import load_event_dynamics
from ufb.infer.rollout import RolloutConfig, rollout_event_model1

import torch
from ufb.infer.predictor_gnn import GNNPredictor
from ufb.models.gnn_py import Model1Net


def load_gnn_model(meta_path: Path, ckpt_path: Path, static, device: str):
    """Load a GNNPredictor from meta.json + state_dict.pt."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols_2d = meta["feature_cols_2d"]
    feature_cols_1d = meta["feature_cols_1d"]

    net = Model1Net(
        in_dim_2d=len(feature_cols_2d),
        in_dim_1d=len(feature_cols_1d),
        hidden_dim=128,
        gnn_layers=3,
        dropout=0.0,
    )
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    predictor = GNNPredictor(
        model=net,
        adj_2d=static.adj_2d,
        device=device,
        predict_delta=True,
    )

    # Build unified feature_cols: 2d cols first, then 1d-only cols
    # rollout_event_model1 uses a single feature_cols list applied to both node types
    # via ensure_cols fallback — columns absent for a node type are zero-filled.
    # The meta stores them separately; we merge into one deduplicated list preserving order.
    seen = set()
    feature_cols = []
    for c in feature_cols_2d + feature_cols_1d:
        if c not in seen:
            feature_cols.append(c)
            seen.add(c)

    return predictor, feature_cols, meta


def main() -> None:
    project_root = Path("/kaggle/working/urbanfloodbench")
    models_root = project_root / "full_dataset" / "Models"
    kaggle_root = project_root / "kaggle_dataset" / "urban-flood-modelling"
    sample_sub = kaggle_root / "sample_submission.csv"

    out_path = project_root / "outputs" / "submission_full.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wp = build_write_plan(sample_sub)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Model_1 GNN ---
    m1_cache_dir = project_root / "data_cache" / "gnn_model1_rollout"
    static1 = load_model_static(models_root, model_id=1, split="test")
    predictor1, feature_cols_1, meta1 = load_gnn_model(
        meta_path=m1_cache_dir / "model1_meta.json",
        ckpt_path=m1_cache_dir / "model1_state_dict.pt",
        static=static1,
        device=device,
    )
    event_map1 = index_event_folders(models_root / "Model_1" / "test")
    print(f"Model_1: {len(feature_cols_1)} feature cols, {len(event_map1)} test events")

    # --- Model_2 GNN ---
    m2_cache_dir = project_root / "data_cache" / "gnn_model2_rollout"
    static2 = load_model_static(models_root, model_id=2, split="test")
    predictor2, feature_cols_2, meta2 = load_gnn_model(
        meta_path=m2_cache_dir / "model2_meta.json",
        ckpt_path=m2_cache_dir / "model2_state_dict.pt",
        static=static2,
        device=device,
    )
    event_map2 = index_event_folders(models_root / "Model_2" / "test")
    print(f"Model_2: {len(feature_cols_2)} feature cols, {len(event_map2)} test events")

    # --- Normalisation stats (needed by rollout for delta denorm) ---
    # GNNPredictor handles denorm internally via predict_both; stats already baked in.

    cfg = RolloutConfig(warmup_steps=10, dtype="float32")

    # --- Parquet streaming writer ---
    schema = pa.schema([
        ("row_id",      pa.int64()),
        ("model_id",    pa.int16()),
        ("event_id",    pa.int16()),
        ("node_type",   pa.int8()),
        ("node_id",     pa.int32()),
        ("water_level", pa.float32()),
    ])

    batch_size_rows = 200_000
    buf_row_id:     list[int]   = []
    buf_model_id:   list[int]   = []
    buf_event_id:   list[int]   = []
    buf_node_type:  list[int]   = []
    buf_node_id:    list[int]   = []
    buf_water_level: list[float] = []

    def flush(writer: pq.ParquetWriter) -> None:
        if not buf_row_id:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(buf_row_id,      type=pa.int64()),
                pa.array(buf_model_id,    type=pa.int16()),
                pa.array(buf_event_id,    type=pa.int16()),
                pa.array(buf_node_type,   type=pa.int8()),
                pa.array(buf_node_id,     type=pa.int32()),
                pa.array(buf_water_level, type=pa.float32()),
            ],
            schema=schema,
        )
        writer.write_table(table)
        buf_row_id.clear(); buf_model_id.clear(); buf_event_id.clear()
        buf_node_type.clear(); buf_node_id.clear(); buf_water_level.clear()

    current_evt   = None
    current_cache = None

    with pq.ParquetWriter(out_path, schema=schema, compression="zstd") as writer:
        for block in wp.blocks:
            evt = (block.model_id, block.event_id)

            if evt != current_evt:
                mid, eid = evt

                H1 = wp.horizons.get((mid, eid, 1))
                H2 = wp.horizons.get((mid, eid, 2))
                if H1 is None or H2 is None:
                    raise KeyError(f"Missing horizon for (model,event)=({mid},{eid})")
                if H1 != H2:
                    raise ValueError(f"H mismatch ({mid},{eid}): H1={H1} H2={H2}")
                H = H1

                if mid == 1:
                    event_dir = event_map1.get(eid)
                    if event_dir is None:
                        raise FileNotFoundError(f"Missing test event for Model_1 event_id={eid}")
                    dyn = load_event_dynamics(event_dir=event_dir, model_id=1,
                                              event_id=eid, split="test")
                    current_cache = rollout_event_model1(
                        predictor=predictor1,
                        feature_cols=feature_cols_1,
                        model_id=1,
                        nodes_1d_static=static1.nodes_1d,
                        nodes_2d_static=static1.nodes_2d,
                        nodes_1d_dyn=dyn.nodes_1d_dyn,
                        nodes_2d_dyn=dyn.nodes_2d_dyn,
                        H=H,
                        cfg=cfg,
                        adj_1d=static1.adj_1d,
                        adj_2d=static1.adj_2d,
                        conn1d_to_2d=static1.conn1d_to_2d,
                    )

                elif mid == 2:
                    event_dir = event_map2.get(eid)
                    if event_dir is None:
                        raise FileNotFoundError(f"Missing test event for Model_2 event_id={eid}")
                    dyn = load_event_dynamics(event_dir=event_dir, model_id=2,
                                              event_id=eid, split="test")
                    current_cache = rollout_event_model1(
                        predictor=predictor2,
                        feature_cols=feature_cols_2,
                        model_id=2,
                        nodes_1d_static=static2.nodes_1d,
                        nodes_2d_static=static2.nodes_2d,
                        nodes_1d_dyn=dyn.nodes_1d_dyn,
                        nodes_2d_dyn=dyn.nodes_2d_dyn,
                        H=H,
                        cfg=cfg,
                        adj_1d=static2.adj_1d,
                        adj_2d=static2.adj_2d,
                        conn1d_to_2d=static2.conn1d_to_2d,
                    )

                else:
                    raise ValueError(f"Unexpected model_id: {mid}")

                # Sanity stats for a couple of events
                if eid in {5, 8}:
                    a1 = np.concatenate([current_cache[(1, i)]
                                         for i in range(len(static1.nodes_1d if mid == 1 else static2.nodes_1d))])
                    a2 = np.concatenate([current_cache[(2, j)]
                                         for j in range(len(static1.nodes_2d if mid == 1 else static2.nodes_2d))])
                    print(f"[STATS] M{mid} E{eid} 1D min/mean/max: "
                          f"{a1.min():.3f}/{a1.mean():.3f}/{a1.max():.3f}")
                    print(f"[STATS] M{mid} E{eid} 2D min/mean/max: "
                          f"{a2.min():.3f}/{a2.mean():.3f}/{a2.max():.3f}")

                print(f"[OK] Rolled out Model_{mid} event {eid} H={H}")
                current_evt = evt

            series = current_cache[(block.node_type, block.node_id)]
            if len(series) != block.length:
                raise ValueError(
                    f"Length mismatch block {block}: expected {block.length}, got {len(series)}"
                )

            for k in range(block.length):
                buf_row_id.append(int(block.start_row + k))
                buf_model_id.append(int(block.model_id))
                buf_event_id.append(int(block.event_id))
                buf_node_type.append(int(block.node_type))
                buf_node_id.append(int(block.node_id))
                buf_water_level.append(float(series[k]))

                if len(buf_row_id) >= batch_size_rows:
                    flush(writer)

        flush(writer)

    print(f"\nWrote submission: {out_path}")
    print("Upload this .parquet to Kaggle.")


if __name__ == "__main__":
    main()
