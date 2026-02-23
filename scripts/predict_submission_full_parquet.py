# scripts/predict_submission_full_parquet.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from xgboost import XGBRegressor

from ufb.io.write_plan import build_write_plan
from ufb.io.events import index_event_folders
from ufb.io.static import load_model_static
from ufb.io.dynamics import load_event_dynamics
from ufb.infer.rollout import RolloutConfig, rollout_event_model1, rollout_event_two_models


def main() -> None:
    project_root = Path(
        r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench"
    )

    models_root = project_root / "full_dataset" / "Models"
    kaggle_root = project_root / "kaggle_dataset" / "urban-flood-modelling"
    sample_sub = kaggle_root / "sample_submission.csv"

    out_path = project_root / "outputs" / "submission_full.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Write plan (authoritative row order + horizons) ---
    wp = build_write_plan(sample_sub)

    # --- Load Model_1 artifacts ---
    m1_dir = project_root / "data_cache" / "model1_train_samples_parquet"
    m1_model_path = m1_dir / "model1_xgb.json"
    m1_feat_path = m1_dir / "feature_cols.csv"

    model1 = XGBRegressor()
    model1.load_model(str(m1_model_path))
    feature_cols_1 = pd.read_csv(m1_feat_path, header=None)[0].tolist()

    static1 = load_model_static(models_root, model_id=1, split="test")
    event_map1 = index_event_folders(models_root / "Model_1" / "test")

    # --- Load Model_2 artifacts (split 1D / 2D models) ---
    m2_dir = project_root / "data_cache" / "model2_train_samples_parquet"
    m2_model_1d_path = m2_dir / "model2_xgb_1d.json"
    m2_model_2d_path = m2_dir / "model2_xgb_2d.json"
    m2_feat_1d_path = m2_dir / "feature_cols_1d.csv"
    m2_feat_2d_path = m2_dir / "feature_cols_2d.csv"

    model2_1d = XGBRegressor()
    model2_1d.load_model(str(m2_model_1d_path))
    feature_cols_2_1d = pd.read_csv(m2_feat_1d_path, header=None)[0].tolist()

    model2_2d = XGBRegressor()
    model2_2d.load_model(str(m2_model_2d_path))
    feature_cols_2_2d = pd.read_csv(m2_feat_2d_path, header=None)[0].tolist()

    static2 = load_model_static(models_root, model_id=2, split="test")
    event_map2 = index_event_folders(models_root / "Model_2" / "test")

    # --- Rollout config ---
    cfg = RolloutConfig(warmup_steps=10, dtype="float32")

    # --- Parquet schema (types are important for size) ---
    schema = pa.schema([
        ("row_id", pa.int64()),
        ("model_id", pa.int16()),
        ("event_id", pa.int16()),
        ("node_type", pa.int8()),
        ("node_id", pa.int32()),
        ("water_level", pa.float32()),
    ])

    # --- Streaming write buffers ---
    batch_size_rows = 200_000  # tune: 100k–500k good
    buf_row_id: list[int] = []
    buf_model_id: list[int] = []
    buf_event_id: list[int] = []
    buf_node_type: list[int] = []
    buf_node_id: list[int] = []
    buf_water_level: list[float] = []

    def flush(writer: pq.ParquetWriter) -> None:
        if not buf_row_id:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(buf_row_id, type=pa.int64()),
                pa.array(buf_model_id, type=pa.int16()),
                pa.array(buf_event_id, type=pa.int16()),
                pa.array(buf_node_type, type=pa.int8()),
                pa.array(buf_node_id, type=pa.int32()),
                pa.array(buf_water_level, type=pa.float32()),
            ],
            schema=schema
        )
        writer.write_table(table)
        buf_row_id.clear()
        buf_model_id.clear()
        buf_event_id.clear()
        buf_node_type.clear()
        buf_node_id.clear()
        buf_water_level.clear()

    # Cache rollout results per (model_id,event_id) so each event is computed once
    current_evt = None
    current_cache = None  # Dict[(node_type,node_id)] -> np.ndarray length H



    with pq.ParquetWriter(out_path, schema=schema, compression="zstd") as writer:
        for block in wp.blocks:
            evt = (block.model_id, block.event_id)

            if evt != current_evt:
                mid, eid = evt

                # Horizon must be consistent across node types for the same event
                H1 = wp.horizons.get((mid, eid, 1))
                H2 = wp.horizons.get((mid, eid, 2))
                if H1 is None or H2 is None:
                    raise KeyError(f"Missing horizon for (model,event)=({mid},{eid}) in write plan.")
                if H1 != H2:
                    raise ValueError(f"H mismatch in template for (model,event)=({mid},{eid}): H1={H1}, H2={H2}")
                H = H1

                if mid == 1:
                    event_dir = event_map1.get(eid)
                    if event_dir is None:
                        raise FileNotFoundError(f"Missing test event folder for Model_1 event_id={eid}")
                    dyn = load_event_dynamics(event_dir=event_dir, model_id=1, event_id=eid, split="test")

                    current_cache = rollout_event_model1(
                        model=model1,
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
                    if eid in {5, 8}:  # pick any 1–2 event_ids you like
                        a1 = np.concatenate([current_cache[(1, i)] for i in range(len(static1.nodes_1d))])
                        a2 = np.concatenate([current_cache[(2, j)] for j in range(len(static1.nodes_2d))])
                        print(f"[STATS] M1 E{eid} 1D min/mean/max: {a1.min():.3f}/{a1.mean():.3f}/{a1.max():.3f}")
                        print(f"[STATS] M1 E{eid} 2D min/mean/max: {a2.min():.3f}/{a2.mean():.3f}/{a2.max():.3f}")
                    print(f"[OK] Rolled out Model_1 event {eid} with H={H}")

                elif mid == 2:
                    event_dir = event_map2.get(eid)
                    if event_dir is None:
                        raise FileNotFoundError(f"Missing test event folder for Model_2 event_id={eid}")
                    dyn = load_event_dynamics(event_dir=event_dir, model_id=2, event_id=eid, split="test")

                    current_cache = rollout_event_two_models(
                        model_1d=model2_1d,
                        feature_cols_1d=feature_cols_2_1d,
                        model_2d=model2_2d,
                        feature_cols_2d=feature_cols_2_2d,
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
                        alpha_1d=1.0,
                        clip_1d=None,
                    )
                    if eid in {5, 8}:  # pick any 1–2 event_ids you like
                        a1 = np.concatenate([current_cache[(1, i)] for i in range(len(static2.nodes_1d))])
                        a2 = np.concatenate([current_cache[(2, j)] for j in range(len(static2.nodes_2d))])
                        print(f"[STATS] M1 E{eid} 1D min/mean/max: {a1.min():.3f}/{a1.mean():.3f}/{a1.max():.3f}")
                        print(f"[STATS] M1 E{eid} 2D min/mean/max: {a2.min():.3f}/{a2.mean():.3f}/{a2.max():.3f}")
                    print(f"[OK] Rolled out Model_2 event {eid} with H={H}")

                else:
                    raise ValueError(f"Unexpected model_id in template: {mid}")

                current_evt = evt

            # Write this block in exact template order
            series = current_cache[(block.node_type, block.node_id)]
            if len(series) != block.length:
                raise ValueError(f"Length mismatch for block {block}: expected {block.length}, got {len(series)}")

            for k in range(block.length):
                row_id = block.start_row + k
                buf_row_id.append(int(row_id))
                buf_model_id.append(int(block.model_id))
                buf_event_id.append(int(block.event_id))
                buf_node_type.append(int(block.node_type))
                buf_node_id.append(int(block.node_id))
                buf_water_level.append(float(series[k]))

                if len(buf_row_id) >= batch_size_rows:
                    flush(writer)

        # final flush
        flush(writer)

    print(f"\nWrote full submission (Parquet): {out_path}")
    print("Upload this .parquet to Kaggle.")


if __name__ == "__main__":
    main()
