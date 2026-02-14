from pathlib import Path
import csv
import pandas as pd
from xgboost import XGBRegressor

from ufb.io.write_plan import build_write_plan
from ufb.io.events import index_event_folders
from ufb.io.static import load_model_static
from ufb.io.dynamics import load_event_dynamics
from ufb.infer.rollout import RolloutConfig, rollout_event_model1


def main():
    project_root = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench")
    models_root = project_root / "full_dataset" / "Models"
    kaggle_root = project_root / "kaggle_dataset" / "urban-flood-modelling"

    sample_sub = kaggle_root / "sample_submission.csv"

    parquet_dir = project_root / "data_cache" / "model1_train_samples_parquet"
    model_path = parquet_dir / "model1_xgb.json"
    feat_path = parquet_dir / "feature_cols.csv"

    out_csv = project_root / "outputs" / "submission_model1_only.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load write plan
    wp = build_write_plan(sample_sub)

    # Load feature columns (exact training order)
    feature_cols = pd.read_csv(feat_path, header=None)[0].tolist()

    # Load model
    model = XGBRegressor()
    model.load_model(str(model_path))

    # Load static for Model_1/test (static files live under split folder in your layout)
    static = load_model_static(models_root, model_id=1, split="test")

    # Index test event folders
    event_map = index_event_folders(models_root / "Model_1" / "test")

    cfg = RolloutConfig(warmup_steps=10, dtype="float32")

    # Cache predictions per (model,event)
    current_evt = None
    current_cache = None

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"])

        for block in wp.blocks:
            # For now: only fill Model_1; skip Model_2 rows (leave blank not allowed, so you must later extend)
            if block.model_id != 1:
                continue

            evt = (block.model_id, block.event_id)
            if evt != current_evt:
                event_dir = event_map.get(block.event_id)
                if event_dir is None:
                    raise FileNotFoundError(f"Missing test event folder for Model_1 event_id={block.event_id}")

                dyn = load_event_dynamics(event_dir=event_dir, model_id=1, event_id=block.event_id, split="test")

                # Horizon from template (should match for type=1 and type=2)
                H1 = wp.horizons[(1, block.event_id, 1)]
                H2 = wp.horizons[(1, block.event_id, 2)]
                if H1 != H2:
                    raise ValueError(f"H mismatch in template for event {block.event_id}: H1={H1}, H2={H2}")
                H = H1

                current_cache = rollout_event_model1(
                    model=model,
                    feature_cols=feature_cols,
                    model_id=1,
                    nodes_1d_static=static.nodes_1d,
                    nodes_2d_static=static.nodes_2d,
                    nodes_1d_dyn=dyn.nodes_1d_dyn,
                    nodes_2d_dyn=dyn.nodes_2d_dyn,
                    H=H,
                    cfg=cfg,
                )
                current_evt = evt
                print(f"[OK] Rolled out Model_1 event {block.event_id} with H={H}")

            series = current_cache[(block.node_type, block.node_id)]
            if len(series) != block.length:
                raise ValueError("Block length mismatch vs predicted series length.")

            for k in range(block.length):
                row_id = block.start_row + k
                w.writerow([row_id, block.model_id, block.event_id, block.node_type, block.node_id, float(series[k])])

    print(f"\nWrote: {out_csv}")
    print("NOTE: This file is incomplete (Model_1 only). Next step is to extend to Model_2 and write all rows.")


if __name__ == "__main__":
    main()
