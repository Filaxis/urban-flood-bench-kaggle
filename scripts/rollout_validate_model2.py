from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

from ufb.io.events import index_event_folders
from ufb.io.static import load_model_static
from ufb.io.dynamics import load_event_dynamics
from ufb.infer.rollout import RolloutConfig, rollout_event_two_models


def main():
    project_root = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench")
    models_root = project_root / "full_dataset" / "Models"
    parquet_dir = project_root / "data_cache" / "model2_train_samples_parquet"
    model_path_1d = parquet_dir / "model2_xgb_1d.json"
    model_path_2d = parquet_dir / "model2_xgb_2d.json"
    feat_path_1d = parquet_dir / "feature_cols_1d.csv"
    feat_path_2d = parquet_dir / "feature_cols_2d.csv"

    # Load model + feature columns
    model_1d = XGBRegressor()
    model_1d.load_model(str(model_path_1d))
    feature_cols_1d = pd.read_csv(feat_path_1d, header=None)[0].tolist()

    model_2d = XGBRegressor()
    model_2d.load_model(str(model_path_2d))
    feature_cols_2d = pd.read_csv(feat_path_2d, header=None)[0].tolist()

    # Static + event folders (TRAIN this time, to compare to truth)
    static = load_model_static(models_root, model_id=2, split="train")
    event_map = index_event_folders(models_root / "Model_2" / "train")

    cfg = RolloutConfig(warmup_steps=10, dtype="float32")

    # Choose a few events to validate rollout (edit as you like)
    # Tip: include a short (94/97), medium (205), long (445)
    validate_event_ids = [1, 5, 15]  # adjust if these exist in your train set

    all_pred = []
    all_true = []
    all_type = []

    for eid in validate_event_ids:
        dyn = load_event_dynamics(event_dir=event_map[eid], model_id=2, event_id=eid, split="train")
        T = len(dyn.timesteps)
        H = T - cfg.warmup_steps

        pred = rollout_event_two_models(
            model_1d=model_1d,
            feature_cols_1d=feature_cols_1d,
            model_2d=model_2d,
            feature_cols_2d=feature_cols_2d,
            model_id=2,
            nodes_1d_static=static.nodes_1d,
            nodes_2d_static=static.nodes_2d,
            nodes_1d_dyn=dyn.nodes_1d_dyn,
            nodes_2d_dyn=dyn.nodes_2d_dyn,
            H=H,
            cfg=cfg,
            adj_1d=static.adj_1d,
            adj_2d=static.adj_2d,
            conn1d_to_2d=static.conn1d_to_2d,
            alpha_1d=1.0,
            clip_1d=None,
        )

        # Build true arrays for comparison
        # 1D true water_level for t=10..T-1
        n1 = len(static.nodes_1d)
        n2 = len(static.nodes_2d)

        df1 = dyn.nodes_1d_dyn.sort_values(["timestep","node_idx"])
        wl1 = df1["water_level"].to_numpy().reshape(T, n1)
        true1 = wl1[cfg.warmup_steps:, :]  # (H,n1)

        df2 = dyn.nodes_2d_dyn.sort_values(["timestep","node_idx"])
        wl2 = df2["water_level"].to_numpy().reshape(T, n2)
        true2 = wl2[cfg.warmup_steps:, :]  # (H,n2)

        # Collect predictions aligned
        pred1 = np.stack([pred[(1,i)] for i in range(n1)], axis=1)  # (H,n1)
        pred2 = np.stack([pred[(2,j)] for j in range(n2)], axis=1)  # (H,n2)

        all_pred.append(pred1.ravel()); all_true.append(true1.ravel()); all_type.append(np.ones(true1.size, dtype=np.int8))
        all_pred.append(pred2.ravel()); all_true.append(true2.ravel()); all_type.append(np.full(true2.size, 2, dtype=np.int8))

        rmse_1d = root_mean_squared_error(true1.ravel(), pred1.ravel())
        rmse_2d = root_mean_squared_error(true2.ravel(), pred2.ravel())
        rmse_all = root_mean_squared_error(np.concatenate([true1.ravel(), true2.ravel()]),
                                           np.concatenate([pred1.ravel(), pred2.ravel()]))

        print(f"[EVENT {eid}] T={T} H={H} | RMSE all={rmse_all:.5f} 1D={rmse_1d:.5f} 2D={rmse_2d:.5f}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    types = np.concatenate(all_type)

    print("\n[SUMMARY]")
    print("RMSE all:", root_mean_squared_error(y_true, y_pred))
    print("RMSE 1D :", root_mean_squared_error(y_true[types==1], y_pred[types==1]))
    print("RMSE 2D :", root_mean_squared_error(y_true[types==2], y_pred[types==2]))


if __name__ == "__main__":
    main()
