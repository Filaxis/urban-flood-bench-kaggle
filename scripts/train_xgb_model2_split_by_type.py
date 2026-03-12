from __future__ import annotations

from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error


# Canonical feature columns — must match rollout.py exactly.
# 2D: no 1D-specific columns, no phantom 'deg'.
FEATURE_COLS_2D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t",
    "nbr_wl_mean_t", "nbr_wl_mean_tm1", "nbr_rain_sum_4",
    "position_x", "position_y",
    "area", "roughness", "min_elevation", "elevation",
    "aspect", "curvature", "flow_accumulation",
]

# 1D: no 2D-specific columns, no phantom 'deg'.
FEATURE_COLS_1D = [
    "wl_t", "wl_tm1", "wl_tm2", "wl_tm3", "wl_tm4", "wl_tm5",
    "rain_t", "rain_tm1", "rain_tm2", "rain_tm3", "rain_sum_4",
    "d_wl_t",
    "nbr_wl_mean_t", "nbr_wl_mean_tm1",
    "conn2d_wl_t", "conn2d_rain_sum_4",
    "position_x", "position_y",
    "depth", "invert_elevation", "surface_elevation", "base_area",
]


def load_event_parquets(event_ids, base_dir: Path, model_id: int):
    dfs = []
    for eid in event_ids:
        path = base_dir / f"model{model_id}_train_event{eid:03d}.parquet"
        dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, axis=0, ignore_index=True)


def train_one(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    out_dir: Path,
    tag: str,
    feature_cols: list[str],
):
    # Select only the canonical feature columns (drop anything else except target)
    missing_train = [c for c in feature_cols if c not in df_train.columns]
    missing_val   = [c for c in feature_cols if c not in df_val.columns]
    if missing_train or missing_val:
        raise ValueError(
            f"[{tag}] Missing columns in parquet.\n"
            f"  train missing: {missing_train}\n"
            f"  val missing:   {missing_val}\n"
            "Rebuild parquets with build_train_parquet_model2_xgb.py first."
        )

    X_train = df_train[feature_cols]
    y_train = df_train["target"]
    X_val   = df_val[feature_cols]
    y_val   = df_val["target"]

    # Save canonical feature columns for this model
    pd.Series(feature_cols).to_csv(
        out_dir / f"feature_cols_{tag}.csv", index=False, header=False
    )

    model = XGBRegressor(
        n_estimators=4000,
        max_depth=10,
        learning_rate=0.03,
        min_child_weight=5,
        reg_lambda=1,
        reg_alpha=0,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )

    model_path = out_dir / f"model2_xgb_{tag}.json"
    interrupted = False
    try:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Interrupted] Saving partially trained model...")
    finally:
        model.save_model(str(model_path))
        print(f"[{tag}] Saved: {model_path}")

    if not interrupted:
        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)
        print(f"[{tag}] Validation RMSE: {rmse:.6f}")
    else:
        print(f"[{tag}] Skipped RMSE (interrupted).")


def main():
    project_root = Path(
        r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench"
    )
    model_id   = 2
    parquet_dir = project_root / "data_cache" / "model2_train_samples_parquet"
    index_df   = pd.read_csv(parquet_dir / "index.csv")

    event_ids      = sorted(index_df["event_id"].tolist())
    n_val          = max(1, int(round(0.2 * len(event_ids))))
    val_event_ids  = event_ids[-n_val:]
    train_event_ids = event_ids[:-n_val]

    print("Train events:", train_event_ids)
    print("Val events  :", val_event_ids)

    print("\nLoading training data...")
    df_train = load_event_parquets(train_event_ids, parquet_dir, model_id)
    print("Loading validation data...")
    df_val   = load_event_parquets(val_event_ids,   parquet_dir, model_id)

    df_train_1d = df_train[df_train["node_type"] == 1].copy()
    df_val_1d   = df_val[df_val["node_type"] == 1].copy()
    df_train_2d = df_train[df_train["node_type"] == 2].copy()
    df_val_2d   = df_val[df_val["node_type"] == 2].copy()

    print(f"\nTraining 1D model  ({len(FEATURE_COLS_1D)} features)...")
    train_one(df_train_1d, df_val_1d, parquet_dir, tag="1d", feature_cols=FEATURE_COLS_1D)

    print(f"\nTraining 2D model  ({len(FEATURE_COLS_2D)} features)...")
    train_one(df_train_2d, df_val_2d, parquet_dir, tag="2d", feature_cols=FEATURE_COLS_2D)


if __name__ == "__main__":
    main()
