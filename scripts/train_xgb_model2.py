from __future__ import annotations

from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error


def load_event_parquets(event_ids, base_dir: Path, model_id: int):
    dfs = []
    for eid in event_ids:
        path = base_dir / f"model{model_id}_train_event{eid:03d}.parquet"
        df = pd.read_parquet(path)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)


def main():
    project_root = Path(
        r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench"
    )

    model_id = 2
    parquet_dir = project_root / "data_cache" / "model2_train_samples_parquet"
    index_df = pd.read_csv(parquet_dir / "index.csv")

    event_ids = sorted(index_df["event_id"].tolist())
    if len(event_ids) < 10:
        raise RuntimeError("Too few events found in index.csv. Did build_train_parquet_model2.py finish?")

    # Event-level split: last 20% for validation
    n_val = max(1, int(round(0.2 * len(event_ids))))
    val_event_ids = event_ids[-n_val:]
    train_event_ids = event_ids[:-n_val]

    print("Train events:", train_event_ids)
    print("Val events  :", val_event_ids)

    print("\nLoading training data...")
    df_train = load_event_parquets(train_event_ids, parquet_dir, model_id)

    print("Loading validation data...")
    df_val = load_event_parquets(val_event_ids, parquet_dir, model_id)

    drop_cols = ["target"]
    X_train = df_train.drop(columns=drop_cols)
    y_train = df_train["target"]

    X_val = df_val.drop(columns=drop_cols)
    y_val = df_val["target"]

    # Save feature column order (for rollout)
    feature_cols = list(X_train.columns)
    pd.Series(feature_cols).to_csv(parquet_dir / "feature_cols.csv", index=False, header=False)

    print("Train shape:", X_train.shape)
    print("Val shape  :", X_val.shape)

    model = XGBRegressor(
        n_estimators=5000,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)
    print(f"\nValidation RMSE: {rmse:.6f}")

    model.save_model(str(parquet_dir / "model2_xgb.json"))
    print("Model saved:", parquet_dir / "model2_xgb.json")


if __name__ == "__main__":
    main()
