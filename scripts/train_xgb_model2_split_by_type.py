from __future__ import annotations

from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error


def load_event_parquets(event_ids, base_dir: Path, model_id: int):
    dfs = []
    for eid in event_ids:
        path = base_dir / f"model{model_id}_train_event{eid:03d}.parquet"
        dfs.append(pd.read_parquet(path))
    return pd.concat(dfs, axis=0, ignore_index=True)


def train_one(df_train: pd.DataFrame, df_val: pd.DataFrame, out_dir: Path, tag: str):
    y_train = df_train["target"]
    X_train = df_train.drop(columns=["target"])

    y_val = df_val["target"]
    X_val = df_val.drop(columns=["target"])

    # Save feature columns for this model
    feature_cols = list(X_train.columns)
    pd.Series(feature_cols).to_csv(out_dir / f"feature_cols_{tag}.csv", index=False, header=False)

    model = XGBRegressor(
        n_estimators=4000,
        max_depth=10,
        learning_rate=0.03,
        min_child_weight = 5,
        reg_lambda = 1,
        reg_alpha = 0,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )

    # model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
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
        # save whatever state we have
        model.save_model(str(model_path))
        print(f"[{tag}] Saved: {model_path}")

    if not interrupted:
        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)
        print(f"[{tag}] Validation RMSE: {rmse:.6f}")
    else:
        print(f"[{tag}] Skipped RMSE because training was interrupted.")



def main():
    project_root = Path(
        r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench"
    )
    model_id = 2
    parquet_dir = project_root / "data_cache" / "model2_train_samples_parquet"
    index_df = pd.read_csv(parquet_dir / "index.csv")

    event_ids = sorted(index_df["event_id"].tolist())
    n_val = max(1, int(round(0.2 * len(event_ids))))
    val_event_ids = event_ids[-n_val:]
    train_event_ids = event_ids[:-n_val]

    print("Train events:", train_event_ids)
    print("Val events  :", val_event_ids)

    print("\nLoading training data...")
    df_train = load_event_parquets(train_event_ids, parquet_dir, model_id)
    print("Loading validation data...")
    df_val = load_event_parquets(val_event_ids, parquet_dir, model_id)

    # Split by node_type
    df_train_1d = df_train[df_train["node_type"] == 1].copy()
    df_val_1d = df_val[df_val["node_type"] == 1].copy()

    df_train_2d = df_train[df_train["node_type"] == 2].copy()
    df_val_2d = df_val[df_val["node_type"] == 2].copy()

    print("\nTraining 1D model...")
    train_one(df_train_1d, df_val_1d, parquet_dir, tag="1d")

    print("\nTraining 2D model...")
    train_one(df_train_2d, df_val_2d, parquet_dir, tag="2d")


if __name__ == "__main__":
    main()
