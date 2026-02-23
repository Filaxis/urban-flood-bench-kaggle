from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error
# from sklearn.model_selection import train_test_split


def load_event_parquets(event_ids, base_dir):
    dfs = []
    for eid in event_ids:
        path = base_dir / f"model1_train_event{eid:03d}.parquet"
        df = pd.read_parquet(path)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)


def main():
    project_root = Path(
        r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench"
    )

    parquet_dir = project_root / "data_cache" / "model1_train_samples_parquet"
    index_df = pd.read_csv(parquet_dir / "index.csv")

    event_ids = sorted(index_df["event_id"].tolist())

    # Event-level split (last 14 for validation)
    val_event_ids = event_ids[-14:]
    train_event_ids = event_ids[:-14]

    print("Train events:", train_event_ids)
    print("Val events:", val_event_ids)

    print("\nLoading training data...")
    df_train = load_event_parquets(train_event_ids, parquet_dir)

    print("Loading validation data...")
    df_val = load_event_parquets(val_event_ids, parquet_dir)

    # Features
    drop_cols = ["target"]
    X_train = df_train.drop(columns=drop_cols)
    y_train = df_train["target"]

    X_val = df_val.drop(columns=drop_cols)
    y_val = df_val["target"]

    # Save feature columns
    feature_cols = list(X_train.columns)
    pd.Series(feature_cols).to_csv(
        parquet_dir / "feature_cols.csv",
        index=False,
        header=False
)

    print("Train shape:", X_train.shape)
    print("Val shape:", X_val.shape)

    model = XGBRegressor(
        n_estimators=4000,
        max_depth=10,
        learning_rate=0.03,
        min_child_weight = 5,
        reg_lambda = 1,
        reg_alpha = 0,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",  # important for speed
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=40,
    )
    model_path = parquet_dir / "model1_xgb.json"
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
        print(f"Model saved: {model_path}")

    if not interrupted:
        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)
        print(f"Model validation RMSE: {rmse:.6f}")
    else:
        print(f"Skipped RMSE because training was interrupted.")


if __name__ == "__main__":
    main()
