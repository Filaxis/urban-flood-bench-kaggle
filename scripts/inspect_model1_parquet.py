from pathlib import Path
import pandas as pd

def main():
    project_root = Path(
        r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench"
    )

    parquet_dir = project_root / "data_cache" / "model1_train_samples_parquet"

    # Pick the first parquet file
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print("No parquet files found.")
        return

    first_file = parquet_files[0]
    print(f"Reading: {first_file.name}")

    df = pd.read_parquet(first_file)

    print("\n--- Column list ---")
    for col in df.columns:
        print(col)

    print("\n--- Lag columns ---")
    print([c for c in df.columns if c.startswith("wl_tm")])
    print([c for c in df.columns if c.startswith("rain_tm")])
    print("d_wl_t present:", "d_wl_t" in df.columns)
    print("rain_sum_4 present:", "rain_sum_4" in df.columns)

    print("\n--- Basic stats (1D only sample) ---")
    df_1d = df[df["node_type"] == 1].head(10000)
    print(df_1d[["wl_t", "wl_tm1"]].describe())

if __name__ == "__main__":
    main()
