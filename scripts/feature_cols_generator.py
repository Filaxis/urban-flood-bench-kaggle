"""
Run this AFTER rebuilding the model2 parquets to regenerate feature_cols.csv.
It auto-discovers all feature columns (everything except 'target').
"""
from pathlib import Path
import pandas as pd

project_root = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench")

for model_id in [1, 2]:
    parquet_dir = project_root / "data_cache" / f"model{model_id}_train_samples_parquet"
    if not parquet_dir.exists():
        print(f"[SKIP] model{model_id} parquet dir not found: {parquet_dir}")
        continue

    example_file = next(parquet_dir.glob(f"model{model_id}_train_event*.parquet"), None)
    if example_file is None:
        print(f"[SKIP] No parquet files found in {parquet_dir}")
        continue

    df = pd.read_parquet(example_file)
    feature_cols = [c for c in df.columns if c != "target"]
    out_path = parquet_dir / "feature_cols.csv"
    pd.Series(feature_cols).to_csv(out_path, index=False, header=False)
    print(f"[OK] model{model_id}: {len(feature_cols)} feature cols -> {out_path}")
    print(f"     New cols present: {[c for c in feature_cols if 'rain_frac' in c or 'rain_steps' in c or 'rain_intensity' in c]}")
