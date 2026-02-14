from pathlib import Path
import pandas as pd

project_root = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench")

parquet_dir = project_root / "data_cache" / "model1_train_samples_parquet"

# Load just one parquet file
example_file = next(parquet_dir.glob("model1_train_event*.parquet"))
df = pd.read_parquet(example_file)

feature_cols = [c for c in df.columns if c != "target"]

(pd.Series(feature_cols)).to_csv(parquet_dir / "feature_cols.csv", index=False, header=False)

print("feature_cols.csv created.")
