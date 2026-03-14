import pandas as pd
from pathlib import Path

project_root = Path(
        r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench"
    )
p = project_root / "data_cache/model2_train_samples_parquet/model2_train_event001.parquet"  # any one file
df = pd.read_parquet(p)
# print(df["target"].describe())
# print(df["wl_t"].describe())
# print(df.columns.tolist())
# print(df[df["node_type"]==2]["target"].describe())
# print(df[df["node_type"]==1]["target"].describe())

# f_cols_1d = project_root / "data_cache/model2_train_samples_parquet/feature_cols_1d.csv"
# f_cols_2d = project_root / "data_cache/model2_train_samples_parquet/feature_cols_2d.csv"
# feat_1d = pd.read_csv(f_cols_1d, header=None)[0].tolist()
# feat_2d = pd.read_csv(f_cols_2d, header=None)[0].tolist()
# print("1D features:", feat_1d)
# print("2D features:", feat_2d)

event_dir = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench\full_dataset\Models\Model_2\train\event_1")  # any event
print(pd.read_csv(event_dir / "1d_nodes_dynamic_all.csv", nrows=2).columns.tolist())
print(pd.read_csv(event_dir / "1d_edges_dynamic_all.csv", nrows=2).columns.tolist())