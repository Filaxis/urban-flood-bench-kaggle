from pathlib import Path
import pandas as pd

from ufb.io.events import index_event_folders


def main():
    root = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench")
    models_root = root / "full_dataset" / "Models"

    split_root = models_root / "Model_1" / "train"
    event_map = index_event_folders(split_root)

    rows = []
    for eid, edir in sorted(event_map.items()):
        ts = pd.read_csv(edir / "timesteps.csv")
        T = len(ts)
        rows.append({"event_id": eid, "T": T})

    df = pd.DataFrame(rows).sort_values("T")
    print(df.to_string(index=False))
    print("\nSummary:")
    print(df["T"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_string())


if __name__ == "__main__":
    main()
