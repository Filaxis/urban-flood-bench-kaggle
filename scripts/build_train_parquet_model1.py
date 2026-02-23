from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

from ufb.io.events import index_event_folders
from ufb.io.static import load_model_static
from ufb.io.dynamics import load_event_dynamics
from ufb.training.samples import SampleConfig, build_event_training_samples


def main() -> None:
    # --- USER PATHS (edit if needed) ---
    project_root = Path(
        r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench"
    )
    models_root = project_root / "full_dataset" / "Models"

    # Output location (recommended: keep under a "data_cache" or "outputs" folder)
    out_dir = project_root / "data_cache" / "model1_train_samples_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- CONFIG ---
    model_id = 1
    split = "train"
    warmup_steps = 10

    # Given your T distribution (94/97/205/445), this is a good baseline:
    cfg = SampleConfig(
        warmup_steps=warmup_steps,
        n_lags=6,
        min_t=9,                 # mimic test regime (warmup 0..9)
        dry_keep_prob=0.15,      # keep 15% of dry steps
        max_timesteps_per_event=180,  # caps 445-step events so they don't dominate
        seed=42,
        dtype="float32",
    )

    print(f"[INFO] Loading static data for Model_{model_id} ({split})...")
    static = load_model_static(models_root, model_id, split=split)

    # Index train event folders
    split_root = models_root / f"Model_{model_id}" / split
    event_map = index_event_folders(split_root)

    print(f"[INFO] Found {len(event_map)} train events for Model_{model_id}.")
    print(f"[INFO] Writing per-event parquet samples to: {out_dir}")

    # Keep an index of outputs
    index_rows = []

    # Process each event
    for eid, event_dir in sorted(event_map.items()):
        print(f"\n[EVENT] Model_{model_id} train event {eid} @ {event_dir.name}")

        # Load dynamics (node dynamics only)
        try:
            dyn = load_event_dynamics(event_dir=event_dir, model_id=model_id, event_id=eid, split=split)
        except TypeError:
            # If your load_event_dynamics signature differs, fall back to positional:
            dyn = load_event_dynamics(event_dir, model_id=model_id, event_id=eid, split=split)

        T = len(dyn.timesteps)
        H = T - warmup_steps
        if H <= 0:
            print(f"[WARN] Event {eid}: T={T} too short for warmup={warmup_steps}; skipping.")
            continue

        # Build samples
        df = build_event_training_samples(
            model_id=model_id,
            nodes_1d_static=static.nodes_1d,
            nodes_2d_static=static.nodes_2d,
            nodes_1d_dyn=dyn.nodes_1d_dyn,
            nodes_2d_dyn=dyn.nodes_2d_dyn,
            cfg=cfg,
            adj_1d=static.adj_1d,
            adj_2d=static.adj_2d,
            conn1d_to_2d=static.conn1d_to_2d,
        )

        if df.empty:
            print(f"[WARN] Event {eid}: produced 0 samples; skipping parquet write.")
            continue

        # Output parquet
        out_path = out_dir / f"model{model_id}_train_event{eid:03d}.parquet"
        df.to_parquet(out_path, index=False)

        print(f"[OK] Event {eid}: T={T}, H={H}, rows={len(df):,} -> {out_path.name}")

        index_rows.append(
            {
                "model_id": model_id,
                "split": split,
                "event_id": eid,
                "T": T,
                "H": H,
                "rows": int(len(df)),
                "parquet_path": str(out_path),
            }
        )

        # Optional: free memory explicitly (usually not needed, but harmless)
        del df

    # Save index
    index_df = pd.DataFrame(index_rows).sort_values(["event_id"])
    index_csv = out_dir / "index.csv"
    index_df.to_csv(index_csv, index=False)
    print(f"\n[INFO] Wrote index: {index_csv}")
    print(index_df[["event_id", "T", "H", "rows"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise
