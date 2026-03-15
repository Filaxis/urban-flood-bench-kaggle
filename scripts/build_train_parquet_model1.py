from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

from ufb.io.events import index_event_folders
from ufb.io.static import load_model_static
from ufb.io.dynamics import load_event_dynamics
from ufb.training.samples import SampleConfig, build_event_training_samples


def main() -> None:
    # --- USER PATHS ---
    project_root = Path(
        r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench"
    )
    models_root = project_root / "full_dataset" / "Models"

    # Node parquets (same location as before — seamless for existing code)
    out_dir = project_root / "data_cache" / "model1_train_samples_parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Edge parquets go into a parallel folder
    out_dir_edges = project_root / "data_cache" / "model1_train_edges_parquet"
    out_dir_edges.mkdir(parents=True, exist_ok=True)

    # --- CONFIG ---
    model_id = 1
    split = "train"
    warmup_steps = 10

    cfg = SampleConfig(
        warmup_steps=warmup_steps,
        n_lags=6,
        min_t=9,
        dry_keep_prob=0.15,
        max_timesteps_per_event=180,
        seed=42,
        dtype="float32",
    )

    print(f"[INFO] Loading static data for Model_{model_id} ({split})...")
    static = load_model_static(models_root, model_id, split=split)

    split_root = models_root / f"Model_{model_id}" / split
    event_map = index_event_folders(split_root)

    print(f"[INFO] Found {len(event_map)} train events for Model_{model_id}.")
    print(f"[INFO] Node parquets  -> {out_dir}")
    print(f"[INFO] Edge parquets  -> {out_dir_edges}")

    index_rows = []

    for eid, event_dir in sorted(event_map.items()):
        print(f"\n[EVENT] Model_{model_id} train event {eid} @ {event_dir.name}")

        dyn = load_event_dynamics(
            event_dir=event_dir, model_id=model_id, event_id=eid, split=split
        )

        T = len(dyn.timesteps)
        H = T - warmup_steps
        if H <= 0:
            print(f"[WARN] Event {eid}: T={T} too short; skipping.")
            continue

        df_nodes, df_edges = build_event_training_samples(
            model_id=model_id,
            nodes_1d_static=static.nodes_1d,
            nodes_2d_static=static.nodes_2d,
            nodes_1d_dyn=dyn.nodes_1d_dyn,
            nodes_2d_dyn=dyn.nodes_2d_dyn,
            cfg=cfg,
            adj_1d=static.adj_1d,
            adj_2d=static.adj_2d,
            conn1d_to_2d=static.conn1d_to_2d,
            edges_1d_static=static.edges_1d,
            edges_1d_dyn=dyn.edges_1d_dyn,
        )

        if df_nodes.empty:
            print(f"[WARN] Event {eid}: produced 0 node samples; skipping.")
            continue

        # Write node parquet
        node_path = out_dir / f"model{model_id}_train_event{eid:03d}.parquet"
        df_nodes.to_parquet(node_path, index=False)

        # Write edge parquet (may be empty if edge dynamics unavailable)
        edge_path = out_dir_edges / f"model{model_id}_train_event{eid:03d}_edges.parquet"
        if not df_edges.empty:
            df_edges.to_parquet(edge_path, index=False)
            edge_rows = len(df_edges)
        else:
            edge_rows = 0
            print(f"[WARN] Event {eid}: no edge samples (edge dynamics missing?).")

        print(
            f"[OK] Event {eid}: T={T} H={H} "
            f"node_rows={len(df_nodes):,} edge_rows={edge_rows:,}"
        )

        index_rows.append({
            "model_id":   model_id,
            "split":      split,
            "event_id":   eid,
            "T":          T,
            "H":          H,
            "node_rows":  int(len(df_nodes)),
            "edge_rows":  int(edge_rows),
            "node_path":  str(node_path),
            "edge_path":  str(edge_path) if edge_rows > 0 else "",
        })

        del df_nodes, df_edges

    index_df = pd.DataFrame(index_rows).sort_values("event_id")
    index_csv = out_dir / "index.csv"
    index_df.to_csv(index_csv, index=False)
    print(f"\n[INFO] Wrote index: {index_csv}")
    print(index_df[["event_id", "T", "H", "node_rows", "edge_rows"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        raise
