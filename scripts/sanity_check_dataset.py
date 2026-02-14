from pathlib import Path

import pandas as pd

from ufb.io.write_plan import build_write_plan
from ufb.io.events import index_event_folders
from ufb.io.static import load_model_static

WARMUP_STEPS = 10


def main():
    root = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench")
    models_root = root / "full_dataset" / "Models"
    sample_sub = root / "kaggle_dataset" / "urban-flood-modelling" / "sample_submission.csv"

    wp = build_write_plan(sample_sub)

    # Build folder indices for each model in test
    event_map = {}
    for mid in [1, 2]:
        split_root = models_root / f"Model_{mid}" / "test"
        event_map[mid] = index_event_folders(split_root)

    # 1) Check every (model,event) exists
    missing = []
    for (mid, eid) in wp.events_in_order:
        if eid not in event_map[mid]:
            missing.append((mid, eid))
    if missing:
        raise RuntimeError(f"Missing event folders for: {missing[:20]} (showing up to 20)")
    print(f"OK: all template (model,event) pairs exist as folders. Count={len(wp.events_in_order)}")

    # 2) Check horizons vs timesteps.csv - warmup
    # We check per (model,event,node_type)
    mismatches = []
    for (mid, eid) in wp.events_in_order:
        event_dir = event_map[mid][eid]
        ts_path = event_dir / "timesteps.csv"
        ts = pd.read_csv(ts_path)
        T = len(ts)
        expected_H = T - WARMUP_STEPS
        for node_type in [1, 2]:
            H_template = wp.horizons.get((mid, eid, node_type))
            if H_template is None:
                continue
            if H_template != expected_H:
                mismatches.append((mid, eid, node_type, H_template, expected_H, T))

    if mismatches:
        print("WARNING: horizon mismatches found (mid,eid,type,H_template,H_expected,T):")
        for row in mismatches[:20]:
            print(row)
        print(f"Total mismatches: {len(mismatches)}")
    else:
        print("OK: template horizons match timesteps.csv (T-10) for all checked events/types.")

    print("\nChecking node count consistency vs static files...")

    for mid in [1, 2]:
        static = load_model_static(models_root, mid, split="test")

        n1_static = len(static.nodes_1d)
        n2_static = len(static.nodes_2d)

        # Count unique nodes in template per model & node_type
        blocks_mid = [b for b in wp.blocks if b.model_id == mid]

        nodes_1_template = {
            (b.event_id, b.node_id)
            for b in blocks_mid if b.node_type == 1
        }
        nodes_2_template = {
            (b.event_id, b.node_id)
            for b in blocks_mid if b.node_type == 2
        }

        # We only check node_ids per event (should be same count per event)
        events_mid = {b.event_id for b in blocks_mid}

        for eid in events_mid:
            n1_template = len({b.node_id for b in blocks_mid
                            if b.event_id == eid and b.node_type == 1})
            n2_template = len({b.node_id for b in blocks_mid
                            if b.event_id == eid and b.node_type == 2})

            if n1_template != n1_static:
                print(f"Mismatch Model {mid} Event {eid} 1D nodes: "
                    f"template={n1_template}, static={n1_static}")

            if n2_template != n2_static:
                print(f"Mismatch Model {mid} Event {eid} 2D nodes: "
                    f"template={n2_template}, static={n2_static}")
    print("Node count consistency check complete.")


if __name__ == "__main__":
    main()
