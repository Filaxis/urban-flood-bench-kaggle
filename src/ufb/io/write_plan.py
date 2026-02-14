from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Optional

import pandas as pd


@dataclass(frozen=True)
class WriteBlock:
    model_id: int
    event_id: int
    node_type: int  # 1 or 2
    node_id: int
    start_row: int  # starting row position in template (0-based)
    length: int     # number of rows for this (m,e,type,node)


@dataclass
class WritePlan:
    template_path: Path
    n_rows: int
    blocks: List[WriteBlock]
    # Convenience indices:
    horizons: Dict[Tuple[int, int, int], int]  # (model_id,event_id,node_type) -> H
    events_in_order: List[Tuple[int, int]]     # (model_id,event_id) in template order


def build_write_plan(sample_submission_csv: Path) -> WritePlan:
    df = pd.read_csv(sample_submission_csv, usecols=["row_id", "model_id", "event_id", "node_type", "node_id"])
    # Ensure correct ordering by row_id (don’t assume file is sorted)
    df = df.sort_values("row_id", kind="stable").reset_index(drop=True)

    # Sanity: row_id should match index after sorting (common in Kaggle)
    # If it doesn't, we still rely on sorted row_id order.
    n_rows = len(df)

    keys = list(zip(df["model_id"].astype(int),
                    df["event_id"].astype(int),
                    df["node_type"].astype(int),
                    df["node_id"].astype(int)))

    blocks: List[WriteBlock] = []
    horizons: Dict[Tuple[int, int, int], int] = {}
    events_in_order: List[Tuple[int, int]] = []

    i = 0
    last_event_pair: Optional[Tuple[int, int]] = None
    while i < n_rows:
        k = keys[i]
        j = i + 1
        while j < n_rows and keys[j] == k:
            j += 1
        length = j - i
        block = WriteBlock(
            model_id=k[0], event_id=k[1], node_type=k[2], node_id=k[3],
            start_row=i, length=length
        )
        blocks.append(block)

        group_key = (k[0], k[1], k[2])
        if group_key not in horizons:
            horizons[group_key] = length
        else:
            # Optional sanity check: horizon constant within (model,event,node_type)
            if horizons[group_key] != length:
                # Don’t crash; just warn. Some comps can have weirdness.
                print(f"WARNING: non-constant horizon for {group_key}: "
                      f"{horizons[group_key]} vs {length} (node_id={k[3]})")

        event_pair = (k[0], k[1])
        if event_pair != last_event_pair:
            events_in_order.append(event_pair)
            last_event_pair = event_pair

        i = j

    return WritePlan(
        template_path=sample_submission_csv,
        n_rows=n_rows,
        blocks=blocks,
        horizons=horizons,
        events_in_order=events_in_order,
    )
