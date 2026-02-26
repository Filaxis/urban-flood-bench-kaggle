# src/ufb/training/gnn_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Snapshot:
    x2d: torch.Tensor
    y2d: torch.Tensor
    x1d: torch.Tensor
    y1d: torch.Tensor


class Model1SnapshotDataset(Dataset):
    """
    Expects a parquet with per-node, per-time rows, including:
      - node_id
      - node_type (optional; or separate parquets)
      - t
      - event_id (or similar)
      - features for 1D and 2D
      - target columns (delta target recommended)

    Minimal assumption:
      - you can compute delta target if wl_t and wl_tp1 exist, or you have a ready target column.
    """

    def __init__(
        self,
        parquet_path: str,
        node_ids_1d: Sequence[int],
        node_ids_2d: Sequence[int],
        feature_cols_1d: Sequence[str],
        feature_cols_2d: Sequence[str],
        group_event_col: str = "event_id",
        group_time_col: str = "t",
        node_id_col: str = "node_id",
        node_type_col: str = "node_type",
        target_col: str = "d_wl_tp1",  # preferred: delta(wl_{t+1}-wl_t)
        wl_t_col: str = "wl_t",
        wl_tp1_col: str = "wl_tp1",
        node_type_1d_value: str = "1d",
        node_type_2d_value: str = "2d",
    ):
        self.df = pd.read_parquet(parquet_path)

        # Infer event column if needed
        if group_event_col not in self.df.columns:
            # common alternatives
            for c in ["event", "event_idx", "event_id_int"]:
                if c in self.df.columns:
                    group_event_col = c
                    break
        self.group_event_col = group_event_col
        self.group_time_col = group_time_col
        self.node_id_col = node_id_col
        self.node_type_col = node_type_col

        self.feature_cols_1d = list(feature_cols_1d)
        self.feature_cols_2d = list(feature_cols_2d)

        self.node_ids_1d = np.asarray(node_ids_1d)
        self.node_ids_2d = np.asarray(node_ids_2d)
        self.idx1 = {int(nid): i for i, nid in enumerate(self.node_ids_1d)}
        self.idx2 = {int(nid): i for i, nid in enumerate(self.node_ids_2d)}

        # Target handling
        if target_col in self.df.columns:
            self.target_col = target_col
        else:
            # compute delta target if possible
            if wl_t_col in self.df.columns and wl_tp1_col in self.df.columns:
                self.df[target_col] = self.df[wl_tp1_col] - self.df[wl_t_col]
                self.target_col = target_col
            else:
                raise ValueError(
                    f"Target column '{target_col}' not found, and cannot compute from "
                    f"'{wl_t_col}' + '{wl_tp1_col}'."
                )

        # Split 1D/2D frames
        if node_type_col in self.df.columns:
            df1 = self.df[self.df[node_type_col] == node_type_1d_value].copy()
            df2 = self.df[self.df[node_type_col] == node_type_2d_value].copy()
        else:
            # If node_type not present, assume parquet is mixed but distinguishable by node_id sets
            df1 = self.df[self.df[node_id_col].isin(self.node_ids_1d)].copy()
            df2 = self.df[self.df[node_id_col].isin(self.node_ids_2d)].copy()

        # Build group keys present in both (event,t)
        g1 = set(zip(df1[self.group_event_col].tolist(), df1[self.group_time_col].tolist()))
        g2 = set(zip(df2[self.group_event_col].tolist(), df2[self.group_time_col].tolist()))
        self.keys = sorted(list(g1.intersection(g2)))

        self.df1 = df1
        self.df2 = df2

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Snapshot:
        ev, t = self.keys[idx]

        a1 = self.df1[(self.df1[self.group_event_col] == ev) & (self.df1[self.group_time_col] == t)]
        a2 = self.df2[(self.df2[self.group_event_col] == ev) & (self.df2[self.group_time_col] == t)]

        # Allocate fixed-size arrays aligned to node ordering
        x1 = np.zeros((len(self.node_ids_1d), len(self.feature_cols_1d)), dtype=np.float32)
        y1 = np.zeros((len(self.node_ids_1d),), dtype=np.float32)

        x2 = np.zeros((len(self.node_ids_2d), len(self.feature_cols_2d)), dtype=np.float32)
        y2 = np.zeros((len(self.node_ids_2d),), dtype=np.float32)

        # Fill 1D
        for _, r in a1.iterrows():
            nid = int(r[self.node_id_col])
            i = self.idx1.get(nid, None)
            if i is None:
                continue
            x1[i, :] = r[self.feature_cols_1d].to_numpy(dtype=np.float32)
            y1[i] = float(r[self.target_col])

        # Fill 2D
        for _, r in a2.iterrows():
            nid = int(r[self.node_id_col])
            i = self.idx2.get(nid, None)
            if i is None:
                continue
            x2[i, :] = r[self.feature_cols_2d].to_numpy(dtype=np.float32)
            y2[i] = float(r[self.target_col])

        return Snapshot(
            x2d=torch.from_numpy(x2),
            y2d=torch.from_numpy(y2),
            x1d=torch.from_numpy(x1),
            y1d=torch.from_numpy(y1),
        )