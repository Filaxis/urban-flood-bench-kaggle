# src/ufb/training/gnn_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Snapshot:
    x2d:    torch.Tensor   # (N2, F2)  — 2D node features
    y2d:    torch.Tensor   # (N2,)     — delta WL target
    x1d:    torch.Tensor   # (N1, F1)  — 1D node features
    y1d:    torch.Tensor   # (N1,)     — delta WL target
    y_inlet: torch.Tensor  # (N1,)     — inlet_flow target (absolute)
    x_edge: torch.Tensor   # (E1, Fe)  — 1D edge features
    y_edge: torch.Tensor   # (E1,)     — edge_flow target (absolute)


class Model1SnapshotDataset(Dataset):
    """
    Loads per-event node parquets and (optionally) edge parquets.
    Returns Snapshot objects containing tensors for all four model outputs.

    Node parquet columns of interest:
      node_type == 1  ->  1D rows  (target, target_inlet_flow)
      node_type == 2  ->  2D rows  (target)

    Edge parquet columns of interest:
      flow_t, flow_tm1, flow_tm2, static edge cols, target_edge_flow
    """

    def __init__(
        self,
        parquet_path: str,
        node_ids_1d: Sequence[int],
        node_ids_2d: Sequence[int],
        feature_cols_1d: Sequence[str],
        feature_cols_2d: Sequence[str],
        feature_cols_edge: Sequence[str],
        n_edges: int,                          # total number of 1D edges
        group_event_col: str = "event_id",
        group_time_col: str = "t",
        node_id_col: str = "node_id",
        node_type_col: str = "node_type",
        target_col: str = "target",
        target_inlet_col: str = "target_inlet_flow",
        node_type_1d_value: int = 1,
        node_type_2d_value: int = 2,
        feature_mean_1d=None,
        feature_std_1d=None,
        feature_mean_2d=None,
        feature_std_2d=None,
        feature_mean_edge=None,
        feature_std_edge=None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        target_inlet_mean: float = 0.0,
        target_inlet_std: float = 1.0,
        target_edge_mean: float = 0.0,
        target_edge_std: float = 1.0,
        edge_parquet_path: Optional[str] = None,
    ):
        # ---- Load node parquet ----
        self.df = pd.read_parquet(parquet_path)

        # Infer event column
        if group_event_col not in self.df.columns:
            for c in ["event", "event_idx", "event_id_int"]:
                if c in self.df.columns:
                    group_event_col = c
                    break

        self.group_event_col  = group_event_col
        self.group_time_col   = group_time_col
        self.node_id_col      = node_id_col
        self.node_type_col    = node_type_col
        self.target_col       = target_col
        self.target_inlet_col = target_inlet_col

        # Normalisation stats — nodes
        self.feature_mean_1d = feature_mean_1d
        self.feature_std_1d  = (
            np.where(feature_std_1d < 1e-6, 1.0, feature_std_1d)
            if feature_std_1d is not None else None
        )
        self.feature_mean_2d = feature_mean_2d
        self.feature_std_2d  = (
            np.where(feature_std_2d < 1e-6, 1.0, feature_std_2d)
            if feature_std_2d is not None else None
        )
        self.target_mean        = float(target_mean)
        self.target_std         = float(target_std) if target_std > 1e-6 else 1.0
        self.target_inlet_mean  = float(target_inlet_mean)
        self.target_inlet_std   = float(target_inlet_std) if target_inlet_std > 1e-6 else 1.0

        # Normalisation stats — edges
        self.feature_cols_edge = list(feature_cols_edge)
        self.n_edges           = n_edges
        self.feature_mean_edge = feature_mean_edge
        self.feature_std_edge  = (
            np.where(feature_std_edge < 1e-6, 1.0, feature_std_edge)
            if feature_std_edge is not None else None
        )
        self.target_edge_mean = float(target_edge_mean)
        self.target_edge_std  = float(target_edge_std) if target_edge_std > 1e-6 else 1.0

        self.feature_cols_1d = list(feature_cols_1d)
        self.feature_cols_2d = list(feature_cols_2d)

        self.node_ids_1d = np.asarray(node_ids_1d)
        self.node_ids_2d = np.asarray(node_ids_2d)
        self.idx1 = {int(nid): i for i, nid in enumerate(self.node_ids_1d)}
        self.idx2 = {int(nid): i for i, nid in enumerate(self.node_ids_2d)}

        # Validate target columns exist
        if target_col not in self.df.columns:
            raise ValueError(f"Target column '{target_col}' not found in node parquet.")

        # Add zero target_inlet_flow if missing (e.g. for 2D rows or fallback)
        if target_inlet_col not in self.df.columns:
            self.df[target_inlet_col] = 0.0

        # Split 1D / 2D
        if node_type_col in self.df.columns:
            df1 = self.df[self.df[node_type_col] == node_type_1d_value].copy()
            df2 = self.df[self.df[node_type_col] == node_type_2d_value].copy()
        else:
            df1 = self.df[self.df[node_id_col].isin(self.node_ids_1d)].copy()
            df2 = self.df[self.df[node_id_col].isin(self.node_ids_2d)].copy()

        # Keys = (event, t) pairs present in both subsystems
        g1 = set(zip(df1[self.group_event_col].tolist(), df1[self.group_time_col].tolist()))
        g2 = set(zip(df2[self.group_event_col].tolist(), df2[self.group_time_col].tolist()))
        self.keys = sorted(list(g1.intersection(g2)))

        self.df1 = df1
        self.df2 = df2

        # ---- Load edge parquet ----
        self.df_edge = None
        if edge_parquet_path is not None and Path(edge_parquet_path).exists():
            self.df_edge = pd.read_parquet(edge_parquet_path)
            if "target_edge_flow" not in self.df_edge.columns:
                self.df_edge["target_edge_flow"] = 0.0
        # If no edge parquet, we return zero tensors for edge outputs

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Snapshot:
        ev, t = self.keys[idx]

        # ---- 1D nodes ----
        a1 = self.df1[
            (self.df1[self.group_event_col] == ev) &
            (self.df1[self.group_time_col]  == t)
        ]
        x1 = np.zeros((len(self.node_ids_1d), len(self.feature_cols_1d)), dtype=np.float32)
        y1 = np.zeros((len(self.node_ids_1d),), dtype=np.float32)
        y_inlet = np.zeros((len(self.node_ids_1d),), dtype=np.float32)

        for _, r in a1.iterrows():
            nid = int(r[self.node_id_col])
            i   = self.idx1.get(nid)
            if i is None:
                continue

            x = r[self.feature_cols_1d].to_numpy(dtype=np.float32)
            if self.feature_mean_1d is not None:
                x = np.where(np.isfinite(x), x, self.feature_mean_1d)
                x = (x - self.feature_mean_1d) / self.feature_std_1d
            else:
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x1[i, :] = x

            y = float(r[self.target_col])
            y1[i] = (y - self.target_mean) / self.target_std

            yi = float(r[self.target_inlet_col])
            y_inlet[i] = (yi - self.target_inlet_mean) / self.target_inlet_std

        # ---- 2D nodes ----
        a2 = self.df2[
            (self.df2[self.group_event_col] == ev) &
            (self.df2[self.group_time_col]  == t)
        ]
        x2 = np.zeros((len(self.node_ids_2d), len(self.feature_cols_2d)), dtype=np.float32)
        y2 = np.zeros((len(self.node_ids_2d),), dtype=np.float32)

        for _, r in a2.iterrows():
            nid = int(r[self.node_id_col])
            i   = self.idx2.get(nid)
            if i is None:
                continue

            x = r[self.feature_cols_2d].to_numpy(dtype=np.float32)
            if self.feature_mean_2d is not None:
                x = np.where(np.isfinite(x), x, self.feature_mean_2d)
                x = (x - self.feature_mean_2d) / self.feature_std_2d
            else:
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            x2[i, :] = x

            y = float(r[self.target_col])
            y2[i] = (y - self.target_mean) / self.target_std

        # ---- 1D edges ----
        x_edge = np.zeros((self.n_edges, len(self.feature_cols_edge)), dtype=np.float32)
        y_edge = np.zeros((self.n_edges,), dtype=np.float32)

        if self.df_edge is not None:
            ae = self.df_edge[
                (self.df_edge[self.group_event_col] == ev) &
                (self.df_edge[self.group_time_col]  == t)
            ]
            for _, r in ae.iterrows():
                eid = int(r["edge_id"])
                if eid < 0 or eid >= self.n_edges:
                    continue

                xe = r[self.feature_cols_edge].to_numpy(dtype=np.float32)
                if self.feature_mean_edge is not None:
                    xe = np.where(np.isfinite(xe), xe, self.feature_mean_edge)
                    xe = (xe - self.feature_mean_edge) / self.feature_std_edge
                else:
                    xe = np.nan_to_num(xe, nan=0.0, posinf=0.0, neginf=0.0)
                x_edge[eid, :] = xe

                ye = float(r["target_edge_flow"])
                y_edge[eid] = (ye - self.target_edge_mean) / self.target_edge_std

        return Snapshot(
            x2d=torch.from_numpy(x2),
            y2d=torch.from_numpy(y2),
            x1d=torch.from_numpy(x1),
            y1d=torch.from_numpy(y1),
            y_inlet=torch.from_numpy(y_inlet),
            x_edge=torch.from_numpy(x_edge),
            y_edge=torch.from_numpy(y_edge),
        )
