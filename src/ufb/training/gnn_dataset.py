"""
Model1SnapshotDataset — fast vectorised version.

Key fixes vs. original:
  - No iterrows(): uses numpy argsort-based indexing → ~100x faster __getitem__
  - Accepts a pre-loaded DataFrame (df=...) so the caller can avoid a double read
  - target column is expected to be TRUE DELTA (wl_{t+1} - wl_t)
  - Normalises features and target with caller-supplied stats
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Snapshot:
    x2d: torch.Tensor   # (N2, F2)  normalised features
    y2d: torch.Tensor   # (N2,)     normalised target deltas
    x1d: torch.Tensor   # (N1, F1)
    y1d: torch.Tensor   # (N1,)


class Model1SnapshotDataset(Dataset):
    """
    One Dataset wraps ONE event's DataFrame (or a concatenation of several).

    Parameters
    ----------
    df : pd.DataFrame
        Already-loaded frame with columns: node_id, node_type, event_id (or
        group_event_col), t (or group_time_col), feature cols, target col.
        Pass ``None`` together with ``parquet_path`` to load from disk.
    parquet_path : str | None
        Only used when df is None.
    """

    def __init__(
        self,
        node_ids_1d: Sequence[int],
        node_ids_2d: Sequence[int],
        feature_cols_1d: Sequence[str],
        feature_cols_2d: Sequence[str],
        df: Optional[pd.DataFrame] = None,
        parquet_path: Optional[str] = None,
        group_event_col: str = "event_id",
        group_time_col: str = "t",
        node_id_col: str = "node_id",
        node_type_col: str = "node_type",
        target_col: str = "target",
        node_type_1d_value: int = 1,
        node_type_2d_value: int = 2,
        feature_mean_1d: Optional[np.ndarray] = None,
        feature_std_1d:  Optional[np.ndarray] = None,
        feature_mean_2d: Optional[np.ndarray] = None,
        feature_std_2d:  Optional[np.ndarray] = None,
        target_mean: float = 0.0,
        target_std:  float = 1.0,
    ):
        if df is None:
            if parquet_path is None:
                raise ValueError("Provide either df or parquet_path.")
            df = pd.read_parquet(parquet_path)

        # ---- normalisation stats ----
        self.feature_mean_1d = feature_mean_1d
        self.feature_std_1d  = (
            np.where(feature_std_1d  < 1e-6, 1.0, feature_std_1d)
            if feature_std_1d is not None else None
        )
        self.feature_mean_2d = feature_mean_2d
        self.feature_std_2d  = (
            np.where(feature_std_2d  < 1e-6, 1.0, feature_std_2d)
            if feature_std_2d is not None else None
        )
        self.target_mean = target_mean
        self.target_std  = target_std if target_std > 1e-6 else 1.0

        self.feature_cols_1d = list(feature_cols_1d)
        self.feature_cols_2d = list(feature_cols_2d)

        # node id → position index
        self.node_ids_1d = np.asarray(node_ids_1d, dtype=np.int64)
        self.node_ids_2d = np.asarray(node_ids_2d, dtype=np.int64)
        n1 = len(self.node_ids_1d)
        n2 = len(self.node_ids_2d)

        # look-up dict: node_id → row index in x1d / x2d matrices
        self._idx1 = {int(nid): i for i, nid in enumerate(self.node_ids_1d)}
        self._idx2 = {int(nid): i for i, nid in enumerate(self.node_ids_2d)}

        # ---- split 1D / 2D ----
        if node_type_col in df.columns:
            df1 = df[df[node_type_col] == node_type_1d_value].copy()
            df2 = df[df[node_type_col] == node_type_2d_value].copy()
        else:
            df1 = df[df[node_id_col].isin(set(self.node_ids_1d))].copy()
            df2 = df[df[node_id_col].isin(set(self.node_ids_2d))].copy()

        # ---- infer event column ----
        if group_event_col not in df1.columns:
            for c in ("event", "event_idx", "event_id_int"):
                if c in df1.columns:
                    group_event_col = c
                    break

        # ---- shared (event, t) keys ----
        g1 = set(zip(df1[group_event_col].tolist(), df1[group_time_col].tolist()))
        g2 = set(zip(df2[group_event_col].tolist(), df2[group_time_col].tolist()))
        self.keys = sorted(g1 & g2)   # list of (event_id, t)

        # ---- pre-pivot into contiguous numpy arrays for O(1) __getitem__ ----
        #
        # For each (event, t) we need a dense (n_nodes, n_feats) matrix.
        # Strategy: sort both frames by (event, t, node_id), then for each key
        # we can slice a contiguous block.  We build an index of (start, length)
        # pairs at __init__ time.
        #
        # This is O(N log N) once at construction and O(1) per __getitem__.

        def _pivot_frame(df_sub, feat_cols, id_col, n_nodes, idx_map):
            """Return (X_all, y_all, offsets) where offsets[i] = row start in X_all for key i."""
            df_sub = df_sub.copy()
            # fill missing feature cols with 0
            for c in feat_cols + [target_col]:
                if c not in df_sub.columns:
                    df_sub[c] = 0.0
            df_sub[feat_cols] = df_sub[feat_cols].fillna(0.0)
            df_sub[target_col] = df_sub[target_col].fillna(0.0)

            # sort so that (event, t, node_id) are in canonical order
            df_sub = df_sub.sort_values([group_event_col, group_time_col, id_col]).reset_index(drop=True)

            X_raw = df_sub[feat_cols].to_numpy(dtype=np.float32)
            y_raw = df_sub[target_col].to_numpy(dtype=np.float32)
            nids  = df_sub[id_col].to_numpy(dtype=np.int64)
            ev_t  = list(zip(df_sub[group_event_col].tolist(), df_sub[group_time_col].tolist()))

            # build per-key offset list aligned to self.keys
            # Each key should correspond to exactly n_nodes rows
            key_to_start: dict = {}
            i = 0
            while i < len(ev_t):
                k = ev_t[i]
                if k not in key_to_start:
                    key_to_start[k] = i
                i += 1

            # Build dense matrices: for each key, arrange rows by node position
            n_keys = len(self.keys)
            X_dense = np.zeros((n_keys * n_nodes, len(feat_cols)), dtype=np.float32)
            y_dense = np.zeros((n_keys * n_nodes,), dtype=np.float32)

            for ki, key in enumerate(self.keys):
                start = key_to_start.get(key)
                if start is None:
                    continue
                base_out = ki * n_nodes
                # walk rows for this key
                j = start
                while j < len(ev_t) and ev_t[j] == key:
                    nid = int(nids[j])
                    pos = idx_map.get(nid)
                    if pos is not None:
                        X_dense[base_out + pos] = X_raw[j]
                        y_dense[base_out + pos] = y_raw[j]
                    j += 1

            return X_dense, y_dense, n_nodes

        self._X1, self._y1, self._n1 = _pivot_frame(
            df1, self.feature_cols_1d, node_id_col, n1, self._idx1
        )
        self._X2, self._y2, self._n2 = _pivot_frame(
            df2, self.feature_cols_2d, node_id_col, n2, self._idx2
        )

        # apply normalisation in-place (done once at construction)
        if self.feature_mean_1d is not None and self.feature_std_1d is not None:
            self._X1 = (self._X1 - self.feature_mean_1d) / self.feature_std_1d
        np.nan_to_num(self._X1, copy=False)

        if self.feature_mean_2d is not None and self.feature_std_2d is not None:
            self._X2 = (self._X2 - self.feature_mean_2d) / self.feature_std_2d
        np.nan_to_num(self._X2, copy=False)

        self._y1 = (self._y1 - self.target_mean) / self.target_std
        self._y2 = (self._y2 - self.target_mean) / self.target_std
        np.nan_to_num(self._y1, copy=False)
        np.nan_to_num(self._y2, copy=False)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> Snapshot:
        b1 = idx * self._n1
        b2 = idx * self._n2
        return Snapshot(
            x2d=torch.from_numpy(self._X2[b2: b2 + self._n2].copy()),
            y2d=torch.from_numpy(self._y2[b2: b2 + self._n2].copy()),
            x1d=torch.from_numpy(self._X1[b1: b1 + self._n1].copy()),
            y1d=torch.from_numpy(self._y1[b1: b1 + self._n1].copy()),
        )
