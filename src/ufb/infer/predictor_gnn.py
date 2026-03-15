"""
GNNPredictor — four-output version.

Handles the new Model1Net which predicts:
  d2    : (N2,)  normalised delta_wl for 2D nodes
  d1    : (N1,)  normalised delta_wl for 1D nodes
  inlet : (N1,)  normalised inlet_flow at t+1 (absolute)
  eflow : (E1,)  normalised edge_flow at t+1  (absolute)

Normalisation is applied inside _forward() so rollout.py passes raw features.
All four outputs are denormalised before being returned to the caller.
"""
from __future__ import annotations

import numpy as np
import torch

from ufb.features.graph_feats import UndirectedEdges
from ufb.infer.predictor_base import Predictor


class GNNPredictor(Predictor):

    def __init__(
        self,
        model: torch.nn.Module,
        adj_2d: UndirectedEdges,
        adj_1d: UndirectedEdges,
        feature_mean_2d: np.ndarray,
        feature_std_2d:  np.ndarray,
        feature_mean_1d: np.ndarray,
        feature_std_1d:  np.ndarray,
        feature_mean_edge: np.ndarray,
        feature_std_edge:  np.ndarray,
        target_mean: float = 0.0,
        target_std:  float = 1.0,
        target_inlet_mean: float = 0.0,
        target_inlet_std:  float = 1.0,
        target_edge_mean:  float = 0.0,
        target_edge_std:   float = 1.0,
        device: str = "cuda",
        dtype: str = "float32",
    ):
        self.model  = model
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.dtype  = dtype

        # Feature normalisation stats
        self._mean2  = feature_mean_2d.astype(np.float32)
        self._std2   = np.where(feature_std_2d  < 1e-6, 1.0, feature_std_2d).astype(np.float32)
        self._mean1  = feature_mean_1d.astype(np.float32)
        self._std1   = np.where(feature_std_1d  < 1e-6, 1.0, feature_std_1d).astype(np.float32)
        self._mean_e = feature_mean_edge.astype(np.float32)
        self._std_e  = np.where(feature_std_edge < 1e-6, 1.0, feature_std_edge).astype(np.float32)

        # Target denormalisation stats
        self._t_mean  = float(target_mean)
        self._t_std   = float(target_std)   if target_std   > 1e-6 else 1.0
        self._ti_mean = float(target_inlet_mean)
        self._ti_std  = float(target_inlet_std)  if target_inlet_std  > 1e-6 else 1.0
        self._te_mean = float(target_edge_mean)
        self._te_std  = float(target_edge_std)   if target_edge_std   > 1e-6 else 1.0

        # Precompute edge indices on device
        src2 = torch.as_tensor(adj_2d.src, dtype=torch.long)
        dst2 = torch.as_tensor(adj_2d.dst, dtype=torch.long)
        self.edge_index_2d = torch.stack([src2, dst2], dim=0).to(self.device)

        src1 = torch.as_tensor(adj_1d.src, dtype=torch.long)
        dst1 = torch.as_tensor(adj_1d.dst, dtype=torch.long)
        self.edge_index_1d = torch.stack([src1, dst1], dim=0).to(self.device)

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _forward(
        self,
        X2: np.ndarray,   # (N2, F2D)  raw features
        X1: np.ndarray,   # (N1, F1D)  raw features
        Xe: np.ndarray,   # (E1, FEDG) raw features
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Normalise
        X2n = (X2 - self._mean2) / self._std2
        X1n = (X1 - self._mean1) / self._std1
        Xen = (Xe - self._mean_e) / self._std_e
        np.nan_to_num(X2n, copy=False)
        np.nan_to_num(X1n, copy=False)
        np.nan_to_num(Xen, copy=False)

        x2 = torch.as_tensor(X2n, dtype=torch.float32, device=self.device)
        x1 = torch.as_tensor(X1n, dtype=torch.float32, device=self.device)
        xe = torch.as_tensor(Xen, dtype=torch.float32, device=self.device)

        d2_norm, d1_norm, inlet_norm, edge_norm = self.model(
            x2, self.edge_index_2d, x1, self.edge_index_1d, xe
        )

        return (
            d2_norm.detach().cpu().numpy().astype(np.float32, copy=False),
            d1_norm.detach().cpu().numpy().astype(np.float32, copy=False),
            inlet_norm.detach().cpu().numpy().astype(np.float32, copy=False),
            edge_norm.detach().cpu().numpy().astype(np.float32, copy=False),
        )

    def predict_all(
        self,
        X2: np.ndarray,
        wl2_t: np.ndarray,
        X1: np.ndarray,
        wl1_t: np.ndarray,
        Xe: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          wl2_next  : (N2,) absolute WL predictions for t+1
          wl1_next  : (N1,) absolute WL predictions for t+1
          inlet_next: (N1,) absolute inlet_flow predictions for t+1
          edge_next : (E1,) absolute edge_flow predictions for t+1
        """
        d2_norm, d1_norm, inlet_norm, edge_norm = self._forward(X2, X1, Xe)

        wl2_next   = wl2_t.astype(np.float32) + d2_norm    * self._t_std  + self._t_mean
        wl1_next   = wl1_t.astype(np.float32) + d1_norm    * self._t_std  + self._t_mean
        inlet_next = inlet_norm * self._ti_std + self._ti_mean
        edge_next  = edge_norm  * self._te_std + self._te_mean

        return (
            wl2_next.astype(self.dtype),
            wl1_next.astype(self.dtype),
            inlet_next.astype(self.dtype),
            edge_next.astype(self.dtype),
        )

    # Keep base class contract — not used directly
    def predict_both(self, X2, wl2_t, X1, wl1_t):
        raise RuntimeError("Use predict_all() for the four-output GNN.")

    def predict_2d(self, X2, wl2_t):
        raise RuntimeError("Use predict_all() for the four-output GNN.")

    def predict_1d(self, X1, wl1_t):
        raise RuntimeError("Use predict_all() for the four-output GNN.")
