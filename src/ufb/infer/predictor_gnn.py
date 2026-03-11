"""
GNNPredictor — fixed version.

Key fixes vs. original:
  - Applies feature normalisation (z-score) inside _forward() using the stats
    saved during training (loaded from model1_meta.json).
  - predict_both() correctly denormalises the delta output before adding to wl_t.
  - predict_delta=True is the only supported / tested mode.
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
        feature_mean_2d: np.ndarray,
        feature_std_2d:  np.ndarray,
        feature_mean_1d: np.ndarray,
        feature_std_1d:  np.ndarray,
        target_mean: float = 0.0,
        target_std:  float = 1.0,
        device: str = "cuda",
        dtype: str = "float32",
    ):
        self.model = model
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.dtype = dtype

        # normalisation stats (kept as float32 arrays)
        self._mean2 = feature_mean_2d.astype(np.float32)
        self._std2  = np.where(feature_std_2d  < 1e-6, 1.0, feature_std_2d).astype(np.float32)
        self._mean1 = feature_mean_1d.astype(np.float32)
        self._std1  = np.where(feature_std_1d  < 1e-6, 1.0, feature_std_1d).astype(np.float32)
        self._target_mean = float(target_mean)
        self._target_std  = float(target_std) if target_std > 1e-6 else 1.0

        src = torch.as_tensor(adj_2d.src, dtype=torch.long)
        dst = torch.as_tensor(adj_2d.dst, dtype=torch.long)
        self.edge_index_2d = torch.stack([src, dst], dim=0).to(self.device)

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _forward(self, X2: np.ndarray, X1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # normalise features
        X2n = (X2 - self._mean2) / self._std2
        X1n = (X1 - self._mean1) / self._std1
        np.nan_to_num(X2n, copy=False)
        np.nan_to_num(X1n, copy=False)

        x2 = torch.as_tensor(X2n, dtype=torch.float32, device=self.device)
        x1 = torch.as_tensor(X1n, dtype=torch.float32, device=self.device)

        out2, out1 = self.model(x2, self.edge_index_2d, x1)  # normalised deltas
        y2 = out2.detach().cpu().numpy().astype(np.float32, copy=False)
        y1 = out1.detach().cpu().numpy().astype(np.float32, copy=False)
        return y2, y1

    def predict_both(
        self,
        X2: np.ndarray,
        wl2_t: np.ndarray,
        X1: np.ndarray,
        wl1_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns absolute water-level predictions for t+1.
          wl_{t+1} = wl_t  +  (normalised_delta * target_std + target_mean)
        """
        d2_norm, d1_norm = self._forward(X2, X1)

        # denormalise deltas
        d2 = d2_norm * self._target_std + self._target_mean
        d1 = d1_norm * self._target_std + self._target_mean

        y2 = (wl2_t.astype(np.float32, copy=False) + d2).astype(self.dtype, copy=False)
        y1 = (wl1_t.astype(np.float32, copy=False) + d1).astype(self.dtype, copy=False)
        return y2, y1

    # keep these so the base class contract is satisfied, but they always error
    def predict_2d(self, X2, wl2_t):
        raise RuntimeError("Use predict_both() for GNN inference.")

    def predict_1d(self, X1, wl1_t):
        raise RuntimeError("Use predict_both() for GNN inference.")
