from __future__ import annotations

import numpy as np
import torch

from ufb.features.graph_feats import UndirectedEdges
from ufb.infer.predictor_base import Predictor


class GNNPredictor(Predictor):
    """
    Backend for a torch model that predicts either:
      - deltas (d2, d1) and we add wl_t, OR
      - absolute next wl (y2, y1)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        adj_2d: UndirectedEdges,
        device: str = "cuda",
        dtype: str = "float32",
        predict_delta: bool = True,
    ):
        self.model = model
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.dtype = dtype
        self.predict_delta = predict_delta

        src = torch.as_tensor(adj_2d.src, dtype=torch.long)
        dst = torch.as_tensor(adj_2d.dst, dtype=torch.long)
        self.edge_index_2d = torch.stack([src, dst], dim=0).to(self.device)

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def _forward(self, X2: np.ndarray, X1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x2 = torch.as_tensor(X2, dtype=torch.float32, device=self.device)
        x1 = torch.as_tensor(X1, dtype=torch.float32, device=self.device)

        out2, out1 = self.model(x2, self.edge_index_2d, x1)  # (N2,), (N1,)
        y2 = out2.detach().cpu().numpy().astype(self.dtype, copy=False)
        y1 = out1.detach().cpu().numpy().astype(self.dtype, copy=False)
        return y2, y1

    def predict_2d(self, X2: np.ndarray, wl2_t: np.ndarray) -> np.ndarray:
        out2, _ = self._forward(X2, self._dummy_x1_for_shape())  # replaced below by cached last X1
        # NOTE: rollout will call predict_2d then predict_1d; easiest is to call a combined method.
        # For minimal safety, we implement combined predictions in rollout (see below).
        raise RuntimeError("Use predict_both() in rollout (see adjusted rollout.py).")

    def predict_1d(self, X1: np.ndarray, wl1_t: np.ndarray) -> np.ndarray:
        raise RuntimeError("Use predict_both() in rollout (see adjusted rollout.py).")

    @torch.inference_mode()
    def predict_both(
        self,
        X2: np.ndarray,
        wl2_t: np.ndarray,
        X1: np.ndarray,
        wl1_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        out2, out1 = self._forward(X2, X1)

        if self.predict_delta:
            y2 = wl2_t.astype(self.dtype, copy=False) + out2
            y1 = wl1_t.astype(self.dtype, copy=False) + out1
        else:
            y2, y1 = out2, out1

        return y2.astype(self.dtype, copy=False), y1.astype(self.dtype, copy=False)

    def _dummy_x1_for_shape(self) -> np.ndarray:
        # Not used if you use predict_both(). Kept to make errors explicit.
        return np.zeros((1, 1), dtype=np.float32)