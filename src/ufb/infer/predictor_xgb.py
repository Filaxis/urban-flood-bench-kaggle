from __future__ import annotations

import numpy as np
from xgboost import XGBRegressor

from ufb.infer.predictor_base import Predictor


class XGBSinglePredictor(Predictor):
    """Single XGB model used for both 2D and 1D (your rollout_event_model1 style)."""

    def __init__(self, model: XGBRegressor, dtype: str = "float32"):
        self.model = model
        self.dtype = dtype

    def predict_2d(self, X2: np.ndarray, wl2_t: np.ndarray) -> np.ndarray:
        y = self.model.predict(X2).astype(self.dtype, copy=False)
        return y

    def predict_1d(self, X1: np.ndarray, wl1_t: np.ndarray) -> np.ndarray:
        y = self.model.predict(X1).astype(self.dtype, copy=False)
        return y


class XGBTwoModelPredictor:
    """Two-model predictor used for rollout_event_two_models (separate 1D/2D)."""

    def __init__(
        self,
        model_2d: XGBRegressor,
        model_1d: XGBRegressor,
        dtype: str = "float32",
        alpha_1d: float = 1.0,
        clip_1d: tuple[float, float] | None = None,
    ):
        self.model_2d = model_2d
        self.model_1d = model_1d
        self.dtype = dtype
        self.alpha_1d = alpha_1d
        self.clip_1d = clip_1d

    def predict_2d(self, X2: np.ndarray, wl2_t: np.ndarray) -> np.ndarray:
        y2 = self.model_2d.predict(X2).astype(self.dtype, copy=False)
        return y2

    def predict_1d(self, X1: np.ndarray, wl1_t: np.ndarray) -> np.ndarray:
        y1 = self.model_1d.predict(X1).astype(self.dtype, copy=False)

        if self.alpha_1d < 1.0:
            y1 = self.alpha_1d * y1 + (1.0 - self.alpha_1d) * wl1_t

        if self.clip_1d is not None:
            y1 = np.clip(y1, self.clip_1d[0], self.clip_1d[1])

        return y1