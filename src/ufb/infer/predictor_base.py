from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BackendConfig:
    backend: str          # "xgb" or "gnn"
    # XGB-specific
    alpha_1d: float = 1.0
    clip_1d: tuple[float, float] | None = None
    # GNN-specific
    device: str = "cuda"  # "cuda" or "cpu"


class Predictor(Protocol):
    """Backend-agnostic predictor interface used by rollout."""

    def predict_2d(self, X2: np.ndarray, wl2_t: np.ndarray) -> np.ndarray: ...
    def predict_1d(self, X1: np.ndarray, wl1_t: np.ndarray) -> np.ndarray: ...
