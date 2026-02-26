# src/ufb/infer/predictors.py
from __future__ import annotations

import numpy as np
import torch

from ufb.models.gnn import Model1Net


@torch.inference_mode()
def predict_next_delta_model1(
    model: Model1Net,
    X2: np.ndarray,
    X1: np.ndarray,
    edge_index_2d: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inputs:
      X2: (N2, F2) numpy float32
      X1: (N1, F1) numpy float32
    Returns:
      d2, d1: numpy float32 vectors (delta wl)
    """
    model.eval()

    x2 = torch.as_tensor(X2, dtype=torch.float32, device=device)
    x1 = torch.as_tensor(X1, dtype=torch.float32, device=device)
    e2 = edge_index_2d.to(device)

    d2, d1 = model(x2d=x2, edge_index_2d=e2, x1d=x1)

    return d2.detach().cpu().numpy().astype(np.float32), d1.detach().cpu().numpy().astype(np.float32)