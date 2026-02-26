from __future__ import annotations

import os
import json
from typing import Any

import torch
from xgboost import XGBRegressor

from ufb.features.graph_feats import UndirectedEdges
from ufb.infer.predictor_base import BackendConfig
from ufb.infer.predictor_gnn import GNNPredictor
from ufb.infer.predictor_xgb import XGBSinglePredictor, XGBTwoModelPredictor


def load_backend_config() -> BackendConfig:
    backend = os.getenv("UFB_BACKEND", "xgb").lower().strip()
    alpha_1d = float(os.getenv("UFB_ALPHA_1D", "1.0"))
    clip_raw = os.getenv("UFB_CLIP_1D", "").strip()  # e.g. "-5,50"
    clip_1d = None
    if clip_raw:
        a, b = clip_raw.split(",")
        clip_1d = (float(a), float(b))

    device = os.getenv("UFB_DEVICE", "cuda")
    predict_delta = os.getenv("UFB_GNN_PREDICT_DELTA", "1") in ("1", "true", "True")

    return BackendConfig(
        backend=backend,
        alpha_1d=alpha_1d,
        clip_1d=clip_1d,
        device=device,
        predict_delta=predict_delta,
    )


def load_xgb_model(path: str) -> XGBRegressor:
    m = XGBRegressor()
    m.load_model(path)
    return m


def load_predictor_model1(
    *,
    cfg: BackendConfig,
    dtype: str,
    adj_2d: UndirectedEdges,
    xgb_model_path: str | None = None,
    gnn_ckpt_path: str | None = None,
    gnn_meta_path: str | None = None,
    torch_model_ctor: Any | None = None,
):
    """
    Returns:
      - XGBSinglePredictor or GNNPredictor
    For GNN you must provide torch_model_ctor that builds the torch model instance.
    """
    if cfg.backend == "xgb":
        if not xgb_model_path:
            raise ValueError("xgb_model_path is required for backend=xgb")
        model = load_xgb_model(xgb_model_path)
        return XGBSinglePredictor(model=model, dtype=dtype)

    if cfg.backend == "gnn":
        if not (gnn_ckpt_path and gnn_meta_path and torch_model_ctor):
            raise ValueError("gnn_ckpt_path, gnn_meta_path, and torch_model_ctor are required for backend=gnn")

        with open(gnn_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # ctor must accept meta or infer dims
        torch_model = torch_model_ctor(meta)

        sd = torch.load(gnn_ckpt_path, map_location="cpu")
        torch_model.load_state_dict(sd)

        return GNNPredictor(
            model=torch_model,
            adj_2d=adj_2d,
            device=cfg.device,
            dtype=dtype,
            predict_delta=cfg.predict_delta,
        )

    raise ValueError(f"Unknown backend: {cfg.backend}")