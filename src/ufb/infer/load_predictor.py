from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
from xgboost import XGBRegressor

from ufb.features.graph_feats import UndirectedEdges
from ufb.infer.predictor_base import BackendConfig
from ufb.infer.predictor_gnn import GNNPredictor
from ufb.infer.predictor_xgb import XGBSinglePredictor, XGBTwoModelPredictor


def load_backend_config() -> BackendConfig:
    backend  = os.getenv("UFB_BACKEND", "xgb").lower().strip()
    alpha_1d = float(os.getenv("UFB_ALPHA_1D", "1.0"))
    clip_raw = os.getenv("UFB_CLIP_1D", "").strip()
    clip_1d  = None
    if clip_raw:
        a, b = clip_raw.split(",")
        clip_1d = (float(a), float(b))
    device = os.getenv("UFB_DEVICE", "cuda")
    return BackendConfig(
        backend=backend,
        alpha_1d=alpha_1d,
        clip_1d=clip_1d,
        device=device,
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
    Returns a Predictor instance for Model_1.

    For GNN: torch_model_ctor(meta) must return a Model1Net instance.
    The meta JSON (saved by train_gnn_model1.py) now includes all normalisation
    stats, which are forwarded to GNNPredictor automatically.
    """
    if cfg.backend == "xgb":
        if not xgb_model_path:
            raise ValueError("xgb_model_path is required for backend=xgb")
        model = load_xgb_model(xgb_model_path)
        return XGBSinglePredictor(model=model, dtype=dtype)

    if cfg.backend == "gnn":
        if not (gnn_ckpt_path and gnn_meta_path and torch_model_ctor):
            raise ValueError(
                "gnn_ckpt_path, gnn_meta_path, and torch_model_ctor are all required for backend=gnn"
            )

        with open(gnn_meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        torch_model = torch_model_ctor(meta)
        sd = torch.load(gnn_ckpt_path, map_location="cpu")
        torch_model.load_state_dict(sd)

        return GNNPredictor(
            model=torch_model,
            adj_2d=adj_2d,
            feature_mean_2d=np.array(meta["feature_mean_2d"], dtype=np.float32),
            feature_std_2d=np.array(meta["feature_std_2d"],  dtype=np.float32),
            feature_mean_1d=np.array(meta["feature_mean_1d"], dtype=np.float32),
            feature_std_1d=np.array(meta["feature_std_1d"],  dtype=np.float32),
            target_mean=float(meta["target_mean"]),
            target_std=float(meta["target_std"]),
            device=cfg.device,
            dtype=dtype,
        )

    raise ValueError(f"Unknown backend: {cfg.backend}")
