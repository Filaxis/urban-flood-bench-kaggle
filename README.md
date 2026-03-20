# UrbanFloodBench: Autoregressive Flood Prediction with XGBoost and GNN

**Kaggle Competition:** [Urban Flood Modelling](https://www.kaggle.com/competitions/urban-flood-modelling)  
**Final Result:** Rank **41 / 264** (top 16%) — achieved with XGBoost; GNN best result was rank ~77/264 (public LB 0.3905, private LB 0.3522)  
**Metric:** Standardised RMSE (lower is better)

> 📌 **This is the `gnn` branch**, containing the Graph Neural Network solution.  
> The XGBoost solution (which achieved the final competition rank) lives in the [`main` branch](https://github.com/Filaxis/urban-flood-bench-kaggle/tree/main).

---

## Competition Overview

UrbanFloodBench is a Kaggle competition on urban flood simulation surrogation. Two hydraulic simulation models (**Model_1** and **Model_2**) represent different urban drainage networks. Participants are given a 10-step warmup window and must predict water levels at all nodes for the remainder of each rainfall event — some events lasting up to 399 timesteps. The key challenge is stable long-horizon autoregressive rollout without error accumulation.

**Dataset structure:**
- Model_1: 3,716 2D surface nodes + 17 1D pipe/channel nodes; 68 training events
- Model_2: 4,299 2D surface nodes + 198 1D pipe/channel nodes; 100 training events

---

## GNN Solution

This branch replaces the XGBoost regressors with a **Graph Neural Network trained end-to-end with rollout supervision (TBPTT)**. The core motivation: XGBoost required hand-crafted topology features to approximate spatial coupling; a GNN learns the message-passing function directly from data, theoretically enabling richer and more generalisable spatial representations.

### Architecture

| Component | Design |
|---|---|
| 2D encoder | GraphSAGE stack (3 layers, hidden=128, LayerNorm + residual) |
| 1D encoder | SAGEStack (2 layers, hidden=128) — upgraded from MLP in final version |
| Output heads | 4 outputs: Δwl_2D, Δwl_1D, inlet_flow, edge_flow (predicted jointly) |
| Training objective | TBPTT with K=6 unroll steps; MSE loss over rollout window |
| Predicted quantities fed back | Δwl_2D, Δwl_1D, inlet_flow, edge_flow all used as lag features next step |

The four-output head is the key architectural innovation: by jointly predicting water level changes *and* flow quantities, the model gains an explicit drainage signal during rollout — it can observe predicted inlet flows decreasing and correlate that with falling water levels, something impossible with water-level-only prediction.

### Features

| Group | Features |
|---|---|
| WL lags | `wl_t` through `wl_tm5` (6 steps) |
| Flow lags | `inlet_flow_t` through `inlet_flow_tm5`, `edge_flow_t` through `edge_flow_tm5` |
| Rainfall context | `rain_frac_remaining`, `rain_steps_since_peak`, `rain_intensity_trend` |
| Graph | neighbour mean WL, neighbour rain accumulation |
| Static | elevation, area, roughness, slope, pipe geometry, node degree |

### Training Protocol

- **Truncated Backpropagation Through Time (TBPTT):** gradients flow through K consecutive rollout steps; lag state detached between windows
- **K=6** selected as optimal for both models (K=12 caused regression on both)
- **Best checkpoints saved in real time** — training interruptions safe
- **Normalisation:** all features and targets standardised; stats stored in `meta.json`

---

## Submission History (GNN branch)

| # | Name | Private LB | Public LB | Notes |
|---|---|---|---|---|
| 7 | GNN-hybrid-v1-broken | 9.4054 | 9.4137 | First GNN attempt; delta/absolute target confusion in rollout |
| 8 | GNN-hybrid-v2-delta-fix | 8.5325 | 8.5389 | Fixed delta targets + feature normalisation; rollout mismatch still present |
| 9 | GNN-hybrid-v3-tbptt | 8.6721 | 8.7682 | TBPTT training, but Model_2 still XGBoost — mismatch persisted |
| 10 | GNN-full-v1 | 0.4311 | 0.5608 | Both models GNN; first functional full-GNN submission |
| 11 | GNN-full-v2-rainfall-k12 | 0.5161 | 0.6245 | Rainfall features added; K=12 used for Model_2 (K=6 run lost to session expiry) |
| 12 | **GNN-full-v3-four-output** | **0.3522** | **0.3905** | Four-output head (Δwl + inlet_flow + edge_flow); SAGEStack 1D encoder |
| 13 | GNN-full-v4-k12 | 0.4842 | 0.4959 | K=12 experiment; confirmed regression vs K=6 |
| 14 | GNN-full-v5-k12-model2 | 0.6160 | 0.6519 | K=12 for Model_2 only; worst GNN result |
| 15 | GNN-full-v6-extended-model2 | 0.3537 | 0.3960 | Model_2 continued +50 epochs with cosine LR warmup; effectively same as #12 |

> Full analysis including failure mode diagnostics: see [`RESULTS.md`](RESULTS.md)

---

## Repository Structure

```
.
├── scripts/
│   ├── build_train_parquet_model1.py       # Parquet builder with GNN features
│   ├── build_train_parquet_model2.py
│   ├── train_gnn_model_rollout.py          # Unified GNN training script (both models)
│   ├── rollout_validate_model1.py          # GNN rollout validation
│   ├── rollout_validate_model2.py
│   ├── predict_submission_full_gnn.py      # Full GNN submission generator
│   └── scan_event_lengths.py               # Event horizon analysis
└── src/ufb/
    ├── io/             # Data loading
    ├── infer/          # Rollout engine (TBPTT-aware)
    │   ├── rollout.py
    │   ├── predictor_gnn.py
    │   └── predictor_base.py
    ├── models/
    │   └── gnn_py.py   # Model1Net: SAGEStack + four-output head
    ├── training/
    │   ├── samples.py       # Feature/sample construction
    │   └── gnn_dataset.py   # Snapshot dataset for GNN
    └── features/
        └── graph_feats.py   # Adjacency + neighbour aggregation
```

---

## Key Findings

**TBPTT is necessary, not optional.** Training on single-step snapshots (as in the initial GNN attempts) produces a model that looks good on validation loss but collapses on multi-step rollout. Gradient must flow through the unroll window for the model to learn rollout-stable representations.

**The four-output head is the most impactful architectural improvement.** Adding inlet_flow and edge_flow as jointly predicted outputs — and feeding them back as lag features — improved the public LB from 0.5608 to 0.3905 (30% improvement). The model can now observe its own hydraulic exchange predictions and use them to inform water level dynamics.

**Long-horizon collapse remains unsolved.** For H=399 events, the model reverts to nearly identical predictions across events — a symptom of missing drainage signal that the rainfall context features failed to fix. The root cause: without inlet_flow, the model cannot distinguish between "water level is stable because equilibrium is reached" and "water level is stable because nothing is happening." The four-output model partially addresses this, but the collapse persists in long events.

**K=6 is the optimal TBPTT window.** K=12 caused consistent regression across multiple experiments. Longer windows increase gradient noise faster than they improve credit assignment at this model scale.

**Rainfall context features alone are insufficient.** `rain_frac_remaining`, `rain_steps_since_peak`, and `rain_intensity_trend` did not improve scores meaningfully. The model correctly learned that rainfall is ending, but without a signal about active drainage, it had no basis to predict falling water levels. These features remain in the architecture as they are free and may help in combination with better drainage signals.

---

## Development Path and Lessons

The GNN development went through three distinct phases:

**Phase 1 (Hybrid — submissions 7–9):** Model_1 replaced with GNN while Model_2 kept XGBoost. All three submissions scored in the 8–9 range due to a target type mismatch: the GNN predicted deltas, but the rollout was handling predictions as absolute values (or vice versa). This is a silent failure — the rollout runs without errors but produces completely wrong predictions.

**Phase 2 (Full GNN — submissions 10–11):** Both models switched to GNN. The first working full-GNN submission (10) scored 0.5608. Submission 11 regressed due to an accidental K=12 training of Model_2 (the K=6 run was lost to Kaggle session expiry).

**Phase 3 (Four-output architecture — submissions 12–15):** The architectural upgrade to jointly predict flow quantities alongside water levels produced the best GNN result (0.3905). Subsequent K tuning experiments confirmed K=6 as optimal. A final attempt to revert to a hybrid model (GNN + XGBoost) failed again with the same negative-prediction issue in Model_2, and was not submitted.

---

## Reproducibility

Training uses the `UFB_MODEL_ID` environment variable to select the model:

```bash
# Train Model_1
UFB_MODEL_ID=1 python scripts/train_gnn_model_rollout.py

# Train Model_2
UFB_MODEL_ID=2 python scripts/train_gnn_model_rollout.py

# Validate rollout
python scripts/rollout_validate_model1.py
python scripts/rollout_validate_model2.py

# Generate submission
python scripts/predict_submission_full_gnn.py
```

Set `PYTHONPATH=src` before running. GPU recommended (T4 or better); each full training run takes 2–6 hours depending on epoch count.

**Dependencies:** PyTorch, PyTorch Geometric, XGBoost, pandas, numpy, pyarrow
