# UrbanFloodBench: Autoregressive Flood Prediction with XGBoost and GNN

**Kaggle Competition:** [Urban Flood Modelling](https://www.kaggle.com/competitions/urban-flood-modelling)  
**Final Result:** Rank **41 / 264** (top 16%) — achieved with the XGBoost solution on this branch  
**Metric:** Standardised RMSE (lower is better)

> 📌 **This is the `main` branch**, containing the XGBoost-based solution.  
> A Graph Neural Network approach is developed in the [`gnn` branch](https://github.com/Filaxis/urban-flood-bench-kaggle/tree/gnn) — it represents a more architecturally ambitious attempt that scored lower (rank ~77/264) but demonstrates GNN-based spatiotemporal modelling on a hydraulic simulation task.

---

## Competition Overview

UrbanFloodBench is a Kaggle competition on urban flood simulation surrogation. Two hydraulic simulation models (**Model_1** and **Model_2**) represent different urban drainage networks. Participants are given a 10-step warmup window and must predict water levels at all nodes for the remainder of each rainfall event — some events lasting up to 399 timesteps. The key challenge is stable long-horizon autoregressive rollout without error accumulation.

**Dataset structure:**
- Model_1: 3,716 2D surface nodes + 17 1D pipe/channel nodes; 68 training events
- Model_2: 4,299 2D surface nodes + 198 1D pipe/channel nodes; 100 training events
- Dynamic features: water level, rainfall (2D), velocities, flow rates
- Static features: elevation, area, roughness, slope, pipe geometry

**Scoring:** Hierarchical standardised RMSE — errors are normalised by the standard deviation of each (model, node\_type) combination before averaging, which prevents the large 2D population from dominating.

---

## Solution: XGBoost with Topology-Aware Features

The winning approach on this branch uses **per-node autoregressive XGBoost** models with hand-crafted spatial coupling features that approximate graph convolution.

### Architecture

- **Model_1:** single XGBRegressor trained on all node types jointly
- **Model_2:** two separate XGBRegressors, one for 1D nodes and one for 2D nodes (split required because 2D rows otherwise dominate training and underfit the 198 1D nodes)

### Feature Engineering

| Feature Group | Features |
|---|---|
| Temporal lags | `wl_t`, `wl_tm1`, ..., `wl_tm5` (6 lags) |
| Rainfall | `rain_t`, `rain_tm1`, `rain_tm2`, `rain_tm3`, `rain_sum_4` |
| Delta | `d_wl_t = wl_t - wl_tm1` |
| Graph (2D) | `nbr_wl_mean_t`, `nbr_wl_mean_tm1`, `nbr_rain_sum_4` |
| Graph (1D) | `nbr_wl_mean_1d_t`, `nbr_wl_mean_1d_tm1` |
| Cross-subsystem | `conn2d_wl_t`, `conn2d_rain_sum_4` |
| Static | elevation, area, roughness, slope, pipe geometry, node degree |

The topology features — neighbour mean water levels computed from the hydraulic graph — are the single largest improvement, reducing Model_1 1D rollout RMSE from ~1.83 to ~0.56 by providing the spatial coupling information the model otherwise had no access to.

### Rollout

- Warmup: first 10 timesteps from ground truth
- Autoregressive: predicted `wl_t+1` feeds back as `wl_t` at next step
- No post-processing, clipping, or damping in the final version

### Training

- XGBoost with `tree_method=hist`, `max_depth=10`, `learning_rate=0.02`, `n_estimators=5000`, early stopping
- All training events used; last 20% of events held out as validation
- Dataset corruption note: competition organiser released a corrected dataset mid-competition; all results from submission 5 onward use the corrected data

---

## Submission History

| # | Name | Private LB | Public LB | Notes |
|---|---|---|---|---|
| 1 | XGB-2lag-baseline | 4.1448 | 4.1781 | Scale mismatch bug: 1D WL clipped to [0, 50], catastrophic for Model_1 (WL ~320–360) |
| 2 | XGB-2lag-no-clipping | 0.5969 | 0.6329 | Removed clipping/damping; revealed true model performance |
| 3 | XGB-4lag | 0.5800 | 0.6091 | 4 WL lags + rainfall accumulation |
| 4 | XGB-6lag | 0.5522 | 0.5637 | 6 WL lags; trained on corrupted dataset |
| 5 | XGB-6lag-corrected | 0.2275 | 0.2412 | Same model, corrected dataset — jump reflects dataset fix |
| 6 | **XGB-6lag-topology** | **0.1505** | **0.1535** | Added graph-neighbour features; **best result, final rank** |

> Detailed analysis of each submission including root-cause explanations: see [`RESULTS.md`](RESULTS.md)

---

## Repository Structure

```
.
├── scripts/
│   ├── build_train_parquet_model1.py   # Feature engineering + parquet builder for Model_1
│   ├── build_train_parquet_model2.py   # Feature engineering + parquet builder for Model_2
│   ├── train_xgb_model1.py             # XGBoost training for Model_1
│   ├── train_xgb_model2.py             # XGBoost training for Model_2 (unified)
│   ├── train_xgb_model2_split_by_type.py  # XGBoost training for Model_2 (1D/2D split)
│   ├── rollout_validate_model1.py      # Autoregressive rollout validation
│   ├── rollout_validate_model2.py
│   ├── predict_submission_full.py      # Full submission generator (CSV)
│   └── predict_submission_full_parquet.py  # Full submission generator (Parquet)
└── src/ufb/
    ├── io/           # Data loading: static, dynamics, events, submission
    ├── infer/        # Rollout engine
    ├── training/     # Sample builders
    ├── features/     # Graph feature computation
    └── eda/          # Rainfall analysis utilities
```

---

## Key Findings

**Topology features are the decisive factor.** Going from 6-lag-only to 6-lag-with-topology dropped the public LB from 0.2412 to 0.1535 — a 36% improvement. The mechanism is straightforward: the 1D pipe network in Model_1 is a small, tightly coupled system (17 nodes), and without information about what neighbouring nodes are doing, each node is essentially modelling itself in isolation. Adding the graph-average neighbour water level provides the missing spatial coupling at almost no implementation cost.

**Dataset correctness matters more than model sophistication.** The corrected dataset alone (submission 4 → 5) improved the score from 0.5637 to 0.2412 with no model changes whatsoever — a 57% improvement from fixing input data quality.

**Splitting Model_2 by node type is necessary.** Model_2's 4,299 2D nodes drown out the 198 1D nodes in a unified model. Training separate regressors per node type was required to stabilise the 1D subsystem.

---

## Why Switch to GNN?

The XGBoost score of 0.1535 was competitive (top 16%). Further improvements along the XGB path were possible — multi-hop neighbourhood aggregation, difference-to-neighbour features, multi-step training objectives — but would still be approximating graph-structured inference through hand-crafted features.

The GNN branch explores whether learning the message-passing function end-to-end from rollout supervision can outperform feature engineering. This was also a deliberate learning goal: the GNN branch represents a first complete implementation of a spatiotemporal GNN on a physics simulation task. The GNN solution ultimately scored lower (best public LB 0.3905), but the gap narrows meaningfully as the architecture improves — and the failure modes are instructive. See the [`gnn` branch](https://github.com/Filaxis/urban-flood-bench-kaggle/tree/gnn) and its [`RESULTS.md`](https://github.com/Filaxis/urban-flood-bench-kaggle/blob/gnn/RESULTS.md) for the full story.

---

## Reproducibility

```bash
# 1. Build training parquets (requires competition data)
python scripts/build_train_parquet_model1.py
python scripts/build_train_parquet_model2.py

# 2. Train models
python scripts/train_xgb_model1.py
python scripts/train_xgb_model2_split_by_type.py

# 3. Validate rollout
python scripts/rollout_validate_model1.py
python scripts/rollout_validate_model2.py

# 4. Generate submission
python scripts/predict_submission_full_parquet.py
```

Set `PYTHONPATH=src` before running.
