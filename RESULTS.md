# Results Analysis — UrbanFloodBench

This document provides a detailed analysis of all 15 submissions across both competition branches. It covers root-cause explanations for score changes, failure mode diagnostics, and the technical reasoning behind each architectural decision.

---

## Score Overview

| # | Name | Private LB | Public LB | Branch |
|---|---|---|---|---|
| 1 | XGB-2lag-baseline | 4.1448 | 4.1781 | main |
| 2 | XGB-2lag-no-clipping | 0.5969 | 0.6329 | main |
| 3 | XGB-4lag | 0.5800 | 0.6091 | main |
| 4 | XGB-6lag | 0.5522 | 0.5637 | main |
| 5 | XGB-6lag-corrected | 0.2275 | 0.2412 | main |
| 6 | XGB-6lag-topology | **0.1505** | **0.1535** | main |
| 7 | GNN-hybrid-v1-broken | 9.4054 | 9.4137 | gnn |
| 8 | GNN-hybrid-v2-delta-fix | 8.5325 | 8.5389 | gnn |
| 9 | GNN-hybrid-v3-tbptt | 8.6721 | 8.7682 | gnn |
| 10 | GNN-full-v1 | 0.4311 | 0.5608 | gnn |
| 11 | GNN-full-v2-rainfall-k12 | 0.5161 | 0.6245 | gnn |
| 12 | GNN-full-v3-four-output | **0.3522** | **0.3905** | gnn |
| 13 | GNN-full-v4-k12 | 0.4842 | 0.4959 | gnn |
| 14 | GNN-full-v5-k12-model2 | 0.6160 | 0.6519 | gnn |
| 15 | GNN-full-v6-extended-model2 | 0.3537 | 0.3960 | gnn |

The final competition rank of **41/264** was determined by submission 6 (XGB with topology features, public LB 0.1535, private LB 0.1505). The best GNN result (submission 12, public LB 0.3905) would have corresponded to approximately rank 77/264.

---

## XGBoost Phase (Submissions 1–6)

### Submission 1 → 2: The Scale Mismatch Bug (4.14 → 0.60)

The jump from 4.14 to 0.60 is the largest single improvement in the competition run, and it came entirely from removing a clipping operation — not from any model improvement.

During early rollout development, a stabilisation step was added that clipped predicted water levels:

```python
y1 = np.clip(y1, 0.0, 50.0)
```

This was a reasonable precaution for Model_2, whose 1D nodes have water levels in the 23–49 range. However, the same clip was inadvertently left active in `rollout_event_model1()` as well. Model_1's water levels are in the 300–360 range. The clip therefore truncated **every single Model_1 1D prediction** to 50 at most — not a gradual degradation but a complete failure of the 1D subsystem.

Damping (`y1 = 0.9 * y1_raw + 0.1 * wl1_t`) was also tested but had no measurable effect on either model, confirming that the Model_2 1D drift was a systematic modelling deficiency rather than numerical instability. The 1D rollout error in Model_2 at this stage was a consequence of the model having very limited information about the underground pipe network dynamics — not a fixable numerical issue.

**Lesson:** Any post-processing applied to predictions must be validated against the actual value range of each subsystem independently. A plausible bound for one model can be catastrophic for another.

### Submission 4 → 5: The Dataset Correction (0.56 → 0.24)

The competition organiser released a corrected dataset mid-competition. The original data contained a misalignment in the node indexing for Model_2 dynamic features — 1D and/or 2D node time series were mapped to the wrong node IDs, meaning the model was trained on systematically shuffled targets.

Submission 5 is submission 4's model retrained on the corrected data, with no other changes. The improvement from 0.5637 to 0.2412 (57%) is entirely attributable to data correctness.

This illustrates a general principle: no amount of model sophistication can compensate for corrupted training targets. The dataset quality issue was invisible during training (RMSE metrics looked reasonable) because the shuffled mapping was consistent — the model learned to predict shuffled values consistently, which happened to match the shuffled ground truth on the validation split.

### Submission 5 → 6: Topology Features (0.24 → 0.15)

Adding graph-neighbourhood features dropped the score by 36%. The changes:

**For 2D nodes:** `nbr_wl_mean_t`, `nbr_wl_mean_tm1`, `nbr_rain_sum_4` — mean water level and rainfall accumulation of spatially adjacent 2D cells.

**For 1D nodes:** `nbr_wl_mean_1d_t`, `nbr_wl_mean_1d_tm1` — mean water level of adjacent pipe nodes, plus `conn2d_wl_t` and `conn2d_rain_sum_4` — water level and rainfall at the 2D surface cell connected to each 1D pipe node.

The largest single gain was in Model_1's 1D subsystem: rollout RMSE for 1D nodes dropped from ~1.83 to ~0.56. This makes hydraulic sense. The 1D network in Model_1 has only 17 nodes, each of which is tightly coupled to its neighbours via pipe flow. A model with no spatial information treats each node as independent and cannot represent the propagation of water through the pipe network. Once neighbour water levels are included as features, the model can implicitly approximate the pipe flow dynamics.

One-step validation RMSE also improved (0.0217 → 0.0178), confirming the feature quality improvement extends beyond rollout stability.

**Rollout comparison (Model_1):**

| Metric | Without topology | With topology |
|---|---|---|
| RMSE all | 0.5789 | 0.6020 |
| RMSE 1D | 1.8310 | 0.5638 |
| RMSE 2D | 0.5669 | 0.6022 |

The 2D RMSE slightly worsened — a minor negative interaction, likely because the neighbourhood aggregation smooths 2D predictions towards local means, slightly reducing sharpness for peak water levels. This trade-off is clearly worth it given the massive 1D improvement.

---

## GNN Hybrid Phase (Submissions 7–9)

All three hybrid submissions scored in the 8–9 range. The architectural intent was correct (GNN for Model_1 while keeping the proven XGBoost for Model_2), but a fundamental integration error prevented them from working.

### The Delta/Absolute Target Confusion

The GNN was trained to predict **delta water levels** (Δwl = wl_{t+1} - wl_t). The XGBoost models on the `main` branch were trained to predict **absolute water levels** (wl_{t+1} directly). The rollout engine had separate code paths for the two models, but in the hybrid setup the code path for Model_1 was handling GNN outputs inconsistently — either treating deltas as absolute values, or not adding the delta back to the current water level correctly.

This is a silent failure. The rollout completes without errors and produces numerical outputs, but those outputs are completely wrong. Submission 7 (public 9.41) was the first to expose this. Submissions 8 and 9 made partial fixes — submission 8 corrected delta target handling and added feature normalisation at inference, submission 9 added TBPTT training — but the core mismatch between the GNN rollout code path and the XGBoost rollout code path was not fully resolved until the full GNN switch in submission 10.

**Lesson:** When mixing two model types in a rollout, the data flow contract (delta vs absolute, normalised vs raw) must be explicitly verified per model at each step, not assumed from training configuration.

### Why Submission 9 (TBPTT) Scored Slightly Worse than 8

Submission 9 introduced proper TBPTT training (K=6, 20 epochs, GPU), which is the correct training paradigm. The slight regression vs submission 8 (8.67 vs 8.53) is consistent with the hypothesis that the rollout integration bug was still present and masked any training quality improvement. TBPTT produces a better-trained model; the integration bug produced wrong predictions regardless of model quality.

---

## Full GNN Phase (Submissions 10–15)

### Submission 10: First Working Full GNN (public 0.5608)

Switching both models to GNN eliminated the hybrid integration issue. The architecture at this point:
- GraphSAGE (3 layers, hidden=128) for 2D nodes
- MLP (2 layers) for 1D nodes
- Single output head predicting Δwl for 2D and 1D separately
- TBPTT K=6, 20 epochs for Model_1, 35 epochs for Model_2

The score of 0.5608 is worse than the XGBoost baseline (0.1535) but represents a genuine and functional GNN rollout — the first time the GNN was actually predicting meaningful water level trajectories.

The remaining gap to XGBoost reflects several limitations: the model had no inlet_flow or edge_flow information during rollout, so it could not learn drainage dynamics; and the 1D encoder (MLP) had no graph structure to exploit despite 1D nodes being connected in a pipe network.

### Submission 11: Rainfall Features + Accidental K=12 (public 0.6245 — regression)

This submission is described in the submission log as using K=6, but the K=6 training run for Model_2 was lost when the Kaggle interactive session expired while unattended. The submitted Model_2 was trained with K=12 and 25 epochs, and the description was not updated to reflect this.

The K=12 Model_2 converged to a best loss of 0.474 vs 0.287 for K=6 in the previous run. This explains the regression. Rainfall context features (rain_frac_remaining, rain_steps_since_peak, rain_intensity_trend) were added correctly and were present in training, but their effect was masked by the K=12 degradation.

Post-hoc analysis of the submission output revealed the long-horizon collapse signature: all Model_2 events with H=399 produced **identical min and max water level values** across events (1D: min=23.017, max=54.732; 2D: min=32.960, max=54.732), while short events (H=51) showed genuine differentiation. This confirmed that the rainfall features alone were insufficient to address the long-horizon collapse — the model correctly learned "rainfall is ending" but had no basis to predict drainage dynamics without inlet_flow signal.

### Submission 12: Four-Output Architecture — Best GNN Result (public 0.3905)

The key change: the model now jointly predicts four quantities per step:
- Δwl_2D (water level delta for 2D nodes)
- Δwl_1D (water level delta for 1D nodes)
- inlet_flow (flow between 1D and 2D at connection points)
- edge_flow (flow along 1D pipe edges)

All four are fed back as lag features at the next rollout step. This gives the model an explicit drainage signal: as predicted inlet_flow decreases, the model can learn to predict falling water levels. Without this, the model is trying to infer drainage from water level trajectories alone, which contain no information about the timescale of hydraulic exchange.

Additional change: the 1D encoder was upgraded from MLP to SAGEStack (2 GraphSAGE layers with LayerNorm and residual), enabling the 1D model to aggregate information along the pipe network.

The combined effect: public LB improved from 0.5608 to 0.3905 (30% improvement). This is the largest single improvement in the GNN phase.

**Training configuration:** 25 epochs each model, K=6, LR=1e-3 with cosine decay.

### Submission 13: K=12 Experiment (public 0.4959 — regression)

A controlled test of K=12 vs K=6, holding all other parameters constant. K=12 regressed from 0.3905 to 0.4959. This is consistent across multiple experiments: longer TBPTT windows do not help at this model scale and actually hurt by introducing more gradient noise and making the training signal harder to optimise.

**Interpretation:** at K=6, the gradient signal covers approximately 6 minutes of simulated time (given typical 1-minute timesteps). This is long enough to capture local flood dynamics but short enough that gradients remain informative. K=12 doubles the temporal credit assignment problem without a corresponding increase in model capacity to exploit it.

### Submission 14: K=12 for Model_2 Only (public 0.6519 — worst GNN result)

Attempting K=12 for Model_2 only (while keeping K=6 for Model_1) produced the worst GNN result. Model_2 is the harder model (larger graph, more complex dynamics) and appears more sensitive to the K=12 degradation than Model_1. This data point confirmed that K=12 should not be used for either model.

### Submission 15: Extended Model_2 Training (public 0.3960 — effectively same as #12)

Model_2 training was extended by 50 additional epochs using a cosine LR schedule with warmup. The result (0.3960) is essentially identical to submission 12 (0.3905), confirming that the model had converged and additional training could not improve it further.

A Manning's formula regularisation term was explored during this phase but showed no measurable effect and was not submitted separately. A final attempt to revert to a hybrid model (GNN Model_1 + XGBoost Model_2) failed again due to the same negative-prediction issue that affected the earlier hybrid submissions, and was not submitted.

---

## Cross-Phase Observations

### Why GNN Underperformed XGBoost

The XGBoost result (0.1535) outperforms the best GNN result (0.3905) by a factor of ~2.5. Several factors contribute:

**Training data efficiency.** XGBoost trains on individual timestep snapshots from all training events simultaneously (millions of rows), whereas the GNN TBPTT training processes events sequentially with a limited number of K-step windows per epoch. XGBoost therefore sees far more diverse examples per training step.

**Spatial receptive field.** The XGBoost model uses pre-computed 1-hop neighbour aggregates. The GNN uses 3-layer message passing, theoretically covering 3 hops — but in practice the SAGE aggregation with residuals may not be using that depth effectively, especially on the small 1D graph (17 nodes in Model_1).

**Long-horizon collapse.** The GNN was unable to produce differentiated predictions for H=399 events, while XGBoost (with hand-crafted topology features) also showed drift but maintained node-level differentiation. The collapse is a fundamental limitation of predicting without explicit drainage dynamics, not a GNN-specific failure.

**Time invested.** This was a first implementation of a spatiotemporal GNN on a physics simulation task. A significant portion of development time was spent resolving infrastructure issues (delta/absolute mismatch, TBPTT implementation, normalisation at inference) rather than model improvement. Given the same investment of time as the XGBoost solution, the GNN could likely be pushed considerably further.

### Private vs Public LB Gap

Several submissions show a meaningful gap between private and public leaderboard scores. Notable examples:

- Submission 10: private 0.4311, public 0.5608 — private is substantially better. The private test set may contain more short-horizon events where the model performs well.
- Submission 12: private 0.3522, public 0.3905 — same direction, smaller gap.

In all cases the private score is better than the public score, consistent with the long-horizon collapse being more exposed in the public test set.
