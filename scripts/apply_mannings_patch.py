"""
apply_mannings_patch.py
=======================
Adds Manning's equation pipe capacity clipping to rollout.py.
Clips predicted edge_flow to physically plausible range each rollout step.
Run once locally, then push to Kaggle.
"""
from pathlib import Path

project_root = Path(
    r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions"
    r"\05_UrbanFloodBench - flood modelling\urbanfloodbench"
)

ROLLOUT = project_root / "src" / "ufb" / "infer" / "rollout.py"
if not ROLLOUT.exists():
    raise FileNotFoundError(f"Not found: {ROLLOUT}")

txt = ROLLOUT.read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Fix 1: Import numpy at top (already imported, but add math for Manning's)
# Add Manning's precomputation before the rollout loop
# ---------------------------------------------------------------------------

OLD_PRECOMPUTE = (
    "    # ---- Precompute rainfall context for all T steps ----\n"
    "    rain_frac, rain_peak, rain_trend = _precompute_rainfall_context(rain2, cfg.dtype)"
)

NEW_PRECOMPUTE = (
    "    # ---- Precompute rainfall context for all T steps ----\n"
    "    rain_frac, rain_peak, rain_trend = _precompute_rainfall_context(rain2, cfg.dtype)\n"
    "\n"
    "    # ---- Precompute Manning's max pipe flow per edge (physical upper bound) ----\n"
    "    # Q_max = (1/n) * A * R^(2/3) * sqrt(S)  for full circular pipe\n"
    "    # All pipes confirmed circular (shape=0) in both models\n"
    "    _d = se[\"diameter\"].to_numpy(dtype=np.float32) if \"diameter\" in se.columns else np.ones(n_edges, np.float32)\n"
    "    _n = se[\"roughness\"].to_numpy(dtype=np.float32) if \"roughness\" in se.columns else np.full(n_edges, 0.02, np.float32)\n"
    "    _S = se[\"slope\"].to_numpy(dtype=np.float32)    if \"slope\"    in se.columns else np.full(n_edges, 0.01, np.float32)\n"
    "    _A = np.pi * _d**2 / 4\n"
    "    _R = _d / 4\n"
    "    q_max = (1.0 / np.maximum(_n, 1e-6)) * _A * _R**(2/3) * np.sqrt(np.maximum(_S, 1e-8))\n"
    "    q_max = q_max.astype(cfg.dtype)"
)

if OLD_PRECOMPUTE not in txt:
    print("WARNING: precompute pattern not found — check rollout.py manually")
else:
    txt = txt.replace(OLD_PRECOMPUTE, NEW_PRECOMPUTE)
    print("[OK] Added Manning's precomputation")

# ---------------------------------------------------------------------------
# Fix 2: Apply clip after eflow_next is sliced back to n_edges
# ---------------------------------------------------------------------------

OLD_CLIP = (
        "        eflow_next = np.nan_to_num(eflow_next, nan=0.0)\n"
        "        # Slice to original n_edges before storing in lag state\n"
        "        eflow_next = eflow_next[:n_edges]"
)

NEW_CLIP = (
        "        eflow_next = np.nan_to_num(eflow_next, nan=0.0)\n"
        "        # Slice to original n_edges before storing in lag state\n"
        "        eflow_next = eflow_next[:n_edges]\n"
        "        # Clip to Manning's physical maximum (both directions)\n"
        "        eflow_next = np.clip(eflow_next, -q_max, q_max)"
)

if OLD_CLIP not in txt:
    print("WARNING: clip pattern not found — check rollout.py manually")
else:
    txt = txt.replace(OLD_CLIP, NEW_CLIP)
    print("[OK] Added Manning's clipping")

ROLLOUT.write_text(txt, encoding="utf-8")

# Verify
txt_check = ROLLOUT.read_text(encoding="utf-8")
checks = ["q_max", "Manning", "np.clip(eflow_next, -q_max, q_max)"]
for c in checks:
    status = "✓" if c in txt_check else "✗ MISSING"
    print(f"  {status}  {c!r}")

print("\nDone. Push rollout.py to Kaggle and run submission script.")
