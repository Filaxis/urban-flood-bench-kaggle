"""
apply_edge_fixes.py
====================
Applies all three edge-tensor size fixes to the two affected files:
  1. scripts/train_gnn_model_rollout.py  (3 fixes)
  2. src/ufb/infer/rollout.py            (2 fixes)

Run once from anywhere:
    python apply_edge_fixes.py
"""
from pathlib import Path

project_root = Path(
    r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions"
    r"\05_UrbanFloodBench - flood modelling\urbanfloodbench"
)

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
TRAIN = project_root / "scripts" / "train_gnn_model_rollout.py"
ROLLOUT = project_root / "src" / "ufb" / "infer" / "rollout.py"

for p in (TRAIN, ROLLOUT):
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def patch(path: Path, old: str, new: str, label: str) -> None:
    txt = path.read_text(encoding="utf-8")
    if old not in txt:
        print(f"  [SKIP] {label} — pattern not found (already patched?)")
        return
    if new in txt:
        print(f"  [SKIP] {label} — fix already present")
        return
    count = txt.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected exactly 1 occurrence, found {count}")
    path.write_text(txt.replace(old, new), encoding="utf-8")
    print(f"  [OK]   {label}")

# ---------------------------------------------------------------------------
# train_gnn_model_rollout.py — Fix 1
# build_x_edge: duplicate Xe rows for both directed edges
# ---------------------------------------------------------------------------
patch(
    TRAIN,
    old=(
        "    np.nan_to_num(Xe, copy=False)\n"
        "        return Xe"
    ),
    new=(
        "    np.nan_to_num(Xe, copy=False)\n"
        "        # Duplicate for both directions (undirected graph: a→b and b→a)\n"
        "        return np.concatenate([Xe, Xe], axis=0)"
    ),
    label="TRAIN fix1: build_x_edge duplicate rows",
)

# ---------------------------------------------------------------------------
# train_gnn_model_rollout.py — Fix 2
# gte: double ground-truth edge tensor to match 394-element prediction
# ---------------------------------------------------------------------------
patch(
    TRAIN,
    old=(
        "        gte  = torch.as_tensor(seq.gt_eflow[ti], dtype=torch.float32, device=device)"
    ),
    new=(
        "        gte  = torch.as_tensor(\n"
        "            np.concatenate([seq.gt_eflow[ti], seq.gt_eflow[ti]]),\n"
        "            dtype=torch.float32, device=device\n"
        "        )"
    ),
    label="TRAIN fix2: double ground-truth edge tensor",
)

# ---------------------------------------------------------------------------
# train_gnn_model_rollout.py — Fix 3
# eflow_new: slice back to n_edges before storing in lag state
# ---------------------------------------------------------------------------
patch(
    TRAIN,
    old=(
        "        eflow_new = edge_next.detach().cpu().numpy()"
    ),
    new=(
        "        # Slice to original n_edges (first half = forward direction)\n"
        "        eflow_new = edge_next.detach().cpu().numpy()[:seq.n_edges]"
    ),
    label="TRAIN fix3: slice eflow_new to n_edges",
)

# ---------------------------------------------------------------------------
# rollout.py — Fix 4
# Xe: duplicate after building for predictor forward pass
# ---------------------------------------------------------------------------
patch(
    ROLLOUT,
    old=(
        "        Xe = df_e.reindex(columns=feature_cols_edge).to_numpy(dtype=np.float32, copy=False)\n"
        "\n"
        "        # ---- Predict ----"
    ),
    new=(
        "        Xe = df_e.reindex(columns=feature_cols_edge).to_numpy(dtype=np.float32, copy=False)\n"
        "        # Duplicate for both directed edges (undirected graph: a→b and b→a)\n"
        "        Xe = np.concatenate([Xe, Xe], axis=0)\n"
        "\n"
        "        # ---- Predict ----"
    ),
    label="ROLLOUT fix4: duplicate Xe rows",
)

# ---------------------------------------------------------------------------
# rollout.py — Fix 5
# eflow_next: slice back to n_edges before shifting lag state
# ---------------------------------------------------------------------------
patch(
    ROLLOUT,
    old=(
        "        eflow_next = np.nan_to_num(eflow_next, nan=0.0)\n"
        "\n"
        "        pred2[k, :] = y2"
    ),
    new=(
        "        eflow_next = np.nan_to_num(eflow_next, nan=0.0)\n"
        "        # Slice to original n_edges before storing in lag state\n"
        "        eflow_next = eflow_next[:n_edges]\n"
        "\n"
        "        pred2[k, :] = y2"
    ),
    label="ROLLOUT fix5: slice eflow_next to n_edges",
)

# ---------------------------------------------------------------------------
print("\nAll done. Verify:")
for p in (TRAIN, ROLLOUT):
    txt = p.read_text(encoding="utf-8")
    checks = {
        TRAIN: [
            "concatenate([Xe, Xe], axis=0)",
            "concatenate([seq.gt_eflow[ti], seq.gt_eflow[ti]])",
            "[:seq.n_edges]",
        ],
        ROLLOUT: [
            "concatenate([Xe, Xe], axis=0)",
            "eflow_next[:n_edges]",
        ],
    }
    for check in checks[p]:
        status = "✓" if check in txt else "✗ MISSING"
        print(f"  {status}  {p.name}: {check!r}")
