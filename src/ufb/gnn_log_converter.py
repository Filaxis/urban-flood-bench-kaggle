import re
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ----------------------------
# Config
# ----------------------------
LOG_PATH = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench\src\ufb\GNN-training-output_01.txt")   # <-- change to your .txt file path
OUTPUT_DIR = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench\src\ufb")
USE_X = "u"  # "u" or "step"  (u is usually better within-event trend)

# Example line:
# [ep 1] file 2/68 model1_train_event002.parquet u=025/100 step=126 loss=106.1528
LINE_RE = re.compile(
    r"^\[ep\s+(?P<epoch>\d+)\]\s+file\s+(?P<file_idx>\d+)/(?P<n_files>\d+)\s+"
    r"(?P<event>.+?)\s+u=(?P<u>\d+)/(?P<u_max>\d+)\s+step=(?P<step>\d+)\s+loss=(?P<loss>[-+0-9.eE]+)"
)

EPOCH_HDR_RE = re.compile(r"^===\s*EPOCH\s+(?P<epoch>\d+)\s*/\s*(?P<total>\d+)\s*===\s*$")


def fit_line(x: np.ndarray, y: np.ndarray):
    """
    Fit y = a + b*x. Returns (a, b, r2).
    For <2 points or degenerate x, returns (nan, nan, nan).
    """
    if len(x) < 2:
        return (math.nan, math.nan, math.nan)
    if np.allclose(x, x[0]):
        return (math.nan, math.nan, math.nan)

    b, a = np.polyfit(x, y, 1)  # polyfit returns slope first for degree=1
    y_hat = a + b * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return (float(a), float(b), float(r2))


def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log file not found: {LOG_PATH.resolve()}")

    # Parse all lines into a flat table
    rows = []
    current_epoch_from_header = None

    with LOG_PATH.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue

            m_hdr = EPOCH_HDR_RE.match(line)
            if m_hdr:
                current_epoch_from_header = int(m_hdr.group("epoch"))
                continue

            m = LINE_RE.match(line)
            if not m:
                continue

            epoch = int(m.group("epoch"))
            # sanity: sometimes header epoch can help detect weird logs, but we trust [ep X] first
            if current_epoch_from_header is not None and current_epoch_from_header != epoch:
                # not fatal; logs sometimes interleave
                pass

            event = m.group("event").strip()
            # Keep only the basename if you want shorter ids:
            event_id = Path(event).name

            rows.append({
                "line_no": line_no,
                "epoch": epoch,
                "file_idx": int(m.group("file_idx")),
                "n_files": int(m.group("n_files")),
                "event": event_id,
                "u": int(m.group("u")),
                "u_max": int(m.group("u_max")),
                "step": int(m.group("step")),
                "loss": float(m.group("loss")),
            })

    if not rows:
        raise ValueError("No training lines matched the expected pattern. Check LINE_RE against your log format.")

    df = pd.DataFrame(rows)

    # Sort within each (epoch, event) by u then step (stable ordering)
    df = df.sort_values(["epoch", "event", "u", "step", "line_no"]).reset_index(drop=True)

    # Aggregate per (epoch, event)
    summary_rows = []
    for (epoch, event), g in df.groupby(["epoch", "event"], sort=True):
        x_col = USE_X
        x = g[x_col].to_numpy(dtype=float)
        y = g["loss"].to_numpy(dtype=float)

        a, b, r2 = fit_line(x, y)

        summary_rows.append({
            "epoch": epoch,
            "event": event,
            "n_points": int(len(g)),
            "x_used": x_col,

            "avg_loss": float(np.mean(y)),
            "std_loss": float(np.std(y, ddof=0)),
            "min_loss": float(np.min(y)),
            "max_loss": float(np.max(y)),
            "start_loss": float(y[0]),
            "end_loss": float(y[-1]),
            "delta_loss_end_minus_start": float(y[-1] - y[0]),

            "fit_a": a,
            "fit_b_slope": b,
            "fit_r2": r2,

            # optional helpful context:
            "u_min": int(g["u"].min()),
            "u_max_seen": int(g["u"].max()),
            "step_min": int(g["step"].min()),
            "step_max": int(g["step"].max()),
            "file_idx_min": int(g["file_idx"].min()),
            "file_idx_max": int(g["file_idx"].max()),
        })

    event_epoch = pd.DataFrame(summary_rows).sort_values(["epoch", "event"]).reset_index(drop=True)

    # Epoch-level rollup (simple weighted averages by n_points)
    def wavg(series, weights):
        return float(np.average(series.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))

    epoch_rows = []
    for epoch, g in event_epoch.groupby("epoch", sort=True):
        weights = g["n_points"]
        epoch_rows.append({
            "epoch": int(epoch),
            "n_events": int(len(g)),
            "total_points": int(weights.sum()),

            "avg_loss_weighted": wavg(g["avg_loss"], weights),
            "median_avg_loss": float(np.median(g["avg_loss"].to_numpy(dtype=float))),
            "avg_slope_b_weighted": wavg(g["fit_b_slope"].fillna(0.0), weights),
            "median_slope_b": float(np.median(g["fit_b_slope"].dropna().to_numpy(dtype=float))) if g["fit_b_slope"].notna().any() else math.nan,

            # how many events show within-event improvement (negative slope) vs worsening
            "n_events_slope_neg": int((g["fit_b_slope"] < 0).sum()),
            "n_events_slope_pos": int((g["fit_b_slope"] > 0).sum()),
            "n_events_slope_nan": int(g["fit_b_slope"].isna().sum()),
        })

    epoch_summary = pd.DataFrame(epoch_rows).sort_values("epoch").reset_index(drop=True)

    # If you have multiple epochs, compute per-event change across epochs (e.g., avg_loss improvement)
    # This is often more meaningful than within-event slopes.
    pivot = event_epoch.pivot_table(index="event", columns="epoch", values="avg_loss", aggfunc="mean")
    if pivot.shape[1] >= 2:
        first_epoch = int(min(pivot.columns))
        last_epoch = int(max(pivot.columns))
        delta = (pivot[last_epoch] - pivot[first_epoch]).rename("avg_loss_last_minus_first")
        event_delta = delta.reset_index().sort_values("avg_loss_last_minus_first")
    else:
        event_delta = pd.DataFrame(columns=["event", "avg_loss_last_minus_first"])

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    event_epoch_path = OUTPUT_DIR / "event_epoch_summary.csv"
    epoch_summary_path = OUTPUT_DIR / "epoch_summary.csv"
    event_delta_path = OUTPUT_DIR / "event_delta_across_epochs.csv"

    event_epoch.to_csv(event_epoch_path, index=False)
    epoch_summary.to_csv(epoch_summary_path, index=False)
    event_delta.to_csv(event_delta_path, index=False)

    # Print a compact human-readable snapshot you can paste into your “main chat”
    # Top 10 best/worst avg_loss per epoch, and steepest negative/positive slopes.
    for epoch in sorted(event_epoch["epoch"].unique()):
        g = event_epoch[event_epoch["epoch"] == epoch].copy()
        print("\n" + "=" * 80)
        print(f"EPOCH {epoch} — events: {len(g)}")
        print("Epoch avg_loss (weighted):", epoch_summary.loc[epoch_summary["epoch"] == epoch, "avg_loss_weighted"].iloc[0])
        print("Epoch median avg_loss:", epoch_summary.loc[epoch_summary["epoch"] == epoch, "median_avg_loss"].iloc[0])

        print("\nBest (lowest) avg_loss:")
        print(g.nsmallest(10, "avg_loss")[["event", "avg_loss", "start_loss", "end_loss", "fit_b_slope", "fit_r2"]].to_string(index=False))

        print("\nWorst (highest) avg_loss:")
        print(g.nlargest(10, "avg_loss")[["event", "avg_loss", "start_loss", "end_loss", "fit_b_slope", "fit_r2"]].to_string(index=False))

        # slope extremes (ignore nan)
        g2 = g[g["fit_b_slope"].notna()].copy()
        if len(g2):
            print("\nSteepest improving within-event (most negative slope b):")
            print(g2.nsmallest(10, "fit_b_slope")[["event", "fit_b_slope", "avg_loss", "fit_r2"]].to_string(index=False))

            print("\nSteepest worsening within-event (most positive slope b):")
            print(g2.nlargest(10, "fit_b_slope")[["event", "fit_b_slope", "avg_loss", "fit_r2"]].to_string(index=False))

    print("\nWrote:")
    print(" -", event_epoch_path.resolve())
    print(" -", epoch_summary_path.resolve())
    print(" -", event_delta_path.resolve())


if __name__ == "__main__":
    main()