"""
Rainfall Pattern Visualization
================================
Reads all train events for Model_1 and Model_2, extracts rainfall and
water level signals, and produces diagnostic plots.

Adjust DATA_ROOT to point to your local dataset root.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench\full_dataset")   # <-- adjust this
OUT_DIR   = Path(r"C:\Users\filax\OneDrive\Desktop\Code-repository\Kaggle\Competitions\05_UrbanFloodBench - flood modelling\urbanfloodbench\rainfall_plots")              # output folder for PNGs
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS    = [1, 2]
SPLIT     = "train"
CHUNKSIZE = 2_000_000
# ──────────────────────────────────────────────────────────────────────────────


def index_events(model_split_root: Path) -> dict:
    mapping = {}
    for p in sorted(model_split_root.glob("event_*")):
        suffix = p.name.split("_", 1)[1]
        try:
            mapping[int(suffix)] = p
        except ValueError:
            continue
    return mapping


def load_rainfall_summary(event_dir: Path) -> pd.DataFrame:
    """Per-timestep: max rainfall across nodes, mean 2D water level, mean 1D water level."""
    p2d = event_dir / "2d_nodes_dynamic_all.csv"
    p1d = event_dir / "1d_nodes_dynamic_all.csv"

    # 2D: rain + wl
    rain_max, rain_sum, wl2d_mean = {}, {}, {}
    for chunk in pd.read_csv(p2d, usecols=["timestep", "rainfall", "water_level"],
                             chunksize=CHUNKSIZE):
        for t, grp in chunk.groupby("timestep"):
            rain_max[t]  = max(grp["rainfall"].max(),  rain_max.get(t, 0.0))
            rain_sum[t]  = grp["rainfall"].sum()       + rain_sum.get(t, 0.0)
            wl2d_mean[t] = grp["water_level"].sum()    + wl2d_mean.get(t, 0.0)

    n2d_counts = {}
    for chunk in pd.read_csv(p2d, usecols=["timestep", "node_idx"], chunksize=CHUNKSIZE):
        for t, grp in chunk.groupby("timestep"):
            n2d_counts[t] = n2d_counts.get(t, 0) + len(grp)

    # 1D: wl only
    wl1d_mean = {}
    n1d_counts = {}
    for chunk in pd.read_csv(p1d, usecols=["timestep", "water_level"], chunksize=CHUNKSIZE):
        for t, grp in chunk.groupby("timestep"):
            wl1d_mean[t]  = grp["water_level"].sum() + wl1d_mean.get(t, 0.0)
            n1d_counts[t] = n1d_counts.get(t, 0)    + len(grp)

    timesteps = sorted(rain_max.keys())
    df = pd.DataFrame({
        "timestep":    timesteps,
        "rain_max":    [rain_max[t]  for t in timesteps],
        "rain_sum":    [rain_sum[t]  for t in timesteps],
        "rain_cum":    np.cumsum([rain_max[t] for t in timesteps]),
        "wl2d_mean":   [wl2d_mean[t] / n2d_counts[t] for t in timesteps],
        "wl1d_mean":   [wl1d_mean.get(t, 0.0) / max(n1d_counts.get(t, 1), 1)
                        for t in timesteps],
    })
    return df


def collect_model_events(model_id: int) -> dict[int, pd.DataFrame]:
    root = DATA_ROOT / "Models" / f"Model_{model_id}" / SPLIT
    events = index_events(root)
    print(f"  Model_{model_id}: {len(events)} events found")
    summaries = {}
    for eid, edir in sorted(events.items()):
        try:
            summaries[eid] = load_rainfall_summary(edir)
        except Exception as e:
            print(f"    [WARN] event {eid}: {e}")
    return summaries


# ── PLOT FUNCTIONS ────────────────────────────────────────────────────────────

def plot_rainfall_timeseries(model_id: int, summaries: dict):
    """All events overlaid: rain_max vs timestep."""
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = cm.tab20(np.linspace(0, 1, len(summaries)))
    for (eid, df), col in zip(sorted(summaries.items()), colors):
        ax.plot(df["timestep"], df["rain_max"], lw=0.8, alpha=0.7,
                color=col, label=f"E{eid}")
    ax.set_title(f"Model_{model_id} — Rainfall intensity over time (all events)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Max rainfall across nodes (mm/h or native unit)")
    ax.legend(fontsize=6, ncol=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"model{model_id}_rainfall_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: model{model_id}_rainfall_timeseries.png")


def plot_cumulative_rainfall(model_id: int, summaries: dict):
    """Cumulative rainfall curves — distinguishes slow/steady vs sudden storms."""
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = cm.tab20(np.linspace(0, 1, len(summaries)))
    for (eid, df), col in zip(sorted(summaries.items()), colors):
        # normalise x to [0,1] so events of different lengths are comparable
        x = df["timestep"] / df["timestep"].max()
        y = df["rain_cum"] / df["rain_cum"].iloc[-1] if df["rain_cum"].iloc[-1] > 0 else df["rain_cum"]
        ax.plot(x, y, lw=0.9, alpha=0.7, color=col, label=f"E{eid}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="uniform reference")
    ax.set_title(f"Model_{model_id} — Normalised cumulative rainfall (event fraction of time)")
    ax.set_xlabel("Fraction of event duration")
    ax.set_ylabel("Fraction of total rainfall delivered")
    ax.legend(fontsize=6, ncol=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"model{model_id}_cumulative_rainfall.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: model{model_id}_cumulative_rainfall.png")


def plot_event_scatter(model_id: int, summaries: dict):
    """Scatter: total rainfall vs event duration, sized by peak intensity."""
    durations, totals, peaks, eids = [], [], [], []
    for eid, df in summaries.items():
        durations.append(len(df))
        totals.append(df["rain_cum"].iloc[-1])
        peaks.append(df["rain_max"].max())
        eids.append(eid)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(durations, totals, c=peaks, s=60, cmap="YlOrRd",
                    edgecolors="k", linewidths=0.4)
    for i, eid in enumerate(eids):
        ax.annotate(str(eid), (durations[i], totals[i]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="Peak rainfall intensity")
    ax.set_title(f"Model_{model_id} — Event duration vs total rainfall")
    ax.set_xlabel("Event duration (timesteps)")
    ax.set_ylabel("Cumulative rainfall (sum of rain_max)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"model{model_id}_event_scatter.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: model{model_id}_event_scatter.png")


def plot_wl_response(model_id: int, summaries: dict):
    """Dual-axis: rainfall (bar) and mean 2D+1D water levels (lines) for each event."""
    n = len(summaries)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3))
    axes = np.array(axes).flatten()

    for ax_i, (eid, df) in enumerate(sorted(summaries.items())):
        ax = axes[ax_i]
        ax2 = ax.twinx()

        ax.bar(df["timestep"], df["rain_max"], color="steelblue",
               alpha=0.4, width=1.0, label="Rain max")
        ax2.plot(df["timestep"], df["wl2d_mean"], color="darkorange",
                 lw=1.0, label="2D WL mean")
        ax2.plot(df["timestep"], df["wl1d_mean"], color="green",
                 lw=1.0, linestyle="--", label="1D WL mean")

        ax.set_title(f"Event {eid}", fontsize=8)
        ax.set_xlabel("Timestep", fontsize=6)
        ax.set_ylabel("Rainfall", fontsize=6, color="steelblue")
        ax2.set_ylabel("WL", fontsize=6)
        ax.tick_params(labelsize=6)
        ax2.tick_params(labelsize=6)

        if ax_i == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=5, loc="upper right")

    # hide unused axes
    for ax in axes[len(summaries):]:
        ax.set_visible(False)

    fig.suptitle(f"Model_{model_id} — Rainfall vs Water Level response per event",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"model{model_id}_wl_response.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: model{model_id}_wl_response.png")


def plot_peak_timing(model_id: int, summaries: dict):
    """For each event: when does rainfall peak vs when does 2D WL peak?"""
    rain_peak_frac, wl_peak_frac, eids = [], [], []
    for eid, df in summaries.items():
        if df["rain_max"].max() == 0:
            continue
        rain_peak_frac.append(df.loc[df["rain_max"].idxmax(), "timestep"] / df["timestep"].max())
        wl_peak_frac.append(df.loc[df["wl2d_mean"].idxmax(), "timestep"] / df["timestep"].max())
        eids.append(eid)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rain_peak_frac, wl_peak_frac, s=60, edgecolors="k", linewidths=0.5)
    for i, eid in enumerate(eids):
        ax.annotate(str(eid), (rain_peak_frac[i], wl_peak_frac[i]),
                    fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="no lag")
    ax.set_title(f"Model_{model_id} — Rainfall peak vs WL peak (fraction of event)")
    ax.set_xlabel("Rainfall peak (fraction of event duration)")
    ax.set_ylabel("2D WL peak (fraction of event duration)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"model{model_id}_peak_timing.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: model{model_id}_peak_timing.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for model_id in MODELS:
        print(f"\n=== Model_{model_id} ===")
        summaries = collect_model_events(model_id)
        if not summaries:
            print("  No events loaded, skipping.")
            continue

        plot_rainfall_timeseries(model_id, summaries)
        plot_cumulative_rainfall(model_id, summaries)
        plot_event_scatter(model_id, summaries)
        plot_wl_response(model_id, summaries)
        plot_peak_timing(model_id, summaries)

    print(f"\nAll plots saved to: {OUT_DIR.resolve()}")
