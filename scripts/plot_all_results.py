"""
DOVE — Plot training and validation accuracy curves for all experiment sets.

Reads *_log.csv files from:
  results_vb100/tables/   — backbone + invariant (bb_inv)
  results_motion/tables/  — motion + invariant (mot_inv)
  results_triple/tables/  — backbone + invariant + motion (triple)

Produces:
  results_comparison/figures/
    curves_bb_inv.png        — one subplot per config, bb+inv sweep
    curves_mot_inv.png       — one subplot per config, mot+inv sweep
    curves_triple.png        — one subplot per config, triple sweep
    best_per_set.png         — best val curve from each set overlaid
    final_test_accuracy.png  — bar chart of test accuracy across all configs
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

REPO_ROOT = Path("/data/DOVE")
OUT_DIR   = REPO_ROOT / "results_comparison" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SETS = {
    "bb_inv":  ("results_vb100",  "Backbone + Invariant"),
    "mot_inv": ("results_motion", "Motion + Invariant"),
    "triple":  ("results_triple", "Backbone + Invariant + Motion"),
}

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_logs(tables_dir: Path) -> dict[str, pd.DataFrame]:
    """Return {exp_name: DataFrame(epoch, train_acc, val_acc)} for all log CSVs."""
    logs = {}
    for f in sorted(tables_dir.glob("*_log.csv")):
        name = f.stem.replace("_log", "")
        df = pd.read_csv(f)
        if {"epoch", "train_acc", "val_acc"}.issubset(df.columns):
            logs[name] = df
    return logs


def load_results(tables_dir: Path) -> pd.DataFrame | None:
    """Load the experiment_results*.csv if it exists."""
    candidates = list(tables_dir.glob("experiment_results*.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def short_name(name: str) -> str:
    """Strip leading 'motion_inv_' or 'triple_' prefixes for cleaner labels."""
    for prefix in ("triple_", "motion_inv_", "mot_inv_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# ── Per-set curve plots ───────────────────────────────────────────────────────

def plot_curves_grid(logs: dict[str, pd.DataFrame], title: str, out_path: Path):
    """One subplot per experiment showing train + val accuracy over epochs."""
    n     = len(logs)
    if n == 0:
        print(f"  No logs found for {title} — skipping")
        return
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                              squeeze=False)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    for ax_idx, (name, df) in enumerate(sorted(logs.items())):
        r, c = divmod(ax_idx, ncols)
        ax   = axes[r][c]
        ax.plot(df["epoch"], df["train_acc"], label="Train", linewidth=1.8)
        ax.plot(df["epoch"], df["val_acc"],   label="Val",   linewidth=1.8,
                linestyle="--")
        best_val = df["val_acc"].max()
        ax.axhline(best_val, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(short_name(name), fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Accuracy", fontsize=8)
        ax.set_ylim(0.5, 1.02)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.text(0.97, 0.05, f"best val={best_val:.3f}",
                transform=ax.transAxes, fontsize=7, ha="right",
                color="gray")

    # Hide unused subplots
    for ax_idx in range(n, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Best-per-set overlay ──────────────────────────────────────────────────────

def plot_best_overlay(all_logs: dict[str, dict], out_path: Path):
    """
    For each experiment set, find the config with highest best-val and
    overlay its train+val curves on a single plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Best Config per Experiment Set — Training & Validation Accuracy",
                 fontsize=13, fontweight="bold")

    for ax, metric, ylabel in zip(axes, ["train_acc", "val_acc"],
                                  ["Train Accuracy", "Validation Accuracy"]):
        for ci, (set_key, (_, set_label)) in enumerate(SETS.items()):
            logs = all_logs.get(set_key, {})
            if not logs:
                continue
            # Pick config with highest final val_acc
            best_name = max(logs, key=lambda n: logs[n]["val_acc"].max())
            df = logs[best_name]
            ax.plot(df["epoch"], df[metric],
                    label=f"{set_label}\n({short_name(best_name)})",
                    color=COLORS[ci], linewidth=2.0)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0.5, 1.02)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Final test accuracy bar chart ────────────────────────────────────────────

def plot_test_bar(all_results: dict[str, pd.DataFrame], out_path: Path):
    """Grouped bar chart of test accuracy for all configs across all sets."""
    frames = []
    for set_key, (_, set_label) in SETS.items():
        df = all_results.get(set_key)
        if df is None or "test_accuracy" not in df.columns:
            continue
        df = df.copy()
        df["set_label"] = set_label
        df["short_name"] = df["name"].apply(short_name)
        frames.append(df[["short_name", "set_label", "test_accuracy"]].dropna())

    if not frames:
        print("  No test results found — skipping bar chart")
        return

    combined = pd.concat(frames, ignore_index=True)
    sets     = combined["set_label"].unique()
    names    = sorted(combined["short_name"].unique())
    x        = np.arange(len(names))
    width    = 0.8 / max(len(sets), 1)

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.9), 5))
    for i, (set_label, color) in enumerate(zip(sets, COLORS)):
        subset = combined[combined["set_label"] == set_label]
        vals   = [subset[subset["short_name"] == n]["test_accuracy"].values
                  for n in names]
        heights = [v[0] if len(v) else np.nan for v in vals]
        bars = ax.bar(x + i * width, heights, width,
                      label=set_label, color=color, alpha=0.85)
        for bar, h in zip(bars, heights):
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=6,
                        rotation=90)

    ax.set_xticks(x + width * (len(sets) - 1) / 2)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("Test Accuracy — All Configurations by Experiment Set",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Val accuracy over epochs — all configs all sets, one panel each ───────────

def plot_val_curves_by_set(all_logs: dict, out_path: Path):
    """
    Three-panel figure. Each panel = one experiment set.
    All configs in that set overlaid as thin lines; best config bolded.
    """
    n_sets = sum(1 for k in SETS if all_logs.get(k))
    if n_sets == 0:
        return
    fig, axes = plt.subplots(1, n_sets, figsize=(6 * n_sets, 4.5), squeeze=False)
    fig.suptitle("Validation Accuracy — All Configs per Experiment Set",
                 fontsize=13, fontweight="bold")

    ax_idx = 0
    for set_key, (_, set_label) in SETS.items():
        logs = all_logs.get(set_key, {})
        if not logs:
            continue
        ax = axes[0][ax_idx]
        best_name = max(logs, key=lambda n: logs[n]["val_acc"].max())
        cmap = cm.get_cmap("tab20", len(logs))
        for ci, (name, df) in enumerate(sorted(logs.items())):
            is_best = name == best_name
            ax.plot(df["epoch"], df["val_acc"],
                    label=short_name(name),
                    color=cmap(ci),
                    linewidth=2.5 if is_best else 1.0,
                    alpha=1.0  if is_best else 0.45,
                    zorder=10  if is_best else 1)
        ax.set_title(set_label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Val Accuracy", fontsize=9)
        ax.set_ylim(0.5, 1.02)
        ax.legend(fontsize=6, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading logs...")
    all_logs    = {}
    all_results = {}

    for set_key, (folder, set_label) in SETS.items():
        tables_dir = REPO_ROOT / folder / "tables"
        if not tables_dir.exists():
            print(f"  {set_label}: no tables dir yet, skipping")
            continue
        logs = load_logs(tables_dir)
        res  = load_results(tables_dir)
        all_logs[set_key]    = logs
        all_results[set_key] = res
        print(f"  {set_label}: {len(logs)} log files")

    if not any(all_logs.values()):
        print("No log files found anywhere — nothing to plot")
        sys.exit(0)

    # Per-set curve grids
    for set_key, (_, set_label) in SETS.items():
        logs = all_logs.get(set_key, {})
        if logs:
            plot_curves_grid(logs, set_label,
                             OUT_DIR / f"curves_{set_key}.png")

    # Val curves — all configs per set overlaid
    plot_val_curves_by_set(all_logs, OUT_DIR / "val_curves_all_configs.png")

    # Best per set overlay
    plot_best_overlay(all_logs, OUT_DIR / "best_per_set.png")

    # Test accuracy bar chart
    plot_test_bar(all_results, OUT_DIR / "final_test_accuracy.png")

    print(f"\nAll plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
