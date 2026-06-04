"""
Plot training/val loss + val accuracy curves for the top-5 best experiments.
Reads from results/tables/{name}_log.csv.
Saves to results/figures/top5_learning_curves.png.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = Path(__file__).parent.parent
TABLES    = REPO_ROOT / "results" / "tables"
FIGS      = REPO_ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# Top-5 by test accuracy from experiment results
TOP5 = [
    ("efficientnet_b3_cross_attention_linear", 0.9409, "EfficientNet-B3 + CrossAttn + Linear"),
    ("mobilenet_cross_attention_linear",       0.9409, "MobileNet + CrossAttn + Linear"),
    ("efficientnet_b3_concat_linear",          0.9376, "EfficientNet-B3 + Concat + Linear"),
    ("mobilenet_cross_attention_mlp",          0.9376, "MobileNet + CrossAttn + MLP"),
    ("mobilenet_concat_mlp",                   0.9376, "MobileNet + Concat + MLP"),
]

COLORS = ["#E63946", "#2A9D8F", "#E9C46A", "#457B9D", "#F4A261"]
STYLES = ["-", "--", "-.", ":", (0,(3,1,1,1))]

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 1, hspace=0.35)

ax_loss = fig.add_subplot(gs[0])
ax_acc  = fig.add_subplot(gs[1])

for (name, test_acc, label), color, ls in zip(TOP5, COLORS, STYLES):
    csv = TABLES / f"{name}_log.csv"
    if not csv.exists():
        print(f"  WARNING: {csv} not found, skipping")
        continue
    df = pd.read_csv(csv)

    ax_loss.plot(df["epoch"], df["train_loss"],
                 color=color, linestyle=ls, linewidth=2,
                 label=f"{label} — train")
    # No val_loss column — plot val_acc on separate axis
    ax_acc.plot(df["epoch"], df["val_acc"] * 100,
                color=color, linestyle=ls, linewidth=2,
                label=f"{label} (test={test_acc*100:.2f}%)")
    ax_acc.plot(df["epoch"], df["train_acc"] * 100,
                color=color, linestyle=ls, linewidth=1.2, alpha=0.4)

ax_loss.set_xlabel("Epoch", fontsize=12)
ax_loss.set_ylabel("Cross-Entropy Loss (train)", fontsize=12)
ax_loss.set_title("Training Loss — Top-5 Architectures", fontsize=14, fontweight="bold")
ax_loss.legend(fontsize=8, ncol=2, loc="upper right")
ax_loss.grid(True, alpha=0.3)
ax_loss.set_xlim(1, 20)

ax_acc.set_xlabel("Epoch", fontsize=12)
ax_acc.set_ylabel("Accuracy (%)", fontsize=12)
ax_acc.set_title("Val Accuracy (solid) vs Train Accuracy (faded) — Top-5 Architectures",
                  fontsize=14, fontweight="bold")
ax_acc.legend(fontsize=8, ncol=2, loc="lower right")
ax_acc.grid(True, alpha=0.3)
ax_acc.set_xlim(1, 20)
ax_acc.set_ylim(60, 101)

# Annotate final val acc for each model
for (name, test_acc, label), color in zip(TOP5, COLORS):
    csv = TABLES / f"{name}_log.csv"
    if not csv.exists():
        continue
    df = pd.read_csv(csv)
    final_val = df["val_acc"].iloc[-1] * 100
    ax_acc.annotate(f"{final_val:.1f}%",
                    xy=(df["epoch"].iloc[-1], final_val),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=7, color=color, ha="center", fontweight="bold")

out = FIGS / "top5_learning_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")
