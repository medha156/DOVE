"""
Plot per-species image counts before and after WeightedRandomSampler rebalancing.
"""
import sys
sys.path.insert(0, "/data/DOVE/src")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO   = Path("/data/DOVE")
SPLITS = REPO / "data" / "splits"
OUT    = REPO / "results" / "figures" / "data_distribution.png"

SPECIES = [
    "Acorn Woodpecker", "American Crow", "American Robin", "Anna's Hummingbird",
    "Black Phoebe", "Brewer's Blackbird", "Bushtit", "California Scrub-Jay",
    "California Towhee", "Chestnut-backed Chickadee", "Cooper's Hawk",
    "Dark-eyed Junco", "House Finch", "Lesser Goldfinch", "Mourning Dove",
    "Northern Mockingbird", "Oak Titmouse", "Red-tailed Hawk",
    "White-crowned Sparrow", "Yellow-rumped Warbler",
]

# ── Load all splits ────────────────────────────────────────────────────────────
df = pd.concat([
    pd.read_csv(SPLITS / f"{s}.csv").assign(split=s)
    for s in ("train", "val", "test")
], ignore_index=True)

# Raw counts per species (all splits combined)
raw = df.groupby("species_id").size().reindex(range(20), fill_value=0)

# Effective training counts: WeightedRandomSampler draws len(train) samples per epoch
# with probability proportional to class_weights[species_id]
train_df  = df[df["split"] == "train"]
n_train   = len(train_df)
cw        = np.load(SPLITS / "class_weights.npy")
sample_w  = np.array([cw[int(r["species_id"])] for _, r in train_df.iterrows()], dtype=np.float32)
# normalise to get expected counts per epoch
if sample_w.sum() > 0:
    prob      = sample_w / sample_w.sum()
    # expected count per species per epoch
    eff_counts = np.zeros(20)
    for i, (_, row) in enumerate(train_df.iterrows()):
        eff_counts[int(row["species_id"])] += prob[i] * n_train
else:
    eff_counts = raw.values.copy().astype(float)

# ── Plot ──────────────────────────────────────────────────────────────────────
short = [s.replace("'", "").replace("-", "-\n") for s in SPECIES]
x     = np.arange(20)
w     = 0.38

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# Top: raw counts
colors_raw = ["#c0392b" if raw[i] == 0 else "#2980b9" for i in range(20)]
bars0 = axes[0].bar(x, raw.values, width=w*2, color=colors_raw, edgecolor="white", linewidth=0.5)
axes[0].set_ylabel("Image count", fontsize=11)
axes[0].set_title("Before augmentation — raw sample counts per species\n(red = missing from dataset)", fontsize=12)
axes[0].axhline(raw[raw > 0].mean(), color="orange", linestyle="--", linewidth=1.2, label=f"Mean = {raw[raw>0].mean():.0f}")
axes[0].legend(fontsize=9)
for bar, val in zip(bars0, raw.values):
    if val > 0:
        axes[0].text(bar.get_x() + bar.get_width()/2, val + 5, str(val),
                     ha="center", va="bottom", fontsize=7)

# Bottom: effective counts after WeightedRandomSampler
colors_eff = ["#c0392b" if raw[i] == 0 else "#27ae60" for i in range(20)]
bars1 = axes[1].bar(x, eff_counts, width=w*2, color=colors_eff, edgecolor="white", linewidth=0.5)
axes[1].set_ylabel("Expected samples / epoch", fontsize=11)
axes[1].set_title("After WeightedRandomSampler rebalancing — effective training distribution per epoch", fontsize=12)
present = eff_counts[eff_counts > 0]
axes[1].axhline(present.mean(), color="orange", linestyle="--", linewidth=1.2, label=f"Mean = {present.mean():.0f}")
axes[1].legend(fontsize=9)
for bar, val in zip(bars1, eff_counts):
    if val > 0:
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.0f}",
                     ha="center", va="bottom", fontsize=7)

axes[1].set_xticks(x)
axes[1].set_xticklabels(SPECIES, rotation=45, ha="right", fontsize=8)

fig.suptitle("DOVE Dataset — Species Distribution (iNaturalist images only, 6,085 total)\n4 species missing: Acorn Woodpecker, Black Phoebe, California Towhee, White-crowned Sparrow",
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved to {OUT}")

# Also print a summary table
print("\nSpecies | Raw count | Eff/epoch")
print("-" * 45)
for i, name in enumerate(SPECIES):
    flag = " ← MISSING" if raw[i] == 0 else ""
    print(f"{name:<32} {raw[i]:>5}   {eff_counts[i]:>7.1f}{flag}")
