"""
DOVE — Stratified train/val/test split generation.

Splits are performed at the video/image level (not frame level).
Stratification is by a combined (species_id, modality) key.
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

SPECIES_NAMES = [
    "Acorn Woodpecker", "American Crow", "American Robin", "Anna's Hummingbird",
    "Black Phoebe", "Brewer's Blackbird", "Bushtit", "California Scrub-Jay",
    "California Towhee", "Chestnut-backed Chickadee", "Cooper's Hawk",
    "Dark-eyed Junco", "House Finch", "Lesser Goldfinch", "Mourning Dove",
    "Northern Mockingbird", "Oak Titmouse", "Red-tailed Hawk",
    "White-crowned Sparrow", "Yellow-rumped Warbler",
]
SPECIES_TO_ID = {name: i for i, name in enumerate(SPECIES_NAMES)}


def _scan_directory(root: Path, modality: str) -> list[dict]:
    """Scan a root directory for images or videos organised by species subfolder."""
    records = []
    if not root.exists():
        logger.warning("Directory %s does not exist — skipping", root)
        return records

    extensions = {".jpg", ".jpeg", ".png"} if modality == "image" else {".mp4", ".avi", ".mov"}

    for species_dir in sorted(root.iterdir()):
        if not species_dir.is_dir():
            continue
        species_name = species_dir.name
        species_id = SPECIES_TO_ID.get(species_name, -1)
        if species_id == -1:
            logger.warning("Unknown species directory: %s", species_name)
            continue
        for fpath in sorted(species_dir.iterdir()):
            if fpath.suffix.lower() in extensions:
                records.append(
                    {
                        "filepath": str(fpath),
                        "species_id": species_id,
                        "species_name": species_name,
                        "modality": modality,
                    }
                )
    return records


def build_manifest(
    inat_root: str | Path,
    vb100_root: str | Path,
) -> pd.DataFrame:
    """Scan both data roots and return a combined manifest DataFrame."""
    records = _scan_directory(Path(inat_root), "image") + _scan_directory(
        Path(vb100_root), "video"
    )
    if not records:
        logger.warning("No data found in %s or %s", inat_root, vb100_root)
        return pd.DataFrame(
            columns=["filepath", "species_id", "species_name", "modality"]
        )
    return pd.DataFrame(records)


def generate_splits(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add a 'split' column with values 'train'/'val'/'test'.
    Stratified by (species_id, modality).
    """
    df = df.copy()
    df["strat_key"] = df["species_id"].astype(str) + "_" + df["modality"]

    n = len(df)
    indices = np.arange(n)
    strat = df["strat_key"].values

    # First peel off test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(indices, strat))

    # Then peel off val from trainval
    val_frac_adjusted = val_frac / (1.0 - test_frac)
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_frac_adjusted, random_state=seed
    )
    sub_strat = strat[trainval_idx]
    train_sub_idx, val_sub_idx = next(
        sss_val.split(trainval_idx, sub_strat)
    )
    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    df["split"] = "train"
    df.iloc[val_idx, df.columns.get_loc("split")] = "val"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"
    df = df.drop(columns=["strat_key"])
    return df


def compute_class_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Inverse-frequency weights: w_c = N_total / (N_classes * N_c).
    Returns array of length num_classes.
    """
    train_df = df[df["split"] == "train"]
    n_classes = df["species_id"].nunique()
    n_total = len(train_df)
    weights = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        n_c = (train_df["species_id"] == c).sum()
        weights[c] = n_total / (n_classes * n_c) if n_c > 0 else 0.0
    return weights


def save_splits(
    df: pd.DataFrame,
    splits_dir: str | Path,
    class_weights: Optional[np.ndarray] = None,
) -> None:
    """Write train/val/test CSVs and optionally class_weights.npy."""
    splits_dir = Path(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        subset = df[df["split"] == split].drop(columns=["split"])
        subset.to_csv(splits_dir / f"{split}.csv", index=False)
        logger.info("Saved %d rows to %s/%s.csv", len(subset), splits_dir, split)
    if class_weights is not None:
        np.save(splits_dir / "class_weights.npy", class_weights)
        logger.info("Saved class weights to %s/class_weights.npy", splits_dir)


if __name__ == "__main__":
    import random
    import tempfile

    logging.basicConfig(level=logging.INFO)
    random.seed(42)
    np.random.seed(42)

    # Build a synthetic manifest
    rng = np.random.default_rng(42)
    rows = []
    for sid, sname in enumerate(SPECIES_NAMES):
        for mod, n in [("image", 30), ("video", 20)]:
            for i in range(n):
                rows.append(
                    {
                        "filepath": f"fake/{sname}/{mod}_{i}.{'jpg' if mod=='image' else 'mp4'}",
                        "species_id": sid,
                        "species_name": sname,
                        "modality": mod,
                    }
                )
    df = pd.DataFrame(rows)
    df = generate_splits(df)
    weights = compute_class_weights(df)

    print("Split counts:", df["split"].value_counts().to_dict())
    print("Class weights shape:", weights.shape)
    print("Class weights (first 5):", weights[:5])

    with tempfile.TemporaryDirectory() as tmp:
        save_splits(df, tmp, weights)
        print("CSVs written to", tmp)
        print("train.csv head:")
        print(pd.read_csv(f"{tmp}/train.csv").head(3))
