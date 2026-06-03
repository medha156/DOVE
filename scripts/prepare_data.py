"""
DOVE — Data preparation script.

Scans data/inaturalist/ and data/vb100/, verifies structure,
generates train/val/test CSVs, computes class weights.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.splits import (
    SPECIES_NAMES,
    SPECIES_TO_ID,
    build_manifest,
    compute_class_weights,
    generate_splits,
    save_splits,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("prepare_data")

REPO_ROOT = Path(__file__).parent.parent
INAT_ROOT = REPO_ROOT / "data" / "inaturalist"
VB100_ROOT = REPO_ROOT / "data" / "vb100"
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def verify_structure(root: Path, modality: str) -> bool:
    """Check that the data directory has species subdirectories."""
    if not root.exists():
        logger.warning("Directory %s does not exist", root)
        return False
    subdirs = [d for d in root.iterdir() if d.is_dir()]
    if not subdirs:
        logger.warning("No subdirectories in %s", root)
        return False
    unknown = [d.name for d in subdirs if d.name not in SPECIES_TO_ID]
    if unknown:
        logger.warning("Unknown species directories in %s: %s", root, unknown)
    logger.info("Verified %s: %d species subdirectories", root, len(subdirs))
    return True


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    logger.info("=" * 60)
    logger.info("DOVE — Data Preparation")
    logger.info("=" * 60)

    inat_ok = verify_structure(INAT_ROOT, "image")
    vb100_ok = verify_structure(VB100_ROOT, "video")

    if not inat_ok and not vb100_ok:
        logger.error("No data found in either directory. Please populate data/ first.")
        sys.exit(1)

    logger.info("Building manifest from %s and %s", INAT_ROOT, VB100_ROOT)
    df = build_manifest(INAT_ROOT, VB100_ROOT)

    if len(df) == 0:
        logger.error("Empty manifest — aborting")
        sys.exit(1)

    logger.info("Total samples: %d", len(df))
    logger.info("Modality breakdown: %s", df["modality"].value_counts().to_dict())
    logger.info("Species breakdown:")
    for sid, sname in enumerate(SPECIES_NAMES):
        n = (df["species_id"] == sid).sum()
        logger.info("  [%2d] %-30s %d", sid, sname, n)

    # Generate splits
    logger.info("Generating stratified splits (70/15/15)…")
    df_split = generate_splits(df, val_frac=0.15, test_frac=0.15, seed=42)
    split_counts = df_split["split"].value_counts().to_dict()
    logger.info("Split sizes: %s", split_counts)

    # Class weights
    weights = compute_class_weights(df_split)
    logger.info("Class weights (min=%.4f, max=%.4f)", weights.min(), weights.max())

    # Save
    save_splits(df_split, SPLITS_DIR, weights)
    logger.info("Splits saved to %s", SPLITS_DIR)
    logger.info("Done.")


if __name__ == "__main__":
    main()
