"""
DOVE — Data preparation script.

Pulls cleaned datasets from the CHIRP repo (https://github.com/medha156/CHIRP)
and generates train/val/test split CSVs for DOVE training.

VB100 frames used here have already been quality-filtered in CHIRP by:
  - Pixel std deviation >= 10  (removes blank/near-black frames)
  - Laplacian variance   >= 5  (removes motion-blurred frames)

Usage:
    python scripts/prepare_data.py [--chirp-dir /path/to/CHIRP] [--clone]
"""
from __future__ import annotations

import argparse
import logging
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.splits import (
    SPECIES_NAMES,
    build_manifest_from_chirp,
    build_manifest,
    compute_class_weights,
    generate_splits,
    save_splits,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("prepare_data")

REPO_ROOT  = Path(__file__).parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"
CHIRP_REPO = "https://github.com/medha156/CHIRP.git"


def clone_or_pull_chirp(chirp_dir: Path) -> None:
    if chirp_dir.exists():
        logger.info("Pulling latest CHIRP at %s", chirp_dir)
        subprocess.run(["git", "-C", str(chirp_dir), "pull", "--ff-only"], check=True)
    else:
        logger.info("Cloning CHIRP into %s", chirp_dir)
        subprocess.run(["git", "clone", "--depth=1", CHIRP_REPO, str(chirp_dir)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chirp-dir", type=Path,
        default=REPO_ROOT.parent / "CHIRP",
        help="Path to local CHIRP repository",
    )
    parser.add_argument(
        "--clone", action="store_true",
        help="Clone/pull CHIRP repo before preparing data",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    logger.info("=" * 60)
    logger.info("DOVE — Data Preparation (CHIRP cleaned datasets)")
    logger.info("=" * 60)

    chirp_dir: Path = args.chirp_dir

    if args.clone:
        clone_or_pull_chirp(chirp_dir)

    if not chirp_dir.exists():
        logger.error(
            "CHIRP directory not found at %s.\n"
            "Run with --clone to clone automatically, or point --chirp-dir at "
            "your local copy of https://github.com/medha156/CHIRP",
            chirp_dir,
        )
        sys.exit(1)

    # ── Locate CHIRP index CSVs ───────────────────────────────────────────
    inat_index  = chirp_dir / "data" / "inaturalist" / "index.csv"
    vb100_index = chirp_dir / "data" / "processed" / "vb100_frames" / "index.csv"

    for p in (inat_index, vb100_index):
        if not p.exists():
            logger.error("Index not found: %s", p)
            logger.error(
                "Make sure the CHIRP repo is up to date and the data preparation "
                "pipeline has been run (see CHIRP/pipelines/filter_vb100_frames.py)."
            )
            sys.exit(1)

    logger.info("iNat  index : %s", inat_index)
    logger.info("VB100 index : %s", vb100_index)

    # ── Build unified manifest ────────────────────────────────────────────
    logger.info("Building manifest from CHIRP indices…")
    df = build_manifest_from_chirp(inat_index, vb100_index)

    if len(df) == 0:
        logger.error("Empty manifest — aborting")
        sys.exit(1)

    logger.info("Total samples : %d", len(df))
    logger.info("Modality breakdown: %s", df["modality"].value_counts().to_dict())
    logger.info("Species breakdown:")
    for sid, sname in enumerate(SPECIES_NAMES):
        n = (df["species_id"] == sid).sum()
        logger.info("  [%2d] %-32s %d", sid, sname, n)

    # ── Stratified splits (video-level for VB100) ─────────────────────────
    logger.info("Generating stratified splits (70 / 15 / 15)…")
    df_split = generate_splits(df, val_frac=0.15, test_frac=0.15, seed=42)
    logger.info("Split sizes: %s", df_split["split"].value_counts().to_dict())

    # ── Class weights ─────────────────────────────────────────────────────
    weights = compute_class_weights(df_split)
    logger.info("Class weights — min=%.4f  max=%.4f", weights.min(), weights.max())

    # ── Save ──────────────────────────────────────────────────────────────
    save_splits(df_split, SPLITS_DIR, weights)
    logger.info("Splits saved to %s", SPLITS_DIR)
    logger.info("Done.")


if __name__ == "__main__":
    main()
