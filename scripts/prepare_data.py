"""
DOVE — Data preparation script.

Accepts either:
  A) A pre-built CHIRP merged index  (--merged-index path/to/merged/index.csv)
  B) Individual CHIRP source indices (--chirp-dir, reads inat + vb100_frames)

The merged index (data/merged/index.csv) is the preferred source — it already
combines VB100 videos, Birds-525 photos, and iNaturalist photos into one file.

VB100 frames used here have already been quality-filtered in CHIRP by:
  - Pixel std deviation >= 10  (removes blank/near-black frames)
  - Laplacian variance   >= 5  (removes motion-blurred frames)
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.splits import (
    SPECIES_NAMES,
    SPECIES_TO_ID,
    SLUG_TO_ID,
    generate_splits,
    compute_class_weights,
    save_splits,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("prepare_data")

REPO_ROOT  = Path(__file__).parent.parent
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def load_merged_index(merged_csv: Path) -> pd.DataFrame:
    """
    Load a CHIRP merged index.csv and normalise to DOVE schema.

    CHIRP columns: path, label, species, source, modality, license, [video_src, frame_idx]
    DOVE columns:  filepath, species_id, species_name, modality, video_src
    """
    df = pd.read_csv(merged_csv)
    logger.info("Loaded %d rows from %s", len(df), merged_csv)

    # Normalise column names
    if "path" in df.columns and "filepath" not in df.columns:
        df = df.rename(columns={"path": "filepath"})
    if "label" in df.columns and "species_id" not in df.columns:
        df = df.rename(columns={"label": "species_id"})

    # Normalise modality: CHIRP uses 'photo'/'video'; DOVE uses 'image'/'video_frame'
    modality_map = {"photo": "image", "video": "video_frame", "image": "image",
                    "video_frame": "video_frame"}
    df["modality"] = df["modality"].map(modality_map).fillna("image")

    # Ensure video_src column exists
    if "video_src" not in df.columns:
        df["video_src"] = ""

    # Add species_name if missing
    if "species_name" not in df.columns:
        if "species" in df.columns:
            df = df.rename(columns={"species": "species_name"})
        else:
            df["species_name"] = df["species_id"].map(
                {i: n for i, n in enumerate(SPECIES_NAMES)})

    # Ensure species_id is int
    df["species_id"] = df["species_id"].astype(int)

    # Drop rows with unknown species
    valid = df["species_id"].between(0, len(SPECIES_NAMES) - 1)
    if (~valid).sum() > 0:
        logger.warning("Dropping %d rows with out-of-range species_id", (~valid).sum())
        df = df[valid].reset_index(drop=True)

    return df[["filepath", "species_id", "species_name", "modality", "video_src"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chirp-dir", type=Path,
        default=None,
        help="Path to CHIRP repo — uses data/merged/index.csv inside it",
    )
    parser.add_argument(
        "--merged-index", type=Path,
        default=None,
        help="Direct path to a CHIRP merged index.csv",
    )
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    logger.info("=" * 60)
    logger.info("DOVE — Data Preparation")
    logger.info("=" * 60)

    # Resolve merged index path
    if args.merged_index:
        merged_csv = args.merged_index
    elif args.chirp_dir:
        merged_csv = args.chirp_dir / "data" / "merged" / "index.csv"
    else:
        logger.error("Provide --chirp-dir or --merged-index")
        sys.exit(1)

    if not merged_csv.exists():
        logger.error("Merged index not found: %s", merged_csv)
        logger.error(
            "Run CHIRP's build_index.py first:\n"
            "  python experiments/build_index.py \\\n"
            "    --vb100-extracted data/raw/vb100/extracted \\\n"
            "    --inat-index data/inaturalist/index.csv \\\n"
            "    --unified-out data/merged/index.csv"
        )
        sys.exit(1)

    df = load_merged_index(merged_csv)

    logger.info("Total samples : %d", len(df))
    logger.info("Modality breakdown: %s", df["modality"].value_counts().to_dict())
    logger.info("Species breakdown:")
    for sid, sname in enumerate(SPECIES_NAMES):
        n = (df["species_id"] == sid).sum()
        flag = "⚠ MISSING" if n == 0 else ""
        logger.info("  [%2d] %-32s %4d  %s", sid, sname, n, flag)

    missing = [SPECIES_NAMES[i] for i in range(len(SPECIES_NAMES))
               if (df["species_id"] == i).sum() == 0]
    if missing:
        logger.warning("Missing species (%d): %s", len(missing), missing)

    logger.info("Generating stratified splits (70 / 15 / 15)…")
    df_split = generate_splits(df, val_frac=0.15, test_frac=0.15, seed=42)
    logger.info("Split sizes: %s", df_split["split"].value_counts().to_dict())

    weights = compute_class_weights(df_split)
    logger.info("Class weights — min=%.4f  max=%.4f", weights.min(), weights.max())

    save_splits(df_split, SPLITS_DIR, weights)
    logger.info("Splits saved to %s", SPLITS_DIR)
    logger.info("Done.")


if __name__ == "__main__":
    main()
