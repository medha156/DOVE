"""
DOVE — Extract motion features from all VB100 videos.
Saves features to data/motion_features.parquet.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.motion.pipeline import MotionFeaturePipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("extract_motion")

REPO_ROOT = Path(__file__).parent.parent
VB100_ROOT = REPO_ROOT / "data" / "vb100"
OUTPUT_PATH = REPO_ROOT / "data" / "motion_features.parquet"


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    logger.info("=" * 60)
    logger.info("DOVE — Motion Feature Extraction")
    logger.info("=" * 60)

    if not VB100_ROOT.exists():
        logger.error("VB100 data directory not found: %s", VB100_ROOT)
        sys.exit(1)

    # Gather all video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    videos = sorted(
        p for p in VB100_ROOT.rglob("*") if p.suffix.lower() in video_extensions
    )
    if not videos:
        logger.warning("No video files found under %s", VB100_ROOT)
        return

    logger.info("Found %d video files", len(videos))
    pipeline = MotionFeaturePipeline(window_size=64)

    all_rows = []
    for vpath in tqdm(videos, desc="Extracting motion features"):
        try:
            df_vid = pipeline.extract_video(str(vpath))
            if df_vid.empty:
                continue
            # Add metadata
            df_vid["filepath"] = str(vpath)
            df_vid["species_name"] = vpath.parent.name
            all_rows.append(df_vid)
        except Exception as exc:
            logger.warning("Failed on %s: %s", vpath, exc)

    if not all_rows:
        logger.warning("No features extracted")
        return

    result = pd.concat(all_rows, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved %d rows × %d cols to %s", len(result), len(result.columns), OUTPUT_PATH)


if __name__ == "__main__":
    main()
