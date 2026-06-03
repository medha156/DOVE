"""
DOVE — Extract invariant features from all images and videos.
Saves embeddings to data/invariant_features.parquet.
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.invariant.pipeline import InvariantFeaturePipeline
from data.augment import get_image_transform

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("extract_invariant")

REPO_ROOT = Path(__file__).parent.parent
INAT_ROOT = REPO_ROOT / "data" / "inaturalist"
VB100_ROOT = REPO_ROOT / "data" / "vb100"
OUTPUT_PATH = REPO_ROOT / "data" / "invariant_features.parquet"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    logger.info("=" * 60)
    logger.info("DOVE — Invariant Feature Extraction")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    pipeline = InvariantFeaturePipeline(k=100, pretrained=True)
    pipeline.eval()
    pipeline.to(device)

    transform = get_image_transform(train=False)
    rows = []

    # Images (iNaturalist)
    if INAT_ROOT.exists():
        images = sorted(p for p in INAT_ROOT.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        logger.info("Found %d images", len(images))
        from PIL import Image as PILImage

        for ipath in tqdm(images, desc="Images"):
            try:
                img = PILImage.open(ipath).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = pipeline.extract(x).cpu().numpy()[0]
                rows.append({
                    "filepath": str(ipath),
                    "species_name": ipath.parent.name,
                    "modality": "image",
                    **{f"feat_{i:04d}": float(v) for i, v in enumerate(emb)},
                })
            except Exception as exc:
                logger.warning("Failed on %s: %s", ipath, exc)

    # Videos (VB100)
    if VB100_ROOT.exists():
        videos = sorted(p for p in VB100_ROOT.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
        logger.info("Found %d videos", len(videos))

        for vpath in tqdm(videos, desc="Videos"):
            try:
                emb = pipeline.extract_from_video(str(vpath), n_frames=8, device=device)
                emb_np = emb.cpu().numpy()[0]
                rows.append({
                    "filepath": str(vpath),
                    "species_name": vpath.parent.name,
                    "modality": "video",
                    **{f"feat_{i:04d}": float(v) for i, v in enumerate(emb_np)},
                })
            except Exception as exc:
                logger.warning("Failed on %s: %s", vpath, exc)

    if not rows:
        logger.warning("No features extracted")
        return

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved %d rows × %d cols to %s", len(df), len(df.columns), OUTPUT_PATH)


if __name__ == "__main__":
    main()
