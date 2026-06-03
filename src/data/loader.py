"""
DOVE — Dataset and DataLoader utilities.

Data comes from the CHIRP repo (https://github.com/medha156/CHIRP):
  - iNaturalist images (modality='image'): raw JPEGs, one per observation
  - VB100 filtered frames (modality='video_frame'): pre-extracted JPEGs that
    have already passed pixel std>=10 and Laplacian variance>=5 quality gates.
    Each JPEG is one quality-filtered frame from a VB100 clip.

Classes:
- ImageLoader      : iNaturalist-style image dataset
- VideoFrameLoader : VB100 pre-extracted frame dataset (loads JPEGs)
- DOVEDataset      : unified wrapper over both modalities
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("cv2 not available — VideoLoader will return zero tensors")

try:
    from PIL import Image as PILImage
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False
    logger.warning("PIL not available — ImageLoader will return zero tensors")


def _load_pil(fpath: str) -> "PILImage.Image":
    """Load a JPEG/PNG as an RGB PIL image, returning a blank image on failure."""
    if _PIL_AVAILABLE and Path(fpath).exists():
        try:
            return PILImage.open(fpath).convert("RGB")
        except Exception:
            pass
    if _PIL_AVAILABLE:
        return PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    return None  # type: ignore


class ImageLoader(Dataset):
    """
    Loads iNaturalist images from a CSV manifest (modality='image').

    CSV must have columns: filepath, species_id  (CHIRP format also accepted:
    path, label, species).
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[Callable] = None,
        root_dir: Optional[str | Path] = None,
    ):
        df = pd.read_csv(csv_path)
        # normalise CHIRP column names
        if "path" in df.columns and "filepath" not in df.columns:
            df = df.rename(columns={"path": "filepath"})
        if "label" in df.columns and "species_id" not in df.columns:
            df = df.rename(columns={"label": "species_id"})
        if "modality" in df.columns:
            df = df[df["modality"] == "image"].reset_index(drop=True)
        self.df = df
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else None
        logger.info("ImageLoader: %d samples", len(self.df))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        fpath = str(row["filepath"])
        if self.root_dir:
            fpath = str(self.root_dir / fpath)
        label = int(row["species_id"])
        img = _load_pil(fpath)
        image_tensor = self.transform(img) if (self.transform and img is not None) \
            else torch.zeros(3, 224, 224)
        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "modality": "image",
            "filepath": fpath,
        }


class VideoFrameLoader(Dataset):
    """
    Loads pre-extracted, quality-filtered VB100 frames (modality='video_frame').

    Each row in the CSV is one JPEG frame that has already passed the CHIRP
    quality gates (pixel std >= 10, Laplacian variance >= 5).  The loader
    simply reads the JPEG — no video decoding required.
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[Callable] = None,
        root_dir: Optional[str | Path] = None,
    ):
        df = pd.read_csv(csv_path)
        if "path" in df.columns and "filepath" not in df.columns:
            df = df.rename(columns={"path": "filepath"})
        if "label" in df.columns and "species_id" not in df.columns:
            df = df.rename(columns={"label": "species_id"})
        if "modality" in df.columns:
            df = df[df["modality"] == "video_frame"].reset_index(drop=True)
        self.df = df
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else None
        logger.info("VideoFrameLoader: %d samples", len(self.df))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        fpath = str(row["filepath"])
        if self.root_dir:
            fpath = str(self.root_dir / fpath)
        label = int(row["species_id"])
        img = _load_pil(fpath)
        image_tensor = self.transform(img) if (self.transform and img is not None) \
            else torch.zeros(3, 224, 224)
        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "modality": "video_frame",
            "filepath": fpath,
        }


# Keep VideoLoader as a thin alias so existing code doesn't break
VideoLoader = VideoFrameLoader


class DOVEDataset(Dataset):
    """
    Unified dataset over iNaturalist images and quality-filtered VB100 frames.

    Both modalities are single JPEGs — no video decoding.  Each item returns:
        {
          "image"   : (C, H, W) tensor  — the loaded/transformed JPEG
          "label"   : long scalar
          "modality": 'image' or 'video_frame'
          "filepath": str
        }
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[Callable] = None,
        root_dir: Optional[str | Path] = None,
    ):
        df = pd.read_csv(csv_path)
        # normalise CHIRP column names
        if "path" in df.columns and "filepath" not in df.columns:
            df = df.rename(columns={"path": "filepath"})
        if "label" in df.columns and "species_id" not in df.columns:
            df = df.rename(columns={"label": "species_id"})
        self.df = df
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else None
        logger.info("DOVEDataset: %d total samples", len(self.df))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        modality = str(row.get("modality", "image"))
        fpath = str(row["filepath"])
        if self.root_dir:
            fpath = str(self.root_dir / fpath)
        label = int(row["species_id"])

        img = _load_pil(fpath)
        image_tensor = self.transform(img) if (self.transform and img is not None) \
            else torch.zeros(3, 224, 224)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "modality": modality,
            "filepath": fpath,
        }


if __name__ == "__main__":
    import tempfile

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create a tiny synthetic CSV
    rows = [
        {"filepath": "fake/Acorn Woodpecker/img_0.jpg", "species_id": 0, "species_name": "Acorn Woodpecker", "modality": "image"},
        {"filepath": "fake/American Crow/vid_0.mp4", "species_id": 1, "species_name": "American Crow", "modality": "video"},
    ]
    df = pd.DataFrame(rows)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f.name, index=False)
        csv_path = f.name

    ds = DOVEDataset(csv_path, n_frames=4)
    for i in range(len(ds)):
        item = ds[i]
        print(
            f"[{i}] modality={item['modality']}, "
            f"image={item['image'].shape}, "
            f"frames={item['frames'].shape}, "
            f"label={item['label'].item()}"
        )
