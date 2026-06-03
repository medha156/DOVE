"""
DOVE — Dataset and DataLoader utilities.

Classes:
- ImageLoader   : iNaturalist-style image dataset
- VideoLoader   : VB100-style video dataset (uniform frame sampling)
- DOVEDataset   : unified wrapper over both modalities
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


class ImageLoader(Dataset):
    """
    Loads single images from a CSV manifest.

    Parameters
    ----------
    csv_path : path to split CSV (columns: filepath, species_id, …)
    transform : torchvision transform applied to each PIL image
    root_dir  : optional prefix prepended to each filepath
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: Optional[Callable] = None,
        root_dir: Optional[str | Path] = None,
    ):
        self.df = pd.read_csv(csv_path)
        # keep only image rows
        if "modality" in self.df.columns:
            self.df = self.df[self.df["modality"] == "image"].reset_index(drop=True)
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

        if _PIL_AVAILABLE and Path(fpath).exists():
            img = PILImage.open(fpath).convert("RGB")
        else:
            img = PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) if _PIL_AVAILABLE else None

        if self.transform is not None and img is not None:
            image_tensor = self.transform(img)
        elif img is not None:
            image_tensor = torch.zeros(3, 224, 224)
        else:
            image_tensor = torch.zeros(3, 224, 224)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "modality": "image",
            "filepath": fpath,
        }


class VideoLoader(Dataset):
    """
    Loads short video clips and uniformly samples n_frames frames.

    Parameters
    ----------
    csv_path  : path to split CSV
    n_frames  : number of frames to uniformly sample
    transform : applied to each frame independently (PIL image → tensor)
    root_dir  : optional path prefix
    """

    def __init__(
        self,
        csv_path: str | Path,
        n_frames: int = 8,
        transform: Optional[Callable] = None,
        root_dir: Optional[str | Path] = None,
    ):
        self.df = pd.read_csv(csv_path)
        if "modality" in self.df.columns:
            self.df = self.df[self.df["modality"] == "video"].reset_index(drop=True)
        self.n_frames = n_frames
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else None
        logger.info("VideoLoader: %d samples, n_frames=%d", len(self.df), n_frames)

    def __len__(self) -> int:
        return len(self.df)

    def _load_frames(self, path: str) -> torch.Tensor:
        """Returns (n_frames, C, H, W) tensor."""
        if not _CV2_AVAILABLE or not Path(path).exists():
            return torch.zeros(self.n_frames, 3, 224, 224)

        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return torch.zeros(self.n_frames, 3, 224, 224)

        indices = np.linspace(0, total - 1, self.n_frames, dtype=int)
        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret or frame is None:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()

        tensors = []
        for f in frames:
            if _PIL_AVAILABLE:
                pil = PILImage.fromarray(f)
                if self.transform:
                    t = self.transform(pil)
                else:
                    t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            else:
                t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
            tensors.append(t)

        return torch.stack(tensors, dim=0)  # (n_frames, C, H, W)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        fpath = str(row["filepath"])
        if self.root_dir:
            fpath = str(self.root_dir / fpath)
        label = int(row["species_id"])
        frames = self._load_frames(fpath)
        return {
            "frames": frames,
            "label": torch.tensor(label, dtype=torch.long),
            "modality": "video",
            "filepath": fpath,
        }


class DOVEDataset(Dataset):
    """
    Unified dataset that wraps both ImageLoader and VideoLoader.

    Each item returns:
        {
          "image"   : (C, H, W) tensor  (zeros if video)
          "frames"  : (n_frames, C, H, W) tensor  (zeros if image)
          "label"   : long scalar
          "modality": 'image' or 'video'
          "filepath": str
        }
    """

    def __init__(
        self,
        csv_path: str | Path,
        n_frames: int = 8,
        transform: Optional[Callable] = None,
        root_dir: Optional[str | Path] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.n_frames = n_frames
        self.transform = transform
        self.root_dir = Path(root_dir) if root_dir else None
        logger.info("DOVEDataset: %d total samples", len(self.df))

        self._img_loader = ImageLoader.__new__(ImageLoader)
        self._img_loader.df = self.df
        self._img_loader.transform = transform
        self._img_loader.root_dir = self.root_dir

        self._vid_loader = VideoLoader.__new__(VideoLoader)
        self._vid_loader.df = self.df
        self._vid_loader.n_frames = n_frames
        self._vid_loader.transform = transform
        self._vid_loader.root_dir = self.root_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        modality = str(row.get("modality", "image"))
        fpath = str(row["filepath"])
        if self.root_dir:
            fpath = str(self.root_dir / fpath)
        label = int(row["species_id"])

        if modality == "image":
            if _PIL_AVAILABLE and Path(fpath).exists():
                img = PILImage.open(fpath).convert("RGB")
            else:
                img = PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)) if _PIL_AVAILABLE else None

            if self.transform and img is not None:
                image_tensor = self.transform(img)
            else:
                image_tensor = torch.zeros(3, 224, 224)
            frames_tensor = torch.zeros(self.n_frames, 3, 224, 224)
        else:
            image_tensor = torch.zeros(3, 224, 224)
            frames_tensor = self._vid_loader._load_frames(fpath)

        return {
            "image": image_tensor,
            "frames": frames_tensor,
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
