"""
DOVE — Invariant feature pipeline: Swin + HSFA.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .backbone import SwinBackboneWithFFA
from .ffa import FFAModule
from .hsfa import HSFAModule

logger = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from PIL import Image as PILImage
    from torchvision import transforms
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


_PREPROCESS = None
if _PIL_AVAILABLE:
    from torchvision import transforms as _T
    _PREPROCESS = _T.Compose([
        _T.Resize(256),
        _T.CenterCrop(224),
        _T.ToTensor(),
        _T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class InvariantFeaturePipeline(nn.Module):
    """
    Full invariant feature pipeline.

    Components: SwinBackboneWithFFA → HSFAModule
    Output: (B, 1440) embeddings.
    """

    def __init__(
        self,
        k: int = 100,
        sim_mode: str = "cosine",
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = SwinBackboneWithFFA(pretrained=pretrained)
        self.ffa = FFAModule(k=k, sim_mode=sim_mode)
        self.hsfa = HSFAModule()

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 224, 224)

        Returns
        -------
        (B, 1440)
        """
        feature_maps = self.backbone(x)
        emb = self.hsfa(feature_maps, ffa_module=self.ffa)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract(x)

    @torch.no_grad()
    def extract_from_video(
        self,
        path: str | Path,
        n_frames: int = 8,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Load n_frames from a video, run extract per frame, mean-pool.

        Returns
        -------
        (1, 1440) tensor
        """
        path = str(path)
        self.eval()
        self.to(device)

        frames_tensor = self._load_video_frames(path, n_frames, device)
        if frames_tensor is None or frames_tensor.shape[0] == 0:
            return torch.zeros(1, 1440, device=device)

        embs = self.extract(frames_tensor)        # (n_frames, 1440)
        return embs.mean(dim=0, keepdim=True)     # (1, 1440)

    def _load_video_frames(
        self,
        path: str,
        n_frames: int,
        device: str,
    ) -> Optional[torch.Tensor]:
        if not _CV2_AVAILABLE or _PREPROCESS is None:
            return None

        import cv2

        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None

        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        tensors = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            if not ret or frame is None:
                tensors.append(torch.zeros(3, 224, 224))
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = PILImage.fromarray(rgb)
            t = _PREPROCESS(pil)
            tensors.append(t)
        cap.release()

        return torch.stack(tensors).to(device)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    pipeline = InvariantFeaturePipeline(k=50, pretrained=False)
    pipeline.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = pipeline.extract(x)
    print("InvariantFeaturePipeline extract output:", out.shape)  # (2, 1440)
