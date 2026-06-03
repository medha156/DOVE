"""
DOVE — Faster R-CNN backbone wrapper.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn

logger = logging.getLogger(__name__)


class FasterRCNNBackbone(nn.Module):
    """
    Faster R-CNN (ResNet-50 + FPN) backbone.

    Returns FPN feature maps from the backbone (not full detection output).
    feature_dim: 256 (FPN output channels).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        detector = fasterrcnn_resnet50_fpn(pretrained=pretrained)
        # Extract just the backbone (ResNet + FPN)
        self.backbone = detector.backbone
        self._feature_dim = 256  # FPN output

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        dict of FPN feature maps {'0': ..., '1': ..., '2': ..., '3': ..., 'pool': ...}
        """
        images = [img for img in x]
        from torchvision.models.detection.image_list import ImageList
        image_list = ImageList(x, [(x.shape[-2], x.shape[-1])] * x.shape[0])
        return self.backbone(image_list.tensors)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    model = FasterRCNNBackbone(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("FasterRCNNBackbone output keys:", list(out.keys()))
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
    print("feature_dim:", model.feature_dim)
