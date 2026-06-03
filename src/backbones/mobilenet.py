"""
DOVE — MobileNetV3-Large backbone.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class MobileNetBackbone(nn.Module):
    """
    MobileNetV3-Large backbone returning feature maps (before classifier).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.mobilenet_v3_large(pretrained=pretrained)
        # Return features (all convolutional layers)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Last feature channel count
        self._feature_dim = 960

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, 960)"""
        feats = self.features(x)
        pooled = self.avgpool(feats)
        return pooled.flatten(1)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    model = MobileNetBackbone(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("MobileNetBackbone output:", out.shape)
    print("feature_dim:", model.feature_dim)
