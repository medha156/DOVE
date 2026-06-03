"""
DOVE — EfficientNet-B3 backbone.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class EfficientNetB3Backbone(nn.Module):
    """
    EfficientNet-B3 backbone returning pooled feature maps.
    feature_dim: 1536 (EfficientNet-B3 penultimate layer channels)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = models.efficientnet_b3(pretrained=pretrained)
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._feature_dim = 1536

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, 1536)"""
        feats = self.features(x)
        pooled = self.avgpool(feats)
        return pooled.flatten(1)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    model = EfficientNetB3Backbone(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("EfficientNetB3Backbone output:", out.shape)
    print("feature_dim:", model.feature_dim)
