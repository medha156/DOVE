"""
DOVE — VGG-19 BN backbone.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class VGG19Backbone(nn.Module):
    """
    VGG-19 with batch normalisation, returning feature maps up to pool4.

    feature_dim: 512 (channels at pool4 output)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        vgg = models.vgg19_bn(pretrained=pretrained)
        # Features up to pool4: layers 0-36 (indices 0..36 inclusive)
        # pool4 is the 4th MaxPool2d, which in vgg19_bn is at index 36
        self.features = nn.Sequential(*list(vgg.features.children())[:37])
        self._feature_dim = 512

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) → (B, 512, H/16, W/16)"""
        return self.features(x)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    model = VGG19Backbone(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("VGG19Backbone output:", out.shape)
    print("feature_dim:", model.feature_dim)
