"""
DOVE — Swin-Tiny backbone (timm).
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False
    logger.warning("timm not available — SwinTBackbone will return zeros")


class SwinTBackbone(nn.Module):
    """
    Swin-Tiny backbone returning global pooled features (768-d).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self._dummy = not _TIMM_AVAILABLE
        if _TIMM_AVAILABLE:
            self.model = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=pretrained,
                num_classes=0,  # remove classifier head
            )
            self._feature_dim = self.model.num_features
        else:
            self.model = None
            self._feature_dim = 768

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) → (B, 768)"""
        if self._dummy or self.model is None:
            return torch.zeros(x.shape[0], 768, device=x.device)
        return self.model(x)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    model = SwinTBackbone(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print("SwinTBackbone output:", out.shape)
    print("feature_dim:", model.feature_dim)
