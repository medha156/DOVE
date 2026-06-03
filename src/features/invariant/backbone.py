"""
DOVE — Swin-T backbone with multi-scale feature extraction.
"""
from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False
    logger.warning("timm not available — SwinBackboneWithFFA will return zero tensors")


class SwinBackboneWithFFA(nn.Module):
    """
    Swin-Tiny backbone that returns feature maps from all 4 stages.

    Output: list of 4 tensors at stages 0-3 with channel dims
    [96, 192, 384, 768] (swin_tiny).
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self._dummy = not _TIMM_AVAILABLE

        if _TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                "swin_tiny_patch4_window7_224",
                features_only=True,
                out_indices=(0, 1, 2, 3),
                pretrained=pretrained,
            )
        else:
            # Fallback placeholder
            self.backbone = None

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, 224, 224)

        Returns
        -------
        list of 4 feature maps
        """
        if self._dummy or self.backbone is None:
            # Return dummy tensors with correct channel dims
            B = x.shape[0]
            return [
                torch.zeros(B, 96, 56, 56, device=x.device),
                torch.zeros(B, 192, 28, 28, device=x.device),
                torch.zeros(B, 384, 14, 14, device=x.device),
                torch.zeros(B, 768, 7, 7, device=x.device),
            ]
        return self.backbone(x)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    model = SwinBackboneWithFFA(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        features = model(x)
    print("SwinBackboneWithFFA output stages:")
    for i, f in enumerate(features):
        print(f"  Stage {i}: {f.shape}")
