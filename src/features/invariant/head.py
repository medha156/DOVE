"""
DOVE — Classification and feature-extraction heads for invariant pipeline.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """
    Linear classifier on top of a 1440-d embedding.

    Parameters
    ----------
    num_classes : int
    in_dim      : int  (default 1440, sum of Swin-T stage dims)
    """

    def __init__(self, num_classes: int = 20, in_dim: int = 1440):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_dim) → (B, num_classes)"""
        return self.fc(x)


class FeatureExtractor(nn.Module):
    """
    Passes through the 1440-d embedding unchanged (for downstream use).

    Optionally applies L2-normalisation.
    """

    def __init__(self, normalise: bool = False):
        super().__init__()
        self.normalise = normalise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1440) → (B, 1440)"""
        if self.normalise:
            x = nn.functional.normalize(x, dim=1)
        return x


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    B = 4
    emb = torch.randn(B, 1440)

    head = ClassificationHead(num_classes=20)
    logits = head(emb)
    print("ClassificationHead output:", logits.shape)  # (4, 20)

    extractor = FeatureExtractor(normalise=True)
    feats = extractor(emb)
    print("FeatureExtractor output:", feats.shape)      # (4, 1440)
    print("L2 norms (should be ~1):", feats.norm(dim=1)[:4])
