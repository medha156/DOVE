"""
DOVE — Linear classification head.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LinearHead(nn.Module):
    """
    Single linear layer: nn.Linear(in_dim, num_classes).
    """

    def __init__(self, in_dim: int = 512, num_classes: int = 20):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    B = 8
    x = torch.randn(B, 512)
    head = LinearHead(in_dim=512, num_classes=20)
    out = head(x)
    print("LinearHead output:", out.shape)  # (8, 20)
