"""
DOVE — MLP classification head.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MLPHead(nn.Module):
    """
    Two-layer MLP head: Linear → GELU → Dropout → Linear.

    Parameters
    ----------
    in_dim     : int  input dimensionality (default 512)
    num_classes: int
    hidden_dim : int  (default 256)
    dropout    : float (default 0.3)
    """

    def __init__(
        self,
        in_dim: int = 512,
        num_classes: int = 20,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    B = 8
    x = torch.randn(B, 512)
    head = MLPHead(in_dim=512, num_classes=20, hidden_dim=256, dropout=0.1)
    out = head(x)
    print("MLPHead output:", out.shape)  # (8, 20)
