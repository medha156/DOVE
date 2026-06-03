"""
DOVE — Concatenation-based fusion of motion and invariant features.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConcatFusion(nn.Module):
    """
    Projects motion (139→d_model) and invariant (1440→d_model) features,
    concatenates them, then projects back to d_model via Linear + LayerNorm.

    Pipeline: [motion_proj(139→512), inv_proj(1440→512)]
              cat → (B, 1024) → Linear(1024, 512) → LayerNorm(512)
    """

    def __init__(
        self,
        motion_dim: int = 139,
        invariant_dim: int = 1440,
        d_model: int = 512,
    ):
        super().__init__()
        self.motion_proj = nn.Sequential(
            nn.Linear(motion_dim, d_model),
            nn.GELU(),
        )
        self.inv_proj = nn.Sequential(
            nn.Linear(invariant_dim, d_model),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        motion: torch.Tensor,
        invariant: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        motion    : (B, 139)
        invariant : (B, 1440)

        Returns
        -------
        (B, 512)
        """
        m = self.motion_proj(motion)      # (B, 512)
        v = self.inv_proj(invariant)      # (B, 512)
        cat = torch.cat([m, v], dim=-1)  # (B, 1024)
        return self.fusion(cat)           # (B, 512)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    B = 4
    motion = torch.randn(B, 139)
    invariant = torch.randn(B, 1440)

    fusion = ConcatFusion()
    out = fusion(motion, invariant)
    print("ConcatFusion output shape:", out.shape)  # (4, 512)
