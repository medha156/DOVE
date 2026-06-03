"""
DOVE — Cross-attention fusion of motion and invariant features.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion where invariant features attend to motion features.

    Architecture:
        motion_proj    : Linear(139, 512)
        invariant_proj : Linear(1440, 512)
        Q = invariant_proj(invariant)
        K = V = motion_proj(motion)
        MultiheadAttention(512, nhead=8)
        output = Q + attn_output  (residual)
        LayerNorm(512)

    Attention weights are stored in self.last_attn_weights.
    """

    def __init__(
        self,
        motion_dim: int = 139,
        invariant_dim: int = 1440,
        d_model: int = 512,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.motion_proj = nn.Linear(motion_dim, d_model)
        self.inv_proj = nn.Linear(invariant_dim, d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.last_attn_weights: Optional[torch.Tensor] = None

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
        # Add sequence dimension: (B, 1, d_model)
        q = self.inv_proj(invariant).unsqueeze(1)    # (B, 1, 512)
        kv = self.motion_proj(motion).unsqueeze(1)   # (B, 1, 512)

        attn_out, attn_weights = self.attn(q, kv, kv)  # (B, 1, 512)
        self.last_attn_weights = attn_weights           # store for inspection

        out = self.norm(q + attn_out)   # residual + norm
        return out.squeeze(1)           # (B, 512)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    B = 4
    motion = torch.randn(B, 139)
    invariant = torch.randn(B, 1440)

    fusion = CrossAttentionFusion()
    out = fusion(motion, invariant)
    print("CrossAttentionFusion output shape:", out.shape)             # (4, 512)
    print("Attention weights shape:", fusion.last_attn_weights.shape)  # (4, 1, 1)
