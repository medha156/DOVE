"""
DOVE — Three-way fusion: backbone + invariant + motion.

Two variants:
  TripleConcatFusion      — project each stream to 512, cat to 1536, Linear+LN→512
  TripleCrossAttentionFusion — invariant attends to backbone (as before),
                               then motion is concatenated before final projection
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TripleConcatFusion(nn.Module):
    """
    backbone_feat (B, d_bb) + inv_feat (B, 1440) + motion_feat (B, 139)
    → (B, 512)

    Each stream independently projected to 512 via Linear+GELU,
    concatenated to (B, 1536), then Linear(1536, 512)+LayerNorm.
    """

    def __init__(self, backbone_dim: int, invariant_dim: int = 1440,
                 motion_dim: int = 139, d_model: int = 512):
        super().__init__()
        self.bb_proj  = nn.Sequential(nn.Linear(backbone_dim, d_model), nn.GELU())
        self.inv_proj = nn.Sequential(nn.Linear(invariant_dim, d_model), nn.GELU())
        self.mot_proj = nn.Sequential(nn.Linear(motion_dim, d_model), nn.GELU())
        self.fusion   = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, backbone: torch.Tensor, invariant: torch.Tensor,
                motion: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.bb_proj(backbone),
                       self.inv_proj(invariant),
                       self.mot_proj(motion)], dim=-1)  # (B, 1536)
        return self.fusion(z)                            # (B, 512)


class TripleCrossAttentionFusion(nn.Module):
    """
    Cross-attention between invariant (Q) and backbone (K/V), then
    motion is concatenated to the attended vector before final projection.

    invariant (B, 1440) → Q
    backbone  (B, d_bb) → K, V
    motion    (B, 139)  → concatenated after attention

    Output: (B, 512)
    """

    def __init__(self, backbone_dim: int, invariant_dim: int = 1440,
                 motion_dim: int = 139, d_model: int = 512, nhead: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.bb_proj  = nn.Linear(backbone_dim, d_model)
        self.inv_proj = nn.Linear(invariant_dim, d_model)
        self.mot_proj = nn.Sequential(nn.Linear(motion_dim, d_model), nn.GELU())
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out  = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, backbone: torch.Tensor, invariant: torch.Tensor,
                motion: torch.Tensor) -> torch.Tensor:
        q  = self.inv_proj(invariant).unsqueeze(1)   # (B, 1, 512)
        kv = self.bb_proj(backbone).unsqueeze(1)     # (B, 1, 512)
        attn_out, _ = self.attn(q, kv, kv)
        attended = self.norm(q + attn_out).squeeze(1)  # (B, 512)
        mot = self.mot_proj(motion)                    # (B, 512)
        return self.out(torch.cat([attended, mot], dim=-1))  # (B, 512)
