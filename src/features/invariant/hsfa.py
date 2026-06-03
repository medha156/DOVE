"""
DOVE — Hierarchical Scale-aware Feature Aggregation (HSFA) module.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HSFAModule(nn.Module):
    """
    Hierarchical Scale-aware Feature Aggregation.

    Aggregates multi-scale feature maps from SwinBackboneWithFFA into a
    single (B, 1440) descriptor.

    Channel dims for swin_tiny stages: [96, 192, 384, 768] → sum = 1440.

    For each stage:
        1. Optionally filter patches via FFA.
        2. Global average pool → (B, C_i).
    Concatenate all stages → (B, 1440).
    """

    def __init__(self, channel_dims: List[int] = None):
        super().__init__()
        if channel_dims is None:
            channel_dims = [96, 192, 384, 768]
        self.channel_dims = channel_dims
        self.out_dim = sum(channel_dims)  # 1440

    def forward(
        self,
        feature_maps: List[torch.Tensor],
        ffa_module: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        feature_maps : list of (B, C_i, H_i, W_i) tensors from Swin stages
        ffa_module   : optional FFAModule to filter patches before pooling

        Returns
        -------
        (B, 1440) aggregated descriptor
        """
        pooled = []
        for fm in feature_maps:
            B, C, H, W = fm.shape
            if ffa_module is not None:
                # Reshape to patches: (B, N, C) where N = H*W
                patches = fm.permute(0, 2, 3, 1).reshape(B, H * W, C)
                filtered_list = []
                for b in range(B):
                    filt = ffa_module(patches[b])  # (k, C)
                    # Pool filtered patches
                    pooled_patch = filt.mean(dim=0)  # (C,)
                    filtered_list.append(pooled_patch)
                p = torch.stack(filtered_list, dim=0)  # (B, C)
            else:
                # Global average pool
                p = F.adaptive_avg_pool2d(fm, (1, 1)).squeeze(-1).squeeze(-1)  # (B, C)
            pooled.append(p)

        out = torch.cat(pooled, dim=1)  # (B, 1440)
        return out


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    B = 2
    feature_maps = [
        torch.randn(B, 96, 56, 56),
        torch.randn(B, 192, 28, 28),
        torch.randn(B, 384, 14, 14),
        torch.randn(B, 768, 7, 7),
    ]
    hsfa = HSFAModule()
    out = hsfa(feature_maps)
    print("HSFA output shape:", out.shape)  # (2, 1440)

    from .ffa import FFAModule
    ffa = FFAModule(k=10)
    out_ffa = hsfa(feature_maps, ffa_module=ffa)
    print("HSFA + FFA output shape:", out_ffa.shape)  # (2, 1440)
