"""
DOVE — Feature Filtering by Activation (FFA) module.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FFAModule(nn.Module):
    """
    Feature Filtering by Activation.

    Given a set of patch vectors, selects the top-k most discriminative
    patches based on a pairwise similarity score.

    Parameters
    ----------
    k        : int   — number of patches to keep
    sim_mode : str   — 'cosine' or 'l2'
    """

    def __init__(self, k: int = 100, sim_mode: str = "cosine"):
        super().__init__()
        self.k = k
        self.sim_mode = sim_mode

    def _discrimination_score(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Compute a discrimination score for each patch vector.

        Score = mean pairwise similarity with all other patches.
        High score → patch is representative / generic.
        We invert: score_i = 1 / (mean_sim + eps), so less generic patches rank higher.

        Parameters
        ----------
        vectors : (N, D) tensor

        Returns
        -------
        scores : (N,) tensor
        """
        N, D = vectors.shape
        if N == 0:
            return torch.zeros(0, device=vectors.device)

        if self.sim_mode == "cosine":
            norm = F.normalize(vectors, dim=1)  # (N, D)
            sim_matrix = norm @ norm.T           # (N, N)
        else:
            # Inverted L2: sim = 1 / (1 + ||a - b||)
            diff = vectors.unsqueeze(1) - vectors.unsqueeze(0)  # (N, N, D)
            dist = diff.norm(dim=-1)                            # (N, N)
            sim_matrix = 1.0 / (1.0 + dist)

        # Exclude self-similarity on diagonal
        mask = ~torch.eye(N, dtype=torch.bool, device=vectors.device)
        mean_sim = (sim_matrix * mask.float()).sum(dim=1) / (N - 1 + 1e-10)

        # Discrimination = inverse of mean similarity
        scores = 1.0 / (mean_sim + 1e-10)
        return scores

    def forward(self, patch_vectors: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        patch_vectors : (N, D) — all patch feature vectors

        Returns
        -------
        top-k patches : (min(k, N), D)
        """
        N = patch_vectors.shape[0]
        if N == 0:
            return patch_vectors

        scores = self._discrimination_score(patch_vectors)
        k = min(self.k, N)
        top_idx = torch.topk(scores, k, largest=True).indices
        return patch_vectors[top_idx]


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    patches = torch.randn(200, 96)
    ffa = FFAModule(k=50, sim_mode="cosine")
    out = ffa(patches)
    print("FFAModule input:", patches.shape, "→ output:", out.shape)

    ffa_l2 = FFAModule(k=50, sim_mode="l2")
    out_l2 = ffa_l2(patches)
    print("FFAModule (l2) input:", patches.shape, "→ output:", out_l2.shape)
