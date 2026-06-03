"""
DOVE — Turn / heading angle features.
"""
from __future__ import annotations

import logging

import numpy as np

from . import compute_statistics

logger = logging.getLogger(__name__)

_N_BINS = 120
_TOP_K = 20


class TurnFeatures:
    """
    Heading-angle feature extractor.

    Extracts 30 features from a 2-D trajectory:
        - compute_statistics(theta values)       → 10 features
        - Top-20 histogram bins by variance      → 20 features
    """

    def __init__(self, n_bins: int = _N_BINS):
        self.n_bins = n_bins

    # ------------------------------------------------------------------
    @staticmethod
    def _heading_angles(traj: np.ndarray) -> np.ndarray:
        """
        Piecewise heading angles using atan2 of successive displacement vectors.
        Returns array of length T-1 (or empty if T < 2).
        """
        if len(traj) < 2:
            return np.array([])
        dx = np.diff(traj[:, 0])
        dy = np.diff(traj[:, 1])
        angles = np.arctan2(dy, dx)
        return angles

    # ------------------------------------------------------------------
    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        trajectory : np.ndarray, shape (T, 2)

        Returns
        -------
        np.ndarray, shape (30,)
        """
        theta = self._heading_angles(trajectory)

        if len(theta) == 0:
            return np.zeros(30, dtype=np.float32)

        stats = compute_statistics(theta)  # 10

        # 120-bin histogram over [-π, π]
        hist, _ = np.histogram(theta, bins=self.n_bins, range=(-np.pi, np.pi))
        hist = hist.astype(float)

        # Top-20 bins by variance (treat each bin count as a 1-sample population)
        # "variance" here: contribution = bin_count^2 (sorted descending, pick top 20)
        top20_idx = np.argsort(hist)[::-1][:_TOP_K]
        top20_vals = hist[top20_idx]

        feats = np.concatenate([stats, top20_vals])  # 10 + 20 = 30
        return feats.astype(np.float32)


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 128)
    traj = np.stack([t, np.sin(t)], axis=1)

    turn = TurnFeatures()
    feats = turn.extract(traj)
    print("Turn features shape:", feats.shape)  # (30,)
    print("First few values:", feats[:5])
