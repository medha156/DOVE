"""
DOVE — CDF (centroid-distance function) features.
"""
from __future__ import annotations

import logging

import numpy as np

from . import compute_statistics

logger = logging.getLogger(__name__)


class CDFFeatures:
    """
    Trajectory centroid-distance feature extractor.

    Computes the distance from each point to the trajectory centroid,
    then summarises with compute_statistics → 10 features.
    """

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        trajectory : np.ndarray, shape (T, 2)

        Returns
        -------
        np.ndarray, shape (10,)
        """
        if len(trajectory) == 0:
            return np.zeros(10, dtype=np.float32)

        traj = np.asarray(trajectory, dtype=float)
        centroid = traj.mean(axis=0)  # (2,)
        dists = np.linalg.norm(traj - centroid, axis=1)  # (T,)
        feats = compute_statistics(dists)
        return feats.astype(np.float32)


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 128)
    traj = np.stack([np.cos(t), np.sin(t)], axis=1)

    cdf = CDFFeatures()
    feats = cdf.extract(traj)
    print("CDF features shape:", feats.shape)  # (10,)
    print("Values:", feats)
