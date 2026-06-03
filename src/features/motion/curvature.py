"""
DOVE — Curvature-bearing features via law of cosines.
"""
from __future__ import annotations

import logging

import numpy as np

from . import compute_statistics

logger = logging.getLogger(__name__)


class CurvatureBearingFeatures:
    """
    Curvature-bearing feature extractor.

    For each interior triplet, computes the cosine of the angle at the middle
    point using the law of cosines.  Then summarises with compute_statistics → 10.
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
        traj = np.asarray(trajectory, dtype=float)
        T = len(traj)
        if T < 3:
            return np.zeros(10, dtype=np.float32)

        cos_thetas = []
        for i in range(1, T - 1):
            a = traj[i - 1]
            b = traj[i]
            c = traj[i + 1]
            # sides of the triangle
            ab = np.linalg.norm(b - a)
            bc = np.linalg.norm(c - b)
            ac = np.linalg.norm(c - a)
            # law of cosines: ac^2 = ab^2 + bc^2 - 2*ab*bc*cos(theta_b)
            denom = 2.0 * ab * bc
            if denom < 1e-10:
                cos_theta = 1.0
            else:
                cos_theta = np.clip((ab ** 2 + bc ** 2 - ac ** 2) / denom, -1.0, 1.0)
            cos_thetas.append(float(cos_theta))

        feats = compute_statistics(np.array(cos_thetas))
        return feats.astype(np.float32)


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 128)
    traj = np.stack([np.cos(t), np.sin(t)], axis=1)

    cb = CurvatureBearingFeatures()
    feats = cb.extract(traj)
    print("CurvatureBearing features shape:", feats.shape)  # (10,)
    print("Values:", feats)
