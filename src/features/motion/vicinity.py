"""
DOVE — Vicinity features (curliness, slope, aspect, linearity) for each triplet.
"""
from __future__ import annotations

import logging

import numpy as np

from . import compute_statistics

logger = logging.getLogger(__name__)


def _normalise(sig: np.ndarray) -> np.ndarray:
    mn, mx = sig.min(), sig.max()
    if mx - mn < 1e-10:
        return np.zeros_like(sig)
    return (sig - mn) / (mx - mn)


class VicinityFeatures:
    """
    Vicinity feature extractor.

    For each interior point triplet (p_{i-1}, p_i, p_{i+1}) computes:
        - curliness  : chord_length / arc_length
        - slope      : atan2(dy, dx) for overall displacement
        - aspect     : (max_y - min_y) / (max_x - min_x + eps)
        - linearity  : variance of perpendicular distances from regression line

    Each of the 4 signals is normalised and summarised with compute_statistics → 10.
    Total: 4 × 10 = 40 features.
    """

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        trajectory : np.ndarray, shape (T, 2)

        Returns
        -------
        np.ndarray, shape (40,)
        """
        traj = np.asarray(trajectory, dtype=float)
        T = len(traj)
        if T < 3:
            return np.zeros(40, dtype=np.float32)

        curlinesses = []
        slopes = []
        aspects = []
        linearities = []

        for i in range(1, T - 1):
            p_prev = traj[i - 1]
            p_curr = traj[i]
            p_next = traj[i + 1]
            triplet = np.array([p_prev, p_curr, p_next])

            # curliness: chord / arc
            chord = np.linalg.norm(p_next - p_prev) + 1e-10
            arc = (np.linalg.norm(p_curr - p_prev) + np.linalg.norm(p_next - p_curr)) + 1e-10
            curliness = chord / arc
            curlinesses.append(curliness)

            # slope of overall displacement p_prev → p_next
            disp = p_next - p_prev
            slope = float(np.arctan2(disp[1], disp[0]))
            slopes.append(slope)

            # aspect of bounding box of triplet
            xs = triplet[:, 0]
            ys = triplet[:, 1]
            dx = xs.max() - xs.min() + 1e-10
            dy = ys.max() - ys.min()
            aspects.append(dy / dx)

            # linearity: perpendicular distance of p_curr from line p_prev—p_next
            v = p_next - p_prev
            norm_v = np.linalg.norm(v) + 1e-10
            perp = abs((p_curr[0] - p_prev[0]) * v[1] - (p_curr[1] - p_prev[1]) * v[0]) / norm_v
            linearities.append(perp)

        curlinesses = _normalise(np.array(curlinesses))
        slopes = _normalise(np.array(slopes))
        aspects = _normalise(np.array(aspects))
        linearities = _normalise(np.array(linearities))

        feats = np.concatenate([
            compute_statistics(curlinesses),   # 10
            compute_statistics(slopes),        # 10
            compute_statistics(aspects),       # 10
            compute_statistics(linearities),   # 10
        ])
        return feats.astype(np.float32)


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    t = np.linspace(0, 6 * np.pi, 128)
    traj = np.stack([t, np.sin(t) + 0.1 * np.random.randn(128)], axis=1)

    vic = VicinityFeatures()
    feats = vic.extract(traj)
    print("Vicinity features shape:", feats.shape)  # (40,)
    print("First few values:", feats[:5])
