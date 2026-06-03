"""
DOVE — Curvature Scale Space (CSS) features.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import compute_statistics

logger = logging.getLogger(__name__)


class CSSFeatures:
    """
    Curvature Scale Space feature extractor.

    Extracts 22 features from a 2-D trajectory:
        - compute_statistics(|K_i|)           → 10 features
        - n_curves                            →  1 feature
        - total_curve_length                  →  1 feature
        - compute_statistics(CSS_maxima)      → 10 features
    """

    def __init__(self, sigmas: tuple = tuple(range(1, 11))):
        self.sigmas = list(sigmas)

    # ------------------------------------------------------------------
    def _compute_curvature(self, traj: np.ndarray) -> np.ndarray:
        """
        Signed curvature K_i = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2).
        Uses np.gradient for first/second derivatives.
        """
        x = traj[:, 0].astype(float)
        y = traj[:, 1].astype(float)
        if len(x) < 3:
            return np.zeros(len(x))
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        denom = (dx ** 2 + dy ** 2) ** 1.5 + 1e-10
        K = (dx * ddy - dy * ddx) / denom
        return K

    # ------------------------------------------------------------------
    def _css_image(self, traj: np.ndarray) -> np.ndarray:
        """
        Compute the CSS image: zero-crossings of curvature at each sigma.

        Returns
        -------
        np.ndarray, shape (len(sigmas),) — number of zero crossings per scale
        """
        x = traj[:, 0].astype(float)
        y = traj[:, 1].astype(float)
        zc_counts = []
        for sigma in self.sigmas:
            xs = gaussian_filter1d(x, sigma=sigma)
            ys = gaussian_filter1d(y, sigma=sigma)
            tmp = np.stack([xs, ys], axis=1)
            K = self._compute_curvature(tmp)
            zc = np.sum(np.diff(np.sign(K)) != 0)
            zc_counts.append(float(zc))
        return np.array(zc_counts)

    # ------------------------------------------------------------------
    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        trajectory : np.ndarray, shape (T, 2)

        Returns
        -------
        np.ndarray, shape (22,)
        """
        if len(trajectory) < 3:
            return np.zeros(22)

        K = self._compute_curvature(trajectory)
        stats_K = compute_statistics(np.abs(K))  # 10

        css_zc = self._css_image(trajectory)  # (10,)

        # n_curves: number of contiguous curve segments (runs where |K| > threshold)
        thresh = np.mean(np.abs(K)) + 1e-8
        above = (np.abs(K) > thresh).astype(int)
        transitions = np.diff(above)
        n_curves = float(np.sum(transitions == 1))

        # total_curve_length: sum of arc-length increments
        diffs = np.diff(trajectory, axis=0)
        total_curve_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

        css_maxima_stats = compute_statistics(css_zc)  # 10

        feats = np.concatenate([
            stats_K,                          # 10
            [n_curves, total_curve_length],   #  2
            css_maxima_stats,                 # 10
        ])
        return feats.astype(np.float32)  # 22


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 128)
    traj = np.stack([np.cos(t), np.sin(t)], axis=1)

    css = CSSFeatures()
    feats = css.extract(traj)
    print("CSS features shape:", feats.shape)  # (22,)
    print("First few values:", feats[:5])
