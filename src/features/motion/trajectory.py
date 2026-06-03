"""
DOVE — Trajectory accumulation and windowing.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TrajectoryModel:
    """
    Accumulates centroid positions over time and provides windowed views.

    Parameters
    ----------
    window_size : int — number of points in each temporal window
    """

    def __init__(self, window_size: int = 64):
        self.window_size = window_size
        self._points: List[np.ndarray] = []

    # ------------------------------------------------------------------
    def add_point(self, cx: float, cy: float) -> None:
        """Append a new (cx, cy) position."""
        self._points.append(np.array([cx, cy], dtype=np.float32))

    # ------------------------------------------------------------------
    def smooth(self) -> np.ndarray:
        """
        Apply a box filter (kernel size 3) independently to x and y.

        Returns
        -------
        np.ndarray, shape (T, 2) — smoothed trajectory
        """
        if len(self._points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.array(self._points, dtype=np.float32)
        kernel = np.ones(3) / 3.0
        xs = np.convolve(pts[:, 0], kernel, mode="same")
        ys = np.convolve(pts[:, 1], kernel, mode="same")
        return np.stack([xs, ys], axis=1)

    # ------------------------------------------------------------------
    def get_window(self, center_idx: int) -> np.ndarray:
        """
        Return a window of *window_size* points centred at *center_idx*.
        Edge-padded by repeating the first/last point as needed.

        Returns
        -------
        np.ndarray, shape (window_size, 2)
        """
        pts = self.smooth()
        T = len(pts)
        if T == 0:
            return np.zeros((self.window_size, 2), dtype=np.float32)

        half = self.window_size // 2
        start = center_idx - half
        end = start + self.window_size

        # Build padded array
        result = []
        for i in range(start, end):
            clamped = max(0, min(T - 1, i))
            result.append(pts[clamped])
        return np.array(result, dtype=np.float32)

    # ------------------------------------------------------------------
    def get_full(self) -> np.ndarray:
        """Return the full unsmoothed trajectory as (T, 2)."""
        if not self._points:
            return np.empty((0, 2), dtype=np.float32)
        return np.array(self._points, dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all accumulated points."""
        self._points = []


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)

    tm = TrajectoryModel(window_size=16)
    for _ in range(64):
        tm.add_point(np.random.randn(), np.random.randn())

    smoothed = tm.smooth()
    print("Smoothed trajectory shape:", smoothed.shape)

    win = tm.get_window(center_idx=32)
    print("Window shape:", win.shape)

    tm.reset()
    print("After reset, full trajectory:", tm.get_full().shape)
