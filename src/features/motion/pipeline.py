"""
DOVE — Motion feature pipeline: 139-d feature vector per window.

Feature breakdown:
    CSS        22
    Turn       30
    Wingbeat   27
    CDF        10
    Vicinity   40
    Curvature  10
    ----------
    Total     139
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .css import CSSFeatures
from .turn import TurnFeatures
from .wingbeat import WingbeatFeatures
from .cdf import CDFFeatures
from .vicinity import VicinityFeatures
from .curvature import CurvatureBearingFeatures
from .loader import TrajectoryLoader, FrameLoader
from .segmentation import SilhouetteExtractor
from .trajectory import TrajectoryModel
from .bbox import fit_oriented_bbox

logger = logging.getLogger(__name__)

_FEATURE_DIMS = {
    "css": 22,
    "turn": 30,
    "wingbeat": 27,
    "cdf": 10,
    "vicinity": 40,
    "curvature": 10,
}
_TOTAL = sum(_FEATURE_DIMS.values())  # 139


class MotionFeaturePipeline:
    """
    Extracts 139-d motion feature vectors from trajectory windows.
    """

    def __init__(self, window_size: int = 64):
        self.window_size = window_size
        self._css = CSSFeatures()
        self._turn = TurnFeatures()
        self._wingbeat = WingbeatFeatures()
        self._cdf = CDFFeatures()
        self._vicinity = VicinityFeatures()
        self._curvature = CurvatureBearingFeatures()

    # ------------------------------------------------------------------
    def extract_window(
        self,
        trajectory: np.ndarray,
        bbox_sequence: Optional[list] = None,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        trajectory    : np.ndarray, shape (T, 2)
        bbox_sequence : list of bbox dicts or tuples, length T  (optional)

        Returns
        -------
        np.ndarray, shape (139,)
        """
        if bbox_sequence is None:
            bbox_sequence = []

        feats = np.concatenate([
            self._css.extract(trajectory),          # 22
            self._turn.extract(trajectory),         # 30
            self._wingbeat.extract(bbox_sequence),  # 27
            self._cdf.extract(trajectory),          # 10
            self._vicinity.extract(trajectory),     # 40
            self._curvature.extract(trajectory),    # 10
        ])
        assert feats.shape == (_TOTAL,), f"Expected {_TOTAL} features, got {feats.shape[0]}"
        return feats.astype(np.float32)

    # ------------------------------------------------------------------
    def extract_video(self, path: str | Path) -> pd.DataFrame:
        """
        Run the pipeline on every frame window in a video.

        Returns
        -------
        pd.DataFrame with columns = feature_names(), one row per frame
        (after warmup).
        """
        path = str(path)
        extractor = SilhouetteExtractor(warmup_frames=30)
        tm = TrajectoryModel(window_size=self.window_size)
        bboxes_full: list = []

        for _, frame in FrameLoader(path):
            res = extractor.extract(frame)
            c = res.get("centroid")
            bb = res.get("bbox")
            if c is not None:
                tm.add_point(c[0], c[1])
                bboxes_full.append(bb)

        traj_full = tm.get_full()
        T = len(traj_full)
        if T < 3:
            logger.warning("Trajectory too short (%d points) for %s", T, path)
            return pd.DataFrame(columns=self.feature_names())

        rows = []
        for i in range(T):
            win = tm.get_window(i)
            bb_win_start = max(0, i - self.window_size // 2)
            bb_win_end = min(T, bb_win_start + self.window_size)
            bb_slice = bboxes_full[bb_win_start:bb_win_end]
            bb_slice_clean = [b for b in bb_slice if b is not None]
            feats = self.extract_window(win, bb_slice_clean)
            rows.append(feats)

        df = pd.DataFrame(rows, columns=self.feature_names())
        return df

    # ------------------------------------------------------------------
    def feature_names(self) -> List[str]:
        """Return the 139 feature names."""
        names: List[str] = []
        for prefix, dim in _FEATURE_DIMS.items():
            for i in range(dim):
                names.append(f"{prefix}_{i:03d}")
        return names


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)

    pipeline = MotionFeaturePipeline(window_size=64)

    # Synthetic trajectory
    t = np.linspace(0, 4 * np.pi, 128)
    traj = np.stack([np.cos(t) * 50 + 100, np.sin(t) * 30 + 80], axis=1)
    bboxes = [{"w": 40 + 5 * np.sin(i / 5), "h": 30 + 3 * np.cos(i / 5)} for i in range(64)]

    feats = pipeline.extract_window(traj[:64], bboxes)
    print("extract_window shape:", feats.shape)  # (139,)

    names = pipeline.feature_names()
    print("Feature names count:", len(names))
    print("First 5 names:", names[:5])
    print("Last 5 names:", names[-5:])
