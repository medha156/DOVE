"""
DOVE — Frame and trajectory loading from video files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("cv2 not available — FrameLoader / TrajectoryLoader will yield empty data")

from .segmentation import SilhouetteExtractor
from .trajectory import TrajectoryModel


class FrameLoader:
    """
    Iterates over frames in a video file.

    Usage
    -----
    for frame_idx, frame_bgr in FrameLoader(path):
        ...
    """

    def __init__(self, path: str | Path, max_frames: int = 0):
        self.path = str(path)
        self.max_frames = max_frames  # 0 = all frames

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        if not _CV2_AVAILABLE:
            logger.error("cv2 not available")
            return

        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", self.path)
            cap.release()
            return

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame
            frame_idx += 1
            if self.max_frames and frame_idx >= self.max_frames:
                break
        cap.release()


class TrajectoryLoader:
    """
    Extracts a full trajectory from a video by applying SilhouetteExtractor
    per frame and accumulating centroids in a TrajectoryModel.

    Returns the trajectory as (T, 2) array and the list of bboxes.
    """

    def __init__(
        self,
        warmup_frames: int = 30,
        window_size: int = 64,
    ):
        self.warmup_frames = warmup_frames
        self.window_size = window_size

    def load(self, path: str | Path) -> Tuple[np.ndarray, list]:
        """
        Parameters
        ----------
        path : video file path

        Returns
        -------
        trajectory : np.ndarray, shape (T, 2)
        bboxes     : list of (x, y, w, h) tuples  (None entries removed)
        """
        extractor = SilhouetteExtractor(warmup_frames=self.warmup_frames)
        model = TrajectoryModel(window_size=self.window_size)
        bboxes = []

        for _, frame in FrameLoader(path):
            result = extractor.extract(frame)
            centroid = result.get("centroid")
            bbox = result.get("bbox")
            if centroid is not None:
                model.add_point(centroid[0], centroid[1])
                bboxes.append(bbox)

        return model.get_full(), bboxes


if __name__ == "__main__":
    import numpy as np
    import tempfile, os

    np.random.seed(42)

    # Synthetic: create a tiny video file if cv2 available
    if _CV2_AVAILABLE:
        import cv2, tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
        tmp.close()
        writer = cv2.VideoWriter(tmp.name, cv2.VideoWriter_fourcc(*"XVID"), 10, (160, 120))
        for i in range(60):
            frame = np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        fl = FrameLoader(tmp.name, max_frames=10)
        frames = list(fl)
        print(f"FrameLoader yielded {len(frames)} frames, shape={frames[0][1].shape}")

        tl = TrajectoryLoader(warmup_frames=5)
        traj, bboxes = tl.load(tmp.name)
        print(f"TrajectoryLoader: trajectory shape={traj.shape}, bboxes={len(bboxes)}")
        os.unlink(tmp.name)
    else:
        print("cv2 not available; skipping smoke test")
