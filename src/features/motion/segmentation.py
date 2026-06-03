"""
DOVE — Silhouette extraction using background subtraction (MOG2).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("cv2 not available — SilhouetteExtractor will return empty results")


class SilhouetteExtractor:
    """
    Background-subtraction silhouette extractor using MOG2.

    Parameters
    ----------
    history       : int  — MOG2 history parameter
    var_threshold : int  — MOG2 varThreshold
    detect_shadows: bool — enable shadow detection
    warmup_frames : int  — frames used only for background model initialisation
    """

    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 16,
        detect_shadows: bool = True,
        warmup_frames: int = 100,
    ):
        self.warmup_frames = warmup_frames
        self._frame_count = 0

        if _CV2_AVAILABLE:
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows,
            )
        else:
            self._subtractor = None

        logger.debug(
            "SilhouetteExtractor init: history=%d, var_threshold=%d, warmup=%d",
            history, var_threshold, warmup_frames,
        )

    def extract(self, frame: np.ndarray) -> Dict:
        """
        Process one BGR frame.

        Returns
        -------
        dict with keys:
            mask     : np.ndarray (H, W) uint8 — foreground mask
            contour  : np.ndarray or None — largest contour points
            bbox     : tuple (x, y, w, h) or None
            centroid : tuple (cx, cy) or None
        """
        empty = {"mask": np.zeros((1, 1), dtype=np.uint8), "contour": None, "bbox": None, "centroid": None}

        if not _CV2_AVAILABLE or self._subtractor is None:
            return empty

        fg_mask = self._subtractor.apply(frame)
        self._frame_count += 1

        if self._frame_count <= self.warmup_frames:
            return {"mask": fg_mask, "contour": None, "bbox": None, "centroid": None}

        # Clean mask: threshold shadows (127 → 0, 255 = foreground)
        _, binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological clean-up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return {"mask": binary, "contour": None, "bbox": None, "centroid": None}

        # Largest contour
        largest = max(contours, key=cv2.contourArea)

        # Bounding box
        x, y, w, h = cv2.boundingRect(largest)

        # Centroid from moments
        M = cv2.moments(largest)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = x + w / 2.0
            cy = y + h / 2.0

        return {
            "mask": binary,
            "contour": largest,
            "bbox": (x, y, w, h),
            "centroid": (cx, cy),
        }

    def reset(self) -> None:
        """Re-initialise the background model."""
        self._frame_count = 0
        if _CV2_AVAILABLE:
            self._subtractor = cv2.createBackgroundSubtractorMOG2()


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)

    extractor = SilhouetteExtractor(warmup_frames=2)

    # Synthesise a few frames
    for i in range(5):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        result = extractor.extract(frame)
        print(
            f"Frame {i}: mask={result['mask'].shape}, "
            f"bbox={result['bbox']}, centroid={result['centroid']}"
        )
