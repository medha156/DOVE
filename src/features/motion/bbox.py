"""
DOVE — Oriented bounding box fitting.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    logger.warning("cv2 not available — fit_oriented_bbox will return zeros")


def fit_oriented_bbox(contour: Optional[np.ndarray]) -> Dict:
    """
    Fit a minimum-area oriented bounding rectangle to a contour.

    Parameters
    ----------
    contour : np.ndarray, shape (N, 1, 2) — OpenCV contour format

    Returns
    -------
    dict:
        center     : (cx, cy)
        width      : float
        height     : float
        angle      : float (degrees)
        hypotenuse : float  (diagonal = sqrt(w^2 + h^2))
        box_points : np.ndarray (4, 2)  — corner points
    """
    if contour is None or not _CV2_AVAILABLE:
        return {
            "center": (0.0, 0.0),
            "width": 0.0,
            "height": 0.0,
            "angle": 0.0,
            "hypotenuse": 0.0,
            "box_points": np.zeros((4, 2), dtype=np.float32),
        }

    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect)

    hypotenuse = math.hypot(float(w), float(h))

    return {
        "center": (float(cx), float(cy)),
        "width": float(w),
        "height": float(h),
        "angle": float(angle),
        "hypotenuse": float(hypotenuse),
        "box_points": box,
    }


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)

    if _CV2_AVAILABLE:
        import cv2

        # Create a synthetic elliptical contour
        img = np.zeros((200, 200), dtype=np.uint8)
        cv2.ellipse(img, (100, 100), (60, 30), 30, 0, 360, 255, -1)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = contours[0]
        result = fit_oriented_bbox(contour)
        print("fit_oriented_bbox result:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    else:
        print("cv2 not available; returning zeros")
        result = fit_oriented_bbox(None)
        for k, v in result.items():
            print(f"  {k}: {v}")
