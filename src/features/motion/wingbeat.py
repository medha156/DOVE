"""
DOVE — Wingbeat frequency features from bounding-box size oscillations.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WingbeatFeatures:
    """
    Extracts 27 wingbeat features from a sequence of bounding boxes.

    Each bbox is a dict (or tuple) with keys/indices: x, y, w, h  (or x,y,w,h).
    Three signals are derived: height, width, diagonal.
    Each signal is zero-padded to 128, FFT'd, and the 9 dominant frequencies kept.
    Total: 3 × 9 = 27.
    """

    _N_FFT = 128
    _N_DOMINANT = 9

    def _parse_bboxes(self, bbox_sequence: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        heights, widths, diags = [], [], []
        for bb in bbox_sequence:
            if isinstance(bb, dict):
                w = float(bb.get("width", bb.get("w", 0)))
                h = float(bb.get("height", bb.get("h", 0)))
            elif hasattr(bb, "__len__") and len(bb) >= 4:
                # (x, y, w, h)
                w, h = float(bb[2]), float(bb[3])
            else:
                w, h = 0.0, 0.0
            diag = float(np.hypot(w, h))
            heights.append(h)
            widths.append(w)
            diags.append(diag)
        return (
            np.array(heights, dtype=np.float32),
            np.array(widths, dtype=np.float32),
            np.array(diags, dtype=np.float32),
        )

    def _dominant_frequencies(self, signal: np.ndarray) -> np.ndarray:
        """Zero-pad to N_FFT, FFT, return magnitudes of 9 dominant frequencies."""
        n = self._N_FFT
        padded = np.zeros(n, dtype=np.float32)
        L = min(len(signal), n)
        padded[:L] = signal[:L]
        spectrum = np.abs(np.fft.rfft(padded))
        # Only positive frequencies (rfft gives n//2+1 values)
        # Pick top N_DOMINANT indices
        if len(spectrum) < self._N_DOMINANT:
            spectrum = np.pad(spectrum, (0, self._N_DOMINANT - len(spectrum)))
        top_idx = np.argsort(spectrum)[::-1][: self._N_DOMINANT]
        return spectrum[np.sort(top_idx)]

    def extract(self, bbox_sequence: list) -> np.ndarray:
        """
        Parameters
        ----------
        bbox_sequence : list of dicts or (x,y,w,h) tuples

        Returns
        -------
        np.ndarray, shape (27,)
        """
        if not bbox_sequence:
            return np.zeros(27, dtype=np.float32)

        heights, widths, diags = self._parse_bboxes(bbox_sequence)
        f_h = self._dominant_frequencies(heights)
        f_w = self._dominant_frequencies(widths)
        f_d = self._dominant_frequencies(diags)

        feats = np.concatenate([f_h, f_w, f_d])  # 9+9+9 = 27
        return feats.astype(np.float32)


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    T = 64
    bboxes = [{"w": 40 + 10 * np.sin(2 * np.pi * i / 20), "h": 30 + 5 * np.sin(2 * np.pi * i / 10)} for i in range(T)]

    wb = WingbeatFeatures()
    feats = wb.extract(bboxes)
    print("Wingbeat features shape:", feats.shape)  # (27,)
    print("First few values:", feats[:5])
