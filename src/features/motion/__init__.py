"""
DOVE — Motion feature utilities.

Provides `compute_statistics` used by all motion feature extractors.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis, entropy as scipy_entropy


def compute_statistics(signal: np.ndarray) -> np.ndarray:
    """
    Compute 10 descriptive statistics for a 1-D signal.

    Returns
    -------
    np.ndarray, shape (10,)
        [mean, std, skewness, kurtosis, entropy,
         min, max, local_minima_count, local_maxima_count, zero_crossings_count]
    """
    s = np.asarray(signal, dtype=float).flatten()
    if len(s) == 0:
        return np.zeros(10)
    mean = np.mean(s)
    std = np.std(s) + 1e-8
    sk = skew(s) if len(s) > 1 else 0.0
    kurt = kurtosis(s) if len(s) > 1 else 0.0
    # entropy on histogram
    hist, _ = np.histogram(s, bins=max(10, len(s) // 5), density=True)
    hist = hist + 1e-10
    ent = scipy_entropy(hist)
    mn = np.min(s)
    mx = np.max(s)
    # local minima/maxima (sign changes in diff)
    d = np.diff(s)
    lmin = np.sum((d[:-1] < 0) & (d[1:] > 0))
    lmax = np.sum((d[:-1] > 0) & (d[1:] < 0))
    zc = np.sum(np.diff(np.sign(s)) != 0)
    return np.array(
        [mean, std, sk, kurt, ent, mn, mx, float(lmin), float(lmax), float(zc)]
    )


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    sig = np.random.randn(128)
    stats = compute_statistics(sig)
    print("compute_statistics output shape:", stats.shape)
    print("values:", stats)
