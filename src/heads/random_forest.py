"""
DOVE — Random Forest classification head.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier


def _to_numpy(x) -> np.ndarray:
    if _TORCH_AVAILABLE:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


class RandomForestHead:
    """
    Sklearn RandomForestClassifier wrapper with tensor→numpy conversion.

    Parameters
    ----------
    n_estimators : int
    random_state : int
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42, **kwargs):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X, y) -> "RandomForestHead":
        X_np = _to_numpy(X)
        y_np = _to_numpy(y)
        self.clf.fit(X_np, y_np)
        logger.info("RandomForestHead fitted on %d samples", len(X_np))
        return self

    def predict(self, X) -> np.ndarray:
        return self.clf.predict(_to_numpy(X))

    def predict_proba(self, X) -> np.ndarray:
        return self.clf.predict_proba(_to_numpy(X))


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    X_train = np.random.randn(200, 512).astype(np.float32)
    y_train = np.random.randint(0, 20, 200)
    X_test = np.random.randn(40, 512).astype(np.float32)

    head = RandomForestHead(n_estimators=10)
    head.fit(X_train, y_train)
    preds = head.predict(X_test)
    proba = head.predict_proba(X_test)
    print("RandomForestHead preds shape:", preds.shape)
    print("RandomForestHead proba shape:", proba.shape)
