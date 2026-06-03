"""
DOVE — Gaussian Naive Bayes classification head.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.naive_bayes import GaussianNB

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _to_numpy(x) -> np.ndarray:
    if _TORCH_AVAILABLE:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    return np.asarray(x)


class NaiveBayesHead:
    """
    Sklearn GaussianNB wrapper with tensor→numpy conversion.
    """

    def __init__(self, var_smoothing: float = 1e-9):
        self.clf = GaussianNB(var_smoothing=var_smoothing)

    def fit(self, X, y) -> "NaiveBayesHead":
        X_np = _to_numpy(X)
        y_np = _to_numpy(y)
        self.clf.fit(X_np, y_np)
        logger.info("NaiveBayesHead fitted on %d samples", len(X_np))
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

    head = NaiveBayesHead()
    head.fit(X_train, y_train)
    preds = head.predict(X_test)
    proba = head.predict_proba(X_test)
    print("NaiveBayesHead preds shape:", preds.shape)
    print("NaiveBayesHead proba shape:", proba.shape)
