"""
DOVE — Classification and detection metrics.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _to_numpy(x) -> np.ndarray:
    if _TORCH_AVAILABLE:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_accuracy(preds, labels) -> float:
    """Top-1 accuracy."""
    return float(accuracy_score(_to_numpy(labels), _to_numpy(preds)))


def compute_top3_accuracy(logits, labels) -> float:
    """Top-3 accuracy from logits."""
    logits_np = _to_numpy(logits)
    labels_np = _to_numpy(labels)
    top3 = np.argsort(logits_np, axis=1)[:, -3:]
    correct = sum(labels_np[i] in top3[i] for i in range(len(labels_np)))
    return correct / max(len(labels_np), 1)


def compute_per_class_metrics(
    preds,
    labels,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Per-class precision, recall, F1, support.

    Returns
    -------
    pd.DataFrame with columns: class_id, class_name, precision, recall, f1, support
    """
    preds_np = _to_numpy(preds)
    labels_np = _to_numpy(labels)
    classes = sorted(set(labels_np))

    precision = precision_score(labels_np, preds_np, labels=classes, average=None, zero_division=0)
    recall = recall_score(labels_np, preds_np, labels=classes, average=None, zero_division=0)
    f1 = f1_score(labels_np, preds_np, labels=classes, average=None, zero_division=0)
    support = np.array([(labels_np == c).sum() for c in classes])

    rows = []
    for i, c in enumerate(classes):
        name = class_names[c] if class_names and c < len(class_names) else str(c)
        rows.append({
            "class_id": c,
            "class_name": name,
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i],
        })
    return pd.DataFrame(rows)


def compute_confusion_matrix(preds, labels) -> np.ndarray:
    """Return sklearn confusion matrix."""
    return confusion_matrix(_to_numpy(labels), _to_numpy(preds))


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU for two boxes [x1, y1, x2, y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area + 1e-10
    return inter_area / union


def compute_map_iou(
    pred_boxes: List[np.ndarray],
    gt_boxes: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute mean Average Precision at a given IoU threshold.

    Parameters
    ----------
    pred_boxes : list of (N_i, 4) arrays — predicted boxes per image
    gt_boxes   : list of (M_i, 4) arrays — ground-truth boxes per image

    Returns
    -------
    float — mAP
    """
    precisions = []
    for preds, gts in zip(pred_boxes, gt_boxes):
        if len(gts) == 0:
            continue
        if len(preds) == 0:
            precisions.append(0.0)
            continue
        matched = np.zeros(len(gts), dtype=bool)
        tp = 0
        for pred in preds:
            best_iou = 0.0
            best_j = -1
            for j, gt in enumerate(gts):
                if matched[j]:
                    continue
                iou = _iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_threshold and best_j >= 0:
                tp += 1
                matched[best_j] = True
        precisions.append(tp / max(len(preds), 1))
    return float(np.mean(precisions)) if precisions else 0.0


if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    N = 100
    preds = np.random.randint(0, 20, N)
    labels = np.random.randint(0, 20, N)
    logits = np.random.randn(N, 20)

    print("Accuracy:", compute_accuracy(preds, labels))
    print("Top-3 accuracy:", compute_top3_accuracy(logits, labels))

    df = compute_per_class_metrics(preds, labels)
    print("Per-class metrics (head):")
    print(df.head(3))

    cm = compute_confusion_matrix(preds, labels)
    print("Confusion matrix shape:", cm.shape)

    pred_boxes = [np.array([[10, 10, 50, 50], [60, 60, 100, 100]])]
    gt_boxes = [np.array([[12, 12, 48, 48]])]
    print("mAP@0.5:", compute_map_iou(pred_boxes, gt_boxes))
