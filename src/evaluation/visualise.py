"""
DOVE — Visualisation utilities.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available — plots will be skipped")

try:
    from sklearn.manifold import TSNE
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from pytorch_grad_cam import GradCAM, ScoreCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False
    logger.warning("pytorch_grad_cam not available — ScoreCAM plot will be skipped")


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str | Path,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib not available; skipping confusion matrix plot")
        return
    save_path = _ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def plot_learning_curve(
    log_csv: str | Path,
    save_path: str | Path,
) -> None:
    """Plot train/val loss and accuracy from a training log CSV."""
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib not available; skipping learning curve")
        return
    save_path = _ensure_dir(save_path)
    df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    if "val_loss" in df.columns:
        axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_acc"], label="train")
    axes[1].plot(df["epoch"], df["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Learning curve saved to %s", save_path)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    modalities: Optional[List[str]],
    save_path: str | Path,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """2-D t-SNE scatter of embeddings coloured by class, shaped by modality."""
    if not _MPL_AVAILABLE or not _SKLEARN_AVAILABLE:
        logger.warning("sklearn/matplotlib not available; skipping t-SNE plot")
        return
    save_path = _ensure_dir(save_path)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", len(unique_labels))
    marker_map = {"image": "o", "video": "^", None: "o"}

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        if modalities is not None:
            for mod in set(modalities):
                sub = mask & np.array([m == mod for m in modalities])
                if sub.sum() > 0:
                    ax.scatter(
                        coords[sub, 0], coords[sub, 1],
                        c=[cmap(i)], marker=marker_map.get(mod, "o"),
                        label=f"cls{lab}-{mod}", alpha=0.6, s=20,
                    )
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)], label=f"cls{lab}", alpha=0.6, s=20)

    ax.set_title("t-SNE Embedding Space")
    ax.legend(fontsize=6, ncol=4, loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("t-SNE plot saved to %s", save_path)


def plot_scorecam(
    model,
    images: np.ndarray,
    labels: List[int],
    save_path: str | Path,
    target_layer=None,
) -> None:
    """
    Generate ScoreCAM visualisations for a batch of images.

    Parameters
    ----------
    model       : nn.Module
    images      : np.ndarray (N, H, W, 3) float32, range [0, 1]
    labels      : list of ground-truth class indices
    save_path   : output figure path
    target_layer: model layer to hook (default: last conv layer)
    """
    if not _GRADCAM_AVAILABLE or not _MPL_AVAILABLE:
        logger.warning("pytorch_grad_cam not available; skipping ScoreCAM")
        return

    import torch

    save_path = _ensure_dir(save_path)
    if target_layer is None:
        # Attempt to auto-detect last conv layer
        for layer in reversed(list(model.modules())):
            if isinstance(layer, torch.nn.Conv2d):
                target_layer = layer
                break
    if target_layer is None:
        logger.warning("No Conv2d found; skipping ScoreCAM")
        return

    cam = GradCAM(model=model, target_layers=[target_layer])
    n = min(len(images), 8)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 5))

    for i in range(n):
        img = images[i]
        input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        vis = show_cam_on_image(img, grayscale_cam, use_rgb=True)

        axes[0, i].imshow(img)
        axes[0, i].set_title(f"GT:{labels[i]}")
        axes[0, i].axis("off")
        axes[1, i].imshow(vis)
        axes[1, i].axis("off")

    plt.suptitle("GradCAM Visualisations")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ScoreCAM saved to %s", save_path)


def plot_attention_weights(
    attn_weights: np.ndarray,
    save_path: str | Path,
) -> None:
    """
    Visualise attention weight matrices.

    Parameters
    ----------
    attn_weights : np.ndarray (B, heads, seq_q, seq_k) or (B, seq_q, seq_k)
    save_path    : output path
    """
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib not available; skipping attention plot")
        return
    save_path = _ensure_dir(save_path)
    attn = np.asarray(attn_weights)
    # Average over batch
    if attn.ndim == 4:
        attn = attn.mean(axis=0)  # (heads, sq, sk)
    if attn.ndim == 3:
        n_heads = attn.shape[0]
        fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
        if n_heads == 1:
            axes = [axes]
        for h, ax in enumerate(axes):
            ax.imshow(attn[h], cmap="viridis", aspect="auto")
            ax.set_title(f"Head {h}")
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(attn, cmap="viridis", aspect="auto")
        ax.set_title("Attention Weights")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Attention weights saved to %s", save_path)


def plot_error_examples(
    images: np.ndarray,
    pred_labels: List[int],
    true_labels: List[int],
    save_path: str | Path,
    class_names: Optional[List[str]] = None,
    max_examples: int = 16,
) -> None:
    """
    Display grid of misclassified examples.

    Parameters
    ----------
    images      : np.ndarray (N, H, W, 3) float32 [0,1]
    pred_labels : list of predicted class indices
    true_labels : list of true class indices
    """
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib not available; skipping error examples")
        return
    save_path = _ensure_dir(save_path)
    errors = [i for i, (p, t) in enumerate(zip(pred_labels, true_labels)) if p != t]
    errors = errors[:max_examples]
    if not errors:
        logger.info("No errors to display")
        return

    ncols = 4
    nrows = (len(errors) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax_i, idx in enumerate(errors):
        axes[ax_i].imshow(np.clip(images[idx], 0, 1))
        p_name = class_names[pred_labels[idx]] if class_names else str(pred_labels[idx])
        t_name = class_names[true_labels[idx]] if class_names else str(true_labels[idx])
        axes[ax_i].set_title(f"P:{p_name}\nT:{t_name}", fontsize=8)
        axes[ax_i].axis("off")
    for ax in axes[len(errors):]:
        ax.axis("off")

    plt.suptitle("Error Examples")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Error examples saved to %s", save_path)


if __name__ == "__main__":
    import tempfile, os
    import numpy as np

    np.random.seed(42)

    with tempfile.TemporaryDirectory() as tmp:
        # Confusion matrix
        cm = np.random.randint(0, 50, (5, 5))
        classes = [f"cls{i}" for i in range(5)]
        plot_confusion_matrix(cm, classes, f"{tmp}/cm.png")
        print("Confusion matrix:", os.path.exists(f"{tmp}/cm.png"))

        # Attention weights
        attn = np.random.rand(4, 1, 1)
        plot_attention_weights(attn, f"{tmp}/attn.png")
        print("Attention plot:", os.path.exists(f"{tmp}/attn.png"))

        # Error examples
        imgs = np.random.rand(20, 64, 64, 3).astype(np.float32)
        preds = np.random.randint(0, 5, 20).tolist()
        trues = np.random.randint(0, 5, 20).tolist()
        plot_error_examples(imgs, preds, trues, f"{tmp}/errors.png")
        print("Error examples:", os.path.exists(f"{tmp}/errors.png"))

        # t-SNE
        embs = np.random.randn(50, 32).astype(np.float32)
        labs = np.random.randint(0, 5, 50)
        plot_tsne(embs, labs, None, f"{tmp}/tsne.png")
        print("t-SNE:", os.path.exists(f"{tmp}/tsne.png"))
