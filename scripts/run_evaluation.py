"""
DOVE — Comprehensive evaluation script for all 16 neural experiments
(4 backbones × 2 fusions × 2 heads).

Computes per-experiment:
  1. Full test-set inference  (logits, preds, labels, 512-d embeddings, images, modalities, attn)
  2. Per-class metrics CSV + full_metrics.csv + normalised confusion matrix PNG
  3. ScoreCAM / GradCAM saliency map grids
  4. t-SNE embedding scatter plots
  5. Common qualitative errors grid
  6. Attention-weight bar chart (cross_attention only)
  7. FLOPs / params / timing CSV
  8. Pareto-optimal accuracy-vs-FLOPs scatter (post all experiments)
"""
from __future__ import annotations

import importlib
import itertools
import logging
import os
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import DOVEDataset
from data.augment import get_image_transform

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger("run_evaluation")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "tables"
FIGS_DIR    = REPO_ROOT / "results" / "figures"
CKPT_DIR    = REPO_ROOT / "results" / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

NUM_CLASSES = 20
SPECIES_NAMES = [
    "Acorn Woodpecker", "American Crow", "American Robin", "Anna's Hummingbird",
    "Black Phoebe", "Brewer's Blackbird", "Bushtit", "California Scrub-Jay",
    "California Towhee", "Chestnut-backed Chickadee", "Cooper's Hawk",
    "Dark-eyed Junco", "House Finch", "Lesser Goldfinch", "Mourning Dove",
    "Northern Mockingbird", "Oak Titmouse", "Red-tailed Hawk",
    "White-crowned Sparrow", "Yellow-rumped Warbler",
]
SHORT_NAMES = [s.split()[-1] for s in SPECIES_NAMES]

BACKBONES    = ["swin_t", "efficientnet_b3", "mobilenet", "vgg19"]
FUSIONS      = ["cross_attention", "concat"]
NEURAL_HEADS = ["mlp", "linear"]

BACKBONE_COLORS = {
    "swin_t": "#1f77b4",
    "efficientnet_b3": "#ff7f0e",
    "mobilenet": "#2ca02c",
    "vgg19": "#d62728",
}

INFERENCE_BATCH = 32

# ── Model definitions (duplicated minimal — do not import run_experiments.py) ──

def build_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    bb_name = config["backbone"]
    fu_name = config["fusion"]
    hd_name = config["head"]

    bb_mod = importlib.import_module(f"backbones.{bb_name}")
    backbone_cls = getattr(
        bb_mod,
        next(x for x in dir(bb_mod) if "Backbone" in x),
    )
    backbone = backbone_cls().to(device)

    from features.invariant.pipeline import InvariantFeaturePipeline
    inv_pipeline = InvariantFeaturePipeline().to(device)

    if fu_name == "cross_attention":
        from fusion.cross_attention import CrossAttentionFusion
        fusion = CrossAttentionFusion().to(device)
    else:
        from fusion.concat import ConcatFusion
        fusion = ConcatFusion().to(device)

    if hd_name == "mlp":
        from heads.mlp_head import MLPHead
        head = MLPHead(in_dim=512, num_classes=NUM_CLASSES).to(device)
    else:
        from heads.linear_head import LinearHead
        head = LinearHead(in_dim=512, num_classes=NUM_CLASSES).to(device)

    return _NeuralModel(backbone, inv_pipeline, fusion, head).to(device)


class _NeuralModel(nn.Module):
    def __init__(self, backbone, inv_pipeline, fusion, head):
        super().__init__()
        self.backbone     = backbone
        self.inv_pipeline = inv_pipeline
        self.fusion       = fusion
        self.head         = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(images)
        if feat_map.dim() > 2:
            feat_map = feat_map.mean(dim=[-2, -1])
        inv_feat     = self.inv_pipeline.extract(images)          # (B, 1440)
        motion_zero  = torch.zeros(images.size(0), 139, device=images.device)
        fused        = self.fusion(motion_zero, inv_feat)         # (B, 512)
        return self.head(fused)                                    # (B, 20)

    def forward_with_embedding(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits, fused_embedding)."""
        feat_map = self.backbone(images)
        if feat_map.dim() > 2:
            feat_map = feat_map.mean(dim=[-2, -1])
        inv_feat     = self.inv_pipeline.extract(images)
        motion_zero  = torch.zeros(images.size(0), 139, device=images.device)
        fused        = self.fusion(motion_zero, inv_feat)
        logits       = self.head(fused)
        return logits, fused


# ── Data ──────────────────────────────────────────────────────────────────────

def make_test_loader(batch_size: int = INFERENCE_BATCH) -> DataLoader:
    test_csv = REPO_ROOT / "data" / "splits" / "test.csv"
    test_ds  = DOVEDataset(test_csv, transform=get_image_transform(train=False))
    return DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )


# ── Inference pass ────────────────────────────────────────────────────────────

def run_inference(
    model: _NeuralModel,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Single pass over the test set collecting:
      all_logits       (N, 20) float32 cpu
      all_preds        (N,)    int64   cpu
      all_labels       (N,)    int64   cpu
      all_embeddings   (N, 512) float32 cpu
      all_images       (N, 3, 224, 224) float32 cpu
      all_modalities   list[str]
      all_attn_weights (N, 8, 1, 1) or None
    """
    model.eval()
    all_logits, all_preds, all_labels = [], [], []
    all_embeddings, all_images, all_modalities = [], [], []
    all_attn_weights = []
    has_attn = hasattr(model.fusion, "last_attn_weights")

    with torch.no_grad():
        for batch in loader:
            imgs      = batch["image"].to(device)
            labels    = batch["label"]
            modalities = batch["modality"]

            logits, embs = model.forward_with_embedding(imgs)
            logits = logits.float().cpu()
            embs   = embs.float().cpu()
            preds  = logits.argmax(dim=1)

            all_logits.append(logits)
            all_preds.append(preds)
            all_labels.append(labels)
            all_embeddings.append(embs)
            all_images.append(imgs.float().cpu())
            all_modalities.extend(list(modalities))

            if has_attn and model.fusion.last_attn_weights is not None:
                all_attn_weights.append(model.fusion.last_attn_weights.float().cpu())

    result = {
        "all_logits"     : torch.cat(all_logits,    dim=0),
        "all_preds"      : torch.cat(all_preds,     dim=0),
        "all_labels"     : torch.cat(all_labels,    dim=0),
        "all_embeddings" : torch.cat(all_embeddings, dim=0),
        "all_images"     : torch.cat(all_images,    dim=0),
        "all_modalities" : all_modalities,
    }
    if all_attn_weights:
        result["all_attn_weights"] = torch.cat(all_attn_weights, dim=0)
    else:
        result["all_attn_weights"] = None
    return result


# ── 2. Metrics ────────────────────────────────────────────────────────────────

def compute_and_save_metrics(
    name: str,
    infer: Dict[str, Any],
    full_metrics_rows: List[Dict],
) -> None:
    from sklearn.metrics import classification_report, precision_recall_fscore_support

    preds  = infer["all_preds"].numpy()
    labels = infer["all_labels"].numpy()
    logits = infer["all_logits"]

    # Top-1
    top1_acc = float((preds == labels).mean())

    # Top-3
    top3 = 0
    for i in range(len(labels)):
        top3_preds = logits[i].topk(3).indices.numpy()
        if labels[i] in top3_preds:
            top3 += 1
    top3_acc = top3 / max(len(labels), 1)

    # Per-class
    prec, rec, f1, supp = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    per_class_df = pd.DataFrame({
        "species"   : SPECIES_NAMES,
        "precision" : prec,
        "recall"    : rec,
        "f1"        : f1,
        "support"   : supp,
    })
    per_class_df.to_csv(RESULTS_DIR / f"per_class_{name}.csv", index=False)

    macro_f1   = float(f1.mean())
    macro_prec = float(prec.mean())
    macro_rec  = float(rec.mean())

    full_metrics_rows.append({
        "name"            : name,
        "top1_acc"        : top1_acc,
        "top3_acc"        : top3_acc,
        "macro_f1"        : macro_f1,
        "macro_precision" : macro_prec,
        "macro_recall"    : macro_rec,
    })

    logger.info("[%s] top1=%.4f  top3=%.4f  macro_f1=%.4f", name, top1_acc, top3_acc, macro_f1)

    # Normalised confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm.astype(float), row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=SHORT_NAMES,
        yticklabels=SHORT_NAMES,
        vmin=0, vmax=1,
        linewidths=0.3,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Normalised Confusion Matrix — {name}", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"confmat_norm_{name}.png", dpi=120)
    plt.close(fig)


# ── 3. ScoreCAM / GradCAM saliency ───────────────────────────────────────────

def _denorm_image(tensor: torch.Tensor) -> np.ndarray:
    """tensor: (3, H, W) float32 normalised → (H, W, 3) uint8."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = tensor.numpy().transpose(1, 2, 0)
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _add_border(ax, color: str, lw: float = 4.0) -> None:
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


def compute_and_save_saliency(
    name: str,
    model: _NeuralModel,
    infer: Dict[str, Any],
    device: torch.device,
    config: Dict[str, Any],
) -> None:
    from pytorch_grad_cam import ScoreCAM, GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    preds  = infer["all_preds"].numpy()
    labels = infer["all_labels"].numpy()
    images = infer["all_images"]   # (N, 3, 224, 224)

    correct_mask   = preds == labels
    incorrect_mask = preds != labels

    # Collect up to 5 correct + 5 incorrect per species
    correct_by_sp   = defaultdict(list)
    incorrect_by_sp = defaultdict(list)
    for idx in range(len(labels)):
        sp = int(labels[idx])
        if correct_mask[idx] and len(correct_by_sp[sp]) < 5:
            correct_by_sp[sp].append(idx)
        elif incorrect_mask[idx] and len(incorrect_by_sp[sp]) < 5:
            incorrect_by_sp[sp].append(idx)

    # Pick species that have at least 1 of each type
    species_with_both = sorted(
        sp for sp in range(NUM_CLASSES)
        if correct_by_sp[sp] and incorrect_by_sp[sp]
    )
    if not species_with_both:
        logger.warning("[%s] No species with both correct and incorrect examples — skipping saliency", name)
        return

    # Cap at 20 rows (all species), but ensure grid isn't > 200 images
    max_species = min(len(species_with_both), 20)
    if max_species * 10 > 200:
        max_species = 200 // 10
    species_to_show = species_with_both[:max_species]

    # Determine CAM method and target layer
    backbone_name = config["backbone"]
    has_conv2d = any(isinstance(m, nn.Conv2d) for m in model.backbone.modules())

    # Gather all unique indices we need
    all_needed_idxs = []
    for sp in species_to_show:
        all_needed_idxs.extend(correct_by_sp[sp][:5])
        all_needed_idxs.extend(incorrect_by_sp[sp][:5])
    all_needed_idxs = sorted(set(all_needed_idxs))

    # Build CAM on subsets
    cam_cache: Dict[int, np.ndarray] = {}

    model.eval()
    if has_conv2d:
        # ScoreCAM: last Conv2d in backbone
        last_conv = None
        for m in model.backbone.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            logger.warning("[%s] No Conv2d found — falling back to GradCAM on inv_pipeline", name)
            has_conv2d = False
        else:
            target_layers = [last_conv]
            cam_cls = ScoreCAM

    if not has_conv2d:
        # GradCAM on last Swin stage (layers_3 of FeatureListNet)
        try:
            target_layers = [model.inv_pipeline.backbone.backbone.layers_3]
        except AttributeError:
            logger.warning("[%s] Cannot access layers_3 — skipping saliency", name)
            return
        cam_cls = GradCAM

    try:
        cam_obj = cam_cls(model=model, target_layers=target_layers)

        BATCH = 8
        for batch_start in range(0, len(all_needed_idxs), BATCH):
            batch_idxs = all_needed_idxs[batch_start: batch_start + BATCH]
            batch_imgs = images[batch_idxs].to(device)  # (b, 3, 224, 224)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grayscale_cams = cam_obj(input_tensor=batch_imgs)  # (b, 224, 224)
            for local_i, global_idx in enumerate(batch_idxs):
                cam_cache[global_idx] = grayscale_cams[local_i]
    except Exception as exc:
        logger.warning("[%s] CAM computation failed: %s — skipping saliency", name, exc)
        return

    # Build figure: rows=species, cols alternating [orig | cam] for correct, then incorrect
    n_rows = len(species_to_show)
    n_cols = 20  # 5 correct pairs + 5 incorrect pairs = 20 axes per row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.8 + 0.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Saliency Maps — {name}", fontsize=14, y=1.01)

    for row_i, sp in enumerate(species_to_show):
        correct_list   = correct_by_sp[sp][:5]
        incorrect_list = incorrect_by_sp[sp][:5]

        col = 0
        for sample_list, color in [(correct_list, "green"), (incorrect_list, "red")]:
            for idx in sample_list:
                orig_np = _denorm_image(images[idx])  # (H, W, 3) uint8

                ax_orig = axes[row_i, col]
                ax_cam  = axes[row_i, col + 1] if col + 1 < n_cols else None

                ax_orig.imshow(orig_np)
                ax_orig.axis("off")
                _add_border(ax_orig, color)

                if ax_cam is not None and idx in cam_cache:
                    cam_img = show_cam_on_image(orig_np.astype(np.float32) / 255.0, cam_cache[idx])
                    ax_cam.imshow(cam_img)
                    ax_cam.axis("off")
                    _add_border(ax_cam, color)

                col += 2

            # Pad remaining slots with blank axes
            while col < (10 if color == "green" else 20):
                if col < n_cols:
                    axes[row_i, col].axis("off")
                col += 2 if col % 2 == 0 else 1

        # Row label
        axes[row_i, 0].set_ylabel(SHORT_NAMES[sp], rotation=0, labelpad=50,
                                   va="center", fontsize=7)

    # Column headers
    axes[0, 0].set_title("Correct", fontsize=8)
    if n_cols > 10:
        axes[0, 10].set_title("Incorrect", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"scorecam_{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("[%s] Saliency map saved", name)


# ── 4. t-SNE ──────────────────────────────────────────────────────────────────

def compute_and_save_tsne(name: str, infer: Dict[str, Any]) -> None:
    from sklearn.manifold import TSNE

    embeddings  = infer["all_embeddings"].numpy()   # (N, 512)
    labels      = infer["all_labels"].numpy()
    modalities  = infer["all_modalities"]

    logger.info("[%s] Running t-SNE on %d embeddings…", name, len(embeddings))
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings)          # (N, 2)

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(12, 9))

    for sp in range(NUM_CLASSES):
        mask = labels == sp
        if not mask.any():
            continue
        for mod, marker in [("image", "o"), ("video_frame", "^")]:
            mod_mask = np.array(modalities) == mod
            sel = mask & mod_mask
            if sel.any():
                ax.scatter(
                    coords[sel, 0], coords[sel, 1],
                    c=[cmap(sp / 20.0)],
                    marker=marker,
                    s=18,
                    alpha=0.7,
                    label=f"{SHORT_NAMES[sp]}/{mod}" if mod == "image" else None,
                )

    handles = [
        mpatches.Patch(color=cmap(sp / 20.0), label=SHORT_NAMES[sp])
        for sp in range(NUM_CLASSES)
    ]
    legend1 = ax.legend(handles=handles, loc="upper right", fontsize=6,
                        ncol=2, title="Species")
    ax.add_artist(legend1)

    marker_handles = [
        plt.Line2D([0], [0], marker="o", color="gray", linestyle="None", label="image"),
        plt.Line2D([0], [0], marker="^", color="gray", linestyle="None", label="video_frame"),
    ]
    ax.legend(handles=marker_handles, loc="upper left", fontsize=8, title="Modality")

    ax.set_title(f"t-SNE Embeddings (512-d fused) — {name}", fontsize=13)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"tsne_{name}.png", dpi=120)
    plt.close(fig)
    logger.info("[%s] t-SNE saved", name)


# ── 5. Error grid ─────────────────────────────────────────────────────────────

def compute_and_save_error_grid(name: str, infer: Dict[str, Any]) -> None:
    preds  = infer["all_preds"].numpy()
    labels = infer["all_labels"].numpy()
    images = infer["all_images"]

    # Count misclassification pairs
    error_pairs = Counter(
        (int(labels[i]), int(preds[i]))
        for i in range(len(labels))
        if preds[i] != labels[i]
    )
    top10_pairs = error_pairs.most_common(10)
    if not top10_pairs:
        logger.info("[%s] No misclassifications — skipping error grid", name)
        return

    # Collect up to 9 images per pair
    pair_images: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i in range(len(labels)):
        pair = (int(labels[i]), int(preds[i]))
        if pair in dict(top10_pairs) and len(pair_images[pair]) < 9:
            pair_images[pair].append(i)

    n_pairs  = len(top10_pairs)
    n_cols   = 9
    fig, axes = plt.subplots(n_pairs, n_cols, figsize=(n_cols * 1.8, n_pairs * 2.0 + 0.5))
    if n_pairs == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(f"Top-10 Misclassification Pairs — {name}", fontsize=14)

    for row_i, (pair, count) in enumerate(top10_pairs):
        true_sp, pred_sp = pair
        idxs = pair_images[pair]
        for col_i in range(n_cols):
            ax = axes[row_i, col_i]
            if col_i < len(idxs):
                img_np = _denorm_image(images[idxs[col_i]])
                ax.imshow(img_np)
                if col_i == 0:
                    ax.set_title(
                        f"T:{SHORT_NAMES[true_sp]}\nP:{SHORT_NAMES[pred_sp]}\n×{count}",
                        fontsize=6,
                    )
                else:
                    ax.set_title(
                        f"T:{SHORT_NAMES[true_sp]}\nP:{SHORT_NAMES[pred_sp]}",
                        fontsize=6,
                    )
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"error_grid_{name}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("[%s] Error grid saved", name)


# ── 6. Attention weight heatmap ───────────────────────────────────────────────

def compute_and_save_attn_map(name: str, infer: Dict[str, Any]) -> None:
    attn = infer.get("all_attn_weights")
    if attn is None:
        return

    # attn: (N, 8, 1, 1) → mean over N → (8, 1, 1)
    mean_attn = attn.mean(dim=0).reshape(-1)  # (8,)

    fig, ax = plt.subplots(figsize=(10, 3))
    bars = ax.bar(range(len(mean_attn)), mean_attn.numpy(), color="#1f77b4", edgecolor="black")
    ax.set_xticks(range(len(mean_attn)))
    ax.set_xticklabels([f"Head {i}" for i in range(len(mean_attn))], fontsize=9)
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title(f"Cross-Attention Head Weights (test set mean) — {name}", fontsize=12)
    for bar, val in zip(bars, mean_attn.numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / f"attn_map_{name}.png", dpi=120)
    plt.close(fig)
    logger.info("[%s] Attention map saved", name)


# ── 7. Compute stats ──────────────────────────────────────────────────────────

def compute_and_save_compute_stats(
    name: str,
    model: _NeuralModel,
    device: torch.device,
    compute_rows: List[Dict],
) -> None:
    try:
        from fvcore.nn import FlopCountAnalysis, parameter_count
    except ImportError:
        logger.warning("[%s] fvcore not available — skipping FLOPs", name)
        return

    dummy = torch.randn(1, 3, 224, 224).to(device)
    model.eval()

    # FLOPs
    try:
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        total_flops = int(flops.total())
    except Exception as exc:
        logger.warning("[%s] FLOPs failed: %s", name, exc)
        total_flops = -1

    # Params
    total_params = sum(p.numel() for p in model.parameters())

    # GPU timing (100 passes)
    gpu_ms = float("nan")
    if device.type == "cuda":
        try:
            inp = torch.randn(1, 3, 224, 224).to(device)
            # Warm up
            for _ in range(5):
                with torch.no_grad():
                    model(inp)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                with torch.no_grad():
                    model(inp)
            torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - t0) / 100 * 1000
        except Exception as exc:
            logger.warning("[%s] GPU timing failed: %s", name, exc)

    # CPU timing (10 passes)
    cpu_ms = float("nan")
    try:
        cpu_device = torch.device("cpu")
        cpu_model  = model.cpu()
        inp_cpu    = torch.randn(1, 3, 224, 224)
        # Warm up
        for _ in range(2):
            with torch.no_grad():
                cpu_model(inp_cpu)
        t0 = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                cpu_model(inp_cpu)
        cpu_ms = (time.perf_counter() - t0) / 10 * 1000
        # Move back to original device
        model.to(device)
    except Exception as exc:
        logger.warning("[%s] CPU timing failed: %s", name, exc)
        model.to(device)

    compute_rows.append({
        "name"    : name,
        "flops"   : total_flops,
        "params"  : total_params,
        "gpu_ms"  : gpu_ms,
        "cpu_ms"  : cpu_ms,
    })
    logger.info(
        "[%s] flops=%s  params=%s  gpu_ms=%.1f  cpu_ms=%.1f",
        name,
        f"{total_flops:,}" if total_flops >= 0 else "N/A",
        f"{total_params:,}",
        gpu_ms if not np.isnan(gpu_ms) else -1,
        cpu_ms if not np.isnan(cpu_ms) else -1,
    )


# ── 8. Pareto plot ────────────────────────────────────────────────────────────

def make_pareto_plot() -> None:
    compute_path = RESULTS_DIR / "compute_stats.csv"
    metrics_path = RESULTS_DIR / "full_metrics.csv"

    if not compute_path.exists() or not metrics_path.exists():
        logger.warning("Pareto: missing CSV files — skipping")
        return

    comp_df = pd.read_csv(compute_path)
    metr_df = pd.read_csv(metrics_path)
    df = comp_df.merge(metr_df, on="name", how="inner")

    if df.empty:
        logger.warning("Pareto: merged dataframe empty — skipping")
        return

    df = df[df["flops"] > 0].dropna(subset=["top1_acc", "flops", "params"])
    if df.empty:
        logger.warning("Pareto: no valid rows after filtering — skipping")
        return

    # Identify Pareto-optimal (max top1_acc for ≤ given flops)
    df_sorted = df.sort_values("flops")
    best_acc   = -np.inf
    pareto_mask = []
    for _, row in df_sorted.iterrows():
        if row["top1_acc"] > best_acc:
            best_acc = row["top1_acc"]
            pareto_mask.append(True)
        else:
            pareto_mask.append(False)
    df_sorted["pareto"] = pareto_mask

    # Extract backbone family from name
    def _bb_family(name: str) -> str:
        for bb in BACKBONES:
            if name.startswith(bb):
                return bb
        return "unknown"

    df_sorted["backbone"] = df_sorted["name"].apply(_bb_family)

    fig, ax = plt.subplots(figsize=(11, 7))

    for bb in BACKBONES:
        sub = df_sorted[df_sorted["backbone"] == bb]
        if sub.empty:
            continue
        sizes = np.clip(sub["params"].values / 1e6, 5, 500)
        ax.scatter(
            sub["flops"].values,
            sub["top1_acc"].values * 100,
            s=sizes,
            c=BACKBONE_COLORS.get(bb, "gray"),
            alpha=0.75,
            edgecolors="black",
            linewidths=0.5,
            label=bb,
        )

    # Label Pareto-optimal
    pareto_df = df_sorted[df_sorted["pareto"]]
    for _, row in pareto_df.iterrows():
        ax.annotate(
            row["name"],
            (row["flops"], row["top1_acc"] * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="black",
        )
    # Pareto frontier line
    if len(pareto_df) >= 2:
        ax.plot(
            pareto_df["flops"].values,
            pareto_df["top1_acc"].values * 100,
            "k--",
            lw=1.2,
            alpha=0.6,
            label="Pareto frontier",
        )

    ax.set_xscale("log")
    ax.set_xlabel("FLOPs (log scale)", fontsize=11)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs. FLOPs — Pareto Plot\n(marker size ∝ #params / 1M)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "pareto.png", dpi=150)
    plt.close(fig)
    logger.info("Pareto plot saved → %s", FIGS_DIR / "pareto.png")


# ── Experiment grid ───────────────────────────────────────────────────────────

def build_neural_grid() -> List[Dict[str, Any]]:
    exps = []
    for bb, fu, hd in itertools.product(BACKBONES, FUSIONS, NEURAL_HEADS):
        exps.append({"backbone": bb, "fusion": fu, "head": hd,
                     "name": f"{bb}_{fu}_{hd}"})
    return exps


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    test_loader = make_test_loader(batch_size=INFERENCE_BATCH)
    grid = build_neural_grid()

    full_metrics_rows: List[Dict] = []
    compute_rows: List[Dict] = []

    for exp_i, config in enumerate(grid, 1):
        name = config["name"]
        ckpt = CKPT_DIR / f"{name}_best.pt"

        logger.info("=" * 68)
        logger.info("[%d/%d]  %s", exp_i, len(grid), name)
        logger.info("=" * 68)

        if not ckpt.exists():
            logger.warning("[%s] Checkpoint not found — skipping", name)
            continue

        try:
            model = build_model(config, device)
            state = torch.load(str(ckpt), map_location=device)
            model.load_state_dict(state)
            model.eval()
            logger.info("[%s] Checkpoint loaded (%s)", name, ckpt.name)
        except Exception as exc:
            logger.error("[%s] Failed to build/load model: %s", name, exc, exc_info=True)
            continue

        # ── 1. Inference ──────────────────────────────────────────────────
        try:
            logger.info("[%s] Running inference…", name)
            infer = run_inference(model, test_loader, device)
            logger.info(
                "[%s] Inference done: N=%d", name, len(infer["all_labels"])
            )
        except Exception as exc:
            logger.error("[%s] Inference failed: %s", name, exc, exc_info=True)
            continue

        # ── 2. Metrics ────────────────────────────────────────────────────
        try:
            compute_and_save_metrics(name, infer, full_metrics_rows)
        except Exception as exc:
            logger.error("[%s] Metrics failed: %s", name, exc, exc_info=True)

        # ── 3. Saliency maps ──────────────────────────────────────────────
        try:
            compute_and_save_saliency(name, model, infer, device, config)
        except Exception as exc:
            logger.error("[%s] Saliency failed: %s", name, exc, exc_info=True)

        # ── 4. t-SNE ──────────────────────────────────────────────────────
        try:
            compute_and_save_tsne(name, infer)
        except Exception as exc:
            logger.error("[%s] t-SNE failed: %s", name, exc, exc_info=True)

        # ── 5. Error grid ─────────────────────────────────────────────────
        try:
            compute_and_save_error_grid(name, infer)
        except Exception as exc:
            logger.error("[%s] Error grid failed: %s", name, exc, exc_info=True)

        # ── 6. Attention map ──────────────────────────────────────────────
        try:
            if config["fusion"] == "cross_attention":
                compute_and_save_attn_map(name, infer)
        except Exception as exc:
            logger.error("[%s] Attn map failed: %s", name, exc, exc_info=True)

        # ── 7. Compute stats ──────────────────────────────────────────────
        try:
            compute_and_save_compute_stats(name, model, device, compute_rows)
        except Exception as exc:
            logger.error("[%s] Compute stats failed: %s", name, exc, exc_info=True)

        # Save running CSVs after each experiment
        if full_metrics_rows:
            pd.DataFrame(full_metrics_rows).to_csv(
                RESULTS_DIR / "full_metrics.csv", index=False
            )
        if compute_rows:
            pd.DataFrame(compute_rows).to_csv(
                RESULTS_DIR / "compute_stats.csv", index=False
            )

    # ── 8. Pareto plot (post all experiments) ─────────────────────────────
    try:
        make_pareto_plot()
    except Exception as exc:
        logger.error("Pareto plot failed: %s", exc, exc_info=True)

    logger.info("All done.  Results → %s", RESULTS_DIR)
    logger.info("Figures  → %s", FIGS_DIR)


if __name__ == "__main__":
    main()
