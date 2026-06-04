"""
DOVE — Feature stream ablation: backbone-only vs invariant-only vs both.

Uses best config: efficientnet_b3 + cross_attention + linear, 20 epochs.
Compares three conditions:
  backbone_only   — backbone (EfficientNet-B3, 1536-d) features, zero invariant
  invariant_only  — Swin-T HSFA (1440-d) features, zero backbone
  both            — backbone + invariant (correct full model)

Results → results/tables/feature_ablation_results.csv
          results/figures/feature_ablation_bar.png
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import DOVEDataset
from data.augment import get_image_transform
from evaluation.metrics import compute_accuracy, compute_top3_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("run_feature_ablations")

REPO_ROOT   = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "tables"
FIGS_DIR    = REPO_ROOT / "results" / "figures"
CKPT_DIR    = REPO_ROOT / "results" / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 20
N_EPOCHS    = 20


class _FeatureAblationModel(nn.Module):
    """
    EfficientNet-B3 backbone + Swin-T InvariantFeaturePipeline + CrossAttentionFusion.
    use_backbone / use_invariant flags zero out each stream for ablation.
    """
    def __init__(self, backbone, inv_pipeline, fusion, head,
                 use_backbone: bool = True, use_invariant: bool = True):
        super().__init__()
        self.backbone      = backbone
        self.inv_pipeline  = inv_pipeline
        self.fusion        = fusion
        self.head          = head
        self.use_backbone  = use_backbone
        self.use_invariant = use_invariant

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        backbone_feat = self.backbone(images)           # (B, 1536)
        if backbone_feat.dim() > 2:
            backbone_feat = backbone_feat.mean(dim=[-2, -1])

        inv_feat = self.inv_pipeline.extract(images)   # (B, 1440)

        if not self.use_backbone:
            backbone_feat = torch.zeros_like(backbone_feat)
        if not self.use_invariant:
            inv_feat = torch.zeros_like(inv_feat)

        fused = self.fusion(backbone_feat, inv_feat)   # (B, 512)
        return self.head(fused)


def build_model(use_backbone: bool, use_invariant: bool, device: torch.device) -> nn.Module:
    from backbones.efficientnet_b3 import EfficientNetB3Backbone
    from features.invariant.pipeline import InvariantFeaturePipeline
    from fusion.cross_attention import CrossAttentionFusion
    from heads.linear_head import LinearHead

    backbone     = EfficientNetB3Backbone(pretrained=True).to(device)
    inv_pipeline = InvariantFeaturePipeline(pretrained=True).to(device)
    fusion       = CrossAttentionFusion(motion_dim=backbone.feature_dim).to(device)
    head         = LinearHead(in_dim=512, num_classes=NUM_CLASSES).to(device)

    return _FeatureAblationModel(
        backbone, inv_pipeline, fusion, head,
        use_backbone=use_backbone, use_invariant=use_invariant,
    ).to(device)


def make_loaders(batch_size: int = 32):
    train_csv = REPO_ROOT / "data" / "splits" / "train.csv"
    val_csv   = REPO_ROOT / "data" / "splits" / "val.csv"
    test_csv  = REPO_ROOT / "data" / "splits" / "test.csv"

    train_ds = DOVEDataset(train_csv, transform=get_image_transform(train=True))
    val_ds   = DOVEDataset(val_csv,   transform=get_image_transform(train=False))
    test_ds  = DOVEDataset(test_csv,  transform=get_image_transform(train=False))

    weights_path = REPO_ROOT / "data" / "splits" / "class_weights.npy"
    if weights_path.exists():
        class_weights = np.load(weights_path)
        sample_weights = np.array(
            [class_weights[int(train_ds.df.iloc[i]["species_id"])] for i in range(len(train_ds))],
            dtype=np.float32,
        )
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, device, exp_name: str):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0
    ckpt_path    = CKPT_DIR / f"feat_abl_{exp_name}_best.pt"

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        correct, total = 0, 0
        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += imgs.size(0)
        scheduler.step()

        val_acc = _eval(model, val_loader, device)
        logger.info("[%s] epoch %2d/%d  train=%.3f  val=%.3f",
                    exp_name, epoch, N_EPOCHS, correct / max(total, 1), val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return best_val_acc


def _eval(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
                logits = model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += imgs.size(0)
    return correct / max(total, 1)


def _eval_top3(model, loader, device) -> float:
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
                logits = model(imgs).cpu()
            all_logits.append(logits)
            all_labels.append(batch["label"])
    return float(compute_top3_accuracy(torch.cat(all_logits), torch.cat(all_labels)))


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    train_loader, val_loader, test_loader = make_loaders(batch_size=32)

    experiments = [
        ("backbone_only",  True,  False, "EfficientNet-B3 features only (zero invariant)"),
        ("invariant_only", False, True,  "Swin-T HSFA invariant features only (zero backbone)"),
        ("both",           True,  True,  "Both backbone + invariant features (full model)"),
    ]

    rows = []
    # baseline reference from prior ablation run
    baseline_acc = 0.9409  # efficientnet_b3_cross_attention_linear best result

    for exp_name, use_bb, use_inv, desc in experiments:
        logger.info("=" * 60)
        logger.info("Experiment: %s", exp_name)
        logger.info("  use_backbone=%s  use_invariant=%s", use_bb, use_inv)
        logger.info("=" * 60)

        model = build_model(use_bb, use_inv, device)
        best_val = train(model, train_loader, val_loader, device, exp_name)
        test_acc = _eval(model, test_loader, device)
        top3_acc = _eval_top3(model, test_loader, device)

        logger.info("[%s] DONE  test_acc=%.4f  top3=%.4f", exp_name, test_acc, top3_acc)
        rows.append({
            "name":         exp_name,
            "description":  desc,
            "use_backbone": use_bb,
            "use_invariant": use_inv,
            "val_acc":      best_val,
            "test_acc":     test_acc,
            "test_top3":    top3_acc,
            "delta_vs_both": None,  # filled below
        })

    # compute delta vs "both"
    both_acc = next(r["test_acc"] for r in rows if r["name"] == "both")
    for r in rows:
        r["delta_vs_both"] = r["test_acc"] - both_acc

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "feature_ablation_results.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Results saved to %s", out_csv)
    print(df[["name", "test_acc", "test_top3", "delta_vs_both"]].to_string(index=False))

    # Bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["#4C72B0", "#55A868", "#C44E52"]
        bars = ax.bar(df["name"], df["test_acc"], color=colors, width=0.5)
        ax.set_ylim(0.85, 1.0)
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Feature Stream Ablation\n(EfficientNet-B3 + CrossAttention + Linear, 20 epochs)")
        for bar, acc in zip(bars, df["test_acc"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{acc:.4f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        fig_path = FIGS_DIR / "feature_ablation_bar.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        logger.info("Bar chart saved to %s", fig_path)
    except Exception as e:
        logger.warning("Plot failed: %s", e)


if __name__ == "__main__":
    main()
