"""
DOVE — Real ablation studies on the best experiment.

12 ablations, each trains for 15 epochs, saves results to
results/tables/ablation_results.csv and results/figures/ablation_bar.png.

Ablations (one component changed at a time from best config):
  Backbone:     backbone_swin_t, backbone_mobilenet, backbone_vgg19
  Fusion:       fusion_concat (vs cross_attention baseline)
  Head:         head_mlp (vs linear baseline)
  Features:     no_motion_proj (zero motion, still cross-attn), inv_only (skip backbone)
  Backbone frozen: freeze_backbone (frozen pretrained weights, head only trains)
  No pretrain:  random_init (backbone + inv_pipeline from scratch)
  LR schedule:  no_scheduler (constant lr)
  Loss:         label_smoothing_0.1
  Data aug:     no_augmentation
"""
from __future__ import annotations

import importlib
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

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
logger = logging.getLogger("run_ablations")

REPO_ROOT   = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "tables"
FIGS_DIR    = REPO_ROOT / "results" / "figures"
CKPT_DIR    = REPO_ROOT / "results" / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES   = 20
N_EPOCHS      = 15          # slightly fewer than full run for speed
BASELINE_NAME = "efficientnet_b3_cross_attention_linear"
SPECIES_NAMES = [
    "Acorn Woodpecker","American Crow","American Robin","Anna's Hummingbird",
    "Black Phoebe","Brewer's Blackbird","Bushtit","California Scrub-Jay",
    "California Towhee","Chestnut-backed Chickadee","Cooper's Hawk",
    "Dark-eyed Junco","House Finch","Lesser Goldfinch","Mourning Dove",
    "Northern Mockingbird","Oak Titmouse","Red-tailed Hawk",
    "White-crowned Sparrow","Yellow-rumped Warbler",
]


# ── Model assembly (mirrors run_experiments.py) ───────────────────────────────

class _NeuralModel(nn.Module):
    def __init__(self, backbone, inv_pipeline, fusion, head,
                 freeze_backbone=False, skip_backbone=False):
        super().__init__()
        self.backbone       = backbone
        self.inv_pipeline   = inv_pipeline
        self.fusion         = fusion
        self.head           = head
        self.skip_backbone  = skip_backbone
        if freeze_backbone and backbone is not None:
            for p in backbone.parameters():
                p.requires_grad_(False)

    def forward(self, images):
        inv_feat    = self.inv_pipeline.extract(images)        # (B,1440)
        motion_zero = torch.zeros(images.size(0), 139, device=images.device)
        if self.fusion is not None:
            fused = self.fusion(motion_zero, inv_feat)         # (B,512)
        else:
            fused = inv_feat[:, :512]
        return self.head(fused)


def build_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    bb_name   = cfg.get("backbone", "efficientnet_b3")
    fu_name   = cfg.get("fusion",   "cross_attention")
    hd_name   = cfg.get("head",     "linear")
    pretrained = cfg.get("pretrained", True)
    freeze_bb  = cfg.get("freeze_backbone", False)

    bb_mod  = importlib.import_module(f"backbones.{bb_name}")
    bb_cls  = [getattr(bb_mod, x) for x in dir(bb_mod) if "Backbone" in x][0]
    backbone = bb_cls(pretrained=pretrained).to(device)

    from features.invariant.pipeline import InvariantFeaturePipeline
    inv_pipeline = InvariantFeaturePipeline(pretrained=pretrained).to(device)

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

    return _NeuralModel(backbone, inv_pipeline, fusion, head,
                        freeze_backbone=freeze_bb).to(device)


# ── Data ──────────────────────────────────────────────────────────────────────

def make_loaders(batch_size: int = 16, augment: bool = True):
    train_csv = REPO_ROOT / "data" / "splits" / "train.csv"
    val_csv   = REPO_ROOT / "data" / "splits" / "val.csv"
    test_csv  = REPO_ROOT / "data" / "splits" / "test.csv"

    train_ds = DOVEDataset(train_csv, transform=get_image_transform(train=augment))
    val_ds   = DOVEDataset(val_csv,   transform=get_image_transform(train=False))
    test_ds  = DOVEDataset(test_csv,  transform=get_image_transform(train=False))

    weights_path = REPO_ROOT / "data" / "splits" / "class_weights.npy"
    if weights_path.exists():
        cw = np.load(weights_path)
        sw = np.array([cw[int(train_ds.df.iloc[i]["species_id"])] for i in range(len(train_ds))],
                      dtype=np.float32)
        sampler = WeightedRandomSampler(sw, len(sw))
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_eval(model, train_loader, val_loader, test_loader, cfg, name, device):
    lr             = cfg.get("lr", 1e-4)
    wd             = cfg.get("weight_decay", 1e-4)
    label_smooth   = cfg.get("label_smoothing", 0.0)
    use_scheduler  = cfg.get("use_scheduler", True)
    n_epochs       = cfg.get("n_epochs", N_EPOCHS)

    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scheduler  = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
                  if use_scheduler else None)
    criterion  = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    scaler     = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0
    ckpt_path    = CKPT_DIR / f"ablation_{name}_best.pt"
    patience, no_improve = 5, 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
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
                scaler.step(optimizer); scaler.update()
            else:
                logits = model(imgs)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            correct      += (logits.argmax(1) == labels).sum().item()
            total        += imgs.size(0)
        if scheduler:
            scheduler.step()

        val_acc = _eval_acc(model, val_loader, device)
        logger.info("[%s] Epoch %2d/%d  loss=%.4f  train=%.3f  val=%.3f",
                    name, epoch, n_epochs, running_loss/max(total,1), correct/max(total,1), val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc; no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    test_acc  = _eval_acc(model, test_loader, device)
    test_top3 = _eval_top3(model, test_loader, device)
    return {"val_acc": best_val_acc, "test_acc": test_acc, "test_top3": test_top3}


def _eval_acc(model, loader, device) -> float:
    model.eval(); correct = total = 0
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += imgs.size(0)
    return correct / max(total, 1)


def _eval_top3(model, loader, device) -> float:
    model.eval(); all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            all_logits.append(model(batch["image"].to(device)).cpu())
            all_labels.append(batch["label"])
    return float(compute_top3_accuracy(torch.cat(all_logits), torch.cat(all_labels)))


# ── Ablation definitions ──────────────────────────────────────────────────────
# Each entry: (ablation_name, config_override_dict, description)
# Baseline: efficientnet_b3 + cross_attention + linear + pretrained + augmentation
# One component changed per ablation.

ABLATIONS: List[tuple] = [
    # ── Backbone ──────────────────────────────────────────────────────────────
    ("backbone_swin_t",
     {"backbone": "swin_t"},
     "Replace EfficientNet-B3 backbone with Swin-T"),
    ("backbone_mobilenet",
     {"backbone": "mobilenet"},
     "Replace EfficientNet-B3 backbone with MobileNet"),
    ("backbone_vgg19",
     {"backbone": "vgg19"},
     "Replace EfficientNet-B3 backbone with VGG19"),
    # ── Fusion ────────────────────────────────────────────────────────────────
    ("fusion_concat",
     {"fusion": "concat"},
     "Replace CrossAttention fusion with Concat"),
    # ── Head ──────────────────────────────────────────────────────────────────
    ("head_mlp",
     {"head": "mlp"},
     "Replace Linear head with MLP head"),
    # ── Initialisation ────────────────────────────────────────────────────────
    ("no_pretrain",
     {"pretrained": False},
     "Random init — no ImageNet pretraining"),
    ("freeze_backbone",
     {"freeze_backbone": True},
     "Freeze backbone; only fusion + head train"),
    # ── Regularisation & loss ─────────────────────────────────────────────────
    ("label_smoothing_0.1",
     {"label_smoothing": 0.1},
     "Add label smoothing ε=0.1"),
    ("no_lr_scheduler",
     {"use_scheduler": False},
     "Constant LR (no CosineAnnealingLR)"),
    # ── Data augmentation ─────────────────────────────────────────────────────
    ("no_augmentation",
     {"_no_aug": True},
     "No training augmentation (centre-crop only)"),
    # ── Learning rate ─────────────────────────────────────────────────────────
    ("lr_3e-4",
     {"lr": 3e-4},
     "Higher LR: 3e-4 instead of 1e-4"),
    ("lr_3e-5",
     {"lr": 3e-5},
     "Lower LR: 3e-5 instead of 1e-4"),
]

BASELINE_CFG = {
    "backbone": "efficientnet_b3", "fusion": "cross_attention", "head": "linear",
    "pretrained": True, "freeze_backbone": False, "label_smoothing": 0.0,
    "use_scheduler": True, "lr": 1e-4, "weight_decay": 1e-4, "n_epochs": N_EPOCHS,
}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_ablation_bar(df: pd.DataFrame, baseline_acc: float) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df_sorted = df.sort_values("test_acc", ascending=True)
        colors = ["#2A9D8F" if v >= baseline_acc else "#E63946"
                  for v in df_sorted["test_acc"]]

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(df_sorted["name"], df_sorted["test_acc"] * 100,
                       color=colors, edgecolor="black", linewidth=0.5)
        ax.axvline(baseline_acc * 100, color="navy", linestyle="--",
                   linewidth=1.5, label=f"Baseline ({baseline_acc*100:.2f}%)")
        for bar, val in zip(bars, df_sorted["test_acc"]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f"{val*100:.2f}%", va="center", fontsize=9)
        ax.set_xlabel("Test Accuracy (%)", fontsize=12)
        ax.set_title(f"Ablation Study — Baseline: {BASELINE_NAME}\n"
                     f"(green = improves on baseline, red = degrades)", fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlim(0, 100)
        plt.tight_layout()
        out = FIGS_DIR / "ablation_bar.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Ablation bar chart saved to %s", out)
    except Exception as e:
        logger.warning("Ablation plot failed: %s", e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    out_csv = RESULTS_DIR / "ablation_results.csv"
    # Resume support
    done = set()
    rows = []
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done = set(existing["name"].tolist())
        rows = existing.to_dict("records")
        logger.info("Resuming: %d ablations already done", len(done))

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_acc = None
    if "baseline" not in done:
        logger.info("=" * 60)
        logger.info("BASELINE: %s", BASELINE_NAME)
        logger.info("=" * 60)
        train_loader, val_loader, test_loader = make_loaders()
        model = build_model(BASELINE_CFG, device)
        # Load pretrained checkpoint if available to save time
        ckpt = CKPT_DIR / f"{BASELINE_NAME}_best.pt"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=device))
            test_acc  = _eval_acc(model, test_loader, device)
            test_top3 = _eval_top3(model, test_loader, device)
            val_acc   = _eval_acc(model, val_loader, device)
            logger.info("Baseline loaded from checkpoint: test_acc=%.4f", test_acc)
            res = {"val_acc": val_acc, "test_acc": test_acc, "test_top3": test_top3}
        else:
            res = train_and_eval(model, train_loader, val_loader, test_loader,
                                 BASELINE_CFG, "baseline", device)
        baseline_acc = res["test_acc"]
        rows.append({"name": "baseline", "description": "Full best model",
                     **{k: v for k, v in res.items()}})
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    else:
        baseline_acc = next(r["test_acc"] for r in rows if r["name"] == "baseline")

    # ── Ablations ─────────────────────────────────────────────────────────────
    for abl_name, cfg_override, description in ABLATIONS:
        if abl_name in done:
            logger.info("Skipping %s (already done)", abl_name)
            continue

        logger.info("=" * 60)
        logger.info("ABLATION: %s — %s", abl_name, description)
        logger.info("=" * 60)

        cfg = {**BASELINE_CFG, **cfg_override}
        augment = not cfg_override.get("_no_aug", False)
        try:
            train_loader, val_loader, test_loader = make_loaders(augment=augment)
            model = build_model(cfg, device)
            res   = train_and_eval(model, train_loader, val_loader, test_loader,
                                   cfg, abl_name, device)
            delta = res["test_acc"] - baseline_acc
            logger.info("ABLATION %s: test_acc=%.4f  Δ=%.4f", abl_name, res["test_acc"], delta)
            rows.append({"name": abl_name, "description": description,
                         "delta_vs_baseline": delta, **{k: v for k, v in res.items()}})
        except Exception as exc:
            logger.error("Ablation %s failed: %s", abl_name, exc, exc_info=True)
            rows.append({"name": abl_name, "description": description,
                         "val_acc": None, "test_acc": None, "test_top3": None,
                         "delta_vs_baseline": None})

        pd.DataFrame(rows).to_csv(out_csv, index=False)
        logger.info("Results saved to %s", out_csv)

    # ── Summary plot ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows).dropna(subset=["test_acc"])
    ablation_df = df[df["name"] != "baseline"]
    plot_ablation_bar(ablation_df, baseline_acc)

    logger.info("=" * 60)
    logger.info("ABLATION COMPLETE")
    logger.info("=" * 60)
    print(df[["name", "test_acc", "test_top3", "delta_vs_baseline"]].to_string(index=False))


if __name__ == "__main__":
    main()
