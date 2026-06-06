"""
DOVE — Original two-stream pipeline: motion (139-d) + invariant (1440-d).

Restores the intended design from the fusion module defaults:
  frames → MotionPipeline  → z_mot ∈ R^139
  frames → InvariantPipeline → z_inv ∈ R^1440
  z_mot, z_inv → Fusion (ConcatFusion | CrossAttentionFusion) → z ∈ R^512
  z → Head → ŷ ∈ R^20

VB100 video frames get a real 139-d motion descriptor (mean-pooled per clip).
iNaturalist images get a zero motion vector (no trajectory available).

Grid: 2 fusions × 2 heads = 4 configs.
Results → results_motion/
"""
from __future__ import annotations

import argparse
import itertools
import logging
import re
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
from features.invariant.pipeline import InvariantFeaturePipeline
from fusion.concat import ConcatFusion
from fusion.cross_attention import CrossAttentionFusion
sys.path.insert(0, str(Path(__file__).parent))
from hpo_defaults import load_hpo_configs, get_hparams

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("run_experiments_motion")

REPO_ROOT      = Path(__file__).parent.parent
RESULTS_DIR    = REPO_ROOT / "results_motion" / "tables"
FIGS_DIR       = REPO_ROOT / "results_motion" / "figures"
CKPT_DIR       = REPO_ROOT / "results_motion" / "checkpoints"
REPORT_PATH    = REPO_ROOT / "results_motion" / "DOVE_motion_report.md"
MOTION_PARQUET = REPO_ROOT / "data" / "motion_features.parquet"

for d in (RESULTS_DIR, FIGS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

FUSIONS      = ["cross_attention", "concat"]
NEURAL_HEADS = ["mlp", "linear"]


# ── Motion feature lookup ─────────────────────────────────────────────────────

def _build_motion_lookup() -> dict[str, np.ndarray]:
    if not MOTION_PARQUET.exists():
        logger.warning("Motion parquet not found — all motion feats will be zero")
        return {}
    df = pd.read_parquet(MOTION_PARQUET)
    feat_cols = [f"feat_{i}" for i in range(139)]
    lookup = {row["video_src"]: row[feat_cols].values.astype(np.float32)
              for _, row in df.iterrows()}
    logger.info("Loaded motion features for %d clips", len(lookup))
    return lookup


def _build_stem_to_motion(lookup: dict) -> dict[str, np.ndarray]:
    """Re-key by clip stem (strip _fNNNNN suffix) for frame→clip mapping."""
    return {Path(k).stem: v for k, v in lookup.items()}


# ── Dataset ───────────────────────────────────────────────────────────────────

class MotionAwareDOVEDataset(DOVEDataset):
    """
    Adds 'motion_feat' (139-d) to each sample.
    VB100 frames → real clip descriptor; images → zero vector.
    """

    def __init__(self, csv_path, transform=None, stem_to_motion=None):
        super().__init__(csv_path, transform=transform)
        self._s2m = stem_to_motion or {}

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if sample.get("modality") == "video_frame":
            stem = re.sub(r"_f\d+$", "", Path(sample.get("filepath", "")).stem)
            feat = self._s2m.get(stem, np.zeros(139, dtype=np.float32))
        else:
            feat = np.zeros(139, dtype=np.float32)
        sample["motion_feat"] = torch.tensor(feat, dtype=torch.float32)
        return sample


# ── Model ─────────────────────────────────────────────────────────────────────

class MotionInvariantModel(nn.Module):
    """
    Original two-stream design:
      z_mot (139-d) + z_inv (1440-d) → Fusion → 512-d → Head → 20 classes
    No backbone.
    """

    def __init__(self, inv_pipeline, fusion, head):
        super().__init__()
        self.inv_pipeline = inv_pipeline
        self.fusion = fusion
        self.head   = head

    def forward(self, images: torch.Tensor,
                motion_feat: torch.Tensor) -> torch.Tensor:
        z_inv = self.inv_pipeline.extract(images)      # (B, 1440)
        z     = self.fusion(motion_feat, z_inv)        # (B, 512)
        return self.head(z)


def build_model(config: dict, device: torch.device) -> MotionInvariantModel:
    inv_pipeline = InvariantFeaturePipeline().to(device)

    if config["fusion"] == "cross_attention":
        # motion (139-d) is Q; invariant (1440-d) is K/V — original intent
        fusion = CrossAttentionFusion(motion_dim=139, invariant_dim=1440).to(device)
    else:
        fusion = ConcatFusion(motion_dim=139, invariant_dim=1440).to(device)

    if config["head"] == "mlp":
        head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(config.get("dropout", 0.316)), nn.Linear(256, 20)
        ).to(device)
    else:
        head = nn.Linear(512, 20).to(device)

    return MotionInvariantModel(inv_pipeline, fusion, head).to(device)


# ── Train / eval ──────────────────────────────────────────────────────────────

def train_and_eval(config, train_loader, val_loader, test_loader,
                   device, n_epochs=10, exp_name="exp"):
    model     = build_model(config, device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.get("lr", 1e-4),
                                  weight_decay=config.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.066))
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val, no_improve = 0.0, 0
    best_ckpt = CKPT_DIR / f"{exp_name}_best.pt"
    log_rows  = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            mfeat  = batch["motion_feat"].to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(imgs, mfeat)
                    loss   = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs, mfeat)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        scheduler.step()
        train_acc = correct / max(total, 1)

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device)
                mfeat  = batch["motion_feat"].to(device)
                with (torch.cuda.amp.autocast() if scaler else torch.no_grad()):
                    logits = model(imgs, mfeat)
                vc += (logits.argmax(1) == labels).sum().item()
                vt += imgs.size(0)
        val_acc = vc / max(vt, 1)
        logger.info("[%s] Epoch %2d/%d  loss=%.4f  train=%.3f  val=%.3f",
                    exp_name, epoch, n_epochs, train_loss / max(total, 1),
                    train_acc, val_acc)
        log_rows.append({"epoch": epoch,
                         "train_loss": train_loss / max(total, 1),
                         "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val:
            best_val = val_acc
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1
            if no_improve >= 5:
                logger.info("[%s] Early stop at epoch %d", exp_name, epoch)
                break

    pd.DataFrame(log_rows).to_csv(RESULTS_DIR / f"{exp_name}_log.csv", index=False)

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    tc, tt, top3c = 0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            mfeat  = batch["motion_feat"].to(device)
            with (torch.cuda.amp.autocast() if scaler else torch.no_grad()):
                logits = model(imgs, mfeat)
            tc    += (logits.argmax(1) == labels).sum().item()
            top3   = logits.topk(3, dim=1).indices
            top3c += (top3 == labels.unsqueeze(1)).any(1).sum().item()
            tt    += imgs.size(0)

    test_acc = tc / max(tt, 1)
    top3_acc = top3c / max(tt, 1)
    logger.info("[%s] TEST  acc=%.4f  top3=%.4f", exp_name, test_acc, top3_acc)
    return {**config, "name": exp_name,
            "test_accuracy": test_acc, "top3_accuracy": top3_acc,
            "best_val": best_val}


# ── Data loaders ──────────────────────────────────────────────────────────────

def make_loaders(stem_to_motion, batch_size=32):
    splits = REPO_ROOT / "data" / "splits"
    kwargs = dict(stem_to_motion=stem_to_motion)
    train_ds = MotionAwareDOVEDataset(splits / "train.csv",
                                      transform=get_image_transform(True), **kwargs)
    val_ds   = MotionAwareDOVEDataset(splits / "val.csv",
                                      transform=get_image_transform(False), **kwargs)
    test_ds  = MotionAwareDOVEDataset(splits / "test.csv",
                                      transform=get_image_transform(False), **kwargs)

    wp = splits / "class_weights.npy"
    if wp.exists():
        cw     = np.load(wp)
        labels = [int(train_ds[i]["label"]) for i in range(len(train_ds))]
        sw     = torch.tensor([cw[l] for l in labels], dtype=torch.float32)
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=WeightedRandomSampler(sw, len(sw)),
                                  num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    stem_to_motion = _build_stem_to_motion(_build_motion_lookup())
    logger.info("Motion lookup: %d stems", len(stem_to_motion))

    n_epochs = 2 if args.smoke_test else args.epochs
    hpo_configs = load_hpo_configs()
    logger.info("Loaded HPO configs for: %s", list(hpo_configs.keys()))

    configs = [{"fusion": fu, "head": hd, "name": f"motion_inv_{fu}_{hd}"}
               for fu, hd in itertools.product(FUSIONS, NEURAL_HEADS)]

    all_results = []
    results_csv = RESULTS_DIR / "experiment_results_motion.csv"

    for i, config in enumerate(configs, 1):
        exp_name = config["name"]
        hparams = get_hparams(exp_name, hpo_configs)
        config.update(hparams)
        logger.info("=" * 60)
        logger.info("[%d/%d] %s  lr=%.2e  wd=%.2e  ls=%.3f  bs=%d",
                    i, len(configs), exp_name,
                    hparams["lr"], hparams["weight_decay"],
                    hparams["label_smoothing"], hparams["batch_size"])
        logger.info("=" * 60)
        train_loader, val_loader, test_loader = make_loaders(stem_to_motion, hparams["batch_size"])
        try:
            row = train_and_eval(config, train_loader, val_loader, test_loader,
                                 device, n_epochs=n_epochs, exp_name=exp_name)
        except Exception as e:
            logger.error("[%s] FAILED: %s", exp_name, e, exc_info=True)
            row = {**config, "test_accuracy": float("nan"), "top3_accuracy": float("nan")}
        all_results.append(row)
        pd.DataFrame(all_results).to_csv(results_csv, index=False)

    df = pd.DataFrame(all_results).sort_values("test_accuracy", ascending=False)
    df.to_csv(results_csv, index=False)
    logger.info("All done.\n%s",
                df[["name", "test_accuracy", "top3_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
