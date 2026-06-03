"""
DOVE — Run the full experiment grid.

Enumerates all combinations of backbone × fusion × head, trains each,
evaluates on the test set, and writes results to:
  results/tables/experiment_results.csv
  results/DOVE_report.md  (updated after each experiment)
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import DOVEDataset
from data.augment import get_image_transform
from evaluation.metrics import compute_accuracy, compute_top3_accuracy, compute_confusion_matrix, compute_per_class_metrics
from evaluation.visualise import plot_confusion_matrix, plot_learning_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("run_experiments")

REPO_ROOT   = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "tables"
FIGS_DIR    = REPO_ROOT / "results" / "figures"
CKPT_DIR    = REPO_ROOT / "results" / "checkpoints"
REPORT_PATH = REPO_ROOT / "results" / "DOVE_report.md"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 20
SPECIES_NAMES = [
    "Acorn Woodpecker", "American Crow", "American Robin", "Anna's Hummingbird",
    "Black Phoebe", "Brewer's Blackbird", "Bushtit", "California Scrub-Jay",
    "California Towhee", "Chestnut-backed Chickadee", "Cooper's Hawk",
    "Dark-eyed Junco", "House Finch", "Lesser Goldfinch", "Mourning Dove",
    "Northern Mockingbird", "Oak Titmouse", "Red-tailed Hawk",
    "White-crowned Sparrow", "Yellow-rumped Warbler",
]


# ── Experiment grid ───────────────────────────────────────────────────────────

BACKBONES    = ["swin_t", "efficientnet_b3", "mobilenet", "vgg19"]
FUSIONS      = ["cross_attention", "concat"]
NEURAL_HEADS = ["mlp", "linear"]
SKLEARN_HEADS = ["random_forest", "naive_bayes"]


def build_grid() -> List[Dict[str, Any]]:
    import itertools
    exps = []
    # Neural: 4 backbones × 2 fusions × 2 heads = 16
    for bb, fu, hd in itertools.product(BACKBONES, FUSIONS, NEURAL_HEADS):
        exps.append({"backbone": bb, "fusion": fu, "head": hd, "name": f"{bb}_{fu}_{hd}"})
    # sklearn (no fusion needed): 4 × 2 = 8
    for bb, hd in itertools.product(BACKBONES, SKLEARN_HEADS):
        exps.append({"backbone": bb, "fusion": "none", "head": hd, "name": f"{bb}_none_{hd}"})
    return exps


# ── Model assembly ────────────────────────────────────────────────────────────

def build_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Assemble backbone + invariant extractor + fusion + head."""
    bb_name = config["backbone"]
    fu_name = config["fusion"]
    hd_name = config["head"]

    # Backbone
    bb_mod = importlib.import_module(f"backbones.{bb_name}")
    backbone_cls = getattr(bb_mod, list(filter(lambda x: "Backbone" in x, dir(bb_mod)))[0])
    backbone = backbone_cls().to(device)

    # Invariant extractor always uses Swin-T
    from features.invariant.pipeline import InvariantFeaturePipeline
    inv_pipeline = InvariantFeaturePipeline().to(device)

    # Fusion
    if fu_name == "cross_attention":
        from fusion.cross_attention import CrossAttentionFusion
        fusion = CrossAttentionFusion().to(device)
    elif fu_name == "concat":
        from fusion.concat import ConcatFusion
        fusion = ConcatFusion().to(device)
    else:
        fusion = None

    # Head
    if hd_name == "mlp":
        from heads.mlp_head import MLPHead
        head = MLPHead(in_dim=512, num_classes=NUM_CLASSES).to(device)
    elif hd_name == "linear":
        from heads.linear_head import LinearHead
        head = LinearHead(in_dim=512, num_classes=NUM_CLASSES).to(device)
    elif hd_name == "random_forest":
        from heads.random_forest import RandomForestHead
        head = RandomForestHead()
        return _SklearnModel(backbone, inv_pipeline, head)
    elif hd_name == "naive_bayes":
        from heads.naive_bayes import NaiveBayesHead
        head = NaiveBayesHead()
        return _SklearnModel(backbone, inv_pipeline, head)
    else:
        raise ValueError(f"Unknown head: {hd_name}")

    return _NeuralModel(backbone, inv_pipeline, fusion, head).to(device)


class _NeuralModel(nn.Module):
    def __init__(self, backbone, inv_pipeline, fusion, head):
        super().__init__()
        self.backbone = backbone
        self.inv_pipeline = inv_pipeline
        self.fusion = fusion
        self.head = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(images)
        # Flatten feature map to (B, C) if spatial dims remain
        if feat_map.dim() > 2:
            feat_map = feat_map.mean(dim=[-2, -1])
        inv_feat = self.inv_pipeline.extract(images)   # (B, 1440)
        motion_zero = torch.zeros(images.size(0), 139, device=images.device)
        if self.fusion is not None:
            fused = self.fusion(motion_zero, inv_feat)  # (B, 512)
        else:
            # project inv_feat directly to 512
            fused = inv_feat[:, :512]
        return self.head(fused)


class _SklearnModel:
    """Thin wrapper so sklearn heads fit the same training loop interface."""
    def __init__(self, backbone, inv_pipeline, head):
        self.backbone = backbone
        self.inv_pipeline = inv_pipeline
        self.head = head
        self.is_sklearn = True

    def extract_features(self, loader, device):
        all_feats, all_labels = [], []
        self.backbone.eval()
        self.inv_pipeline.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device)
                feats = self.inv_pipeline.extract(imgs).cpu().numpy()
                all_feats.append(feats)
                all_labels.append(batch["label"].numpy())
        return np.vstack(all_feats), np.concatenate(all_labels)

    def fit(self, train_loader, device):
        X, y = self.extract_features(train_loader, device)
        self.head.fit(X, y)

    def predict(self, loader, device):
        X, _ = self.extract_features(loader, device)
        return self.head.predict(X), self.head.predict_proba(X)


# ── Training ──────────────────────────────────────────────────────────────────

def train_neural(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    n_epochs: int = 20,
    patience: int = 5,
    exp_name: str = "exp",
) -> Dict[str, Any]:
    lr = config.get("lr", 1e-4)
    wd = config.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0
    best_ckpt = CKPT_DIR / f"{exp_name}_best.pt"
    no_improve = 0
    log_rows = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()

        val_acc = _eval_accuracy(model, val_loader, device)
        avg_loss = train_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        logger.info("[%s] Epoch %2d/%d  loss=%.4f  train=%.3f  val=%.3f",
                    exp_name, epoch, n_epochs, avg_loss, train_acc, val_acc)
        log_rows.append({"epoch": epoch, "train_loss": avg_loss,
                         "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("[%s] Early stopping at epoch %d", exp_name, epoch)
                break

    pd.DataFrame(log_rows).to_csv(RESULTS_DIR / f"{exp_name}_log.csv", index=False)
    # Restore best
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    return {"best_val_acc": best_val_acc, "log": log_rows}


def _eval_accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            with torch.cuda.amp.autocast() if device.type == "cuda" else torch.no_grad():
                logits = model(imgs)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return correct / max(total, 1)


def _eval_full(model, loader, device) -> Dict[str, Any]:
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"]
            logits = model(imgs).cpu()
            all_preds.append(logits.argmax(1).numpy())
            all_labels.append(labels.numpy())
            all_logits.append(logits.numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    logits = np.concatenate(all_logits)
    acc    = (preds == labels).mean()
    top3   = compute_top3_accuracy(torch.tensor(logits), torch.tensor(labels))
    cm     = compute_confusion_matrix(preds, labels)
    return {"accuracy": float(acc), "top3_accuracy": float(top3),
            "confusion_matrix": cm, "preds": preds, "labels": labels}


# ── Data loaders ──────────────────────────────────────────────────────────────

def make_loaders(batch_size: int = 32) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_csv = REPO_ROOT / "data" / "splits" / "train.csv"
    val_csv   = REPO_ROOT / "data" / "splits" / "val.csv"
    test_csv  = REPO_ROOT / "data" / "splits" / "test.csv"

    train_ds = DOVEDataset(train_csv, transform=get_image_transform(train=True))
    val_ds   = DOVEDataset(val_csv,   transform=get_image_transform(train=False))
    test_ds  = DOVEDataset(test_csv,  transform=get_image_transform(train=False))

    # WeightedRandomSampler from saved class weights
    weights_path = REPO_ROOT / "data" / "splits" / "class_weights.npy"
    if weights_path.exists():
        class_weights = np.load(weights_path)
        sample_weights = np.array([class_weights[int(train_ds.df.iloc[i]["species_id"])]
                                   for i in range(len(train_ds))], dtype=np.float32)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


# ── Report helpers ────────────────────────────────────────────────────────────

def init_report(n_experiments: int) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(f"# DOVE — Results Report\n\n")
        f.write(f"_Auto-generated. {n_experiments} experiments._\n\n")
        f.write("## 1. Experiment Results\n\n")
        f.write("| # | Experiment | Test Acc | Top-3 Acc |\n")
        f.write("|---|-----------|----------|----------|\n")


def append_result_row(idx: int, name: str, acc: float, top3: float) -> None:
    with open(REPORT_PATH, "a") as f:
        f.write(f"| {idx} | `{name}` | {acc:.4f} | {top3:.4f} |\n")


def finalise_report(df: pd.DataFrame) -> None:
    with open(REPORT_PATH, "a") as f:
        f.write("\n## 2. Top-10 Configurations\n\n")
        top10 = df.nlargest(10, "test_accuracy")[
            ["name", "backbone", "fusion", "head", "test_accuracy", "top3_accuracy"]]
        f.write(top10.to_markdown(index=False))
        f.write("\n\n## 3. Conclusions\n\n")
        best = df.loc[df["test_accuracy"].idxmax()]
        f.write(f"Best configuration: `{best['name']}` with "
                f"**{best['test_accuracy']:.2%}** test accuracy "
                f"(top-3: {best['top3_accuracy']:.2%}).\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",  default=None, help="Run single experiment by name")
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--smoke-test",  action="store_true",
                        help="2-epoch quick sanity check, small dataset subset")
    parser.add_argument("--neural-only", action="store_true",
                        help="Run only neural-head experiments (skip random_forest/naive_bayes)")
    parser.add_argument("--skip-done",   action="store_true",
                        help="Skip experiments already present in experiment_results.csv")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    grid = build_grid()
    if args.experiment:
        grid = [e for e in grid if e["name"] == args.experiment]
        if not grid:
            logger.error("Experiment '%s' not found in grid.", args.experiment)
            sys.exit(1)
    if args.neural_only:
        grid = [e for e in grid if e["head"] not in ("random_forest", "naive_bayes")]
        logger.info("--neural-only: %d experiments remaining", len(grid))

    # Load existing results so we can skip already-done experiments and merge
    done_results: list[dict] = []
    results_csv = RESULTS_DIR / "experiment_results.csv"
    done_names: set[str] = set()
    if args.skip_done and results_csv.exists():
        existing = pd.read_csv(results_csv)
        done_names = set(existing["name"].dropna())
        done_results = existing.to_dict("records")
        logger.info("--skip-done: skipping %d already-completed experiments", len(done_names))
        grid = [e for e in grid if e["name"] not in done_names]

    n_epochs = 2 if args.smoke_test else args.epochs
    batch_size = args.batch_size

    train_loader, val_loader, test_loader = make_loaders(batch_size)

    all_results = list(done_results)
    init_report(len(grid))

    for i, config in enumerate(grid, 1):
        exp_name = config["name"]
        logger.info("=" * 60)
        logger.info("[%d/%d] %s", i, len(grid), exp_name)
        logger.info("=" * 60)

        t0 = time.time()
        try:
            model = build_model(config, device)

            if getattr(model, "is_sklearn", False):
                model.backbone.to(device)
                model.inv_pipeline.to(device)
                model.fit(train_loader, device)
                test_preds, test_proba = model.predict(test_loader, device)
                val_preds,  _          = model.predict(val_loader, device)
                val_labels  = np.concatenate([b["label"].numpy() for b in val_loader])
                test_labels = np.concatenate([b["label"].numpy() for b in test_loader])
                val_acc  = (val_preds  == val_labels).mean()
                test_acc = (test_preds == test_labels).mean()
                top3_acc = 0.0   # not straightforward for sklearn
                cm = compute_confusion_matrix(test_preds, test_labels)
                best_val_acc = val_acc
            else:
                train_info = train_neural(model, train_loader, val_loader, config,
                                          device, n_epochs=n_epochs, exp_name=exp_name)
                best_val_acc = train_info["best_val_acc"]
                eval_result  = _eval_full(model, test_loader, device)
                test_acc     = eval_result["accuracy"]
                top3_acc     = eval_result["top3_accuracy"]
                cm           = eval_result["confusion_matrix"]

                # Save confusion matrix plot
                try:
                    plot_confusion_matrix(cm, SPECIES_NAMES,
                                         FIGS_DIR / f"confmat_{exp_name}.png")
                    plot_learning_curve(RESULTS_DIR / f"{exp_name}_log.csv",
                                        FIGS_DIR / f"curve_{exp_name}.png")
                except Exception as e:
                    logger.warning("Visualisation failed for %s: %s", exp_name, e)

            elapsed = time.time() - t0
            row = {**config, "test_accuracy": test_acc, "top3_accuracy": top3_acc,
                   "best_val_acc": best_val_acc, "elapsed_s": elapsed}
            all_results.append(row)
            append_result_row(i, exp_name, test_acc, top3_acc)

            # Save rolling results after every experiment
            pd.DataFrame(all_results).to_csv(
                RESULTS_DIR / "experiment_results.csv", index=False)
            logger.info("[%s] done  acc=%.4f  top3=%.4f  %.0fs",
                        exp_name, test_acc, top3_acc, elapsed)

        except Exception as exc:
            logger.error("[%s] FAILED: %s", exp_name, exc, exc_info=True)
            all_results.append({**config, "test_accuracy": float("nan"),
                                 "top3_accuracy": float("nan"),
                                 "best_val_acc": float("nan"), "elapsed_s": 0.0})

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "experiment_results.csv", index=False)
    finalise_report(df)
    logger.info("All experiments done. Results → %s", RESULTS_DIR)
    logger.info("Report → %s", REPORT_PATH)

    if len(df) > 0 and not df["test_accuracy"].isna().all():
        best = df.loc[df["test_accuracy"].idxmax()]
        logger.info("Best: %s  acc=%.4f", best["name"], best["test_accuracy"])


if __name__ == "__main__":
    main()
