"""
DOVE — Hyperparameter optimisation with Optuna (TPE sampler).

Runs 15 Optuna trials × 5 best architectures = 75 trials.
Each trial trains for 10 epochs (fast proxy), then the best config
per model is retrained for 20 epochs to get final test accuracy.

Search space:
  lr              log-uniform [1e-5, 5e-3]
  weight_decay    log-uniform [1e-6, 1e-2]
  label_smoothing uniform     [0.0,  0.20]
  dropout         uniform     [0.0,  0.40]   (MLPHead only; ignored for linear)
  batch_size      categorical [8, 16, 32]

Results saved to:
  results/tables/hpo_results.csv         — all trials
  results/tables/hpo_best_configs.csv    — best config per model
  results/figures/hpo_parallel_coords.png — parallel-coords plot
  results/figures/hpo_scatter.png         — lr vs val_acc scatter per model
"""
from __future__ import annotations

import importlib
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from data.loader import DOVEDataset
from data.augment import get_image_transform
from evaluation.metrics import compute_top3_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger("run_hpo")

REPO_ROOT   = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "tables"
FIGS_DIR    = REPO_ROOT / "results" / "figures"
CKPT_DIR    = REPO_ROOT / "results" / "checkpoints"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES  = 20
N_HPO_EPOCHS = 10   # proxy training length for each trial
N_FULL_EPOCHS = 20  # full retraining of best config
N_TRIALS     = 15   # Optuna trials per model

# Top-5 by test accuracy from the main experiment sweep
TOP5_MODELS = [
    {"name": "efficientnet_b3_cross_attention_linear",
     "backbone": "efficientnet_b3", "fusion": "cross_attention", "head": "linear"},
    {"name": "mobilenet_cross_attention_linear",
     "backbone": "mobilenet",       "fusion": "cross_attention", "head": "linear"},
    {"name": "efficientnet_b3_concat_linear",
     "backbone": "efficientnet_b3", "fusion": "concat",          "head": "linear"},
    {"name": "mobilenet_cross_attention_mlp",
     "backbone": "mobilenet",       "fusion": "cross_attention", "head": "mlp"},
    {"name": "mobilenet_concat_mlp",
     "backbone": "mobilenet",       "fusion": "concat",          "head": "mlp"},
]

SPECIES_NAMES = [
    "Acorn Woodpecker","American Crow","American Robin","Anna's Hummingbird",
    "Black Phoebe","Brewer's Blackbird","Bushtit","California Scrub-Jay",
    "California Towhee","Chestnut-backed Chickadee","Cooper's Hawk",
    "Dark-eyed Junco","House Finch","Lesser Goldfinch","Mourning Dove",
    "Northern Mockingbird","Oak Titmouse","Red-tailed Hawk",
    "White-crowned Sparrow","Yellow-rumped Warbler",
]


# ── Model assembly ────────────────────────────────────────────────────────────

class _NeuralModel(nn.Module):
    def __init__(self, backbone, inv_pipeline, fusion, head):
        super().__init__()
        self.backbone     = backbone
        self.inv_pipeline = inv_pipeline
        self.fusion       = fusion
        self.head         = head

    def forward(self, images):
        backbone_feat = self.backbone(images)
        if backbone_feat.dim() > 2:
            backbone_feat = backbone_feat.mean(dim=[-2, -1])
        inv_feat = self.inv_pipeline.extract(images)
        fused    = self.fusion(backbone_feat, inv_feat) if self.fusion else inv_feat[:, :512]
        return self.head(fused)


def build_model(arch: Dict[str, Any], dropout: float, device: torch.device) -> nn.Module:
    bb_mod   = importlib.import_module(f"backbones.{arch['backbone']}")
    bb_cls   = [getattr(bb_mod, x) for x in dir(bb_mod) if "Backbone" in x][0]
    backbone = bb_cls(pretrained=True).to(device)

    from features.invariant.pipeline import InvariantFeaturePipeline
    inv_pipeline = InvariantFeaturePipeline(pretrained=True).to(device)

    if arch["fusion"] == "cross_attention":
        from fusion.cross_attention import CrossAttentionFusion
        fusion = CrossAttentionFusion(motion_dim=backbone.feature_dim).to(device)
    else:
        from fusion.concat import ConcatFusion
        fusion = ConcatFusion(motion_dim=backbone.feature_dim).to(device)

    if arch["head"] == "mlp":
        from heads.mlp_head import MLPHead
        head = MLPHead(in_dim=512, num_classes=NUM_CLASSES, dropout=dropout).to(device)
    else:
        from heads.linear_head import LinearHead
        head = LinearHead(in_dim=512, num_classes=NUM_CLASSES).to(device)

    return _NeuralModel(backbone, inv_pipeline, fusion, head).to(device)


# ── Data ──────────────────────────────────────────────────────────────────────

_loaders_cache: Dict[int, tuple] = {}

def get_loaders(batch_size: int):
    if batch_size in _loaders_cache:
        return _loaders_cache[batch_size]
    train_csv = REPO_ROOT / "data" / "splits" / "train.csv"
    val_csv   = REPO_ROOT / "data" / "splits" / "val.csv"
    test_csv  = REPO_ROOT / "data" / "splits" / "test.csv"

    train_ds = DOVEDataset(train_csv, transform=get_image_transform(train=True))
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

    val_loader  = DataLoader(val_ds,  batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4)
    result = (train_loader, val_loader, test_loader)
    _loaders_cache[batch_size] = result
    return result


# ── Training utilities ────────────────────────────────────────────────────────

def train_epochs(model, train_loader, val_loader, lr, wd, label_smoothing,
                 n_epochs, device, trial=None) -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_val_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in train_loader:
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    loss = criterion(model(imgs), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss = criterion(model(imgs), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        scheduler.step()

        val_acc = _eval_acc(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Optuna pruning
        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_acc


def _eval_acc(model, loader, device) -> float:
    model.eval(); c = t = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            c += (logits.argmax(1) == batch["label"].to(device)).sum().item()
            t += batch["label"].size(0)
    return c / max(t, 1)


def _eval_top3(model, loader, device) -> float:
    model.eval(); ll, lb = [], []
    with torch.no_grad():
        for batch in loader:
            ll.append(model(batch["image"].to(device)).cpu())
            lb.append(batch["label"])
    return float(compute_top3_accuracy(torch.cat(ll), torch.cat(lb)))


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(arch: Dict[str, Any], device: torch.device):
    def objective(trial: optuna.Trial) -> float:
        lr             = trial.suggest_float("lr",             1e-5, 5e-3, log=True)
        weight_decay   = trial.suggest_float("weight_decay",   1e-6, 1e-2, log=True)
        label_smoothing= trial.suggest_float("label_smoothing",0.0,  0.20)
        dropout        = trial.suggest_float("dropout",        0.0,  0.40)
        batch_size     = trial.suggest_categorical("batch_size", [8, 16, 32])

        train_loader, val_loader, _ = get_loaders(batch_size)
        model = build_model(arch, dropout, device)

        try:
            val_acc = train_epochs(model, train_loader, val_loader,
                                   lr, weight_decay, label_smoothing,
                                   N_HPO_EPOCHS, device, trial=trial)
        except optuna.TrialPruned:
            raise
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return val_acc
    return objective


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_hpo_results(all_rows: List[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.DataFrame(all_rows)
        df = df[df["val_acc"].notna()]

        # ── Scatter: lr vs val_acc per model ──────────────────────────────────
        fig, axes = plt.subplots(1, len(TOP5_MODELS), figsize=(5*len(TOP5_MODELS), 4),
                                  sharey=True)
        cmap = plt.cm.viridis
        for ax, mdl in zip(axes, TOP5_MODELS):
            sub = df[df["model"] == mdl["name"]]
            if sub.empty:
                continue
            sc = ax.scatter(sub["lr"], sub["val_acc"] * 100,
                            c=sub["label_smoothing"], cmap=cmap,
                            s=sub["batch_size"] * 2, alpha=0.8, edgecolors="k", linewidths=0.4)
            best = sub.loc[sub["val_acc"].idxmax()]
            ax.axhline(best["val_acc"] * 100, color="red", linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set_title(mdl["name"].replace("_", "\n"), fontsize=7)
            ax.set_xlabel("Learning Rate")
            if ax == axes[0]:
                ax.set_ylabel("Val Accuracy (%)")
        plt.colorbar(sc, ax=axes[-1], label="Label Smoothing")
        fig.suptitle("HPO: LR vs Val Accuracy (colour=label_smoothing, size=batch_size)", y=1.01)
        plt.tight_layout()
        fig.savefig(FIGS_DIR / "hpo_fixed_scatter.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("HPO scatter saved")

        # ── Bar: best configs comparison ──────────────────────────────────────
        best_rows = df.loc[df.groupby("model")["val_acc"].idxmax()]
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        colors = ["#E63946","#2A9D8F","#E9C46A","#457B9D","#F4A261"]
        bars = ax2.bar(range(len(best_rows)), best_rows["val_acc"] * 100,
                       color=colors[:len(best_rows)], edgecolor="black")
        ax2.set_xticks(range(len(best_rows)))
        ax2.set_xticklabels([r["model"].replace("_", "\n") for _, r in best_rows.iterrows()],
                             fontsize=8)
        ax2.set_ylabel("Best Val Accuracy (%)")
        ax2.set_title("Best HPO Val Accuracy per Architecture")
        for bar, (_, row) in zip(bars, best_rows.iterrows()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"lr={row['lr']:.1e}\nbs={int(row['batch_size'])}",
                     ha="center", va="bottom", fontsize=7)
        plt.tight_layout()
        fig2.savefig(FIGS_DIR / "hpo_fixed_best_configs.png", dpi=150)
        plt.close(fig2)
        logger.info("HPO best configs plot saved")
    except Exception as e:
        logger.warning("HPO plot failed: %s", e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("HPO device: %s", device)

    all_csv  = RESULTS_DIR / "hpo_fixed_results.csv"
    best_csv = RESULTS_DIR / "hpo_fixed_best_configs.csv"

    all_rows  = []
    best_rows = []

    # Resume support
    done_models = set()
    if all_csv.exists():
        existing = pd.read_csv(all_csv)
        all_rows = existing.to_dict("records")
        done_models = set(existing[existing["is_best_retrain"] == True]["model"].tolist())
        logger.info("Resuming HPO: %d models already fully done", len(done_models))

    for arch in TOP5_MODELS:
        name = arch["name"]
        if name in done_models:
            logger.info("Skipping %s (already completed)", name)
            continue

        logger.info("=" * 60)
        logger.info("HPO for: %s  (%d trials × %d epochs)", name, N_TRIALS, N_HPO_EPOCHS)
        logger.info("=" * 60)

        # Remove trials already run for this model (partial resume)
        existing_trials = [r for r in all_rows if r.get("model") == name and not r.get("is_best_retrain")]
        n_done = len(existing_trials)

        sampler = optuna.samplers.TPESampler(seed=42)
        pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        # Seed study with existing trials if resuming
        for r in existing_trials:
            try:
                study.add_trial(optuna.trial.create_trial(
                    params={k: r[k] for k in ["lr","weight_decay","label_smoothing","dropout","batch_size"]},
                    distributions={
                        "lr":              optuna.distributions.FloatDistribution(1e-5, 5e-3, log=True),
                        "weight_decay":    optuna.distributions.FloatDistribution(1e-6, 1e-2, log=True),
                        "label_smoothing": optuna.distributions.FloatDistribution(0.0,  0.20),
                        "dropout":         optuna.distributions.FloatDistribution(0.0,  0.40),
                        "batch_size":      optuna.distributions.CategoricalDistribution([8,16,32]),
                    },
                    value=r["val_acc"],
                ))
            except Exception:
                pass

        remaining = max(0, N_TRIALS - n_done)
        if remaining > 0:
            objective = make_objective(arch, device)
            study.optimize(objective, n_trials=remaining,
                           callbacks=[lambda s, t: logger.info(
                               "  [%s] trial %d: val=%.4f  lr=%.2e  wd=%.2e  ls=%.3f  bs=%d",
                               name, t.number, t.value or 0,
                               t.params.get("lr",0), t.params.get("weight_decay",0),
                               t.params.get("label_smoothing",0), t.params.get("batch_size",0)
                           )])

            for t in study.trials:
                if t.state == optuna.trial.TrialState.COMPLETE:
                    all_rows.append({
                        "model": name, "trial": t.number,
                        "val_acc": t.value, "is_best_retrain": False,
                        **t.params,
                    })
            pd.DataFrame(all_rows).to_csv(all_csv, index=False)

        # ── Retrain best config for full 20 epochs ────────────────────────────
        best_trial = study.best_trial
        best_params = best_trial.params
        logger.info("Best trial for %s: val=%.4f  params=%s",
                    name, best_trial.value, best_params)

        train_loader, val_loader, test_loader = get_loaders(int(best_params["batch_size"]))
        model = build_model(arch, float(best_params["dropout"]), device)
        val_acc = train_epochs(
            model, train_loader, val_loader,
            float(best_params["lr"]), float(best_params["weight_decay"]),
            float(best_params["label_smoothing"]),
            N_FULL_EPOCHS, device
        )
        test_acc  = _eval_acc(model, test_loader, device)
        test_top3 = _eval_top3(model, test_loader, device)

        # Save best checkpoint
        torch.save(model.state_dict(), CKPT_DIR / f"hpo_{name}_best.pt")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        row = {"model": name, "trial": "best_retrain",
               "val_acc": val_acc, "test_acc": test_acc, "test_top3": test_top3,
               "is_best_retrain": True, **best_params}
        all_rows.append(row)
        best_rows.append(row)

        pd.DataFrame(all_rows).to_csv(all_csv, index=False)
        pd.DataFrame(best_rows).to_csv(best_csv, index=False)
        logger.info("[%s] HPO DONE — test_acc=%.4f  (was %.4f before HPO)",
                    name, test_acc, 0.9409 if "efficientnet_b3_cross_attention" in name else 0.9376)

    # ── Plots ─────────────────────────────────────────────────────────────────
    trial_rows = [r for r in all_rows if not r.get("is_best_retrain")]
    plot_hpo_results(trial_rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("HPO COMPLETE")
    if best_rows:
        df = pd.DataFrame(best_rows)
        print(df[["model","test_acc","test_top3","lr","weight_decay",
                  "label_smoothing","dropout","batch_size"]].to_string(index=False))


if __name__ == "__main__":
    main()
