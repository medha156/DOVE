"""
DOVE — Generic Trainer with AdamW, CosineAnnealingLR, early stopping, AMP.
"""
from __future__ import annotations

import csv
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Full training loop with:
        - AdamW optimiser
        - CosineAnnealingLR scheduler
        - Early stopping on val accuracy
        - Gradient clipping
        - AMP (automatic mixed precision)
        - CSV logging to results/tables/{name}_log.csv

    Parameters
    ----------
    model        : nn.Module
    train_loader : DataLoader
    val_loader   : DataLoader
    config       : dict — keys: lr, weight_decay, dropout, n_epochs, patience,
                          grad_clip, mixed_precision, seed, num_classes,
                          name (experiment name, default 'dove')
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
    ):
        _set_seeds(config.get("seed", 42))
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("n_epochs", 50),
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.scaler = GradScaler(enabled=config.get("mixed_precision", True) and torch.cuda.is_available())

        self.n_epochs = config.get("n_epochs", 50)
        self.patience = config.get("patience", 5)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.mixed_precision = config.get("mixed_precision", True) and torch.cuda.is_available()
        self.name = config.get("name", "dove")

        # Log file
        log_dir = Path("results/tables")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{self.name}_log.csv"
        self._init_log()

    def _init_log(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    def _log_row(self, row: list) -> None:
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        """Run one epoch. Returns (loss, accuracy)."""
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                # Support both DOVEDataset dicts and plain (x, y) tuples
                if isinstance(batch, dict):
                    x = batch.get("image", batch.get("frames"))
                    if x is None:
                        continue
                    if x.dim() == 5:
                        # video: (B, T, C, H, W) → mean pool over frames
                        B, T, C, H, W = x.shape
                        x = x.view(B * T, C, H, W)
                    y = batch["label"]
                else:
                    x, y = batch

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad()

                with autocast(enabled=self.mixed_precision):
                    logits = self.model(x)
                    loss = self.criterion(logits, y)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    def train(self) -> float:
        """
        Run the full training loop.

        Returns
        -------
        float — best validation accuracy
        """
        best_val_acc = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            logger.info(
                "Epoch %3d/%d | train_loss=%.4f acc=%.4f | val_loss=%.4f acc=%.4f | lr=%.2e | %.1fs",
                epoch, self.n_epochs, train_loss, train_acc, val_loss, val_acc, lr, elapsed,
            )
            self._log_row([epoch, train_loss, train_acc, val_loss, val_acc, lr])

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)
        return best_val_acc


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Tiny synthetic dataset
    N, D, C = 64, 32, 5
    X = torch.randn(N, D)
    y = torch.randint(0, C, (N,))
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=16)

    model = nn.Linear(D, C)
    config = {
        "lr": 1e-3, "weight_decay": 0, "n_epochs": 3,
        "patience": 10, "grad_clip": 1.0,
        "mixed_precision": False, "seed": 42, "name": "smoke_test",
    }
    trainer = Trainer(model, loader, loader, config)
    best_acc = trainer.train()
    print(f"Best val accuracy: {best_acc:.4f}")
