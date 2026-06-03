"""
DOVE — Filtration-Distillation Learning (FDL) Trainer.

Extends Trainer with:
- FFA-based filtration of proposals below tau
- Knowledge distillation loss: L_total = L_task + alpha * L_KD
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from .trainer import Trainer

logger = logging.getLogger(__name__)


def _kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 4.0,
) -> torch.Tensor:
    """
    Hinton-style knowledge distillation loss (KL divergence between soft targets).
    """
    T = temperature
    s_log = F.log_softmax(student_logits / T, dim=1)
    t_soft = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s_log, t_soft, reduction="batchmean") * (T ** 2)


class FDLTrainer(Trainer):
    """
    Filtration-Distillation Learning trainer.

    Parameters
    ----------
    teacher_model : nn.Module  — frozen teacher
    student_model : nn.Module  — trainable student
    train_loader  : DataLoader
    val_loader    : DataLoader
    config        : dict       — same as Trainer, plus fdl keys:
                                  tau (float), temperature (float), alpha (float)
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
    ):
        # Trainer is initialised with the student model
        super().__init__(student_model, train_loader, val_loader, config)
        self.teacher = teacher_model.to(self.device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.tau = config.get("tau", 0.1)
        self.temperature = config.get("temperature", 4.0)
        self.alpha = config.get("alpha", 0.5)
        logger.info(
            "FDLTrainer: tau=%.2f, temperature=%.1f, alpha=%.2f",
            self.tau, self.temperature, self.alpha,
        )

    def _ffa_filter(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Suppress proposals whose max softmax score is below tau.
        Returns a boolean mask: True = keep.
        """
        scores = F.softmax(logits, dim=1).max(dim=1).values
        return scores >= self.tau

    def _compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """L_total = L_task + alpha * L_KD (on filtered proposals)."""
        # Filtration: only distil on samples where teacher is confident
        mask = self._ffa_filter(teacher_logits)
        if mask.sum() == 0:
            mask = torch.ones_like(mask)  # keep all if none pass filter

        l_task = self.criterion(student_logits, labels)

        # KD only on filtered subset
        l_kd = _kd_loss(
            student_logits[mask],
            teacher_logits[mask],
            temperature=self.temperature,
        )
        return l_task + self.alpha * l_kd

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        """Override to include KD loss during training."""
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                if isinstance(batch, dict):
                    x = batch.get("image", batch.get("frames"))
                    if x is None:
                        continue
                    if x.dim() == 5:
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
                    student_logits = self.model(x)
                    with torch.no_grad():
                        teacher_logits = self.teacher(x)

                    if train:
                        loss = self._compute_loss(student_logits, teacher_logits, y)
                    else:
                        loss = self.criterion(student_logits, y)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * y.size(0)
                preds = student_logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        if total == 0:
            return 0.0, 0.0
        return total_loss / total, correct / total


if __name__ == "__main__":
    import random
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    N, D, C = 64, 32, 5
    X = torch.randn(N, D)
    y = torch.randint(0, C, (N,))
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=16)

    teacher = nn.Linear(D, C)
    student = nn.Linear(D, C)

    config = {
        "lr": 1e-3, "weight_decay": 0, "n_epochs": 3,
        "patience": 10, "grad_clip": 1.0,
        "mixed_precision": False, "seed": 42,
        "tau": 0.1, "temperature": 4.0, "alpha": 0.5,
        "name": "fdl_smoke",
    }
    trainer = FDLTrainer(teacher, student, loader, loader, config)
    best_acc = trainer.train()
    print(f"FDLTrainer best val accuracy: {best_acc:.4f}")
