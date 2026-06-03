"""
DOVE — Hyperparameter optimisation with Optuna.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict

import yaml

logger = logging.getLogger(__name__)

try:
    import optuna
    from optuna.visualization import plot_parallel_coordinate
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    logger.warning("optuna not available — HyperparamOptimizer will not run")

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


_SEARCH_SPACE = {
    "lr": ("float_log", 1e-5, 1e-2),
    "weight_decay": ("float_log", 1e-6, 1e-2),
    "dropout": ("float", 0.1, 0.5),
    "batch_size": ("categorical", [16, 32, 64]),
    "hidden_dim": ("categorical", [128, 256, 512]),
    "n_epochs": ("int", 10, 50),
    "patience": ("int", 3, 10),
    "grad_clip": ("float", 0.5, 5.0),
    "fusion": ("categorical", ["concat", "cross_attention"]),
    "backbone": ("categorical", ["swin_t", "efficientnet_b3", "mobilenet", "vgg19"]),
    "head": ("categorical", ["mlp", "linear"]),
    "tau": ("float", 0.05, 0.5),
    "temperature": ("float", 1.0, 10.0),
    "alpha": ("float", 0.0, 1.0),
}


def _suggest(trial: "optuna.Trial", name: str, spec: tuple) -> Any:
    kind = spec[0]
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    elif kind == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    elif kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    elif kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"Unknown spec kind: {kind}")


class HyperparamOptimizer:
    """
    Optuna-based hyperparameter optimiser.

    Parameters
    ----------
    experiment_fn    : Callable[[dict], float]  — takes a config dict, returns val accuracy
    experiment_name  : str
    n_trials         : int
    """

    def __init__(
        self,
        experiment_fn: Callable[[Dict[str, Any]], float],
        experiment_name: str = "dove",
        n_trials: int = 50,
    ):
        self.experiment_fn = experiment_fn
        self.experiment_name = experiment_name
        self.n_trials = n_trials

    def optimize(self) -> Dict[str, Any]:
        """
        Run the Optuna study.

        Returns
        -------
        dict — best hyperparameters
        """
        if not _OPTUNA_AVAILABLE:
            logger.error("optuna not available — cannot optimise")
            return {}

        def objective(trial: "optuna.Trial") -> float:
            config = {name: _suggest(trial, name, spec) for name, spec in _SEARCH_SPACE.items()}
            return self.experiment_fn(config)

        study = optuna.create_study(
            direction="maximize",
            study_name=self.experiment_name,
        )
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        best_params = study.best_params

        # Save best params
        cfg_dir = Path("configs/experiments")
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / f"{self.experiment_name}_best.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)
        logger.info("Best params saved to %s", cfg_path)

        # Save parallel coordinates plot
        fig_dir = Path("results/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        try:
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(str(fig_dir / f"{self.experiment_name}_parallel_coord.png"))
            logger.info("Parallel coordinates plot saved")
        except Exception as exc:
            logger.warning("Could not save parallel coordinates plot: %s", exc)

        return best_params


if __name__ == "__main__":
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)

    def dummy_experiment(config: dict) -> float:
        # Simulate a random accuracy
        return random.uniform(0.4, 0.9)

    if _OPTUNA_AVAILABLE:
        opt = HyperparamOptimizer(dummy_experiment, "smoke_test", n_trials=3)
        best = opt.optimize()
        print("Best params:", best)
    else:
        print("optuna not installed — skipping smoke test")
