"""
DOVE — Ablation study runner (12 ablations).
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# All 12 ablations as (name, config_delta)
_ABLATIONS: List[Dict[str, Any]] = [
    # Backbone ablations
    {"name": "backbone_vgg19",         "model.backbone": "vgg19"},
    {"name": "backbone_mobilenet",     "model.backbone": "mobilenet"},
    {"name": "backbone_efficientnet",  "model.backbone": "efficientnet_b3"},
    {"name": "backbone_swin_t",        "model.backbone": "swin_t"},
    # Fusion ablations
    {"name": "fusion_concat",          "model.fusion": "concat"},
    {"name": "fusion_cross_attention", "model.fusion": "cross_attention"},
    # Head ablations
    {"name": "head_mlp",               "model.head": "mlp"},
    {"name": "head_linear",            "model.head": "linear"},
    {"name": "head_rf",                "model.head": "random_forest"},
    {"name": "head_nb",                "model.head": "naive_bayes"},
    # Feature ablations
    {"name": "no_motion",              "ablate.motion": False},
    {"name": "no_invariant",           "ablate.invariant": False},
]


def _apply_delta(config: dict, delta: dict) -> dict:
    """Apply a flat delta dict (dot-separated keys) to a nested config."""
    cfg = copy.deepcopy(config)
    for dotkey, val in delta.items():
        if dotkey == "name":
            continue
        parts = dotkey.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return cfg


class AblationRunner:
    """
    Runs all 12 ablations from a base config and a best experiment config.

    Parameters
    ----------
    base_config           : dict — base hyperparameters
    best_experiment_name  : str — name of the best experiment (used for logging)
    experiment_fn         : Callable[[dict, str], float]
                            takes (config, ablation_name) → val_accuracy
    """

    def __init__(
        self,
        base_config: Dict[str, Any],
        best_experiment_name: str,
        experiment_fn: Optional[Callable[[Dict[str, Any], str], float]] = None,
    ):
        self.base_config = base_config
        self.best_experiment_name = best_experiment_name
        self.experiment_fn = experiment_fn or self._dummy_experiment

    @staticmethod
    def _dummy_experiment(config: dict, name: str) -> float:
        """Placeholder experiment function."""
        logger.warning("No experiment_fn provided; using dummy for ablation '%s'", name)
        import random
        return random.uniform(0.4, 0.85)

    def run_all(self) -> pd.DataFrame:
        """
        Run all 12 ablations.

        Returns
        -------
        pd.DataFrame — columns: ablation_name, accuracy, ...config fields
        """
        results = []
        for ab in _ABLATIONS:
            name = ab["name"]
            delta = {k: v for k, v in ab.items() if k != "name"}
            cfg = _apply_delta(self.base_config, delta)
            cfg["name"] = name

            logger.info("Running ablation: %s", name)
            try:
                acc = self.experiment_fn(cfg, name)
            except Exception as exc:
                logger.error("Ablation %s failed: %s", name, exc)
                acc = float("nan")

            row = {"ablation_name": name, "accuracy": acc}
            row.update({k: v for k, v in delta.items()})
            results.append(row)
            logger.info("Ablation %s → accuracy=%.4f", name, acc)

        df = pd.DataFrame(results)

        # Save
        out_dir = Path("results/tables")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "ablations.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Ablation results saved to %s", csv_path)
        return df


if __name__ == "__main__":
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)

    base = {
        "model": {"backbone": "swin_t", "fusion": "cross_attention", "head": "mlp"},
        "training": {"lr": 1e-4, "n_epochs": 50},
        "seed": 42,
    }

    def fake_experiment(config: dict, name: str) -> float:
        return random.uniform(0.5, 0.85)

    runner = AblationRunner(base, "best_experiment", experiment_fn=fake_experiment)
    df = runner.run_all()
    print("Ablation results:")
    print(df.to_string())
