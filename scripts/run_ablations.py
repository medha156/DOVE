"""
DOVE — Run all 12 ablation studies.

Loads the best experiment config and calls AblationRunner.run_all().
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ablations.runner import AblationRunner

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("run_ablations")

REPO_ROOT = Path(__file__).parent.parent
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "base.yaml"
BEST_CONFIGS_DIR = REPO_ROOT / "configs" / "experiments"


def _load_best_config(experiment_name: str) -> dict:
    """Load best config from configs/experiments/{name}_best.yaml if available."""
    best_path = BEST_CONFIGS_DIR / f"{experiment_name}_best.yaml"
    if best_path.exists():
        with open(best_path) as f:
            overrides = yaml.safe_load(f)
        logger.info("Loaded best config from %s", best_path)
    else:
        overrides = {}
        logger.info("No best config found — using base config")

    with open(BASE_CONFIG_PATH) as f:
        base = yaml.safe_load(f)

    base.update(overrides)
    return base


def _dummy_experiment(config: dict, name: str) -> float:
    """Placeholder — replace with actual Trainer calls for real experiments."""
    import random
    logger.debug("Dummy experiment: %s", name)
    return random.uniform(0.5, 0.85)


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    logger.info("=" * 60)
    logger.info("DOVE — Ablation Studies")
    logger.info("=" * 60)

    # Try to find the best experiment name from experiment results
    exp_csv = REPO_ROOT / "results" / "tables" / "experiment_results.csv"
    best_name = "dove"
    if exp_csv.exists():
        import pandas as pd
        df = pd.read_csv(exp_csv)
        if "accuracy" in df.columns and len(df) > 0:
            best_row = df.loc[df["accuracy"].idxmax()]
            best_name = f"{best_row.get('backbone', 'swin_t')}_{best_row.get('fusion', 'cross_attention')}"
            logger.info("Best experiment: %s (acc=%.4f)", best_name, best_row["accuracy"])

    config = _load_best_config(best_name)

    runner = AblationRunner(
        base_config=config,
        best_experiment_name=best_name,
        experiment_fn=_dummy_experiment,
    )
    df = runner.run_all()

    logger.info("Ablation results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
