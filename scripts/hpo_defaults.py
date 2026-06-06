"""
Shared helper: load best HPO hyperparameters from the fixed-model HPO run.
For configs not covered by HPO, falls back to medians from the search.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HPO_CSV = Path("/data/DOVE/results/tables/hpo_fixed_best_configs.csv")

# Medians computed from the 5 HPO best configs
_FALLBACK = {
    "lr":              3.096e-05,
    "weight_decay":    5.415e-06,
    "label_smoothing": 0.066,
    "dropout":         0.316,
    "batch_size":      8,
}

def load_hpo_configs() -> dict[str, dict]:
    """Return {model_name: {lr, weight_decay, label_smoothing, dropout, batch_size}}."""
    if not HPO_CSV.exists():
        return {}
    df = pd.read_csv(HPO_CSV)
    df = df[df["trial"] == "best_retrain"]
    out = {}
    for _, row in df.iterrows():
        out[row["model"]] = {
            "lr":              float(row["lr"]),
            "weight_decay":    float(row["weight_decay"]),
            "label_smoothing": float(row["label_smoothing"]),
            "dropout":         float(row["dropout"]),
            "batch_size":      int(row["batch_size"]),
        }
    return out


def get_hparams(config_name: str, hpo_configs: dict) -> dict:
    """
    Return hyperparameters for a config, matching by name.
    For motion/triple configs, strips 'motion_inv_' or 'triple_' prefix
    before looking up, so e.g. 'triple_efficientnet_b3_concat_linear'
    reuses the HPO result for 'efficientnet_b3_concat_linear'.
    """
    # Strip experiment-set prefix
    lookup_name = config_name
    for prefix in ("triple_", "motion_inv_"):
        if config_name.startswith(prefix):
            lookup_name = config_name[len(prefix):]
            break

    if lookup_name in hpo_configs:
        return hpo_configs[lookup_name]

    # For motion+inv configs (no backbone), match on fusion+head suffix
    for key, hparams in hpo_configs.items():
        suffix = "_".join(key.split("_")[-2:])  # e.g. 'concat_linear'
        if config_name.endswith(suffix):
            return hparams

    return dict(_FALLBACK)
