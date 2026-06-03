"""
DOVE — Compute (FLOPs, params, inference time) and Pareto plot.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from fvcore.nn import FlopCountAnalysis
    _FVCORE_AVAILABLE = True
except ImportError:
    _FVCORE_AVAILABLE = False
    logger.warning("fvcore not available — count_flops will return -1")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def count_flops(
    model: "nn.Module",
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
) -> int:
    """
    Count FLOPs using fvcore.

    Returns -1 if fvcore is unavailable.
    """
    if not _FVCORE_AVAILABLE or not _TORCH_AVAILABLE:
        return -1
    import torch
    model.eval()
    dummy = torch.zeros(*input_size)
    try:
        flops = FlopCountAnalysis(model, dummy)
        return int(flops.total())
    except Exception as exc:
        logger.warning("FlopCountAnalysis failed: %s", exc)
        return -1


def count_params(model: "nn.Module") -> int:
    """Count trainable parameters."""
    if not _TORCH_AVAILABLE:
        return -1
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def measure_inference_time(
    model: "nn.Module",
    n_runs: int = 100,
    device: str = "cpu",
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
) -> float:
    """
    Measure mean inference time (ms/image).

    Returns
    -------
    float — mean milliseconds per forward pass
    """
    if not _TORCH_AVAILABLE:
        return -1.0
    import torch

    model.eval()
    model.to(device)
    dummy = torch.zeros(*input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times))


def plot_compute_pareto(
    results_df: pd.DataFrame,
    save_path: str | Path,
    flops_col: str = "flops",
    acc_col: str = "accuracy",
    name_col: str = "name",
) -> None:
    """
    Scatter plot of FLOPs vs accuracy (Pareto front).

    Parameters
    ----------
    results_df : DataFrame with at least flops_col, acc_col, name_col columns
    save_path  : output path
    """
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib not available; skipping Pareto plot")
        return

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = results_df[flops_col].values
    y = results_df[acc_col].values
    names = results_df[name_col].values if name_col in results_df.columns else [""] * len(x)

    ax.scatter(x, y, c="steelblue", alpha=0.7, s=60)
    for xi, yi, name in zip(x, y, names):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)

    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Compute vs Accuracy Pareto")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Pareto plot saved to %s", save_path)


if __name__ == "__main__":
    import tempfile, os
    import numpy as np
    import pandas as pd

    np.random.seed(42)

    if _TORCH_AVAILABLE:
        import torch
        import torch.nn as nn

        torch.manual_seed(42)
        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 20))

        flops = count_flops(model, input_size=(1, 3, 224, 224))
        params = count_params(model)
        inf_time = measure_inference_time(model, n_runs=10, device="cpu")
        print(f"FLOPs: {flops}, Params: {params}, Inference: {inf_time:.2f} ms")

    # Pareto plot
    df = pd.DataFrame({
        "name": [f"model_{i}" for i in range(10)],
        "flops": np.random.randint(1e6, 1e9, 10),
        "accuracy": np.random.uniform(0.5, 0.9, 10),
    })
    with tempfile.TemporaryDirectory() as tmp:
        plot_compute_pareto(df, f"{tmp}/pareto.png")
        print("Pareto plot created:", os.path.exists(f"{tmp}/pareto.png"))
