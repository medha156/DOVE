"""
DOVE — ATSS (Adaptive Training Sample Selection) backbone wrapper.

If mmdet is available, uses the mmdet ATSS detector backbone.
Otherwise, falls back to Faster R-CNN.
"""
from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    import mmdet  # noqa: F401
    _MMDET_AVAILABLE = True
except ImportError:
    _MMDET_AVAILABLE = False
    logger.info("mmdet not available — ATSSBackbone falls back to FasterRCNN")


class ATSSBackbone(nn.Module):
    """
    ATSS detector backbone (ResNet-50 + FPN).

    Falls back to Faster R-CNN backbone when mmdet is not installed.
    feature_dim: 256.

    Note: Full ATSS requires mmdet >= 3.0. Without it, this class wraps
    FasterRCNNBackbone as an equivalent FPN-based feature extractor.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        if _MMDET_AVAILABLE:
            try:
                from mmdet.models import build_backbone
                cfg = dict(
                    type="ResNet",
                    depth=50,
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    frozen_stages=1,
                    norm_cfg=dict(type="BN", requires_grad=True),
                    norm_eval=True,
                    style="pytorch",
                    init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50") if pretrained else None,
                )
                self.backbone = build_backbone(cfg)
                self.backbone.init_weights()
                self._use_mmdet = True
                self._feature_dim = 256
                logger.info("ATSSBackbone using mmdet ResNet-50")
            except Exception as exc:
                logger.warning("mmdet ATSS build failed (%s), falling back", exc)
                self._build_fallback(pretrained)
        else:
            self._build_fallback(pretrained)

    def _build_fallback(self, pretrained: bool) -> None:
        from .faster_rcnn import FasterRCNNBackbone
        self.backbone = FasterRCNNBackbone(pretrained=pretrained)
        self._use_mmdet = False
        self._feature_dim = 256

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> Dict:
        return self.backbone(x)


if __name__ == "__main__":
    import torch

    torch.manual_seed(42)
    model = ATSSBackbone(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    if isinstance(out, dict):
        print("ATSSBackbone output keys:", list(out.keys()))
        for k, v in out.items():
            print(f"  {k}: {v.shape}")
    else:
        print("ATSSBackbone output:", [o.shape for o in out])
    print("feature_dim:", model.feature_dim)
