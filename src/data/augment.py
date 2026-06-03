"""
DOVE — Augmentation pipelines.

Provides:
- get_image_transform(train)   : torchvision transform for single frames / images
- TrajectoryAugment            : time-reversal and Gaussian noise for trajectory arrays
"""
from __future__ import annotations

import logging
import random

import numpy as np
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet statistics
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_image_transform(train: bool = True) -> transforms.Compose:
    """
    Return a torchvision Compose pipeline for single images.

    Train:
        RandomHorizontalFlip(0.5) → RandomResizedCrop(224, scale=(0.7,1.0))
        → ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        → ToTensor → Normalize(ImageNet)

    Val / test:
        Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=_MEAN, std=_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )


class TrajectoryAugment:
    """
    Augmentations for 2-D trajectory arrays of shape (T, 2) — columns are (x, y).

    - Time reversal  : p=0.3, reverses the temporal order of points.
    - Gaussian noise : p=0.5, adds i.i.d. N(0, sigma) to each coordinate.
    """

    def __init__(self, reverse_p: float = 0.3, noise_p: float = 0.5, sigma: float = 0.5):
        self.reverse_p = reverse_p
        self.noise_p = noise_p
        self.sigma = sigma

    def __call__(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        trajectory : np.ndarray, shape (T, 2)

        Returns
        -------
        np.ndarray, shape (T, 2)
        """
        traj = trajectory.copy().astype(np.float32)
        if random.random() < self.reverse_p:
            traj = traj[::-1].copy()
            logger.debug("TrajectoryAugment: applied time reversal")
        if random.random() < self.noise_p:
            noise = np.random.normal(0.0, self.sigma, traj.shape).astype(np.float32)
            traj = traj + noise
            logger.debug("TrajectoryAugment: applied Gaussian noise sigma=%.2f", self.sigma)
        return traj


if __name__ == "__main__":
    import torch
    from PIL import Image

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Image transforms
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )
    for mode in (True, False):
        t = get_image_transform(train=mode)
        out = t(dummy_img)
        print(f"Image transform (train={mode}) output shape: {out.shape}")

    # Trajectory augment
    traj = np.random.randn(64, 2).astype(np.float32)
    aug = TrajectoryAugment()
    out_traj = aug(traj)
    print(f"TrajectoryAugment input shape: {traj.shape}, output shape: {out_traj.shape}")
