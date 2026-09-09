"""
Data Transforms and Lightweight Augmentations for Cadastral Segmentation.
"""

import random
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch


class CadastreAugmentation:
    """
    Lightweight geometric and photometric augmentations for paired image and boundary mask.
    """

    def __init__(
        self,
        is_train: bool = True,
        target_size: Optional[Tuple[int, int]] = None,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.5,
        brightness_contrast_prob: float = 0.3,
        brightness_limit: float = 0.15,
        contrast_limit: float = 0.15,
    ):
        self.is_train = is_train
        self.target_size = target_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        self.brightness_contrast_prob = brightness_contrast_prob
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to image (H, W, 3) uint8 and mask (H, W) uint8 {0, 1}.
        Returns:
            img_tensor: (3, H, W) float32 in [0, 1]
            mask_tensor: (1, H, W) float32 in {0.0, 1.0}
        """
        # Ensure float32 work copies
        img = image.astype(np.float32) / 255.0
        m = mask.astype(np.float32)
        if len(m.shape) == 2:
            m = np.expand_dims(m, axis=-1) # (H, W, 1)

        # Optional resize
        if self.target_size is not None:
            th, tw = self.target_size
            if img.shape[0] != th or img.shape[1] != tw:
                import cv2
                img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
                m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
                if len(m.shape) == 2:
                    m = np.expand_dims(m, axis=-1)

        if self.is_train:
            # 1. Horizontal Flip
            if random.random() < self.hflip_prob:
                img = np.fliplr(img).copy()
                m = np.fliplr(m).copy()

            # 2. Vertical Flip
            if random.random() < self.vflip_prob:
                img = np.flipud(img).copy()
                m = np.flipud(m).copy()

            # 3. Random 90-degree rotations (0, 90, 180, 270 degrees)
            if random.random() < self.rot90_prob:
                k = random.randint(1, 3)
                img = np.rot90(img, k=k, axes=(0, 1)).copy()
                m = np.rot90(m, k=k, axes=(0, 1)).copy()

            # 4. Subtle Brightness & Contrast
            if random.random() < self.brightness_contrast_prob:
                alpha = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
                beta = random.uniform(-self.brightness_limit, self.brightness_limit)
                img = np.clip(alpha * img + beta, 0.0, 1.0)

        # Convert to PyTorch tensors (C, H, W)
        img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(np.transpose(m, (2, 0, 1))).float()

        return img_tensor, mask_tensor


def get_train_transforms(config: Optional[Dict[str, Any]] = None) -> CadastreAugmentation:
    aug_cfg = config.get("augmentation", {}) if config else {}
    data_cfg = config.get("data", {}) if config else {}
    img_size = data_cfg.get("image_size", None)
    target_size = tuple(img_size) if img_size else None
    return CadastreAugmentation(
        is_train=True,
        target_size=target_size,
        hflip_prob=aug_cfg.get("horizontal_flip_prob", 0.5),
        vflip_prob=aug_cfg.get("vertical_flip_prob", 0.5),
        rot90_prob=aug_cfg.get("rotate90_prob", 0.5),
        brightness_contrast_prob=aug_cfg.get("brightness_contrast_prob", 0.3),
        brightness_limit=aug_cfg.get("brightness_limit", 0.15),
        contrast_limit=aug_cfg.get("contrast_limit", 0.15),
    )


def get_val_transforms(config: Optional[Dict[str, Any]] = None) -> CadastreAugmentation:
    data_cfg = config.get("data", {}) if config else {}
    img_size = data_cfg.get("image_size", None)
    target_size = tuple(img_size) if img_size else None
    return CadastreAugmentation(is_train=False, target_size=target_size)
