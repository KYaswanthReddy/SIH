"""
PyTorch Dataset and DataLoader Factories for Cadastral Segmentation.
"""

import os
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import get_train_transforms, get_val_transforms, CadastreAugmentation


class CadastreDataset(Dataset):
    """
    Cadastral boundary segmentation dataset reading from metadata CSV.
    """

    def __init__(
        self,
        metadata_csv: str,
        processed_dir: str = "data/processed",
        transform: Optional[CadastreAugmentation] = None,
        max_samples: Optional[int] = None,
    ):
        self.metadata_csv = metadata_csv
        self.processed_dir = processed_dir
        self.transform = transform
        
        if not os.path.exists(metadata_csv):
            raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
            
        self.df = pd.read_csv(metadata_csv)
        if max_samples is not None and max_samples < len(self.df):
            self.df = self.df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_full_path = os.path.join(self.processed_dir, row["image_path"])
        mask_full_path = os.path.join(self.processed_dir, row["mask_path"])

        # Load RGB image
        img_pil = Image.open(img_full_path).convert("RGB")
        img_np = np.array(img_pil, dtype=np.uint8)

        # Load single-channel binary mask
        mask_pil = Image.open(mask_full_path).convert("L")
        mask_np = np.array(mask_pil, dtype=np.uint8)

        if self.transform is not None:
            img_tensor, mask_tensor = self.transform(img_np, mask_np)
        else:
            img_tensor = torch.from_numpy(np.transpose(img_np.astype(np.float32) / 255.0, (2, 0, 1))).float()
            mask_tensor = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).float()

        return {
            "image": img_tensor,           # Shape: (3, 512, 512), range: [0.0, 1.0]
            "mask": mask_tensor,           # Shape: (1, 512, 512), values: {0.0, 1.0}
            "patch_id": row["patch_id"],
            "split": row["split"],
            "is_urban": bool(row["is_urban"]),
            "boundary_pixels": int(row["boundary_pixels"]),
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
        }


def create_dataloaders(
    config: Dict[str, Any],
    sanity: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        config: Configuration dictionary
        sanity: If True, uses a small sample subset and smaller batch size for fast verification
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    processed_dir = config["data"]["processed_dir"]
    train_csv = config["data"]["train_metadata"]
    val_csv = config["data"]["val_metadata"]
    test_csv = config["data"]["test_metadata"]

    train_aug = get_train_transforms(config)
    val_aug = get_val_transforms()

    if sanity:
        max_samples = config.get("sanity_check", {}).get("num_samples", 128)
        batch_size = config.get("sanity_check", {}).get("batch_size", 8)
        num_workers = 0 # Avoid multi-processing overhead in tiny sanity runs
    else:
        max_samples = None
        batch_size = config["training"]["batch_size"]
        # Use reasonable num_workers (e.g. 2-4, or 0 on non-CUDA systems)
        num_workers = config["training"].get("num_workers", 2)
        if not torch.cuda.is_available():
            num_workers = 0

    pin_memory = config["training"].get("pin_memory", True) and torch.cuda.is_available()

    train_ds = CadastreDataset(
        metadata_csv=train_csv,
        processed_dir=processed_dir,
        transform=train_aug,
        max_samples=max_samples
    )

    val_max_samples = (max_samples // 4) if sanity and max_samples else None
    val_ds = CadastreDataset(
        metadata_csv=val_csv,
        processed_dir=processed_dir,
        transform=val_aug,
        max_samples=val_max_samples
    )

    test_ds = CadastreDataset(
        metadata_csv=test_csv,
        processed_dir=processed_dir,
        transform=val_aug,
        max_samples=val_max_samples
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_ds) > batch_size
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
