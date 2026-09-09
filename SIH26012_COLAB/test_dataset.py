"""
Tests for CadastreDataset, Augmentations, and DataLoaders.
"""

import os
import pytest
import torch
import yaml
from ai.dataset import CadastreDataset, create_dataloaders
from ai.transforms import get_train_transforms, get_val_transforms


@pytest.fixture
def config():
    with open("configs/baseline_train_config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_cadastre_dataset_loading(config):
    train_csv = config["data"]["train_metadata"]
    processed_dir = config["data"]["processed_dir"]
    
    assert os.path.exists(train_csv), f"Missing metadata: {train_csv}"
    
    ds = CadastreDataset(metadata_csv=train_csv, processed_dir=processed_dir, max_samples=5)
    assert len(ds) == 5
    
    item = ds[0]
    assert "image" in item
    assert "mask" in item
    assert "patch_id" in item
    
    img = item["image"]
    mask = item["mask"]
    
    # Verify shapes
    assert img.shape == (3, 512, 512), f"Unexpected image shape: {img.shape}"
    assert mask.shape == (1, 512, 512), f"Unexpected mask shape: {mask.shape}"
    
    # Verify data types and value ranges
    assert img.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert 0.0 <= img.min() <= img.max() <= 1.0
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})


def test_dataset_augmentations(config):
    train_csv = config["data"]["train_metadata"]
    processed_dir = config["data"]["processed_dir"]
    
    transform = get_train_transforms(config)
    ds = CadastreDataset(metadata_csv=train_csv, processed_dir=processed_dir, transform=transform, max_samples=4)
    
    for i in range(len(ds)):
        item = ds[i]
        assert item["image"].shape == (3, 512, 512)
        assert item["mask"].shape == (1, 512, 512)


def test_dataloaders_creation(config):
    train_loader, val_loader, test_loader = create_dataloaders(config, sanity=True)
    
    batch = next(iter(train_loader))
    expected_batch_size = config.get("sanity_check", {}).get("batch_size", 8)
    assert batch["image"].shape[0] == expected_batch_size
    assert batch["image"].shape[1:] == (3, 512, 512)
    assert batch["mask"].shape[1:] == (1, 512, 512)
