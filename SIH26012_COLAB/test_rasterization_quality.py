"""
Tests for Rasterization Quality, Boundary Mask Values, and Geometry Validity.
"""

import os
import pytest
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from gis.rasterizer import rasterize_cadastral_boundaries, compute_mask_statistics
from shapely.geometry import LineString


@pytest.fixture
def config():
    with open("configs/dataset_config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_synthetic_rasterization():
    """Test unit rasterization logic with known synthetic LineStrings."""
    # 512x512 patch covering [0, 0] to [128, 128] in world coords
    # 1 pixel = 0.25m -> transform = [0.25, 0, 0, 0, -0.25, 128]
    transform = [0.25, 0.0, 0.0, 0.0, -0.25, 128.0]
    
    # Line across diagonal: (0, 128) to (128, 0)
    line1 = LineString([(0, 128), (128, 0)])
    
    mask = rasterize_cadastral_boundaries(
        geometries=[line1],
        patch_shape=(512, 512),
        patch_transform=transform,
        line_thickness=2
    )
    
    assert mask.shape == (512, 512)
    assert set(np.unique(mask)).issubset({0, 1})
    assert np.sum(mask == 1) > 0, "Rasterized mask is empty!"
    
    stats = compute_mask_statistics(mask)
    assert not stats["is_empty"]
    assert stats["boundary_pixels"] > 500


def test_dataset_rasterization_masks(config):
    metadata_csv = os.path.join(config["paths"]["metadata_dir"], "patches_all.csv")
    processed_dir = config["paths"]["processed_dir"]
    
    if not os.path.exists(metadata_csv):
        pytest.skip("Patches metadata not found. Run prepare_data.py first.")
        
    df = pd.read_csv(metadata_csv)
    
    # 1. Check boundary-containing patches have non-empty masks
    boundary_samples = df[df["num_lines"] > 5].sample(min(20, len(df)), random_state=42)
    for _, row in boundary_samples.iterrows():
        mask_path = os.path.join(processed_dir, row["mask_path"])
        mask_arr = np.array(Image.open(mask_path))
        
        # Values must be binary {0, 1}
        unique_vals = set(np.unique(mask_arr))
        assert unique_vals.issubset({0, 1}), f"Non-binary values {unique_vals} found in {mask_path}"
        assert np.sum(mask_arr == 1) > 0, f"Expected non-empty mask for {row['patch_id']}"
        assert row["boundary_pixels"] == np.sum(mask_arr == 1)
