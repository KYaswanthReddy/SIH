"""
Tests for Patch Spatial Alignment, Affine Transforms, and Dimensions.
"""

import os
import pytest
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from gis.crs import pixel_to_geo, geo_to_pixel


@pytest.fixture
def config():
    with open("configs/dataset_config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_patch_files_and_dimensions(config):
    metadata_csv = os.path.join(config["paths"]["metadata_dir"], "patches_all.csv")
    processed_dir = config["paths"]["processed_dir"]
    
    if not os.path.exists(metadata_csv):
        pytest.skip("Patches metadata not found. Run prepare_data.py first.")
        
    df = pd.read_csv(metadata_csv)
    sample_df = df.sample(min(25, len(df)), random_state=42)
    
    for _, row in sample_df.iterrows():
        img_path = os.path.join(processed_dir, row["image_path"])
        mask_path = os.path.join(processed_dir, row["mask_path"])
        
        assert os.path.exists(img_path), f"Missing patch image: {img_path}"
        assert os.path.exists(mask_path), f"Missing patch mask: {mask_path}"
        
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        # Check dimensions
        assert img.size == (512, 512), f"Expected (512, 512) for {img_path}, got {img.size}"
        assert mask.size == (512, 512), f"Expected (512, 512) for {mask_path}, got {mask.size}"
        
        # Check modes
        assert img.mode == "RGB", f"Expected RGB mode for {img_path}, got {img.mode}"
        assert mask.mode == "L", f"Expected grayscale mode for {mask_path}, got {mask.mode}"


def test_patch_georeferencing_alignment(config):
    metadata_csv = os.path.join(config["paths"]["metadata_dir"], "patches_all.csv")
    if not os.path.exists(metadata_csv):
        pytest.skip("Patches metadata not found. Run prepare_data.py first.")
        
    df = pd.read_csv(metadata_csv)
    sample_df = df.sample(min(20, len(df)), random_state=42)
    
    for _, row in sample_df.iterrows():
        # Top-left corner in pixel coords: (0, 0)
        # In EPSG:28992, transform maps (col=0, row=0) to (minx, maxy)
        transform = eval(row["transform"]) if isinstance(row["transform"], str) else row["transform"]
        top_left_x, top_left_y = pixel_to_geo(0, 0, transform)
        bottom_right_x, bottom_right_y = pixel_to_geo(512, 512, transform)
        
        assert abs(top_left_x - row["minx"]) < 1e-3, f"Mismatch in minx for {row['patch_id']}"
        assert abs(top_left_y - row["maxy"]) < 1e-3, f"Mismatch in maxy for {row['patch_id']}"
        assert abs(bottom_right_x - row["maxx"]) < 1e-3, f"Mismatch in maxx for {row['patch_id']}"
        assert abs(bottom_right_y - row["miny"]) < 1e-3, f"Mismatch in miny for {row['patch_id']}"
