"""
Tests for Split Integrity, Validation against sample_tiles_10k_split, and Zero Train/Val/Test Leakage.
"""

import os
import sqlite3
import pytest
import pandas as pd
import yaml
from shapely.geometry import box


@pytest.fixture
def config():
    with open("configs/dataset_config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_split_definitions_match_gpkg(config):
    sample_gpkg = config["paths"]["sample_tiles_split"]
    conn = sqlite3.connect(sample_gpkg)
    cur = conn.cursor()
    
    for split_name, expected_tiles in config["splits"].items():
        for tile in expected_tiles:
            # Note: sample_tiles_10k_split uses 'validate' instead of 'val'
            gpkg_split_name = "validate" if split_name == "val" else split_name
            cur.execute("SELECT split FROM tiles_copy WHERE filename_10k = ?;", (tile,))
            row = cur.fetchone()
            assert row is not None, f"Tile {tile} not found in sample_tiles_10k_split.gpkg"
            assert row[0] == gpkg_split_name, (
                f"Mismatch in split for {tile}: expected {gpkg_split_name}, got {row[0]} in GPKG"
            )
    conn.close()


def test_no_tile_overlap_between_splits(config):
    train_tiles = set(config["splits"]["train"])
    val_tiles = set(config["splits"]["val"])
    test_tiles = set(config["splits"]["test"])
    
    assert len(train_tiles.intersection(val_tiles)) == 0, "Leakage detected between train and val tiles!"
    assert len(train_tiles.intersection(test_tiles)) == 0, "Leakage detected between train and test tiles!"
    assert len(val_tiles.intersection(test_tiles)) == 0, "Leakage detected between val and test tiles!"


def test_no_patch_leakage_in_metadata(config):
    metadata_dir = config["paths"]["metadata_dir"]
    train_csv = os.path.join(metadata_dir, "patches_train.csv")
    val_csv = os.path.join(metadata_dir, "patches_val.csv")
    test_csv = os.path.join(metadata_dir, "patches_test.csv")
    
    if not (os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv)):
        pytest.skip("Metadata CSV files not yet generated. Run prepare_data.py first.")
        
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    
    # 1. Check patch IDs are unique
    train_ids = set(df_train["patch_id"])
    val_ids = set(df_val["patch_id"])
    test_ids = set(df_test["patch_id"])
    
    assert len(train_ids.intersection(val_ids)) == 0, "Patch ID overlap between train and val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Patch ID overlap between train and test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Patch ID overlap between val and test!"
    
    # 2. Check source TIFFs in each split match configuration
    assert set(df_train["source_tiff"].unique()).issubset(set(config["splits"]["train"]))
    assert set(df_val["source_tiff"].unique()).issubset(set(config["splits"]["val"]))
    assert set(df_test["source_tiff"].unique()).issubset(set(config["splits"]["test"]))
