"""
Tests for Data Integrity: TIFFs, GPKGs, Dimensions, and CRS.
"""

import os
import sqlite3
import pytest
import rasterio
import yaml
from gis.crs import validate_crs, get_raster_metadata


@pytest.fixture
def config():
    with open("configs/dataset_config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_all_9_tiffs_exist(config):
    images_dir = config["paths"]["images_dir"]
    all_expected_tiffs = (
        config["splits"]["train"] +
        config["splits"]["val"] +
        config["splits"]["test"]
    )
    assert len(all_expected_tiffs) == 9, "Expected exactly 9 TIFF files in configuration."
    
    for tif_file in all_expected_tiffs:
        full_path = os.path.join(images_dir, tif_file)
        assert os.path.exists(full_path), f"TIFF file missing: {full_path}"
        assert os.path.getsize(full_path) > 100_000_000, f"TIFF file suspiciously small: {full_path}"


def test_all_3_gpkgs_exist(config):
    gpkg_paths = [
        config["paths"]["brk_reference"],
        config["paths"]["brt_bebouwde_kom"],
        config["paths"]["sample_tiles_split"]
    ]
    for p in gpkg_paths:
        assert os.path.exists(p), f"GPKG file missing: {p}"
        assert os.path.getsize(p) > 1000, f"GPKG file suspiciously small: {p}"


def test_tiffs_dimensions_and_crs(config):
    images_dir = config["paths"]["images_dir"]
    all_expected_tiffs = (
        config["splits"]["train"] +
        config["splits"]["val"] +
        config["splits"]["test"]
    )
    
    for tif_file in all_expected_tiffs:
        full_path = os.path.join(images_dir, tif_file)
        meta = get_raster_metadata(full_path)
        
        # Dimensions: 40002 x 40002 x 3
        assert meta["width"] == 40002, f"Unexpected width {meta['width']} for {tif_file}"
        assert meta["height"] == 40002, f"Unexpected height {meta['height']} for {tif_file}"
        assert meta["count"] == 3, f"Unexpected channel count {meta['count']} for {tif_file}"
        assert meta["resolution"] == (0.25, 0.25), f"Unexpected resolution {meta['resolution']} for {tif_file}"
        
        # CRS check
        assert validate_crs(meta["crs"], expected_epsg=28992), f"Invalid CRS {meta['crs']} for {tif_file}"


def test_gpkg_crs_and_tables(config):
    gpkg_checks = [
        (config["paths"]["brk_reference"], "brk_reference"),
        (config["paths"]["brt_bebouwde_kom"], "brk_bebouwde_kom"),
        (config["paths"]["sample_tiles_split"], "tiles_copy"),
    ]
    
    for gpkg_path, expected_table in gpkg_checks:
        conn = sqlite3.connect(gpkg_path)
        cur = conn.cursor()
        cur.execute("SELECT table_name, srs_id FROM gpkg_contents WHERE table_name = ?;", (expected_table,))
        res = cur.fetchone()
        assert res is not None, f"Table {expected_table} not found in {gpkg_path}"
        assert res[1] == 28992, f"Table {expected_table} has unexpected SRS ID {res[1]} (expected 28992)"
        conn.close()
