"""
Phase 1 Data Preparation Pipeline Execution CLI.
Processes all 9 aerial tiles into train, validation, and test patches with spatial indexing and metadata catalogs.
"""

import os
import sys
import json
import time
import yaml
import pandas as pd

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gis.crs import validate_crs, get_raster_metadata
from gis.spatial_index import load_urban_polygons
from gis.patch_extractor import PatchExtractor


def load_config(config_path: str = "configs/dataset_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("==================================================")
    print(" SIH26012 — PHASE 1: DATA PREPARATION PIPELINE")
    print("==================================================")
    start_time = time.time()
    
    config = load_config()
    
    images_dir = config["paths"]["images_dir"]
    brk_path = config["paths"]["brk_reference"]
    urban_path = config["paths"]["brt_bebouwde_kom"]
    metadata_dir = config["paths"]["metadata_dir"]
    os.makedirs(metadata_dir, exist_ok=True)

    # 1. Load Urban Index once
    print("\n[Step 1/3] Initializing Urban Area Spatial Index (brt_bebouwde_kom.gpkg)...")
    urban_index = load_urban_polygons(urban_path)
    print(f"Urban index ready with {len(urban_index.geometries)} polygons.")

    # 2. Initialize Patch Extractor
    extractor = PatchExtractor(config, urban_index=urban_index)

    splits = config["splits"]
    targets = config["patch_extraction"]["targets"]

    all_split_dfs = {}

    print("\n[Step 2/3] Extracting Patches across Train, Validation, and Test Splits...")
    for split_name, tile_list in splits.items():
        split_target_total = targets[split_name]
        per_tile_target = split_target_total // len(tile_list)
        print(f"\n>>> Starting Split: {split_name.upper()} ({len(tile_list)} tiles, target: {split_target_total} patches, ~{per_tile_target}/tile) <<<")
        
        split_dfs = []
        for tile_file in tile_list:
            tile_path = os.path.join(images_dir, tile_file)
            if not os.path.exists(tile_path):
                raise FileNotFoundError(f"TIFF file not found: {tile_path}")
            
            df_tile = extractor.process_tile(
                tile_path=tile_path,
                brk_gpkg_path=brk_path,
                split=split_name,
                target_count=per_tile_target
            )
            split_dfs.append(df_tile)

        split_df = pd.concat(split_dfs, ignore_index=True)
        csv_path = os.path.join(metadata_dir, f"patches_{split_name}.csv")
        split_df.to_csv(csv_path, index=False)
        print(f"Saved metadata catalog for {split_name}: {csv_path} ({len(split_df)} patches)")
        all_split_dfs[split_name] = split_df

    # 3. Master Summary
    print("\n[Step 3/3] Generating Master Dataset Summary...")
    total_df = pd.concat(all_split_dfs.values(), ignore_index=True)
    master_csv = os.path.join(metadata_dir, "patches_all.csv")
    total_df.to_csv(master_csv, index=False)

    summary = {
        "dataset_name": config["dataset"]["name"],
        "crs": config["dataset"]["crs"],
        "ground_sampling_distance_m": config["dataset"]["ground_sampling_distance_m"],
        "patch_dimensions": config["patch_extraction"]["patch_size"],
        "total_source_images": sum(len(tiles) for tiles in splits.values()),
        "tile_counts": {split: len(tiles) for split, tiles in splits.items()},
        "total_patches_generated": len(total_df),
        "split_patch_counts": {split: len(df) for split, df in all_split_dfs.items()},
        "useful_boundary_patches": int(sum(total_df["boundary_pixels"] >= config["patch_extraction"]["min_boundary_pixels"])),
        "empty_patches": int(sum(total_df["is_empty"])),
        "empty_patches_percentage": round(float(sum(total_df["is_empty"]) / len(total_df) * 100), 2),
        "useful_patches_percentage": round(float(sum(total_df["boundary_pixels"] >= config["patch_extraction"]["min_boundary_pixels"]) / len(total_df) * 100), 2),
        "urban_patches_count": int(sum(total_df["is_urban"])),
        "urban_patches_percentage": round(float(sum(total_df["is_urban"]) / len(total_df) * 100), 2),
        "average_boundary_density": round(float(total_df["boundary_density"].mean()), 4),
        "max_boundary_density": round(float(total_df["boundary_density"].max()), 4),
        "execution_time_seconds": round(time.time() - start_time, 2),
    }

    summary_json_path = os.path.join(metadata_dir, "dataset_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==================================================")
    print(" DATA PREPARATION COMPLETE")
    print("==================================================")
    print(json.dumps(summary, indent=2))
    print(f"\nSummary saved to {summary_json_path}")


if __name__ == "__main__":
    main()
