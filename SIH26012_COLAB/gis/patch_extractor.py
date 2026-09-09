"""
Tile-by-Tile Windowed Patch Extraction and Dataset Generation Engine.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import rasterio
from rasterio.windows import Window
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .crs import get_raster_metadata
from .spatial_index import TileSpatialIndex, UrbanAreaIndex
from .rasterizer import rasterize_cadastral_boundaries, compute_mask_statistics


class PatchExtractor:
    """
    Extracts image and label patches from a large aerial tile with spatial indexing and quality filtering.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        urban_index: Optional[UrbanAreaIndex] = None
    ):
        self.config = config
        self.urban_index = urban_index
        self.patch_size = tuple(config["patch_extraction"]["patch_size"])
        self.stride = config["patch_extraction"]["stride"]
        self.line_thickness = config["patch_extraction"]["line_thickness"]
        self.min_boundary_pixels = config["patch_extraction"]["min_boundary_pixels"]
        self.empty_patch_ratio = config["patch_extraction"]["empty_patch_ratio"]
        self.patches_output_dir = config["paths"]["patches_dir"]

    def process_tile(
        self,
        tile_path: str,
        brk_gpkg_path: str,
        split: str,
        target_count: int
    ) -> pd.DataFrame:
        """
        Process a single 10km x 10km aerial tile and extract useful patches.
        
        Args:
            tile_path: Path to aerial TIFF file
            brk_gpkg_path: Path to brk_reference.gpkg
            split: 'train', 'val', or 'test'
            target_count: Number of patches to extract from this tile
            
        Returns:
            pd.DataFrame: Metadata records for extracted patches
        """
        tile_name = os.path.basename(tile_path)
        tile_id = os.path.splitext(tile_name)[0]
        print(f"\n--- Processing Tile [{split.upper()}]: {tile_name} ---")

        with rasterio.open(tile_path) as src:
            tile_width = src.width
            tile_height = src.height
            tile_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
            tile_transform = src.transform
            tile_crs = str(src.crs)

        # 1. Build spatial index for this tile
        spatial_index = TileSpatialIndex(brk_gpkg_path, tile_bounds)

        # 2. Generate candidate grid coordinates
        patch_h, patch_w = self.patch_size
        rows = list(range(0, tile_height - patch_h + 1, self.stride))
        cols = list(range(0, tile_width - patch_w + 1, self.stride))
        
        print(f"Total candidate grid locations: {len(rows) * len(cols)} (grid {len(rows)}x{len(cols)})")

        candidates = []
        for r in rows:
            for c in cols:
                win = Window(col_off=c, row_off=r, width=patch_w, height=patch_h)
                p_bounds = rasterio.windows.bounds(win, tile_transform)
                p_geoms = spatial_index.query_patch(p_bounds)
                num_lines = len(p_geoms)
                
                candidates.append({
                    "row_off": r,
                    "col_off": c,
                    "bounds": p_bounds,
                    "window": win,
                    "num_lines": num_lines,
                })

        # Separate candidates into boundary-containing and background/empty
        boundary_candidates = [c for c in candidates if c["num_lines"] > 0]
        empty_candidates = [c for c in candidates if c["num_lines"] == 0]

        print(f"Candidates with cadastral lines: {len(boundary_candidates)}, background candidates: {len(empty_candidates)}")

        # Calculate target counts
        num_empty_target = int(target_count * self.empty_patch_ratio)
        num_boundary_target = target_count - num_empty_target

        np.random.seed(42 + hash(tile_name) % 10000)
        
        # Sort boundary candidates by line count with some randomization for diversity
        # Select boundary-rich and medium-boundary patches evenly
        if len(boundary_candidates) > num_boundary_target:
            # Sort by line density, sample across quantiles for balanced representation
            boundary_candidates.sort(key=lambda x: x["num_lines"], reverse=True)
            # Sample across top 80% to avoid only taking edge-cases
            indices = np.linspace(0, len(boundary_candidates) - 1, num_boundary_target, dtype=int)
            selected_boundary = [boundary_candidates[i] for i in indices]
        else:
            selected_boundary = boundary_candidates

        # Sample empty candidates
        if len(empty_candidates) > num_empty_target:
            empty_idx = np.random.choice(len(empty_candidates), size=num_empty_target, replace=False)
            selected_empty = [empty_candidates[i] for i in empty_idx]
        else:
            selected_empty = empty_candidates

        selected_candidates = selected_boundary + selected_empty
        print(f"Selected {len(selected_candidates)} patches for extraction ({len(selected_boundary)} boundary, {len(selected_empty)} background)")

        # Create output directories for this split
        img_out_dir = os.path.join(self.patches_output_dir, split, "images")
        mask_out_dir = os.path.join(self.patches_output_dir, split, "masks")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)

        patch_records = []

        with rasterio.open(tile_path) as src:
            for idx, cand in enumerate(tqdm(selected_candidates, desc=f"Extracting {tile_id}")):
                win = cand["window"]
                bounds = cand["bounds"]
                r = cand["row_off"]
                c = cand["col_off"]
                
                # 1. Read aerial image window
                img_data = src.read(window=win) # shape (3, H, W)
                # Convert (C, H, W) -> (H, W, C)
                img_rgb = np.transpose(img_data, (1, 2, 0))

                # 2. Query geometries and rasterize mask
                geoms = spatial_index.query_patch(bounds)
                patch_transform = rasterio.windows.transform(win, tile_transform)
                mask = rasterize_cadastral_boundaries(
                    geometries=geoms,
                    patch_shape=(patch_h, patch_w),
                    patch_transform=patch_transform,
                    line_thickness=self.line_thickness
                )

                # 3. Check urban area intersection
                is_urban = False
                if self.urban_index is not None:
                    is_urban = self.urban_index.intersects_urban(bounds)

                # 4. Compute statistics
                stats = compute_mask_statistics(mask)

                # 5. Save files
                patch_name = f"{tile_id}_r{r:05d}_c{c:05d}"
                img_filename = f"{patch_name}.png"
                mask_filename = f"{patch_name}_mask.png"
                
                img_path = os.path.join(img_out_dir, img_filename)
                mask_path = os.path.join(mask_out_dir, mask_filename)

                # Save RGB image
                Image.fromarray(img_rgb).save(img_path, format="PNG")
                
                # Save single-channel binary mask (values 0 and 1)
                Image.fromarray(mask).save(mask_path, format="PNG")

                patch_records.append({
                    "patch_id": patch_name,
                    "tile_id": tile_id,
                    "source_tiff": tile_name,
                    "split": split,
                    "row_off": r,
                    "col_off": c,
                    "width": patch_w,
                    "height": patch_h,
                    "minx": bounds[0],
                    "miny": bounds[1],
                    "maxx": bounds[2],
                    "maxy": bounds[3],
                    "crs": tile_crs,
                    "transform": [float(x) for x in list(patch_transform)],
                    "image_path": os.path.relpath(img_path, self.config["paths"]["processed_dir"]),
                    "mask_path": os.path.relpath(mask_path, self.config["paths"]["processed_dir"]),
                    "boundary_pixels": stats["boundary_pixels"],
                    "boundary_density": stats["boundary_density"],
                    "num_lines": len(geoms),
                    "is_urban": is_urban,
                    "is_empty": stats["is_empty"],
                })

        return pd.DataFrame(patch_records)
