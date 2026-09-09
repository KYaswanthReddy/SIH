"""
Cadastral Boundary Line Rasterization and Mask Statistics Engine.
"""

from typing import List, Tuple, Dict, Any, Union
import numpy as np
import cv2
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiLineString, GeometryCollection


def rasterize_cadastral_boundaries(
    geometries: List[Union[LineString, MultiLineString]],
    patch_shape: Tuple[int, int],
    patch_transform: Union[Affine, Tuple[float, ...]],
    line_thickness: int = 2
) -> np.ndarray:
    """
    Rasterize vector cadastral boundary LineStrings into a binary mask of shape patch_shape.
    
    Args:
        geometries: List of shapely LineString or MultiLineString geometries
        patch_shape: (height, width), e.g. (512, 512)
        patch_transform: Affine transform for the patch window
        line_thickness: Line thickness in pixels (default 2)
        
    Returns:
        np.ndarray: Binary mask of shape (height, width) with values in {0, 1}
    """
    height, width = patch_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if not geometries:
        return mask
    
    if isinstance(patch_transform, (tuple, list)):
        patch_transform = Affine(*patch_transform[:6])
    inv_transform = ~patch_transform

    for geom in geometries:
        if geom is None or geom.is_empty:
            continue
        
        # Extract individual linestrings from MultiLineString or GeometryCollection
        lines = []
        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        elif isinstance(geom, GeometryCollection):
            lines = [g for g in geom.geoms if isinstance(g, (LineString, MultiLineString))]
        
        for line in lines:
            coords = np.array(line.coords)
            if len(coords) < 2:
                continue
            # Convert world coordinates (x, y) to pixel coordinates (col, row)
            xs = coords[:, 0]
            ys = coords[:, 1]
            cols = inv_transform.a * xs + inv_transform.b * ys + inv_transform.c
            rows = inv_transform.d * xs + inv_transform.e * ys + inv_transform.f
            
            # Form points array: shape (N, 1, 2) with (x, y) = (col, row)
            pts = np.column_stack((cols, rows)).reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=line_thickness)

    # Ensure strictly binary {0, 1}
    mask = (mask > 0).astype(np.uint8)
    return mask


def compute_mask_statistics(mask: np.ndarray) -> Dict[str, Any]:
    """
    Compute summary statistics for a generated binary mask.
    
    Args:
        mask: 2D binary numpy array
        
    Returns:
        Dict with boundary pixel count, density ratio, is_empty flag
    """
    total_pixels = mask.size
    boundary_pixels = int(np.sum(mask > 0))
    density = float(boundary_pixels / total_pixels) if total_pixels > 0 else 0.0
    
    return {
        "boundary_pixels": boundary_pixels,
        "boundary_density": density,
        "is_empty": boundary_pixels == 0,
    }
