"""
CRS Validation and Coordinate Transformation Utilities.
"""

from typing import Tuple, Dict, Any, Union
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
import pyproj


def validate_crs(crs_input: Union[str, CRS, pyproj.CRS, int], expected_epsg: int = 28992) -> bool:
    """
    Validate whether a given CRS matches the expected EPSG code (default: EPSG:28992).
    """
    if crs_input is None:
        return False
    try:
        if isinstance(crs_input, int):
            return crs_input == expected_epsg
        if isinstance(crs_input, (CRS, pyproj.CRS)):
            epsg = crs_input.to_epsg()
            if epsg is not None:
                return epsg == expected_epsg
            # Check authority / name
            return "28992" in str(crs_input)
        
        # String representation
        crs_str = str(crs_input).upper()
        if f"EPSG:{expected_epsg}" in crs_str or f"EPSG::{expected_epsg}" in crs_str:
            return True
        
        proj_crs = pyproj.CRS.from_user_input(crs_input)
        return proj_crs.to_epsg() == expected_epsg
    except Exception:
        return False


def get_raster_metadata(raster_path: str) -> Dict[str, Any]:
    """
    Read and return geospatial metadata from a raster file.
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        meta = {
            "path": raster_path,
            "width": src.width,
            "height": src.height,
            "shape": (src.height, src.width),
            "count": src.count,
            "crs": str(src.crs),
            "epsg": src.crs.to_epsg() if src.crs else None,
            "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
            "transform": list(src.transform),
            "resolution": src.res,
            "dtypes": src.dtypes,
        }
    return meta


def pixel_to_geo(col: float, row: float, transform: Union[Affine, Tuple[float, ...]]) -> Tuple[float, float]:
    """
    Convert (col, row) pixel indices to (x, y) geospatial coordinates.
    """
    if isinstance(transform, (tuple, list)):
        transform = Affine(*transform[:6])
    x = transform.a * col + transform.b * row + transform.c
    y = transform.d * col + transform.e * row + transform.f
    return float(x), float(y)


def geo_to_pixel(x: float, y: float, transform: Union[Affine, Tuple[float, ...]]) -> Tuple[float, float]:
    """
    Convert (x, y) geospatial coordinates to (col, row) pixel coordinates.
    """
    if isinstance(transform, (tuple, list)):
        transform = Affine(*transform[:6])
    inv = ~transform
    col = inv.a * x + inv.b * y + inv.c
    row = inv.d * x + inv.e * y + inv.f
    return float(col), float(row)
