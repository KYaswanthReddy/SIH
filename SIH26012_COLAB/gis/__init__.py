"""
SIH26012 GIS Module
Geospatial processing, spatial indexing, coordinate transformation, and boundary rasterization.
"""

from .crs import validate_crs, get_raster_metadata, pixel_to_geo, geo_to_pixel
from .spatial_index import TileSpatialIndex, load_urban_polygons
from .rasterizer import rasterize_cadastral_boundaries, compute_mask_statistics
from .patch_extractor import PatchExtractor

__all__ = [
    "validate_crs",
    "get_raster_metadata",
    "pixel_to_geo",
    "geo_to_pixel",
    "TileSpatialIndex",
    "load_urban_polygons",
    "rasterize_cadastral_boundaries",
    "compute_mask_statistics",
    "PatchExtractor",
]
