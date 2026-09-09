"""
Spatial Indexing and Windowed Geospatial Query Engine.
"""

from typing import Tuple, List, Optional, Any, Dict
import time
import geopandas as gpd
import pyogrio
from shapely.geometry import box, LineString, Polygon, MultiPolygon
from shapely.strtree import STRtree
import numpy as np


class TileSpatialIndex:
    """
    Manages spatial indexing and geometry retrieval for a specific tile.
    Uses tile-level bbox filtering on GPKG and in-memory STRtree for fast patch queries.
    """

    def __init__(self, gpkg_path: str, tile_bounds: Tuple[float, float, float, float], layer: str = "brk_reference"):
        """
        Initialize the tile spatial index.
        
        Args:
            gpkg_path: Path to brk_reference.gpkg
            tile_bounds: Bounding box tuple (minx, miny, maxx, maxy)
            layer: Name of feature layer
        """
        self.gpkg_path = gpkg_path
        self.tile_bounds = tile_bounds
        self.layer = layer
        self.geometries: List[LineString] = []
        self.tree: Optional[STRtree] = None
        self._load_and_index()

    def _load_and_index(self):
        """Fetch intersecting geometries from GPKG using spatial windowing and build STRtree."""
        t0 = time.time()
        # Read only the geometries intersecting the tile bounds
        gdf = pyogrio.read_dataframe(
            self.gpkg_path,
            layer=self.layer,
            bbox=self.tile_bounds,
            columns=["id_1"]
        )
        
        # Filter valid geometries
        valid_geoms = [g for g in gdf.geometry.values if g is not None and not g.is_empty]
        self.geometries = valid_geoms
        
        if len(self.geometries) > 0:
            self.tree = STRtree(self.geometries)
        else:
            self.tree = None
        
        elapsed = time.time() - t0
        print(f"Loaded and indexed {len(self.geometries)} cadastral features for tile in {elapsed:.2f}s")

    def query_patch(self, patch_bounds: Tuple[float, float, float, float]) -> List[LineString]:
        """
        Query geometries intersecting the given patch bounding box.
        
        Args:
            patch_bounds: (minx, miny, maxx, maxy)
            
        Returns:
            List of intersecting LineString geometries
        """
        if self.tree is None or len(self.geometries) == 0:
            return []
        
        patch_box = box(*patch_bounds)
        indices = self.tree.query(patch_box)
        
        # Return matched geometries
        return [self.geometries[i] for i in indices]

    @property
    def feature_count(self) -> int:
        return len(self.geometries)


class UrbanAreaIndex:
    """
    Manages spatial lookup against urban built-up area polygons (brt_bebouwde_kom.gpkg).
    """

    def __init__(self, urban_gpkg_path: str):
        self.urban_gpkg_path = urban_gpkg_path
        self.gdf = pyogrio.read_dataframe(urban_gpkg_path)
        self.geometries = [g for g in self.gdf.geometry.values if g is not None and not g.is_empty]
        self.tree = STRtree(self.geometries) if len(self.geometries) > 0 else None

    def intersects_urban(self, bounds: Tuple[float, float, float, float]) -> bool:
        """Check if bounding box intersects any urban built-up area polygon."""
        if self.tree is None:
            return False
        query_box = box(*bounds)
        candidate_indices = self.tree.query(query_box)
        for idx in candidate_indices:
            if self.geometries[idx].intersects(query_box):
                return True
        return False


def load_urban_polygons(urban_gpkg_path: str) -> UrbanAreaIndex:
    return UrbanAreaIndex(urban_gpkg_path)
