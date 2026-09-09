"""
Unified Raster-to-Vector Cadastral Conversion and Topology Pipeline.
"""

from typing import Dict, Any, Optional, Union, Tuple
import json
import numpy as np
import geopandas as gpd
from rasterio.transform import Affine

from .refinement import refine_boundary_mask
from .skeletonization import fast_skeletonize, zhang_suen_thinning
from .vectorization import vectorize_skeleton
from .topology import validate_cadastral_topology
from .export import export_geodataframe


class RasterToVectorGISPipeline:
    """
    End-to-end converter from raw raster probability maps to clean, georeferenced, topology-validated GIS vectors.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_component_size: int = 50,
        simplify_tolerance: float = 0.5,
        min_length_m: float = 3.5,
        snap_tolerance_m: float = 0.5,
        merge_contiguous: bool = True,
    ):
        self.threshold = threshold
        self.min_component_size = min_component_size
        self.simplify_tolerance = simplify_tolerance
        self.min_length_m = min_length_m
        self.snap_tolerance_m = snap_tolerance_m
        self.merge_contiguous = merge_contiguous

    def run(
        self,
        prob_map: np.ndarray,
        transform: Union[Affine, Tuple[float, ...]],
        crs: str = "EPSG:28992",
        output_geojson_path: Optional[str] = None,
        output_report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute full GIS pipeline on probability map.
        
        Args:
            prob_map: 2D numpy array in [0.0, 1.0]
            transform: Affine transform for the raster
            crs: Source CRS
            output_geojson_path: Optional output path for GeoJSON
            output_report_path: Optional output path for topology report JSON
            
        Returns:
            Dict containing GeoDataFrame, GeoJSON dict, topology report, and intermediate masks
        """
        # 1. Morphological refinement & thresholding
        refined_mask = refine_boundary_mask(
            prob_map=prob_map,
            threshold=self.threshold,
            min_component_size=self.min_component_size
        )

        # 2. Skeletonization (1-pixel medial axis)
        skeleton = fast_skeletonize(refined_mask)

        # 3. Vectorization (Graph tracing & Douglas-Peucker simplification)
        gdf = vectorize_skeleton(
            skeleton=skeleton,
            transform=transform,
            prob_map=prob_map,
            simplify_tolerance=self.simplify_tolerance,
            min_length_pixels=self.min_length_m / (abs(transform[0]) if isinstance(transform, (tuple, list)) else abs(transform.a)),
            merge_contiguous=self.merge_contiguous,
            crs=crs
        )

        # 4. GIS Topology Validation
        topology_report = validate_cadastral_topology(
            gdf=gdf,
            snap_tolerance_m=self.snap_tolerance_m,
            min_length_m=self.min_length_m,
            output_report_path=output_report_path
        )

        # 5. Export to GeoJSON
        geojson_str = gdf.to_json() if not gdf.empty else json.dumps({"type": "FeatureCollection", "features": []})
        geojson_dict = json.loads(geojson_str)

        if output_geojson_path:
            export_geodataframe(gdf, output_geojson_path)

        return {
            "gdf": gdf,
            "geojson": geojson_dict,
            "topology_report": topology_report,
            "refined_mask": refined_mask,
            "skeleton": skeleton,
            "total_lines": len(gdf),
            "total_length_m": topology_report["total_length_m"],
        }
