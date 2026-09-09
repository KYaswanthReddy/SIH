"""
Backend GIS Service for Vectorization, Topology Audit, and Supporting Layers.
"""

import os
import json
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, mapping, box, Polygon, LineString
from rasterio.transform import Affine

from gis.pipeline import RasterToVectorGISPipeline
from gis.topology import validate_cadastral_topology
from gis.export import export_geodataframe


class GISService:
    """
    Manages vectorization, topology auditing, supporting layer synthesis, and GeoJSON export.
    """

    def __init__(self, metadata_path: str = "data/processed/metadata/patches_all.csv"):
        self.metadata_df = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else pd.DataFrame()
        self.pipeline = RasterToVectorGISPipeline()

    def get_patch_metadata(self, patch_id: str) -> Optional[Dict[str, Any]]:
        if self.metadata_df.empty:
            return None
        match = self.metadata_df[self.metadata_df["patch_id"] == patch_id]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        if isinstance(row.get("transform"), str):
            row["transform"] = eval(row["transform"])
        return row

    def vectorize_and_audit(
        self,
        prob_map: np.ndarray,
        transform: Union[Affine, List[float], Tuple[float, ...]],
        threshold: float = 0.5,
        simplify_tolerance: float = 0.5,
        min_length_m: float = 3.5,
        min_component_size: int = 50,
        merge_contiguous: bool = True,
        source_crs: str = "EPSG:28992"
    ) -> Dict[str, Any]:
        """
        Convert predicted probability map to GeoJSON with topology analysis.
        """
        if isinstance(transform, (list, tuple)):
            transform = Affine(*transform[:6])

        self.pipeline.threshold = threshold
        self.pipeline.simplify_tolerance = simplify_tolerance
        self.pipeline.min_length_m = min_length_m
        self.pipeline.min_component_size = min_component_size
        self.pipeline.merge_contiguous = merge_contiguous

        # Run pipeline in source CRS
        res = self.pipeline.run(prob_map=prob_map, transform=transform, crs=source_crs)
        gdf = res["gdf"]

        # Convert to WGS84 (EPSG:4326) for WebGIS Leaflet display
        if not gdf.empty:
            gdf_wgs84 = gdf.to_crs("EPSG:4326")
            geojson_wgs84 = json.loads(gdf_wgs84.to_json())
        else:
            geojson_wgs84 = {"type": "FeatureCollection", "features": []}

        # Compute patch bounding box in WGS84
        c00_x, c00_y = transform.c, transform.f
        c01_x, c01_y = transform.a * 512 + transform.c, transform.d * 512 + transform.f
        c10_x, c10_y = transform.b * 512 + transform.c, transform.e * 512 + transform.f
        c11_x, c11_y = transform.a * 512 + transform.b * 512 + transform.c, transform.d * 512 + transform.e * 512 + transform.f
        min_x = min(c00_x, c01_x, c10_x, c11_x)
        max_x = max(c00_x, c01_x, c10_x, c11_x)
        min_y = min(c00_y, c01_y, c10_y, c11_y)
        max_y = max(c00_y, c01_y, c10_y, c11_y)
        patch_box = gpd.GeoDataFrame(geometry=[box(min_x, min_y, max_x, max_y)], crs=source_crs).to_crs("EPSG:4326")
        b = patch_box.total_bounds
        wgs84_bounds = [[float(b[1]), float(b[0])], [float(b[3]), float(b[2])]]

        return {
            "geojson_wgs84": geojson_wgs84,
            "topology_report": res["topology_report"],
            "total_lines": res["total_lines"],
            "total_length_m": res["total_length_m"],
            "wgs84_bounds": wgs84_bounds,
        }

    def get_patch_wgs84_bounds(self, patch_id: str) -> Optional[List[List[float]]]:
        """
        Calculates exact [[south_lat, west_lon], [north_lat, east_lon]] bounds for a patch.
        """
        meta = self.get_patch_metadata(patch_id)
        if not meta:
            return None
        t = meta.get("transform")
        if isinstance(t, (list, tuple)):
            t = Affine(*t[:6])
        elif not isinstance(t, Affine):
            return None
        
        c00_x, c00_y = t.c, t.f
        c01_x, c01_y = t.a * 512 + t.c, t.d * 512 + t.f
        c10_x, c10_y = t.b * 512 + t.c, t.e * 512 + t.f
        c11_x, c11_y = t.a * 512 + t.b * 512 + t.c, t.d * 512 + t.e * 512 + t.f
        min_x = min(c00_x, c01_x, c10_x, c11_x)
        max_x = max(c00_x, c01_x, c10_x, c11_x)
        min_y = min(c00_y, c01_y, c10_y, c11_y)
        max_y = max(c00_y, c01_y, c10_y, c11_y)
        patch_box = gpd.GeoDataFrame(geometry=[box(min_x, min_y, max_x, max_y)], crs=meta.get("crs", "EPSG:28992")).to_crs("EPSG:4326")
        b = patch_box.total_bounds
        return [[float(b[1]), float(b[0])], [float(b[3]), float(b[2])]]

    def audit_edited_geojson(self, geojson_data: Dict[str, Any], source_crs: str = "EPSG:4326") -> Dict[str, Any]:
        """
        Validate topology of human-edited GeoJSON vectors.
        """
        try:
            gdf = gpd.GeoDataFrame.from_features(geojson_data.get("features", []), crs=source_crs)
            # Project to metric CRS for accurate distance calculations
            if not gdf.empty and gdf.crs.is_geographic:
                gdf_metric = gdf.to_crs("EPSG:28992")
            else:
                gdf_metric = gdf

            report = validate_cadastral_topology(gdf_metric, snap_tolerance_m=0.5, min_length_m=2.0)
            return report
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "total_lines": 0,
                "warnings": [{"issue_type": "PARSING_ERROR", "severity": "HIGH", "message": str(e)}]
            }

    def get_supporting_layers(self, patch_id: str, layer_type: str) -> Dict[str, Any]:
        """
        Returns modular supporting GIS layers (Building Footprints or Roads) for context.
        Clearly labeled as supporting layers.
        """
        meta = self.get_patch_metadata(patch_id)
        if not meta:
            return {"type": "FeatureCollection", "features": [], "layer_type": layer_type, "is_supporting_layer": True}

        # Bounding box in EPSG:28992
        minx, miny, maxx, maxy = meta["minx"], meta["miny"], meta["maxx"], meta["maxy"]
        
        # Synthetic supporting geometries within the patch for visual context in demo
        features = []
        if layer_type == "buildings":
            # Generate 2-3 sample building polygons within the bounds
            cx, cy = (minx + maxx) / 2.0, (miny + maxy) / 2.0
            b1 = box(cx - 15, cy - 15, cx + 15, cy + 15)
            b2 = box(cx + 25, cy + 10, cx + 45, cy + 30)
            gdf_b = gpd.GeoDataFrame([
                {"building_id": "BLD_001", "type": "Residential", "layer": "Supporting Building Footprint", "geometry": b1},
                {"building_id": "BLD_002", "type": "Outbuilding", "layer": "Supporting Building Footprint", "geometry": b2}
            ], crs="EPSG:28992")
            gdf_wgs = gdf_b.to_crs("EPSG:4326")
            res = json.loads(gdf_wgs.to_json())
            res["layer_info"] = "Supporting Building Footprints (Context Layer)"
            return res

        elif layer_type == "roads":
            # Generate sample road centerline
            r1 = LineString([(minx, miny + 20), (maxx, miny + 20)])
            gdf_r = gpd.GeoDataFrame([
                {"road_id": "ROAD_001", "class": "Local Access Road", "layer": "Supporting Road Network", "geometry": r1}
            ], crs="EPSG:28992")
            gdf_wgs = gdf_r.to_crs("EPSG:4326")
            res = json.loads(gdf_wgs.to_json())
            res["layer_info"] = "Supporting Road Network (Context Layer)"
            return res

        return {"type": "FeatureCollection", "features": [], "layer_type": layer_type, "layer_info": "Supporting Layer", "is_supporting_layer": True}

    def get_ground_truth_vector(self, patch_id: str) -> Dict[str, Any]:
        """
        Extract crisp vector GeoJSON from ground-truth mask for ground-truth overlay.
        """
        from PIL import Image
        from gis.skeletonization import fast_skeletonize
        from gis.vectorization import vectorize_skeleton

        meta = self.get_patch_metadata(patch_id)
        if not meta:
            return {"type": "FeatureCollection", "features": []}

        mask_path = os.path.join("data/processed", meta["mask_path"])
        if not os.path.exists(mask_path):
            return {"type": "FeatureCollection", "features": []}

        mask_np = (np.array(Image.open(mask_path)) > 0).astype(np.uint8)
        if mask_np.sum() == 0:
            return {"type": "FeatureCollection", "features": []}

        skel = fast_skeletonize(mask_np)
        transform = meta["transform"]

        gdf_gt = vectorize_skeleton(
            skel,
            transform=transform,
            simplify_tolerance=0.25,
            min_length_m=1.5,
            merge_contiguous=True,
            crs=meta.get("crs", "EPSG:28992")
        )
        if gdf_gt.empty:
            return {"type": "FeatureCollection", "features": []}

        gdf_gt_wgs = gdf_gt.to_crs("EPSG:4326")
        geojson = json.loads(gdf_gt_wgs.to_json())
        for feat in geojson.get("features", []):
            feat["properties"]["is_ground_truth"] = True
            feat["properties"]["mean_confidence"] = 1.0
            feat["properties"]["status"] = "Ground Truth"
        return geojson

