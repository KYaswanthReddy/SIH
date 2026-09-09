"""
GeoJSON and Shapefile Spatial Export Utilities.
"""

import os
from typing import Dict, Any, Optional
import geopandas as gpd


def export_geodataframe(
    gdf: gpd.GeoDataFrame,
    output_path: str,
    target_crs: Optional[str] = None
) -> str:
    """
    Export GeoDataFrame to GeoJSON or Shapefile format with optional CRS reprojection.
    
    Args:
        gdf: GeoDataFrame to export
        output_path: Output file path ending in .geojson or .shp
        target_crs: Optional target CRS (e.g. 'EPSG:4326' for WGS84 web standard)
        
    Returns:
        output_path: Absolute path to saved file
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    export_gdf = gdf.copy()
    if target_crs is not None and export_gdf.crs != target_crs:
        export_gdf = export_gdf.to_crs(target_crs)

    if output_path.endswith(".geojson") or output_path.endswith(".json"):
        export_gdf.to_file(output_path, driver="GeoJSON")
    elif output_path.endswith(".shp"):
        export_gdf.to_file(output_path, driver="ESRI Shapefile")
    else:
        export_gdf.to_file(output_path, driver="GeoJSON")

    return os.path.abspath(output_path)
