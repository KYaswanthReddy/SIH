"""
GIS Topology Validation and Geometric Quality Assurance Engine.
Audits cadastral line vector geometries for self-intersections, duplicates, tiny fragments, gaps, and invalid shapes.
"""

from typing import Dict, Any, List, Tuple, Optional
import json
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.strtree import STRtree


def validate_cadastral_topology(
    gdf: gpd.GeoDataFrame,
    snap_tolerance_m: float = 0.5,
    min_length_m: float = 2.0,
    output_report_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform rigorous GIS topology audit on vectorized cadastral LineStrings.
    
    Args:
        gdf: GeoDataFrame containing vectorized cadastral LineStrings
        snap_tolerance_m: Maximum gap distance in meters to check for undershoots/dangles
        min_length_m: Minimum allowed line segment length in meters
        output_report_path: Optional file path to save JSON topology audit report
        
    Returns:
        report: Detailed dictionary of topological quality metrics and warnings
    """
    if gdf.empty:
        empty_report = {
            "total_lines": 0,
            "total_length_m": 0.0,
            "average_length_m": 0.0,
            "invalid_geometries_count": 0,
            "self_intersections_count": 0,
            "duplicate_lines_count": 0,
            "tiny_fragments_count": 0,
            "disconnected_segments_count": 0,
            "crossing_geometries_count": 0,
            "fragmentation_index": 0.0,
            "status": "EMPTY",
            "warnings": []
        }
        if output_report_path:
            with open(output_report_path, "w") as f:
                json.dump(empty_report, f, indent=2)
        return empty_report

    warnings = []
    invalid_count = 0
    self_intersect_count = 0
    tiny_fragments_count = 0
    duplicate_count = 0
    crossing_count = 0

    geometries = list(gdf.geometry)
    tree = STRtree(geometries)

    # 1. Inspect Individual Geometries
    for idx, row in gdf.iterrows():
        geom = row.geometry
        line_id = row.get("line_id", f"LINE_{idx}")

        # Check validity
        if not geom.is_valid:
            invalid_count += 1
            warnings.append({
                "line_id": line_id,
                "issue_type": "INVALID_GEOMETRY",
                "severity": "HIGH",
                "message": f"Geometry is not valid according to OGC standards."
            })

        # Check self-intersection
        if not geom.is_simple:
            self_intersect_count += 1
            warnings.append({
                "line_id": line_id,
                "issue_type": "SELF_INTERSECTION",
                "severity": "HIGH",
                "message": f"Line intersects itself in its interior."
            })

        # Check tiny fragments
        if geom.length < min_length_m:
            tiny_fragments_count += 1
            warnings.append({
                "line_id": line_id,
                "issue_type": "TINY_FRAGMENT",
                "severity": "MEDIUM",
                "length_m": round(float(geom.length), 3),
                "message": f"Line length ({geom.length:.2f}m) is below minimum threshold ({min_length_m}m)."
            })

    # 2. Duplicate Check
    seen_lines = set()
    for idx, row in gdf.iterrows():
        geom = row.geometry
        line_id = row.get("line_id", f"LINE_{idx}")
        coords = tuple(geom.coords)
        rev_coords = tuple(reversed(geom.coords))
        
        # Round coords to 2 decimals for duplicate matching
        norm_coords = tuple((round(x, 2), round(y, 2)) for x, y in coords)
        norm_rev = tuple((round(x, 2), round(y, 2)) for x, y in rev_coords)

        if norm_coords in seen_lines or norm_rev in seen_lines:
            duplicate_count += 1
            warnings.append({
                "line_id": line_id,
                "issue_type": "DUPLICATE_LINE",
                "severity": "HIGH",
                "message": f"Duplicate overlapping boundary line detected."
            })
        else:
            seen_lines.add(norm_coords)

    # 3. Crossing and Disconnected Checks
    endpoints_arr = np.array([pt[:2] for g in geometries for pt in (g.coords[0], g.coords[-1])], dtype=np.float64)
    N_pts = len(endpoints_arr)

    disconnected_count = 0
    for idx, g in enumerate(geometries):
        if N_pts > 0:
            p_start = endpoints_arr[2 * idx]
            p_end = endpoints_arr[2 * idx + 1]

            other_mask = (np.arange(N_pts) // 2) != idx

            d_start = np.hypot(endpoints_arr[other_mask, 0] - p_start[0], endpoints_arr[other_mask, 1] - p_start[1])
            start_connected = bool(np.any(d_start < snap_tolerance_m)) if len(d_start) > 0 else False

            d_end = np.hypot(endpoints_arr[other_mask, 0] - p_end[0], endpoints_arr[other_mask, 1] - p_end[1])
            end_connected = bool(np.any(d_end < snap_tolerance_m)) if len(d_end) > 0 else False
        else:
            start_connected = False
            end_connected = False

        if not start_connected and not end_connected:
            disconnected_count += 1
            warnings.append({
                "line_id": gdf.iloc[idx].get("line_id", f"LINE_{idx}"),
                "issue_type": "ISOLATED_SEGMENT",
                "severity": "LOW",
                "message": f"Line has neither start nor end point connected to other parcel lines within {snap_tolerance_m}m."
            })

        # Crossing check with other lines
        candidates = tree.query(g)
        for cand_idx in candidates:
            if cand_idx > idx:
                other_geom = geometries[cand_idx]
                if g.crosses(other_geom):
                    crossing_count += 1
                    warnings.append({
                        "line_id": gdf.iloc[idx].get("line_id", f"LINE_{idx}"),
                        "other_line_id": gdf.iloc[cand_idx].get("line_id", f"LINE_{cand_idx}"),
                        "issue_type": "INTERIOR_CROSSING",
                        "severity": "MEDIUM",
                        "message": f"Line crosses another cadastral boundary in its interior."
                    })

    total_len = float(gdf.geometry.length.sum())
    avg_len = float(gdf.geometry.length.mean()) if len(gdf) > 0 else 0.0
    frag_index = round(float(len(gdf) / (total_len + 1e-6) * 100), 4)

    report = {
        "status": "VALID" if len(warnings) == 0 else "TOPOLOGY_WARNINGS",
        "is_valid": len(warnings) == 0,
        "total_lines": len(gdf),
        "total_length_m": round(total_len, 2),
        "average_length_m": round(avg_len, 2),
        "fragmentation_index": frag_index,
        "invalid_geometries_count": invalid_count,
        "self_intersections_count": self_intersect_count,
        "duplicate_lines_count": duplicate_count,
        "tiny_fragments_count": tiny_fragments_count,
        "disconnected_segments_count": disconnected_count,
        "crossing_geometries_count": crossing_count,
        "total_warnings": len(warnings),
        "warnings": warnings,
    }

    if output_report_path:
        with open(output_report_path, "w") as f:
            json.dump(report, f, indent=2)

    return report
