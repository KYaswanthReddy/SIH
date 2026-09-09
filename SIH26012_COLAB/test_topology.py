"""
Tests for GIS Topology Validation Engine.
"""

import pytest
import geopandas as gpd
from shapely.geometry import LineString
from gis.topology import validate_cadastral_topology


def test_topology_validation_clean():
    # Two connected non-overlapping perpendicular lines
    l1 = LineString([(0, 0), (10, 0)])
    l2 = LineString([(10, 0), (10, 10)])

    gdf = gpd.GeoDataFrame([
        {"line_id": "L1", "geometry": l1},
        {"line_id": "L2", "geometry": l2}
    ], crs="EPSG:28992")

    report = validate_cadastral_topology(gdf, snap_tolerance_m=0.5, min_length_m=1.0)
    assert report["invalid_geometries_count"] == 0
    assert report["self_intersections_count"] == 0
    assert report["duplicate_lines_count"] == 0
    assert report["total_lines"] == 2
    assert report["total_length_m"] == 20.0


def test_topology_validation_with_issues():
    # 1. Tiny fragment (length 0.5m < 2.0m)
    tiny = LineString([(0, 0), (0.5, 0)])
    # 2. Duplicate line
    l1 = LineString([(0, 0), (10, 0)])
    l1_dup = LineString([(0, 0), (10, 0)])
    # 3. Crossing line
    cross1 = LineString([(5, -5), (5, 5)])

    gdf = gpd.GeoDataFrame([
        {"line_id": "L_TINY", "geometry": tiny},
        {"line_id": "L1", "geometry": l1},
        {"line_id": "L1_DUP", "geometry": l1_dup},
        {"line_id": "L_CROSS", "geometry": cross1},
    ], crs="EPSG:28992")

    report = validate_cadastral_topology(gdf, min_length_m=2.0)
    assert report["tiny_fragments_count"] >= 1
    assert report["duplicate_lines_count"] >= 1
    assert report["crossing_geometries_count"] >= 1
    assert report["status"] == "TOPOLOGY_WARNINGS"
