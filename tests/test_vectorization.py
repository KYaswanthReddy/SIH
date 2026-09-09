"""
Tests for Raster-to-Vector Pipeline, Skeletonization, Graph Tracing, and GeoJSON Export.
"""

import os
import json
import pytest
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
from rasterio.transform import Affine

from gis.refinement import refine_boundary_mask
from gis.skeletonization import fast_skeletonize, detect_endpoints_and_junctions
from gis.vectorization import vectorize_skeleton, merge_contiguous_lines
from gis.pipeline import RasterToVectorGISPipeline
from gis.export import export_geodataframe


def test_refinement_and_skeletonization():
    # Synthetic probability map with a thick rectangle
    prob_map = np.zeros((100, 100), dtype=np.float32)
    prob_map[20:25, 20:80] = 0.8  # Top horizontal line
    prob_map[20:80, 20:25] = 0.8  # Left vertical line

    refined = refine_boundary_mask(prob_map, threshold=0.5)
    assert refined.shape == (100, 100)
    assert (refined > 0).sum() > 0

    skel = fast_skeletonize(refined)
    assert skel.shape == (100, 100)
    assert (skel > 0).sum() < (refined > 0).sum(), "Skeleton must be thinner than refined mask"

    endpoints, junctions = detect_endpoints_and_junctions(skel)
    assert len(endpoints) >= 1


def test_vectorization_geodataframe():
    # Create simple cross skeleton
    skel = np.zeros((100, 100), dtype=np.uint8)
    skel[50, 10:90] = 1 # Horizontal
    skel[10:90, 50] = 1 # Vertical

    transform = Affine(0.25, 0.0, 100000.0, 0.0, -0.25, 470000.0)
    gdf = vectorize_skeleton(skel, transform=transform, simplify_tolerance=0.1, min_length_m=0.0, crs="EPSG:28992")

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert len(gdf) > 0
    assert "geometry" in gdf.columns
    assert "length_m" in gdf.columns
    assert gdf.crs == "EPSG:28992"

    # Verify georeferencing
    sample_geom = gdf.geometry.iloc[0]
    assert 99990.0 <= sample_geom.coords[0][0] <= 100100.0


def test_vectorization_min_length_filtering():
    """Verify that short noise fragments (< min_length_m) are pruned while longer lines are kept."""
    skel = np.zeros((100, 100), dtype=np.uint8)
    # Long line: 60 pixels = 15.0 meters at 0.25m GSD
    skel[20, 20:80] = 1
    # Tiny noise fragments: 4 pixels = 1.0 meter
    skel[50, 10:14] = 1
    skel[70, 80:84] = 1

    transform = Affine(0.25, 0.0, 100000.0, 0.0, -0.25, 470000.0)

    # Without filter
    gdf_raw = vectorize_skeleton(skel, transform=transform, min_length_m=0.0)
    assert len(gdf_raw) >= 3

    # With min_length_m = 3.5m filter
    gdf_filtered = vectorize_skeleton(skel, transform=transform, min_length_m=3.5)
    assert len(gdf_filtered) == 1
    assert gdf_filtered.iloc[0]["length_m"] >= 14.0


def test_vectorization_merge_contiguous():
    """Verify contiguous colinear line segments are merged into unified polylines."""
    skel = np.zeros((100, 100), dtype=np.uint8)
    # Continuous long line spanning 80 pixels
    skel[30, 10:90] = 1

    transform = Affine(0.25, 0.0, 100000.0, 0.0, -0.25, 470000.0)
    gdf_unmerged = vectorize_skeleton(skel, transform=transform, min_length_m=0.0, merge_contiguous=False)
    gdf_merged = vectorize_skeleton(skel, transform=transform, min_length_m=0.0, merge_contiguous=True)

    assert len(gdf_merged) <= len(gdf_unmerged)
    assert gdf_merged.iloc[0]["length_m"] >= 19.0


def test_raster_to_vector_pipeline():
    prob_map = np.zeros((200, 200), dtype=np.float32)
    prob_map[50:53, 50:150] = 0.9
    prob_map[50:150, 150:153] = 0.9

    transform = Affine(0.25, 0.0, 100000.0, 0.0, -0.25, 470000.0)
    pipeline = RasterToVectorGISPipeline(threshold=0.5, simplify_tolerance=0.2, min_length_m=3.5, min_component_size=50)

    result = pipeline.run(prob_map, transform=transform)
    assert "gdf" in result
    assert "geojson" in result
    assert "topology_report" in result
    assert result["total_lines"] > 0
    assert result["total_length_m"] > 0.0
