"""
Raster-to-Vector Cadastral Boundary Tracing and Polyline Generation Engine.
Converts medial axis skeleton graphs into georeferenced Shapely LineStrings and GeoDataFrames.
"""

from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from rasterio.transform import Affine
import pyproj

from .crs import pixel_to_geo


def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """
    Convert a 1-pixel binary skeleton mask into an undirected spatial graph.
    """
    G = nx.Graph()
    skel_pts = np.argwhere(skeleton > 0)
    
    # 8-neighborhood offsets (only check positive half to avoid duplicate undirected edges)
    offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]
    h, w = skeleton.shape

    for r, c in skel_pts:
        G.add_node((r, c))
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if skeleton[nr, nc] > 0:
                    dist = np.sqrt(dr*dr + dc*dc)
                    G.add_edge((r, c), (nr, nc), weight=dist)

    return G


def trace_graph_branches(G: nx.Graph) -> List[List[Tuple[int, int]]]:
    """
    Trace maximal contiguous non-branching paths/branches and isolated cycles in the graph in linear time.
    """
    paths = []
    visited_edges = set()

    # Identify special nodes: degree != 2 (endpoints has degree 1, junctions have degree >= 3)
    special_nodes = [node for node in G.nodes() if G.degree(node) != 2]

    # Trace paths starting from each special node
    for start_node in special_nodes:
        for neighbor in G.neighbors(start_node):
            edge = tuple(sorted([start_node, neighbor]))
            if edge in visited_edges:
                continue

            current_path = [start_node, neighbor]
            visited_edges.add(edge)
            prev = start_node
            curr = neighbor

            # Follow the path while degree is 2
            while G.degree(curr) == 2 and curr not in special_nodes:
                next_neighbors = [n for n in G.neighbors(curr) if n != prev]
                if not next_neighbors:
                    break
                next_node = next_neighbors[0]
                e = tuple(sorted([curr, next_node]))
                visited_edges.add(e)
                current_path.append(next_node)
                prev = curr
                curr = next_node

            if len(current_path) >= 2:
                paths.append(current_path)

    # Trace any remaining unvisited edges (e.g. isolated degree-2 cycles)
    for u, v in G.edges():
        e = tuple(sorted([u, v]))
        if e in visited_edges:
            continue

        current_path = [u, v]
        visited_edges.add(e)
        prev = u
        curr = v

        while True:
            next_neighbors = [n for n in G.neighbors(curr) if n != prev]
            if not next_neighbors:
                break
            next_node = next_neighbors[0]
            edge = tuple(sorted([curr, next_node]))
            if edge in visited_edges:
                current_path.append(next_node)
                break
            visited_edges.add(edge)
            current_path.append(next_node)
            prev = curr
            curr = next_node
            if curr == u:
                break

        if len(current_path) >= 2:
            paths.append(current_path)

    return paths


def snap_nearby_endpoints(gdf: gpd.GeoDataFrame, snap_tolerance_m: float = 1.0) -> gpd.GeoDataFrame:
    """
    Snap close endpoints of separate LineStrings together within snap_tolerance_m.
    This bridges micro-gaps along genuine parcel boundaries, enabling linemerge to unify them into continuous lines.
    """
    if gdf.empty or len(gdf) <= 1 or snap_tolerance_m <= 0.0:
        return gdf

    coords_list = [list(g.coords) for g in gdf.geometry if isinstance(g, LineString) and len(g.coords) >= 2]
    if len(coords_list) <= 1:
        return gdf

    n_lines = len(coords_list)
    endpoints = []
    for i in range(n_lines):
        endpoints.append((i, 0, np.array(coords_list[i][0][:2], dtype=np.float64)))
        endpoints.append((i, -1, np.array(coords_list[i][-1][:2], dtype=np.float64)))

    for idx_a in range(len(endpoints)):
        line_a, pt_a_idx, pt_a = endpoints[idx_a]
        for idx_b in range(idx_a + 1, len(endpoints)):
            line_b, pt_b_idx, pt_b = endpoints[idx_b]
            if line_a == line_b:
                continue
            dist = np.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])
            if 0 < dist <= snap_tolerance_m:
                mid = ((pt_a[0] + pt_b[0]) / 2.0, (pt_a[1] + pt_b[1]) / 2.0)
                coords_list[line_a][pt_a_idx] = mid
                coords_list[line_b][pt_b_idx] = mid

    new_geoms = [LineString(c) for c in coords_list]
    return gpd.GeoDataFrame(gdf.drop(columns=["geometry"]), geometry=new_geoms, crs=gdf.crs)


def merge_contiguous_lines(gdf: gpd.GeoDataFrame, snap_tolerance_m: float = 1.0) -> gpd.GeoDataFrame:
    """
    Merge contiguous connected line segments into maximal continuous polylines.
    """
    if gdf.empty or len(gdf) <= 1:
        return gdf

    # Snap endpoints within tolerance to bridge broken boundary fragments
    if snap_tolerance_m > 0.0:
        gdf = snap_nearby_endpoints(gdf, snap_tolerance_m=snap_tolerance_m)

    from shapely.ops import linemerge
    lines = list(gdf.geometry)
    merged = linemerge(lines)

    if isinstance(merged, LineString):
        merged_list = [merged]
    elif isinstance(merged, MultiLineString):
        merged_list = list(merged.geoms)
    else:
        merged_list = lines

    overall_mean_conf = float(gdf["mean_confidence"].mean()) if "mean_confidence" in gdf.columns else 0.85

    # Check if spatial index exists for fast bounding-box filtering
    sindex = getattr(gdf, "sindex", None) if len(gdf) > 10 else None

    records = []
    line_id = 1
    for geom in merged_list:
        if not isinstance(geom, LineString) or geom.length <= 0:
            continue

        if sindex is not None:
            try:
                possible_indices = list(sindex.intersection(geom.bounds))
                matching_conf = [
                    gdf.iloc[idx]["mean_confidence"] for idx in possible_indices
                    if "mean_confidence" in gdf.iloc[idx] and geom.intersects(gdf.iloc[idx].geometry)
                ]
            except Exception:
                matching_conf = [overall_mean_conf]
        else:
            matching_conf = [
                row["mean_confidence"] for _, row in gdf.iterrows()
                if "mean_confidence" in row and geom.intersects(row.geometry)
            ]

        mean_conf = float(np.mean(matching_conf)) if matching_conf else overall_mean_conf

        records.append({
            "line_id": f"CAD_LINE_{line_id:05d}",
            "length_m": round(float(geom.length), 3),
            "mean_confidence": round(mean_conf, 4),
            "num_vertices": len(geom.coords),
            "geometry": geom,
        })
        line_id += 1

    return gpd.GeoDataFrame(records, crs=gdf.crs)


def vectorize_skeleton(
    skeleton: np.ndarray,
    transform: Union[Affine, Tuple[float, ...]],
    prob_map: Optional[np.ndarray] = None,
    simplify_tolerance: float = 0.5,
    min_length_m: Optional[float] = None,
    min_length_pixels: Optional[float] = None,
    merge_contiguous: bool = True,
    crs: str = "EPSG:28992"
) -> gpd.GeoDataFrame:
    """
    Convert a 1-pixel skeleton into a georeferenced GeoPandas GeoDataFrame of LineStrings.
    
    Args:
        skeleton: 2D binary numpy array of medial axis skeleton
        transform: Affine transform mapping pixel (col, row) to world coordinates (x, y)
        prob_map: Optional probability map for computing line confidence
        simplify_tolerance: Douglas-Peucker simplification tolerance in world units (meters)
        min_length_m: Minimum path length in meters to filter tiny noise spurs (default: 3.5m)
        min_length_pixels: Optional minimum path length in pixels
        merge_contiguous: If True, merges adjacent contiguous segments into unified polylines
        crs: Coordinate Reference System string
        
    Returns:
        gdf: GeoDataFrame containing vectorized cadastral boundary LineStrings
    """
    if isinstance(transform, (tuple, list)):
        transform = Affine(*transform[:6])

    if skeleton.sum() == 0:
        return gpd.GeoDataFrame(columns=["line_id", "length_m", "mean_confidence", "num_vertices", "geometry"], crs=crs)

    # 1. Build spatial graph
    G = skeleton_to_graph(skeleton)

    # 2. Trace paths
    paths = trace_graph_branches(G)

    records = []
    line_id = 1

    # Resolve min length threshold in meters
    pixel_size = abs(transform.a) if hasattr(transform, "a") else 0.25
    if min_length_m is not None:
        min_len_meters = float(min_length_m)
    elif min_length_pixels is not None:
        min_len_meters = float(min_length_pixels) * pixel_size
    else:
        min_len_meters = 3.5

    for path in paths:
        if len(path) < 2:
            continue

        # Convert pixel coordinates (row, col) to world coordinates (x, y)
        world_coords = []
        confidences = []
        for r, c in path:
            x, y = pixel_to_geo(c, r, transform)
            world_coords.append((x, y))
            if prob_map is not None:
                confidences.append(float(prob_map[r, c]))

        if len(world_coords) < 2:
            continue

        line_geom = LineString(world_coords)
        
        # Check minimum length
        if line_geom.length < min_len_meters:
            continue

        # 3. Douglas-Peucker simplification preserving cadastral geometry
        if simplify_tolerance > 0.0:
            simplified = line_geom.simplify(simplify_tolerance, preserve_topology=True)
            if isinstance(simplified, LineString) and len(simplified.coords) >= 2:
                line_geom = simplified

        if line_geom.length < min_len_meters:
            continue

        mean_conf = float(np.mean(confidences)) if confidences else 1.0

        records.append({
            "line_id": f"CAD_LINE_{line_id:05d}",
            "length_m": round(float(line_geom.length), 3),
            "mean_confidence": round(mean_conf, 4),
            "num_vertices": len(line_geom.coords),
            "geometry": line_geom,
        })
        line_id += 1

    gdf = gpd.GeoDataFrame(records, crs=crs)
    if merge_contiguous and not gdf.empty:
        gdf = merge_contiguous_lines(gdf)
    return gdf

