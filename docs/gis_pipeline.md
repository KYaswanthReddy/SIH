# SIH26012 Raster-to-Vector GIS Pipeline & Topology Engine

## 1. Overview
The raster-to-vector pipeline converts continuous AI probability rasters into GIS-compliant vector geometries (`LineString` and `Polygon`) while strictly maintaining spatial georeferencing and topological validity.

```
AI Probability Map [0.0, 1.0]
        ↓
Morphological Closing (3×3 Kernel) & Area Filtering (min 15 px)
        ↓
Guo-Hall Thinning / Medial Axis Skeletonization (1 px width)
        ↓
Junction and Endpoint Node Detection
        ↓
NetworkX Spatial Graph Construction (8-connected adjacency)
        ↓
Cycle and Branch LineString Tracing
        ↓
Douglas-Peucker Geometric Simplification (ε = 0.5m)
        ↓
Raster-to-World Affine Transformation (EPSG:28992)
        ↓
GeoPandas GeoDataFrame Construction & EPSG:4326 Reprojection
        ↓
Topology Audit Engine (STRtree Intersection Analysis)
        ↓
GeoJSON / ESRI Shapefile Export
```

---

## 2. Pipeline Modules

### A. Refinement (`gis/refinement.py`)
- Applies morphological closing with a 3×3 elliptical structuring element to bridge sub-pixel gaps.
- Removes disconnected speckle noise smaller than `min_component_size` (default: 15 pixels).

### B. Skeletonization (`gis/skeletonization.py`)
- Reduces binary boundary regions into 1-pixel wide medial centerlines.
- Uses convolutional 8-neighborhood kernel operations to classify each pixel into:
  - **Endpoints**: Degree 1
  - **Pathway pixels**: Degree 2
  - **Junctions**: Degree $\ge 3$

### C. Vectorization (`gis/vectorization.py`)
- Constructs an undirected topological graph in NetworkX.
- Traces simple paths between junction and endpoint nodes.
- Applies Douglas-Peucker line simplification ($\epsilon = 0.5\text{m}$) to eliminate collinear vertices while preserving parcel corners.
- Applies Affine transformation matrices to map pixel coordinates $(r, c)$ to projected spatial coordinates $(X, Y)$ in Dutch RD New (`EPSG:28992`), reprojecting to WGS84 (`EPSG:4326`) for web mapping.

---

## 3. Topology Validation Engine (`gis/topology.py`)
Audits all generated vector geometries against core GIS integrity constraints:

1. **Invalid Geometries**: Checks `geom.is_valid` (e.g., self-touching rings, degenerate lines).
2. **Self-Intersections**: Checks `geom.is_simple`.
3. **Duplicate Overlaps**: Flags coincident lines sharing the same vertex sequence.
4. **Tiny Fragments**: Flags line segments shorter than `min_length_m` (default: 2.0 meters).
5. **Interior Crossings**: Uses `shapely.STRtree` bounding-box acceleration to flag non-junction line intersections.
6. **Isolated Segments**: Flags floating segments not connected to any adjacent parcel boundary.
