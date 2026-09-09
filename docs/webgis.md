# SIH26012 FastAPI & Leaflet WebGIS Application

## 1. Overview
The SIH26012 WebGIS interface provides an interactive, full-stack platform for AI inference, layer visualization, real-time topology auditing, human-in-the-loop geometric editing, and GIS vector export.

---

## 2. System Architecture

```
+----------------------------------------------------------------+
|                        Web Browser                             |
|  - Leaflet Map (ESRI Satellite, EPSG:3857)                    |
|  - Real-time Heatmap Canvas Overlay                            |
|  - Vector Boundary Layer (Editable GeoJSON)                    |
|  - Supporting Layers (Buildings & Roads)                       |
|  - Topology Health Dashboard & Warning Center                  |
+-------------------------------+--------------------------------+
                                | HTTP REST / JSON
+-------------------------------v--------------------------------+
|                   FastAPI Backend Server                       |
|  - /api/health                                                 |
|  - /api/demo-samples                                           |
|  - /api/predict                                                |
|  - /api/vectorize                                              |
|  - /api/topology-audit                                         |
|  - /api/supporting-layers                                      |
+-------------------------------+--------------------------------+
                                |
             +------------------+------------------+
             |                                     |
+------------v-----------+              +----------v-------------+
|    AI Inference        |              |     GIS Processing     |
| - PyTorch (MPS / CUDA) |              | - GeoPandas & Shapely  |
| - CadastreUNetFull     |              | - NetworkX & Rasterio  |
+------------------------+              +------------------------+
```

---

## 3. Running the Full Stack Application

### Start the FastAPI Server
```bash
.venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Accessing the WebGIS Interface
Open your browser to:
[http://localhost:8000](http://localhost:8000)

### Key User Features:
1. **Curated Demo Sample Selector**: Instantly load representative Urban, High-Density Boundary, Rural, and Validation patches.
2. **AI Model Selector**: Switch dynamically between `Full (PMG + DG + Conn)` and `Baseline (Residual U-Net)`.
3. **Interactive Inference**: Generates base64 PNG prediction heatmaps on demand.
4. **Vectorization & GIS Extraction**: Converts probability heatmaps to clean vector polylines with configurable probability thresholds and Douglas-Peucker simplification tolerance.
5. **Supporting Context Layers**: Toggle Supporting Building Footprints (amber) and Supporting Road Networks (cyan) to provide visual context without conflating them with cadastral parcel boundaries.
6. **Topology Audit Dashboard**: Live metric counters displaying invalid geometries, duplicate lines, tiny fragments, and self-intersections.
7. **GeoJSON Export**: One-click download of GIS-compliant GeoJSON parcel boundary features for downstream CAD / GIS workflows.
