# SIH26012 FastAPI REST API Documentation

> **Smart India Hackathon (SIH26012)**  
> **API Title**: SIH26012 Cadastral Feature Extraction WebGIS API  
> **Base URL**: `http://localhost:8000` (Local) / `https://<tunnel-url>` (Cloud/Ngrok)  
> **OpenAPI Schema**: Accessible live at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`

---

## 1. REST Endpoint Summary Table

| Method | Endpoint | Purpose | Request Payload | Response Type | Authentication | Status |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: |
| `GET` | `/api/health` | System health, device detection, and checkpoint state | None | `application/json` | None (Public) | `IMPLEMENTED` |
| `GET` | `/api/demo-samples` | List curated demo patches across urban, rural, and test splits | None | `application/json` | None (Public) | `IMPLEMENTED` |
| `GET` | `/api/sample-image/{patch_id}` | Serves $512 \times 512$ RGB aerial orthoimage | URL path param | `image/png` | None (Public) | `IMPLEMENTED` |
| `GET` | `/api/sample-mask/{patch_id}` | Serves ground truth registered boundary mask | URL path param | `image/png` | None (Public) | `IMPLEMENTED` |
| `POST` | `/api/predict` | Runs neural network inference, produces probability map | JSON: `{"patch_id", "model_name"}` | `application/json` | None (Public) | `IMPLEMENTED` |
| `POST` | `/api/vectorize` | Vectorizes raster to GeoJSON lines & parcel polygons | JSON: `{"patch_id", "threshold", ...}` | `application/json` | None (Public) | `IMPLEMENTED` |
| `POST` | `/api/topology-audit` | Audits topological validity of user-edited GeoJSON | JSON: `{"geojson": {...}}` | `application/json` | None (Public) | `IMPLEMENTED` |
| `GET` | `/api/supporting-layers/{type}/{id}`| Serves context layers (buildings / roads) | URL path params | `application/json` | None (Public) | `IMPLEMENTED` |
| `GET` | `/api/ground-truth-vector/{patch_id}`| Serves registered cadastral lines as GeoJSON | URL path param | `application/json` | None (Public) | `IMPLEMENTED` |

---

## 2. Detailed Endpoint Specifications

### 2.1 Health Check & Runtime Telemetry
- **Method & Path**: `GET /api/health`
- **Purpose**: Verifies that the FastAPI backend is operational, detects the active compute device (NVIDIA CUDA / Apple Silicon MPS / CPU), lists loaded neural network models, and reports checkpoint metadata.
- **Called by**: WebGIS frontend on load, automated test suite (`tests/test_backend_api.py`), container health probes.

#### Example Request
```bash
curl -X GET "http://localhost:8000/api/health"
```

#### Example Response (200 OK)
```json
{
  "status": "healthy",
  "system": "SIH26012 Cadastral Extraction System",
  "device": "mps",
  "available_models": [
    "baseline",
    "pmg",
    "pmg_dg",
    "full"
  ],
  "checkpoint_info": {
    "path": "experiments/full_512_fixed/results/checkpoints/best_model.pth",
    "epoch": 14,
    "val_f1": 0.1422,
    "selected_threshold": 0.5,
    "device": "mps"
  }
}
```

---

### 2.2 Demo Samples Catalog
- **Method & Path**: `GET /api/demo-samples`
- **Purpose**: Returns curated high-value demonstration patches representing diverse terrain (Residential Blocks, High-Density Urban Grid, Rural Agricultural Lands, and Held-Out Benchmark Test Samples) with computed WGS84 spatial bounds.
- **Called by**: WebGIS sample selector dropdown on page initialization.

#### Example Request
```bash
curl -X GET "http://localhost:8000/api/demo-samples"
```

#### Example Response (200 OK)
```json
{
  "samples": [
    {
      "patch_id": "tile-190000-310000_r24576_c02048",
      "tile_id": "tile-190000-310000",
      "split": "test",
      "is_urban": false,
      "boundary_pixels": 4512,
      "boundary_density": 0.0172,
      "image_url": "/api/sample-image/tile-190000-310000_r24576_c02048",
      "mask_url": "/api/sample-mask/tile-190000-310000_r24576_c02048",
      "wgs84_bounds": [
        [51.8152, 4.8821],
        [51.8164, 4.8840]
      ]
    }
  ]
}
```

---

### 2.3 Cadastral Boundary Inference
- **Method & Path**: `POST /api/predict`
- **Purpose**: Executes deep learning inference for a designated patch. Computes continuous boundary probabilities, populates the in-memory prediction cache, and generates a skeletonized high-contrast overlay.
- **Request Body**:
  ```json
  {
    "patch_id": "tile-190000-310000_r24576_c02048",
    "model_name": "full"
  }
  ```
- **Processing Logic**:
  1. Loads RGB patch from `data/processed/patches/`.
  2. Applies `preprocess_for_inference()`: casts to `float32`, divides by 255.0, resizes via `INTER_LINEAR` to native training resolution, converts to CHW tensor.
  3. Executes `CadastreUNetFull` forward pass under `torch.no_grad()`.
  4. Caches raw floating-point probability map in `PREDICTION_CACHE[patch_id]`.
  5. Generates a base64 grayscale probability heatmap and a 2-layer high-contrast RGBA skeleton image (5px black halo + 2px lime-green core).

#### Example Request
```bash
curl -X POST "http://localhost:8000/api/predict" \
     -H "Content-Type: application/json" \
     -d '{"patch_id": "tile-190000-310000_r24576_c02048", "model_name": "full"}'
```

#### Example Response (200 OK)
```json
{
  "patch_id": "tile-190000-310000_r24576_c02048",
  "model_name": "full",
  "mean_confidence": 0.0812,
  "max_confidence": 0.9421,
  "heatmap_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAg...",
  "skeleton_image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAg...",
  "wgs84_bounds": [
    [51.8152, 4.8821],
    [51.8164, 4.8840]
  ]
}
```

---

### 2.4 Vectorization, Closed Parcel Extraction & Topology Audit
- **Method & Path**: `POST /api/vectorize`
- **Purpose**: Converts the cached raster probability map into survey-grade continuous **LineStrings** and **closed property parcel Polygons**, executes an automated ISO 19107 topology audit, and outputs reprojected WGS84 GeoJSON.
- **Request Body**:
  ```json
  {
    "patch_id": "tile-190000-310000_r24576_c02048",
    "threshold": 0.45,
    "simplify_tolerance": 0.35,
    "min_length_m": 3.5,
    "min_component_size": 40,
    "merge_contiguous": true
  }
  ```
- **Internal Processing Stages**:
  1. **Morphological Closing**: Applies a $5 \times 5$ rectangular structuring element to close micro-fissures along parcel perimeters.
  2. **Medial Axis Thinning**: Uses fast morphological erosion/dilation kernel passes to extract 1-pixel wide line centerlines.
  3. **Spatial Graph Tracing**: Constructs an 8-connected undirected graph in NetworkX; identifies endpoints ($\text{degree}=1$) and junctions ($\text{degree}\ge3$), tracing maximal paths between them.
  4. **Douglas-Peucker Simplification**: Smooths lines and removes redundant collinear vertices while preserving parcel corners ($\epsilon = 0.35\text{m}$).
  5. **Closed Parcel Polygonization**: Inverts the closed boundary mask, applies OpenCV contour extraction (`RETR_CCOMP`), filters polygons by realistic property size ($20\text{ m}^2 \le \text{Area} \le 25,000\text{ m}^2$), calculates exact metric area ($\text{m}^2$) and perimeter ($\text{m}$).
  6. **Endpoint Snapping & Polyline Merging**: Snaps nearby line endpoints within $1.8\text{m}$ and merges contiguous LineStrings via `shapely.ops.linemerge` into solid continuous borders.
  7. **ISO 19107 STRtree Topology Audit**: Audits geometries for self-intersections, duplicates, tiny fragments, and dangling segments.
  8. **Reprojection**: Transforms all coordinates from Dutch metric (`EPSG:28992`) to standard WebGIS coordinates (`EPSG:4326`).

#### Example Request
```bash
curl -X POST "http://localhost:8000/api/vectorize" \
     -H "Content-Type: application/json" \
     -d '{
       "patch_id": "tile-190000-310000_r24576_c02048",
       "threshold": 0.45,
       "simplify_tolerance": 0.35,
       "min_length_m": 3.5,
       "min_component_size": 40,
       "merge_contiguous": true
     }'
```

#### Example Response (200 OK)
```json
{
  "patch_id": "tile-190000-310000_r24576_c02048",
  "total_lines": 34,
  "total_length_m": 842.15,
  "total_parcels": 8,
  "geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "LineString",
          "coordinates": [
            [4.88214, 51.81532],
            [4.88245, 51.81551],
            [4.88298, 51.81580]
          ]
        },
        "properties": {
          "line_id": "CAD_LINE_0001",
          "length_m": 48.62,
          "mean_confidence": 0.88,
          "num_vertices": 3
        }
      }
    ]
  },
  "parcels_geojson": {
    "type": "FeatureCollection",
    "features": [
      {
        "type": "Feature",
        "geometry": {
          "type": "Polygon",
          "coordinates": [[[4.8821, 51.8153], [4.8825, 51.8153], [4.8825, 51.8157], [4.8821, 51.8157], [4.8821, 51.8153]]]
        },
        "properties": {
          "parcel_id": "PARCEL_0001",
          "area_m2": 420.5,
          "perimeter_m": 84.2
        }
      }
    ]
  },
  "topology_report": {
    "status": "VALID",
    "is_valid": true,
    "total_lines": 34,
    "total_length_m": 842.15,
    "total_parcels": 8,
    "invalid_geometries_count": 0,
    "self_intersections_count": 0,
    "duplicate_lines_count": 0,
    "tiny_fragments_count": 0,
    "disconnected_segments_count": 0,
    "fragmentation_index": 4.037,
    "warnings": []
  },
  "wgs84_bounds": [
    [51.8152, 4.8821],
    [51.8164, 4.8840]
  ]
}
```

---

### 2.5 Real-Time Topology Audit of User Edits
- **Method & Path**: `POST /api/topology-audit`
- **Purpose**: Enables live human-in-the-loop geometric auditing. Revenue surveyors can modify, draw, or delete parcel boundaries in a GIS editor and submit the resulting GeoJSON to be audited against OGC / ISO 19107 geometric integrity rules in real time.
- **Request Body**:
  ```json
  {
    "geojson": {
      "type": "FeatureCollection",
      "features": [...]
    }
  }
  ```

#### Example Response (200 OK)
```json
{
  "topology_report": {
    "status": "VALID",
    "is_valid": true,
    "total_lines": 12,
    "total_length_m": 312.4,
    "invalid_geometries_count": 0,
    "self_intersections_count": 0,
    "duplicate_lines_count": 0,
    "tiny_fragments_count": 0,
    "warnings": []
  }
}
```

---

### 2.6 Supporting Context Layers (Buildings & Roads)
- **Method & Path**: `GET /api/supporting-layers/{layer_type}/{patch_id}`
- **Parameters**: `layer_type` (`buildings` or `roads`), `patch_id` (string).
- **Purpose**: Provides modular context layers clearly distinguished from cadastral boundaries. Serves building footprints or road network polylines to provide visual reference for human surveyors without conflating them with legal land property boundaries.

#### Example Request
```bash
curl -X GET "http://localhost:8000/api/supporting-layers/buildings/tile-190000-310000_r24576_c02048"
```

#### Example Response (200 OK)
```json
{
  "type": "FeatureCollection",
  "layer_type": "buildings",
  "is_supporting_layer": true,
  "layer_info": "Supporting Building Footprints (Context Layer)",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[4.8822, 51.8154], [4.8826, 51.8154], [4.8826, 51.8158], [4.8822, 51.8158], [4.8822, 51.8154]]]
      },
      "properties": {
        "building_id": "BLD_001",
        "type": "Residential",
        "layer": "Supporting Building Footprint"
      }
    }
  ]
}
```

---

### 2.7 Registered Ground Truth Cadastral Vectors
- **Method & Path**: `GET /api/ground-truth-vector/{patch_id}`
- **Purpose**: Extracts authentic cadastral boundaries from the official Kadaster BRK reference dataset for the designated patch and returns them as clean vector LineStrings. Enables side-by-side visual comparison in the WebGIS interface.

---

## 3. Error Handling & HTTP Status Codes

The API uses standard HTTP error codes:

| HTTP Code | Name | Cause | Example Response Body |
| :---: | :--- | :--- | :--- |
| `400` | Bad Request | Invalid parameter (e.g. invalid layer type) | `{"detail": "Invalid layer type. Choose 'buildings' or 'roads'."}` |
| `404` | Not Found | Patch ID does not exist in spatial catalog | `{"detail": "Patch not found"}` |
| `422` | Unprocessable Entity | Pydantic JSON validation failure | `{"detail": [{"loc": ["body", "patch_id"], "msg": "field required"}]}` |
| `500` | Internal Server Error | Unhandled runtime exception | `{"detail": "Inference failure: CUDA out of memory"}` |
