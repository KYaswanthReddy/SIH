# SIH26012 System Architecture & Technical Flow

> **Smart India Hackathon (SIH26012)**  
> **Problem Title**: Automated Cadastral Feature Extraction & Urban Parcel Mapping System  
> **System Classification**: Deep Learning Geospatial Computer Vision & Computational GIS Platform

---

## 1. System Topology Overview

The SIH26012 platform is architected as an end-to-end decoupled system comprising a high-performance **FastAPI asynchronous backend**, a modern **WebGIS Leaflet client**, a modular **PyTorch deep learning inference core**, and an **algorithmic computational GIS topology engine**.

```mermaid
flowchart TD
    subgraph ClientLayer["Client Layer (WebGIS Frontend)"]
        UI["Leaflet 1.9.4 Map Dashboard\n(Tailwind CSS + Lucide Icons)"]
        CONTROL["Collapsible Control Decks\n(Sample Selector & Threshold Sliders)"]
        VECTOR_RENDER["Vector Canvas & SVG Renderer\n(Solid Border Lines & Shaded Parcels)"]
        AUDIT_UI["Real-Time Topology Inspector\n(Health Counters & Audit Logs)"]
    end

    subgraph APILayer["API Gateway & Service Layer (FastAPI Backend)"]
        ROUTER["REST API Router (backend/main.py)"]
        CORS_MID["CORS & No-Cache Middleware"]
        INF_SVC["Inference Service\n(backend/services/inference_service.py)"]
        GIS_SVC["GIS & Topology Service\n(backend/services/gis_service.py)"]
    end

    subgraph AIEngine["Deep Learning Core (PyTorch Engine)"]
        PREPROC["Authoritative Preprocessor\n(Float32 / 255 -> Resized Tensor)"]
        MODEL["CadastreUNetFull (12.38M Params)\n(PMG + MixStyle DG + 8-Way Connectivity)"]
        FUSION["Refinement Fusion Layer\n(Boundary Logits + Affinity Tensor)"]
    end

    subgraph GISEngine["Computational GIS Engine"]
        MORPH["Morphological Refinement\n(3x3 Gap-Closing Kernel)"]
        SKEL["Medial Axis Thinning\n(Guo-Hall / Zhang-Suen 1px Skeletons)"]
        GRAPH["NetworkX Spatial Graph Tracing\n(Junction & Endpoint Traversal)"]
        SIMPLIFY["Douglas-Peucker Simplification\n(Epsilon = 0.35m - 0.50m)"]
        POLYGONIZE["Contour Inversion Polygonizer\n(Area & Perimeter Metrics)"]
        TOPO_AUDIT["ISO 19107 Topology Validator\n(Shapely STRtree Spatial Index)"]
    end

    subgraph DataStore["Data & Persistence Layer"]
        TILES["9 Raw GeoTIFF Tiles\n(10km x 10km, 0.25m GSD, EPSG:28992)"]
        CATALOG["CSV Spatial Metadata Catalogs\n(3,700 Georeferenced Patches)"]
        CKPT["A100 Trained Checkpoint\n(best_model.pth, 141 MB)"]
        GEOJSON["Survey-Grade GeoJSON Exports\n(QGIS / ArcGIS / SVAMITVA)"]
    end

    %% Client to API
    UI -->|HTTP POST /api/predict| ROUTER
    UI -->|HTTP POST /api/vectorize| ROUTER
    UI -->|HTTP POST /api/topology-audit| ROUTER
    UI -->|HTTP GET /api/demo-samples| ROUTER
    CONTROL --> UI

    %% API to Services
    ROUTER --> CORS_MID
    CORS_MID --> INF_SVC
    CORS_MID --> GIS_SVC

    %% Services to Processing Engines
    INF_SVC --> PREPROC
    PREPROC --> MODEL
    MODEL --> FUSION
    FUSION -->|Probability Heatmap| INF_SVC

    GIS_SVC --> MORPH
    MORPH --> SKEL
    SKEL --> GRAPH
    GRAPH --> SIMPLIFY
    SIMPLIFY --> POLYGONIZE
    POLYGONIZE --> TOPO_AUDIT

    %% Data connections
    INF_SVC -.->|Load Weights| CKPT
    GIS_SVC -.->|Read Patch Metadata| CATALOG
    GIS_SVC -.->|Load Imagery & Masks| TILES
    TOPO_AUDIT -->|Generate Vector| GEOJSON
    GEOJSON --> VECTOR_RENDER
    TOPO_AUDIT --> AUDIT_UI
```

---

## 2. Detailed Multi-Layer Architecture

### 2.1 Frontend Architecture (`frontend/index.html`)
- **Core Technologies**: Vanilla JavaScript (ES6+), HTML5 Canvas, Tailwind CSS (via CDN), Google Fonts (*Inter* for UI typography, *JetBrains Mono* for coordinates and technical metrics).
- **Map Rendering Engine**: **Leaflet.js 1.9.4** mapping library.
- **Base Layers**:
  - Primary: ESRI World Imagery (High-Resolution Satellite Basemap).
  - Secondary / Fallback: OpenStreetMap Carto Standard.
- **Dynamic Overlays**:
  - **AI Probability Heatmap Overlay**: Live base64-encoded PNG rendered as a `L.imageOverlay` with interactive opacity slider.
  - **Continuous Solid Cadastral Boundaries**: High-contrast rendering featuring a $4.4\text{px}$ dark casing halo (`#000000`, 0.8 opacity) and a $2.5\text{px}$ electric cyan core line (`#00F0FF`, 0.95 opacity).
  - **Closed Cadastral Parcel Polygons**: Semi-transparent teal fills (`#0D9488`, 0.22 opacity) with fine borders (`#2DD4BF`, 1.8px width) outlining interior property plots.
  - **Ground Truth Cadastral Vectors**: High-visibility spring green polylines (`#22C55E`, 2.5px width) toggled on demand to visually benchmark model precision against authentic land registry records.
  - **Supporting Context Layers**: Distinctly styled building footprints (amber `#F59E0B`) and road network centerlines (sky blue `#38BDF8`).
- **Interactive Capabilities**:
  - Feature inspection modal popups displaying Feature ID, metric length ($\text{m}$), model confidence score ($\%$), parcel plot area ($\text{m}^2$), and perimeter ($\text{m}$).
  - Human-in-the-Loop **Feature Approval Action** allowing surveyors to review, approve, and stage individual boundary features.
  - One-click GeoJSON export downloading survey-grade geometries directly for desktop GIS analysis.

---

### 2.2 Backend API Architecture (`backend/main.py`)
- **Framework**: **FastAPI** (Python 3.10+) running under ASGI server **Uvicorn**.
- **Design Pattern**: Service-Oriented Architecture (SOA) separating request routing from business logic services.
- **Key Modules**:
  - `backend/main.py`: Endpoint definition, request deserialization via Pydantic, HTTP exception handling, and static file mounting.
  - `backend/services/inference_service.py`: Hardware device abstraction (CUDA / Apple Silicon MPS / CPU), checkpoint state management, and inference execution.
  - `backend/services/gis_service.py`: Coordinate space conversions, contour polygonization, and topology orchestration.
- **Security & Middleware**:
  - CORS middleware allowing cross-origin integration for distributed deployments.
  - Strict `no-cache` HTTP header middleware preventing browser caching of dynamic prediction heatmaps.

---

### 2.3 Computational GIS & Topology Pipeline (`gis/`)

```mermaid
flowchart TD
    RAW_PROB["Model Probability Map [0.0, 1.0]\n(512x512 Float32 Array)"]
    
    subgraph Refinement["Stage 1: Morphological Cleanup"]
        BIN["Dynamic Thresholding\n(Default: 0.45 - 0.50)"]
        CLOSE["Morphological Closing\n(3x3 Structuring Element)"]
        FILTER["Component Area Filter\n(Prune Blobs < 40px)"]
    end

    subgraph Centerline["Stage 2: Medial Axis Skeletonization"]
        THIN["Zhang-Suen / Guo-Hall Thinning\n(Sub-pixel 1px Medial Axis)"]
        NODES["Node Classification\n(Endpoints: Deg 1 | Junctions: Deg >= 3)"]
    end

    subgraph Vectorization["Stage 3: Spatial Graph Vectorization"]
        GRAPH_BUILD["NetworkX Undirected Graph\n(8-Connected Adjacency)"]
        TRACE["Contiguous Branch Tracing\n(Cycle & Segment Traversal)"]
        SIMP["Douglas-Peucker Simplification\n(Epsilon = 0.35m Tolerance)"]
        AFFINE["Affine Coordinate Projection\n(Pixel (c, r) -> EPSG:28992 Metric XY)"]
        SNAP["Endpoint Snapping\n(Bridge Micro-Gaps <= 1.8m)"]
        MERGE["Contiguous Polyline Merge\n(linemerge to Solid Continuous Lines)"]
    end

    subgraph ParcelExtraction["Stage 4: Closed Parcel Polygonization"]
        INV["Boundary Mask Inversion\n(Interior Property Plots)"]
        CONTOUR["OpenCV Topological Contour Tracing\n(RETR_CCOMP / CHAIN_APPROX_TC89_L1)"]
        METRIC_CALC["Metric Calculation\n(Area m²: px * 0.0625 | Perimeter m: px * 0.25)"]
        AREA_FILTER["Area Filter\n(20 m² <= Land Area <= 25,000 m²)"]
        POLY_GDF["Parcel GeoDataFrame\n(EPSG:28992 -> EPSG:4326 Reprojection)"]
    end

    subgraph TopologyAudit["Stage 5: ISO 19107 Topology Audit"]
        STRTREE["Shapely STRtree Spatial Index\n(Pairwise Bounding-Box Pruning)"]
        VALID_CHECK["OGC Validity & Self-Intersection Audit\n(geom.is_valid & geom.is_simple)"]
        DUP_CHECK["Duplicate Line Detection\n(Normalized Vertex Sequences)"]
        DANGLING["Dangling Segment Analysis\n(Endpoint Proximity < 0.5m)"]
        REPORT["Generate JSON Audit Report\n(Status: VALID | 0 Errors)"]
    end

    RAW_PROB --> BIN
    BIN --> CLOSE
    CLOSE --> FILTER
    FILTER --> THIN
    THIN --> NODES
    NODES --> GRAPH_BUILD
    GRAPH_BUILD --> TRACE
    TRACE --> SIMP
    SIMP --> AFFINE
    AFFINE --> SNAP
    SNAP --> MERGE

    CLOSE --> INV
    INV --> CONTOUR
    CONTOUR --> METRIC_CALC
    METRIC_CALC --> AREA_FILTER
    AREA_FILTER --> POLY_GDF

    MERGE --> STRTREE
    STRTREE --> VALID_CHECK
    VALID_CHECK --> DUP_CHECK
    DUP_CHECK --> DANGLING
    DANGLING --> REPORT
```

---

### 2.4 Deep Learning Architecture (`ai/models/full_model.py`)

The neural network is a specialized geospatial architecture designated **`CadastreUNetFull`**:

```mermaid
flowchart TD
    INPUT["Input Aerial Image\n(3 x 512 x 512, Float32 [0, 1])"]
    
    subgraph Encoder["Encoder Backbone (Residual U-Net)"]
        INC["DoubleConv Block 0\n(3 -> 32 channels)"]
        DOWN1["Downsample Stage 1\n(32 -> 64 channels, 256x256)"]
        DG1["MixStyle Domain Generalization 1\n(p=0.5, Beta(0.2, 0.2) Feature Perturbation)"]
        DOWN2["Downsample Stage 2\n(64 -> 128 channels, 128x128)"]
        DG2["MixStyle Domain Generalization 2\n(p=0.5, Beta(0.2, 0.2) Feature Perturbation)"]
        DOWN3["Downsample Stage 3\n(128 -> 256 channels, 64x64)"]
        DOWN4["Downsample Stage 4\n(256 -> 512 channels, 32x32)"]
    end

    subgraph PMG["Progressive Multiscale Generator (ai/models/pmg.py)"]
        DIL1["Branch 1 (d=1, 2)\n(Channels: 64 -> 128)"]
        DIL2["Branch 2 (d=1, 2, 4)\n(Channels: 128 -> 128)"]
        DIL3["Branch 3 (d=1, 2, 4)\n(Channels: 256 -> 128)"]
        DIL4["Branch 4 (d=1, 2, 4)\n(Channels: 512 -> 128)"]
        FUSION_TD["Progressive Coarse-to-Fine Top-Down Fusion\n(Interpolate + Lateral Residual Addition)"]
        ADAPTERS["1x1 Convolutional Adapters\n(Inject Context to Skip Connections)"]
    end

    subgraph Decoder["Decoder Backbone (Residual Skip Connections)"]
        UP1["Upsample Stage 1\n(512 + 256 -> 256 channels)"]
        UP2["Upsample Stage 2\n(256 + 128 -> 128 channels)"]
        UP3["Upsample Stage 3\n(128 + 64 -> 64 channels)"]
        UP4["Upsample Stage 4\n(64 + 32 -> 32 channels, 512x512)"]
    end

    subgraph DualHeads["Dual Prediction Heads & Refinement"]
        HEAD_B["Primary Boundary Head\n(1x1 Conv -> Sigmoid -> 1x512x512)"]
        HEAD_C["8-Way Connectivity Head\n(3x3 Conv + BN + 1x1 Conv -> 8x512x512)"]
        FUSION_REFINE["Connectivity-Refined Fusion Layer\n(Concat [B, C] -> 3x3 Conv -> BN -> ReLU -> 1x1 Conv)"]
        OUT_REFINED["Refined Boundary Probability Map\n(1 x 512 x 512 in [0.0, 1.0])"]
    end

    INPUT --> INC
    INC --> DOWN1
    DOWN1 --> DG1
    DG1 --> DOWN2
    DOWN2 --> DG2
    DG2 --> DOWN3
    DOWN3 --> DOWN4

    DG1 -.-> DIL1
    DG2 -.-> DIL2
    DOWN3 -.-> DIL3
    DOWN4 -.-> DIL4

    DIL1 & DIL2 & DIL3 & DIL4 --> FUSION_TD
    FUSION_TD --> ADAPTERS

    DOWN4 --> UP1
    ADAPTERS -.->|Multiscale Skip 4| UP1
    UP1 --> UP2
    ADAPTERS -.->|Multiscale Skip 3| UP2
    UP2 --> UP3
    ADAPTERS -.->|Multiscale Skip 2| UP3
    UP3 --> UP4
    ADAPTERS -.->|Multiscale Skip 1| UP4

    UP4 --> HEAD_B
    UP4 --> HEAD_C
    HEAD_B & HEAD_C --> FUSION_REFINE
    FUSION_REFINE --> OUT_REFINED
```

---

## 3. Data Flow & Request-Response Sequence

```mermaid
sequenceDiagram
    autonumber
    actor User as Revenue Surveyor / GIS Officer
    participant Browser as Leaflet WebGIS Client
    participant API as FastAPI Backend (/api/*)
    participant InfService as Inference Service
    participant GPU as PyTorch Engine (CUDA / MPS)
    participant GISService as GIS & Topology Service
    participant Disk as Processed Storage & Metadata

    User->>Browser: Select Curated Demo Sample (e.g. Residential Block)
    Browser->>API: GET /api/sample-image/{patch_id}
    API->>Disk: Read PNG (512x512 RGB)
    Disk-->>API: Binary Image Stream
    API-->>Browser: Image Displayed on Leaflet Map
    
    User->>Browser: Click "Run AI Cadastral Extraction (1-Click)"
    Browser->>API: POST /api/predict {patch_id, model_name: "full"}
    API->>InfService: predict(image_np, model_name="full")
    InfService->>GPU: Forward Pass CadastreUNetFull
    GPU-->>InfService: refined_prob (512x512 Float32)
    InfService-->>API: prob_map, confidence metrics, base64 skeleton
    API-->>Browser: 200 OK (prob_map cached, prediction stats)

    Browser->>API: POST /api/vectorize {patch_id, threshold: 0.45, simplify: 0.35}
    API->>GISService: vectorize_and_audit(prob_map, transform)
    GISService->>GISService: Medial axis thinning & NetworkX graph tracing
    GISService->>GISService: Extract closed parcel polygons (interior contours)
    GISService->>GISService: Run ISO 19107 STRtree topology audit
    GISService-->>API: geojson_wgs84, parcels_geojson_wgs84, topology_report
    API-->>Browser: 200 OK (GeoJSON Features & Audit Metrics)

    Browser->>Browser: Render Solid Cyan Lines & Teal Shaded Parcel Polygons
    Browser->>Browser: Update Topology Dashboard (0 Violations, 100% VALID)
    User->>Browser: Click Parcel -> View Land Area (m²) and Perimeter (m)
    User->>Browser: Click "Export Clean GeoJSON" -> Download File for QGIS
```

---

## 4. Deployment Architecture

The system supports both local workstation execution and containerized cloud/server deployment.

```mermaid
flowchart TD
    subgraph DeploymentEnvironments["Supported Runtime Environments"]
        LOCAL["Local Workstation\n(macOS Apple Silicon MPS / Linux CUDA)"]
        COLAB["Google Colab Pro\n(NVIDIA T4 / V100 / A100 GPU)"]
        DOCKER["Docker Container\n(uvicorn ASGI + Python 3.10)"]
    end

    subgraph StorageLayer["Data & Assets Storage"]
        FS_DATA["data/processed/patches/\n(3,700 512x512 PNG tiles)"]
        FS_META["data/processed/metadata/\n(Spatial CSV Catalogs)"]
        FS_WEIGHTS["experiments/full_512_fixed/checkpoints/\n(141 MB PyTorch best_model.pth)"]
    end

    subgraph ServiceEngine["Application Container / Process"]
        UVICORN["Uvicorn High-Concurrency ASGI Server\n(Port 8000)"]
        FASTAPI["FastAPI REST Application Instance"]
        TORCH_RUNTIME["PyTorch C++ Backend\n(Native Float32 Tensor Operations)"]
    end

    subgraph PublicAccess["Client Entry Points"]
        LOCAL_BROWSER["Local Browser\n(http://localhost:8000)"]
        NGROK["Ngrok Secure Tunnel\n(https://*.ngrok-free.app)"]
        REVERSE_PROXY["Nginx Reverse Proxy / SSL\n(Production Staging)"]
    end

    LOCAL --> UVICORN
    COLAB --> UVICORN
    DOCKER --> UVICORN

    UVICORN --> FASTAPI
    FASTAPI --> TORCH_RUNTIME
    FASTAPI -.-> FS_DATA
    FASTAPI -.-> FS_META
    TORCH_RUNTIME -.-> FS_WEIGHTS

    UVICORN --> LOCAL_BROWSER
    UVICORN --> NGROK
    UVICORN --> REVERSE_PROXY
```

---

## 5. Architectural Distinction: Cadastral Boundaries vs Supporting Context Layers

> [!IMPORTANT]
> **Fundamental Cadastral Governance Principle**:
> A cadastral boundary delineates **legal land ownership and property rights**. It is invisible in many real-world scenarios (e.g., unmarked inheritance partitions, open agricultural field boundaries, rights-of-way).
> 
> Many hackathon projects conflate building footprints with cadastral parcel boundaries. In SIH26012:
> 1. **Cadastral Parcel Boundaries**: The primary output of our neural network and GIS pipeline, representing property demarcations registered with the national land authority.
> 2. **Building Footprints & Road Networks**: Treated strictly as **Supporting Context Layers** served via `/api/supporting-layers/{type}/{patch_id}`. They provide situational context to revenue surveyors without corrupting legal property deeds.
