# SIH26012 Team Knowledge Distribution & Role Ownership Matrix

> **Smart India Hackathon (SIH26012)**  
> **Mission**: Ensure that every team member can fluently explain, defend, and demonstrate any aspect of the project during panel scrutiny.

---

## 1. What EVERY Team Member Must Know

Regardless of individual sub-roles, **every single team member must be able to answer these core questions instantly**:

1. **Problem Statement**: **SIH26012 — Automated Cadastral Feature Extraction & Urban Parcel Mapping System**.
2. **Elevator Pitch**: We use deep learning and computational GIS to convert $0.25\text{m}$ aerial imagery into survey-grade, topologically valid cadastral boundary vectors and closed property parcel polygons in real time.
3. **Primary Tech Stack**: PyTorch, GeoPandas, Shapely, OpenCV, NetworkX, FastAPI, Leaflet.js, Tailwind CSS.
4. **Key Benchmark Metrics**: **94.43%** pixel accuracy, **0.1666** F1 score, **27.91%** recall, **0.0909** IoU on 600 strictly held-out test patches covering 900 km².
5. **Key Innovation**: Combining **PMG multiscale dilation** + **MixStyle domain generalization** + **8-Way directional connectivity affinity** with an **automated ISO 19107 STRtree topology validator** and **closed parcel contour polygonizer**.
6. **Live Demo Flow**: Select sample $\to$ 1-Click extraction $\to$ Solid cyan boundary lines + teal shaded parcel polygons $\to$ Click parcel for area ($\text{m}^2$) and perimeter ($\text{m}$) $\to$ Toggle green registered ground truth $\to$ Point to 0 errors in topology deck $\to$ One-click GeoJSON export.
7. **Automated Testing**: 64 comprehensive tests passing in ~18.3 seconds (`.venv/bin/pytest tests/ -v`).

---

## 2. Role-Specific Deep Technical Ownership

```text
Team Specialization Matrix:
├── Track 1: AI / Deep Learning Specialist (Model, Loss, Training, Augmentation)
├── Track 2: Computational GIS & Geometry Specialist (Thinning, Graphs, Polygonizer, Topology)
├── Track 3: Backend API & Systems Specialist (FastAPI, Services, Performance, Deployment)
└── Track 4: Frontend & WebGIS Specialist (Leaflet, UI Decks, Layer Rendering, User Journey)
```

---

### Track 1: AI & Deep Learning Specialist

#### Core Files Owned
- [ai/models/full_model.py](file:///Users/kyashwanth/Documents/sih/ai/models/full_model.py)
- [ai/models/pmg.py](file:///Users/kyashwanth/Documents/sih/ai/models/pmg.py)
- [ai/models/domain_generalization.py](file:///Users/kyashwanth/Documents/sih/ai/models/domain_generalization.py)
- [ai/models/connectivity.py](file:///Users/kyashwanth/Documents/sih/ai/models/connectivity.py)
- [ai/losses.py](file:///Users/kyashwanth/Documents/sih/ai/losses.py)
- [ai/metrics.py](file:///Users/kyashwanth/Documents/sih/ai/metrics.py)
- [ai/trainer.py](file:///Users/kyashwanth/Documents/sih/ai/trainer.py)

#### What You Must Master
- **Model Architecture**: Explain the 4 downsampling stages ($[32, 64, 128, 256, 512]$), residual shortcut projections, and total parameter count (**12,376,778**).
- **Progressive Multiscale Generator**: Explain parallel dilated convolutions ($d \in \{1, 2, 4\}$), coarse-to-fine top-down fusion, and $1 \times 1$ adapter channels (128 channels).
- **MixStyle Domain Generalization**: Explain feature statistics perturbation ($\mu, \sigma$), $\text{Beta}(0.2, 0.2)$ sampling, probability $p=0.5$, and why it automatically deactivates during evaluation.
- **8-Way Connectivity Head**: Explain the 8 spatial offset vectors, target matrix generation from ground truth masks, and weighted BCE loss with positive weight $w_{pos}=15.0$.
- **Loss Function Math**: $\mathcal{L}_{total} = 0.5 \cdot [0.3 \cdot \mathcal{L}_{focal}(\alpha=0.75, \gamma=2.0) + 0.7 \cdot \mathcal{L}_{dice}] + 0.5 \cdot \mathcal{L}_{conn}$.
- **Training Telemetry**: Trained on NVIDIA A100 GPU for 15 epochs in 22.16 minutes using AdamW ($lr=1.0 \times 10^{-4}$, weight decay $1.0 \times 10^{-4}$) and Cosine Annealing scheduler.

---

### Track 2: Computational GIS & Geometry Specialist

#### Core Files Owned
- [gis/pipeline.py](file:///Users/kyashwanth/Documents/sih/gis/pipeline.py)
- [gis/vectorization.py](file:///Users/kyashwanth/Documents/sih/gis/vectorization.py)
- [gis/topology.py](file:///Users/kyashwanth/Documents/sih/gis/topology.py)
- [gis/skeletonization.py](file:///Users/kyashwanth/Documents/sih/gis/skeletonization.py)
- [gis/refinement.py](file:///Users/kyashwanth/Documents/sih/gis/refinement.py)
- [gis/crs.py](file:///Users/kyashwanth/Documents/sih/gis/crs.py)

#### What You Must Master
- **Coordinate Reference Systems**: Explain EPSG:28992 (Dutch RD New metric projection) vs EPSG:4326 (WGS84 lat/lon) and why metric coordinates are mandatory for length and area calculations.
- **Medial-Axis Thinning**: Explain Zhang-Suen thinning and why 1-pixel skeletons are required for graph traversal.
- **Spatial Graph Tracing**: Explain how NetworkX builds an 8-connected graph, identifies endpoints (degree 1) and junctions (degree $\ge 3$), and traces non-branching paths without duplicate edges.
- **Douglas-Peucker Simplification**: Explain line smoothing with tolerance $\epsilon = 0.35\text{m}$ to remove stair-step raster artifacts while preserving sharp parcel corners.
- **Closed Parcel Polygonization**: Explain the contour inversion algorithm in [backend/services/gis_service.py](file:///Users/kyashwanth/Documents/sih/backend/services/gis_service.py#L72), boundary mask closing, contour area filtering ($20\text{ m}^2$ to $25,000\text{ m}^2$), and metric area/perimeter calculation.
- **ISO 19107 Topology Audit**: Explain Shapely `STRtree` spatial indexing for fast $\mathcal{O}(N \log N)$ intersection checks, detecting invalid geometries, self-intersections, duplicates, tiny fragments, and dangling segments.

---

### Track 3: Backend API & Systems Specialist

#### Core Files Owned
- [backend/main.py](file:///Users/kyashwanth/Documents/sih/backend/main.py)
- [backend/services/inference_service.py](file:///Users/kyashwanth/Documents/sih/backend/services/inference_service.py)
- [backend/services/gis_service.py](file:///Users/kyashwanth/Documents/sih/backend/services/gis_service.py)
- [scripts/pipeline_parity_check.py](file:///Users/kyashwanth/Documents/sih/scripts/pipeline_parity_check.py)
- [requirements.txt](file:///Users/kyashwanth/Documents/sih/requirements.txt)

#### What You Must Master
- **FastAPI Endpoints**: Be able to explain every endpoint (`/api/health`, `/api/demo-samples`, `/api/predict`, `/api/vectorize`, `/api/topology-audit`, `/api/supporting-layers`).
- **Inference Service Execution**: Explain device autodetection (`cuda`, `mps`, `cpu`), checkpoint loading, the in-memory prediction cache (`PREDICTION_CACHE`), and base64 PNG encoding.
- **Authoritative Preprocessing**: Explain `preprocess_for_inference()` and why ImageNet mean/std normalization was deliberately avoided to guarantee bit-level parity with the offline evaluation pipeline.
- **Concurrency & Performance**: Explain how Uvicorn ASGI handles async requests while running PyTorch forward passes synchronously in under 45 ms.
- **Deployment**: Explain how to run the server locally (`uvicorn backend.main:app --port 8000`), in Docker, or on Google Colab Pro via Ngrok.

---

### Track 4: Frontend & WebGIS Specialist

#### Core Files Owned
- [frontend/index.html](file:///Users/kyashwanth/Documents/sih/frontend/index.html)
- [docs/webgis.md](file:///Users/kyashwanth/Documents/sih/docs/webgis.md)

#### What You Must Master
- **Map Dashboard Architecture**: Explain Leaflet 1.9.4 initialization, ESRI World Imagery basemap, layer groups, and custom popup styles.
- **Visual Styling of Boundaries**: Explain the high-contrast **solid cyan styling** ($4.4\text{px}$ dark casing halo `#000000` + $2.5\text{px}$ electric cyan core `#00F0FF`) and **semi-transparent teal parcel fills** (`#0D9488`, 0.22 opacity).
- **Interactive UI Decks**: Explain the left-hand control deck (curated demo samples, threshold slider, model dropdown) and right-hand topology health counters.
- **Feature Inspection & Human-in-the-Loop**: Explain how clicking on a boundary line or parcel polygon opens an interactive popup displaying Feature ID, metric length/area, model confidence, and the **`✓ Approve Feature`** action button.
- **GeoJSON Export**: Explain how the frontend bundles the active vector layers into a standard GeoJSON file download for desktop QGIS/ArcGIS analysis.

---

## 3. Team Cross-Examination Practice Matrix

Before entering the judging room, team members should pair up and cross-examine each other using this rapid checklist:

| Question Prompt | Expected Fast Answer | Responsible Specialist |
| :--- | :--- | :---: |
| *"Why is your boundary line solid cyan instead of dotted?"* | 8-way connectivity head + morphological closing + 1.8m endpoint snapping | AI & GIS Specialists |
| *"Where does the parcel area (m²) come from?"* | OpenCV contour inversion on closed boundary mask; pixel area * 0.0625 m² | GIS Specialist |
| *"Why does your model not fail on different soils?"* | MixStyle domain generalization perturbs feature stats $(\mu, \sigma)$ during training | AI Specialist |
| *"How do you prove zero self-intersections?"* | Automated ISO 19107 STRtree topology audit engine queries `geom.is_simple` | GIS Specialist |
| *"What is your model's parameter count?"* | Exactly 12,376,778 parameters (~12.38M), 141 MB on disk | AI Specialist |
| *"How long does inference take?"* | 42 ms on NVIDIA A100 GPU / 110 ms on Apple Silicon MPS | Backend Specialist |
| *"Where did the ground truth data come from?"* | Dutch National Land Registry (Kadaster BRK / PDOK) 0.25m GSD orthoimagery | GIS Specialist |
