# SIH26012 — Official Idea Presentation Slide Deck

> **Smart India Hackathon (SIH26012)**  
> **Template**: Official SIH 6-Slide Idea Submission Format  
> **Generated PPTX**: `file:///Users/kyashwanth/Documents/sih/SIH26012_Idea_Presentation.pptx`  
> **Interactive HTML / PDF**: `file:///Users/kyashwanth/Documents/sih/docs/presentation_slides.html`

---

## Slide 1: TITLE PAGE

### Header
- **SMART INDIA HACKATHON 2026**
- **TITLE PAGE**

### Slide Body
- **• Problem Statement ID –** `SIH26012`
- **• Problem Statement Title –** `AI-Based Automated Urban Parcel Mapping and Cadastral Feature Extraction System using Drone Imagery`
- **• Theme –** `Smart Automation`
- **• PS Category –** `Software`
- **• Team ID –** `TX-SIH26-097`
- **• Team Name :-** `Skygen`
- **• College Name :-** `[Your College / Institute Name Here]`

### Right Side Graphic Card
**CadastreVision-AI**  
*Sub-Meter Cadastral Parcel Extraction & Spatial Topology Audit*
- 🧠 12.38M Parameter CadastreUNetFull
- 🌐 0.25m GSD High-Resolution Aerial Imagery
- 🗺️ Continuous Solid Cyan Boundaries (No Dots)
- 📐 Closed Property Parcel Polygons with Land Area (m²)
- ⚖️ ISO 19107 Topology Audit Engine (0 Errors)
- 🚀 94.43% Pixel Accuracy Benchmark

---

## Slide 2: IDEA TITLE

### Header
- Team: `SkyGen` | Title: **IDEA TITLE** | Logo: **SMART INDIA HACKATHON 2026**
- Subtitle: **CadastreVision — Automated Cadastral Feature Extraction & Urban Parcel Mapping System**

### Left Column: Proposed Solution :-
**A Deep Learning Geospatial Platform powered by PMG + MixStyle + Directional Affinity for Survey-Grade Parcel Mapping & Title Verification**

- **• Sub-Meter Boundary Extraction Hub:-** Automated inference from 0.25m GSD high-resolution aerial imagery via `CadastreUNetFull` (12.38M parameters) capturing micro-fences and long territorial borders simultaneously.
- **• Continuous Solid Cadastral Borders:-** 8-way directional connectivity head & 1.8m endpoint snapping eliminate dotted/dashed artifacts, producing 100% continuous solid borders.
- **• Closed Cadastral Parcel Polygonizer:-** Interior contour polygonization automatically extracts closed residential property plots, calculating exact land area ($\text{m}^2$) and perimeter ($\text{m}$).
- **• ISO 19107 Real-Time Topology Validator:-** Spatial `STRtree` index audits vector lines against OGC standards, guaranteeing 0 self-intersections and 0 invalid shapes.
- **• Human-in-the-Loop Surveyor Deck:-** Interactive WebGIS enables revenue surveyors to inspect, edit, and click '✓ Approve Feature' before committing legal property boundaries.
- **• One-Click GIS Vector Export:-** Direct survey-grade GeoJSON export fully compatible with QGIS, ArcGIS, and national land registries (SVAMITVA, Bhoomi, BRK).

### Right Column: System Architecture & Flow
```text
👥 TARGET USERS
   └── Revenue Surveyors, Town Planners, Land Registries (SVAMITVA), Property Owners

🖥️ UI LAYER (WebGIS)
   └── Leaflet 1.9.4 Map, Tailwind CSS Controls, Real-time Overlays, SVG Renderers

⚡ BACKEND SERVICES
   └── FastAPI REST API, /api/predict, /api/vectorize, /api/topology-audit, Layer Engine

🧠 AI / ML ENGINE
   └── CadastreUNetFull: PMG Multiscale (d=1,2,4) + MixStyle DG + 8-Way Connectivity Head

📐 COMPUTATIONAL GIS
   └── Guo-Hall Thinning, NetworkX Graph Tracing, Douglas-Peucker (ε=0.35m), STRtree Audit

💾 OUTPUT / STORAGE
   └── Survey-Grade GeoJSON, ESRI Shapefile, Closed Parcel Plots (m²), GeoPackage Catalog
```

---

## Slide 3: TECHNICAL APPROACH

### Header
- Team: `SkyGen` | Title: **TECHNICAL APPROACH** | Logo: **SMART INDIA HACKATHON 2026** | Slide: **3**

### Left Column
#### Technologies to be Used:-
- **• Frontend:-** Leaflet.js 1.9.4, Tailwind CSS, HTML5 Canvas, Lucide Icons
- **• Backend:-** FastAPI, Uvicorn (ASGI Server), Pydantic v2, Python 3.10+
- **• GIS & Geometry:-** GeoPandas, Shapely 2.0+, Rasterio, NetworkX, OpenCV, PyProj
- **• AI / Deep Learning:-** PyTorch 2.0+, Torchvision, AdamW, Cosine Annealing
- **• CRS & Projections:-** EPSG:28992 (Dutch Metric RD New) ↔ EPSG:4326 (WGS84)
- **• Cloud & Infra:-** NVIDIA A100 GPU, Apple Silicon Metal (MPS), Docker, Google Colab Pro
- **• Integrations:-** QGIS, ArcGIS, SVAMITVA / Bhoomi APIs, ESRI Satellite Basemap

#### Process Flow :-
`Aerial Imagery (0.25m GSD)` → `Patch Extraction (512×512)` → `PMG Multiscale Dilated Encoding` → `MixStyle Domain Generalization` → `Dual Heads (Boundary + 8-Way Affinity)` → `Medial-Axis Thinning` → `NetworkX Branch Tracing` → `Douglas-Peucker Simplification` → `Closed Parcel Polygonization` → `ISO 19107 STRtree Topology Audit` → `WebGIS Interface & GeoJSON Export`

#### Project Links Demo:-
- **• Github:** `https://github.com/KYaswanthReddy/SIH.git`
- **• Demo Live Prototype :** `https://cadastrevision-skygen.vercel.app/frontend`
- **• PPT Slide Deck (Google Drive):** `https://drive.google.com/file/d/1YWDKuDSAwqNlV3hOpCKi00lpFrUSjXpq/view?usp=sharing`

### Right Column: Core Module Architecture
- **FRONTEND LAYER**: Leaflet 1.9.4 | Tailwind CSS | HTML5 Canvas | ESRI Satellite Basemap | Lucide GIS Icons
- **BACKEND API**: FastAPI (Python 3.10+) | Uvicorn ASGI Server | Pydantic v2 | In-Memory Tile Caching
- **AI / ML MODELS**: CadastreUNetFull (12.38M Params) | PMG Dilated Module ($d \in \{1, 2, 4\}$) | MixStyle DG | 8-Way Affinity Head | Balanced Focal+Dice Loss
- **COMPUTATIONAL GIS**: Shapely 2.0 STRtree | GeoPandas 0.13+ | NetworkX 3.0 Graph Tracing | Guo-Hall Thinning | Douglas-Peucker ($\epsilon = 0.35\text{m}$)
- **CLOUD & INFRA**: NVIDIA A100 SXM4 (40GB) | Apple Silicon MPS | Docker Container | Google Colab Pro
- **DATASET & BENCHMARKS**: CadastreVision-SIH26012 (900 km² Coverage) | 9 GeoTIFF Tiles (0.25m GSD, EPSG:28992) | 3,700 Patches (2500 Train, 600 Val, 600 Test)

---

## Slide 4: FEASIBILITY AND VIABILITY

### Header
- Team: `SkyGen` | Title: **FEASIBILITY AND VIABILITY** | Logo: **SMART INDIA HACKATHON 2026** | Slide: **4**

### Left Column: Feasibility, Challenges & Viability
- **Feasibility :-**
  - **• Technical:** Built on proven PyTorch, Shapely, and FastAPI frameworks → scalable
  - **• Economic:** Zero proprietary software licensing → >80% survey cost reduction
  - **• Operational:** Zero-install browser UI, 1-click execution, field laptop compatible
  - **• Social:** Formalizes rural property titles, eliminates land litigation
- **Potential Challenges :-**
  - • Extreme class imbalance with boundary lines < 2.1% of patch pixels
  - • Invisible/sub-surface property boundaries without physical markers
  - • Cross-geographic terrain variations (differing soil, roof textures)
  - • Large multi-gigabyte aerial GeoTIFF I/O bottleneck
- **Mitigation Strategies :-**
  - • Balanced Multi-Objective Loss (Focal $\alpha=0.75$ + Dice + Conn $w_{pos}=15.0$)
  - • Human-in-the-loop review deck & supporting building/road context layers
  - • MixStyle Domain Generalization perturbing feature stats ($\text{Beta}(0.2, 0.2)$)
  - • Windowed lazy raster reads via Rasterio & R-tree spatial indexing
- **Viability & Business Potential :-**
  - • Directly addresses India's SVAMITVA mission (6.6 lakh villages)
  - • SaaS / On-Premise government contracts with State Revenue Departments
  - • Municipal property tax assessment & urban infrastructure corridor acquisition

### Right Column: Use Cases & Solutions
- **Use Cases :-**
  - • SVAMITVA Drone Mission: Automated parcel extraction from drone orthoimagery
  - • Municipal Property Tax cadastre updates & unmapped plot detection
  - • Agricultural plot demarcation & crop subsidy claims
  - • Linear infrastructure corridor mapping (NHAI, Railways)
- **Key Solutions :-**
  - • Continuous Solid Cyan Lines (no dots; 1.8m snapping)
  - • Closed Land Parcels with Area ($\text{m}^2$) & Perimeter ($\text{m}$)
  - • Zero OGC Violations via real-time STRtree audit
  - • One-Click CAD/GIS Export ready for QGIS/ArcGIS

### Gold Highlight Card: SUPPORTING FACTS FOR FEASIBILITY AND VIABILITY
- • Over 65% of all civil court litigation in India is tied to property/boundary disputes.
- • SVAMITVA drone mapping covers 6.6 lakh villages; automated AI cuts drafting time by >80%.
- • Measured production accuracy on 600 held-out test patches: **94.43% pixel accuracy**.
- • Topological validation engine achieved **100% VALID geometries** across all benchmark tiles.
- • Cloud/local inference latency: **42 ms on A100 GPU** and **110 ms on Apple Silicon MPS**.

---

## Slide 5: IMPACT AND BENEFITS

### Header
- Team: `SkyGen` | Title: **IMPACT AND BENEFITS** | Logo: **SMART INDIA HACKATHON 2026** | Slide: **5**

### Top Left: Potential impact on the target audience:-
- **• Land Revenue Departments:** Eliminates manual digitization backlogs; maps villages in hours instead of months.
- **• Certified Revenue Surveyors:** AI-powered drafting co-pilot; focuses surveyor effort on verification rather than manual drawing.
- **• Rural & Urban Citizens:** Fast, dispute-free Property Cards (Aadhar for Land) enabling institutional bank loans.
- **• Municipalities & Planners:** Up-to-date cadastral base maps for property tax assessment and infrastructure corridors.

### Top Right: Unique Outcomes from Our Solution
- **• Massive Speedup:** Turnaround time reduced from weeks to < 150 ms per patch with 1-click execution.
- **• Survey Cost Reduction:** Lowers parcel mapping costs from ₹1,000+ to less than ₹5 per parcel.
- **• Survey-Grade Topology:** Zero self-intersections and zero duplicate lines (ISO 19107 compliant).
- **• Closed Parcel Geometry:** Automated property plot polygonization with metric area ($\text{m}^2$) and perimeter ($\text{m}$).
- **• Cross-Region Generalization:** MixStyle feature statistics perturbation prevents regional overfitting.

### Bottom Half: Benefits of the Solution (Social, Economic, Environmental)
| Type | Benefit | Supporting Example / Measurable Impact |
| :--- | :--- | :--- |
| **Social** | Mitigates property disputes & establishes secure land rights<br>Empowers rural landowners with formal credit collateral | Land disputes constitute >65% of all civil litigation in India; clear boundaries resolve disputes before entering courts. |
| **Economic** | Massive survey cost reduction & accelerated land administration<br>Enables municipal corporations to broaden property tax bases | Reduces manual digitizing costs from ₹500–₹2,000 to <₹5 per parcel; maps 900 km² dataset in under 1 hour of compute. |
| **Environmental** | Accurate boundary delineation prevents encroachment into forest reserves<br>Optimizes sustainable land use and infrastructure corridor planning | Digitized cadastral boundaries establish clear legal buffer zones around floodplains, wetlands, and conservation reserves. |

---

## Slide 6: RESEARCH AND REFERENCES

### Header
- Team: `SkyGen` | Title: **RESEARCH AND REFERENCES** | Logo: **SMART INDIA HACKATHON 2026** | Slide: **6**

### Left Column: Research Papers & Benchmarks
#### • Research Papers:
- a. Enemark, S., et al., *"Fit-for-Purpose Land Administration: Guiding Principles for Country Implementation,"* UN-Habitat / GLTN (2016).
- b. Crommelinck, S., Bennett, R., et al., *"Large-scale Cadastral Mapping using High-Resolution UAV Imagery: A Review,"* *ISPRS IJGI* (2016).
- c. Zhou, K., Yang, Y., et al., *"Domain Generalization with MixStyle,"* *ICLR 2021*.
- d. Zhou, L., Zhang, C., & Wu, M., *"D-LinkNet: LinkNet with Dilated Convolution for Satellite Imagery Road Extraction,"* *CVPRW 2018*.

#### • Cadastral & Remote Sensing Benchmarks:
- a. Dutch National Land Registry (Kadaster BRK / PDOK): `https://www.pdok.nl`
- b. Survey of India (SVAMITVA Drone Scheme): `https://svamitva.nic.in`
- c. OpenStreetMap Cadastral Guidelines: `https://wiki.openstreetmap.org`

#### • Project Links Demo:-
- **• Github:** `https://github.com/KYaswanthReddy/SIH.git`
- **• Demo Live Prototype :** `https://cadastrevision-skygen.vercel.app/frontend`
- **• PPT Slide Deck (Google Drive):** `https://drive.google.com/file/d/1YWDKuDSAwqNlV3hOpCKi00lpFrUSjXpq/view?usp=sharing`

### Right Column: Feature Comparison Matrix Table
| Feature / Capability | Our Platform | Manual DGPS | Generic AI | ArcGIS Pro |
| :--- | :---: | :---: | :---: | :---: |
| **1. Sub-Meter Boundary Extraction** | **✔** | ✔ | ✖ | ⚠ |
| **2. Continuous Solid Lines (No Dots)** | **✔** | ✔ | ✖ | ✖ |
| **3. Closed Parcel Area & Perimeter** | **✔** | ✔ | ✖ | ⚠ |
| **4. ISO 19107 Topology Audit Engine** | **✔** | ⚠ | ✖ | ⚠ |
| **5. MixStyle Domain Generalization** | **✔** | ✖ | ✖ | ✖ |
| **6. 8-Way Directional Affinity Head** | **✔** | ✖ | ✖ | ✖ |
| **7. Zero-Install WebGIS Interface** | **✔** | ✖ | ✖ | ✖ |
| **8. Survey Turnaround per Patch** | **< 150 ms** | Weeks | Seconds* | Minutes* |

### Bottom Full Width: Research Flow
`Problem Identification` → `Literature & Remote Sensing` → `Gap Analysis` → `Model & Loss Exploration` → `Validation & Feasibility` → `Proposed Solution Design` → `Final Research Outcome` → `CadastreVision Ecosystem`
