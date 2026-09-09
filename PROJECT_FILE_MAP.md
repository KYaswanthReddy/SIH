# SIH26012 — Master Repository File & Module Map

> **Smart India Hackathon (SIH26012)**  
> **Project Title**: Automated Cadastral Feature Extraction & Urban Parcel Mapping System  
> **Verification Standard**: Rigorously audited against actual codebase syntax, imports, and AST definitions.

---

## 1. Directory Tree Overview

```text
/Users/kyashwanth/Documents/sih/
├── ai/                                     # PyTorch Deep Learning Core
│   ├── models/                             # Neural Network Architectures
│   │   ├── __init__.py                     # Package export declarations
│   │   ├── unet_baseline.py                # Residual U-Net Backbone (12.18M params)
│   │   ├── pmg.py                          # Progressive Multiscale Generator Module
│   │   ├── domain_generalization.py        # MixStyle Feature Perturbation Module
│   │   ├── connectivity.py                 # 8-Directional Affinity Head & Loss
│   │   └── full_model.py                   # Integrated CadastreUNetFull (12.38M params)
│   ├── __init__.py                         # AI module root initialization
│   ├── dataset.py                          # CadastreDataset & PyTorch DataLoader Factory
│   ├── evaluate.py                         # Offline validation & test metric evaluation
│   ├── losses.py                           # FocalLoss, DiceLoss, CombinedFocalDiceLoss
│   ├── metrics.py                          # Precision, Recall, F1, IoU, Pixel Accuracy tracker
│   ├── trainer.py                          # Production PyTorch training loop with early stopping
│   ├── transforms.py                       # Paired geometric & photometric augmentations
│   └── visualizer.py                       # Multi-panel prediction visualizer & loss plotting
├── backend/                                # FastAPI Production WebGIS REST Engine
│   ├── __init__.py
│   ├── main.py                             # FastAPI application, CORS, middleware, routes
│   └── services/                           # Business logic services
│       ├── __init__.py
│       ├── inference_service.py            # Checkpoint loader & PyTorch MPS/CUDA inference
│       └── gis_service.py                  # Raster-to-vector, parcel extraction & topology audit
├── configs/                                # Declarative YAML Configurations
│   ├── baseline_train_config.yaml          # Phase 2 baseline U-Net configuration
│   ├── dataset_config.yaml                 # Phase 1 dataset & patch extraction configuration
│   ├── full.yaml                           # Phase 5 full architecture config (standard)
│   ├── full_512_fixed.yaml                 # Production A100 512x512 training configuration
│   ├── pmg.yaml                            # Phase 3 PMG module training configuration
│   ├── quick_full.yaml                     # Fast multi-epoch training config
│   ├── quick_full_15epoch.yaml             # 15-epoch quick configuration
│   ├── quick_full_fixed.yaml               # Validated 256x256 training configuration
│   ├── sanity_full.yaml                    # Ultra-fast 1-epoch sanity configuration
│   └── smoke_512_fixed.yaml                # GPU smoke test configuration
├── data/                                   # Geospatial Raw & Processed Assets
│   ├── images/                             # 9 Raw 10km x 10km GeoTIFF aerial tiles (0.25m GSD)
│   │   ├── tile-100000-470000.tif          # Train tile (470 MB)
│   │   ├── tile-100000-500000.tif          # Train tile (564 MB)
│   │   ├── tile-160000-490000.tif          # Validation tile (437 MB)
│   │   ├── tile-170000-460000.tif          # Test tile (501 MB)
│   │   ├── tile-180000-360000.tif          # Train tile (453 MB)
│   │   ├── tile-190000-310000.tif          # Test tile (415 MB)
│   │   ├── tile-190000-350000.tif          # Validation tile (466 MB)
│   │   ├── tile-220000-450000.tif          # Train tile (470 MB)
│   │   └── tile-240000-460000.tif          # Train tile (513 MB)
│   ├── references/                         # Official Survey Reference GeoPackages
│   │   ├── brk_reference.gpkg              # Kadaster BRK registered cadastral boundaries
│   │   ├── brt_bebouwde_kom.gpkg           # Official Dutch urban delineation polygon
│   │   └── sample_tiles_10k_split.gpkg     # Official geographic split partition definition
│   └── processed/                          # Curated & preprocessed datasets
│       ├── metadata/                       # CSV spatial catalogs & dataset statistics
│       │   ├── dataset_summary.json        # Global dataset extraction statistics
│       │   ├── patches_all.csv             # Full catalog of 3,700 georeferenced patches
│       │   ├── patches_train.csv           # 2,500 training patch metadata records
│       │   ├── patches_val.csv             # 600 validation patch metadata records
│       │   └── patches_test.csv            # 600 held-out test patch metadata records
│       └── patches/                        # 512x512 RGB images and 1-channel binary masks
├── docs/                                   # Technical Architecture & Pipeline Guides
│   ├── architecture.md                     # High-level architecture specification
│   ├── evaluation.md                       # Metric definitions & validation results
│   ├── gis_pipeline.md                     # Raster-to-vector & topology audit engine
│   ├── training.md                         # Workstation & Colab Pro training instructions
│   └── webgis.md                           # FastAPI & Leaflet application guide
├── experiments/                            # Checkpoints, Visualizations & Metrics
│   ├── full_512_fixed/                     # Production A100 512x512 Model Artifacts
│   │   └── results/
│   │       ├── checkpoints/                # best_model.pth (141 MB), latest_checkpoint.pth
│   │       ├── metrics/                    # metrics_test.json, metrics_val.json, training_history.json
│   │       ├── summary/                    # experiment_summary.json
│   │       └── visualizations/             # metrics_progress.png, prediction overlays
│   ├── ablation/                           # Ablation experiment evaluations & graphs
│   ├── quick_full_fixed/                   # Validated 256x256 checkpoints & metrics
│   ├── sanity/                             # Phase 2 2-epoch sanity verification runs
│   └── test_sanity/                        # Test runner sanity checkpoints
├── frontend/                               # WebGIS Client Interface
│   └── index.html                          # Full-screen Leaflet + Tailwind CSS Dashboard
├── gis/                                    # Computational GIS & Topology Module
│   ├── __init__.py                         # GIS module initialization
│   ├── crs.py                              # Coordinate reference system & affine transforms
│   ├── export.py                           # GeoPandas GeoJSON & ESRI Shapefile exporter
│   ├── patch_extractor.py                  # Grid sliding-window patch extraction engine
│   ├── pipeline.py                         # Unified RasterToVectorGISPipeline class
│   ├── rasterizer.py                       # Vector-to-raster line burn-in engine (GDAL/rasterio)
│   ├── refinement.py                       # Morphological closing & component filtering
│   ├── skeletonization.py                  # Zhang-Suen & morphological medial axis thinning
│   ├── spatial_index.py                    # R-tree bounding box query acceleration
│   ├── topology.py                         # ISO 19107 geometric integrity validator (STRtree)
│   └── vectorization.py                    # NetworkX graph builder, branch tracer & Douglas-Peucker
├── notebooks/                              # Jupyter Notebooks for Google Colab Pro
│   ├── train_baseline_colab.ipynb          # Phase 2 baseline training notebook
│   ├── train_pmg_colab.ipynb               # Phase 3 PMG training notebook
│   ├── train_sanity_full_colab.ipynb       # Phase 4 1-minute sanity verification notebook
│   └── train_full_colab.ipynb              # Phase 5 master production training notebook
├── scripts/                                # Standalone CLI Automation Scripts
│   ├── build_colab_notebook.py             # Automates generation of Colab notebooks
│   ├── evaluate_baseline.py                # Standalone baseline evaluation script
│   ├── generate_visualizations.py          # Generates 4-panel prediction comparison plots
│   ├── pipeline_parity_check.py            # Verifies exact numerical parity between WebGIS and eval
│   ├── prepare_data.py                     # Master data extraction CLI runner
│   ├── run_ablation_experiments.py         # Multi-model ablation benchmark runner
│   ├── run_phase1_verification.py          # Phase 1 data pipeline verification runner
│   ├── train_baseline.py                   # Baseline U-Net training runner
│   ├── train_full.py                       # Full CadastreUNet production training runner
│   ├── train_sanity.py                     # Rapid 2-epoch sanity training runner
│   └── vectorize_resolution_trace.py       # Traces coordinate resolution across vectorizer
├── tests/                                  # Automated Test Suite (64 Passing Tests)
│   ├── __init__.py
│   ├── test_backend_api.py                 # FastAPI endpoint unit & integration tests
│   ├── test_connectivity.py                # 8-direction target generation & head tests
│   ├── test_data_integrity.py              # File existence, patch dimensions, and mask ranges
│   ├── test_dataset.py                     # PyTorch Dataset and DataLoader tensor tests
│   ├── test_domain_generalization.py       # MixStyle feature perturbation tests
│   ├── test_full_model.py                  # CadastreUNetFull forward pass & parameter count
│   ├── test_losses.py                      # FocalLoss, DiceLoss, and CombinedLoss math tests
│   ├── test_metrics.py                     # Precision, Recall, F1, IoU computation tests
│   ├── test_patch_alignment.py             # Affine coordinate alignment against spatial bounds
│   ├── test_pipeline_parity.py             # Preprocessing & inference parity tests
│   ├── test_pmg.py                         # Multi-scale dilation and top-down fusion tests
│   ├── test_rasterization_quality.py       # Line thickness and boundary density tests
│   ├── test_splits_and_leakage.py          # Zero geographic leakage between splits
│   ├── test_topology.py                    # Self-intersection, duplicate, and fragment audit tests
│   ├── test_trainer_sanity.py              # Single-batch loss descent verification
│   └── test_vectorization.py               # Graph tracing, line simplification, and GeoDataFrame tests
├── SIH26012_COLAB/                         # Self-contained Google Colab Pro Deployment Bundle
│   ├── README_COLAB.md                     # Complete Colab setup & execution walkthrough
│   └── ...                                 # Cloned runtime environment for cloud execution
├── requirements.txt                        # Pinned Python package dependencies
└── README.md                               # Project master documentation
```

---

## 2. Complete File-by-File Technical Specification

| File / Module | Purpose | Important Functions / Classes | Inputs | Outputs | Dependencies | Implementation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **`ai/models/full_model.py`** | Integrated neural network combining U-Net, PMG, MixStyle DG, and 8-Way Connectivity Head | `CadastreUNetFull`, `FullCadastreLoss` | `(B, 3, H, W)` RGB image tensor | `refined_prob` `(B, 1, H, W)`, `conn_prob` `(B, 8, H, W)` | `torch`, `torch.nn`, `.unet_baseline`, `.pmg`, `.domain_generalization`, `.connectivity` | `IMPLEMENTED` |
| **`ai/models/unet_baseline.py`** | Lightweight residual encoder-decoder baseline network | `CadastreUNetBaseline`, `DoubleConv`, `Down`, `Up`, `build_baseline_model` | `(B, 3, H, W)` RGB image tensor | `prob_map` `(B, 1, H, W)` | `torch`, `torch.nn`, `torch.nn.functional` | `IMPLEMENTED` |
| **`ai/models/pmg.py`** | Progressive Multiscale Generator with multi-kernel dilated convolutions | `PMGModule`, `MultiscaleFeatureBranch`, `CadastreUNetPMG` | Multi-level encoder feature list `[x2, x3, x4, x5]` | Re-weighted multiscale enhanced feature representations | `torch`, `torch.nn`, `torch.nn.functional` | `IMPLEMENTED` |
| **`ai/models/domain_generalization.py`** | MixStyle feature statistics perturbation module for cross-geographic resilience | `MixStyleDomainPerturbation`, `DomainAwareFeatureAlignment`, `CadastreUNetDG` | Feature tensor `x` `(B, C, H, W)` | Perturbed feature tensor with interpolated $(\mu, \sigma)$ | `torch`, `torch.nn`, `torch.distributions.Beta` | `IMPLEMENTED` |
| **`ai/models/connectivity.py`** | Directional affinity head predicting 8-neighborhood topological connectivity | `ConnectivityHead`, `ConnectivityLoss`, `generate_connectivity_targets` | Boundary feature representation / binary mask | `(B, 8, H, W)` directional affinity probability tensor | `torch`, `torch.nn`, `torch.nn.functional` | `IMPLEMENTED` |
| **`ai/dataset.py`** | PyTorch Dataset and DataLoader factories for windowed cadastral patches | `CadastreDataset`, `create_dataloaders` | Metadata CSV path, patch image and mask files | Dict with `'image'`, `'mask'`, `'patch_id'`, `'is_urban'` | `torch.utils.data`, `pandas`, `PIL`, `numpy` | `IMPLEMENTED` |
| **`ai/evaluate.py`** | Standalone model evaluation runner on validation and test splits | `evaluate_model`, `evaluate_checkpoint` | Model checkpoint `.pth`, test DataLoader | JSON evaluation dictionary with precision, recall, F1, IoU | `torch`, `numpy`, `.metrics`, `tqdm` | `IMPLEMENTED` |
| **`ai/losses.py`** | Class-imbalance robust loss functions | `FocalLoss`, `DiceLoss`, `CombinedFocalDiceLoss`, `build_loss` | Prediction tensor `(B, 1, H, W)`, ground truth `(B, 1, H, W)` | Scalar loss tensor | `torch`, `torch.nn`, `torch.nn.functional` | `IMPLEMENTED` |
| **`ai/metrics.py`** | Strict segmentation metrics accumulator and threshold sweep engine | `compute_binary_metrics`, `SegmentationMetricsTracker`, `compute_all_metrics` | Predicted probabilities and ground-truth tensors | Precision, Recall, F1, IoU, Pixel Accuracy, Confusion Matrix (TP, FP, FN, TN) | `torch`, `numpy` | `IMPLEMENTED` |
| **`ai/trainer.py`** | Full-featured training loop with validation threshold sweep and checkpointing | `CadastreTrainer`, `train_model` | Model, DataLoaders, Loss, Optimizer, YAML configuration | Best checkpoint file, training curves, history CSV | `torch`, `torch.optim`, `tqdm`, `yaml` | `IMPLEMENTED` |
| **`ai/transforms.py`** | Paired spatial and photometric image/mask augmentations | `CadastreAugmentation`, `get_train_transforms`, `get_val_transforms` | Image `(H, W, 3)` uint8, Mask `(H, W)` uint8 | Augmented PyTorch tensors in `[0.0, 1.0]` | `numpy`, `torch`, `cv2`, `random` | `IMPLEMENTED` |
| **`ai/visualizer.py`** | 4-panel and 5-panel prediction overlay generator | `visualize_predictions`, `plot_training_curves` | RGB image, ground truth mask, predicted probability | Matplotlib visual comparison figure saved to disk | `matplotlib.pyplot`, `numpy`, `torch` | `IMPLEMENTED` |
| **`backend/main.py`** | FastAPI REST application entry point with CORS and static file mount | `app`, `health_check`, `predict_cadastral_boundaries`, `vectorize_prediction`, `audit_topology` | HTTP REST requests from WebGIS client | JSON responses, base64 heatmaps, GeoJSON FeatureCollections | `fastapi`, `pydantic`, `uvicorn`, `PIL`, `numpy` | `IMPLEMENTED` |
| **`backend/services/inference_service.py`** | PyTorch model checkpoint manager and GPU/MPS inference runner | `InferenceService`, `preprocess_for_inference` | `(H, W, 3)` RGB numpy array, model name string | Dict with `prob_map` `(orig_H, orig_W)`, `mean_confidence` | `torch`, `cv2`, `numpy`, `ai.models.*` | `IMPLEMENTED` |
| **`backend/services/gis_service.py`** | Vectorization bridge, contour polygonizer, and supporting layer server | `GISService` | Probability map, Affine transform, threshold parameters | GeoJSON boundary LineStrings and closed parcel Polygons | `geopandas`, `shapely`, `cv2`, `gis.pipeline` | `IMPLEMENTED` |
| **`frontend/index.html`** | Interactive WebGIS client with Leaflet map and dynamic control decks | Single-page WebGIS application | User map interaction, slider tweaks, sample selection | Visual map overlays, GeoJSON rendering, GeoJSON file download | `Leaflet.js`, `Tailwind CSS CDN`, `Lucide icons` | `IMPLEMENTED` |
| **`gis/crs.py`** | CRS validation, metadata extraction, and coordinate transformation utilities | `validate_crs`, `get_raster_metadata`, `pixel_to_geo`, `geo_to_pixel` | Pixel indices `(col, row)` or world coordinates `(x, y)` | Transformed spatial coordinates in target CRS | `rasterio`, `rasterio.crs`, `rasterio.transform`, `pyproj` | `IMPLEMENTED` |
| **`gis/export.py`** | Export utilities for GeoPandas GeoDataFrames | `export_geodataframe` | GeoDataFrame, output file path | Serialized GeoJSON (`.geojson`) or ESRI Shapefile (`.shp`) | `geopandas`, `os` | `IMPLEMENTED` |
| **`gis/patch_extractor.py`** | Windowed sliding grid patch extraction engine with spatial index filtering | `PatchExtractor` | Large 10km x 10km GeoTIFF, Cadastral GeoPackage | Processed $512 \times 512$ PNG image/mask tiles and CSV metadata | `rasterio`, `rasterio.windows`, `numpy`, `pandas`, `PIL` | `IMPLEMENTED` |
| **`gis/pipeline.py`** | Unified end-to-end raster probability to georeferenced vector pipeline | `RasterToVectorGISPipeline` | Raw probability map `(512, 512)`, Affine transform | Dict with GeoDataFrame, GeoJSON, and topology audit report | `.refinement`, `.skeletonization`, `.vectorization`, `.topology` | `IMPLEMENTED` |
| **`gis/rasterizer.py`** | Vector-to-raster line burn-in engine with sub-pixel line rasterization | `rasterize_cadastral_boundaries`, `compute_mask_statistics` | Cadastral LineString geometries, window transform | Binary raster mask `(H, W)` with 2px line thickness | `rasterio.features.rasterize`, `numpy` | `IMPLEMENTED` |
| **`gis/refinement.py`** | Morphological closing and small connected component filtering | `refine_boundary_mask` | Raw probability map, binarization threshold | Cleaned binary mask with bridged micro-gaps | `cv2`, `numpy` | `IMPLEMENTED` |
| **`gis/skeletonization.py`** | Medial axis thinning and graph node (endpoint/junction) detection | `zhang_suen_thinning`, `fast_skeletonize`, `detect_endpoints_and_junctions` | Binary boundary mask | 1-pixel wide continuous skeleton mask and node lists | `cv2`, `numpy` | `IMPLEMENTED` |
| **`gis/spatial_index.py`** | R-tree spatial index wrapper for rapid bounding-box intersection queries | `TileSpatialIndex`, `UrbanAreaIndex` | GeoPackage cadastral layer, patch bounding box | Filtered subset of intersecting Shapely LineString geometries | `geopandas`, `shapely.geometry`, `pyogrio` | `IMPLEMENTED` |
| **`gis/topology.py`** | Automated ISO 19107 geometric integrity validator and error reporter | `validate_cadastral_topology` | GeoDataFrame of cadastral LineStrings | Comprehensive topology audit dictionary with warning list | `geopandas`, `shapely.strtree.STRtree`, `shapely.geometry` | `IMPLEMENTED` |
| **`gis/vectorization.py`** | Medial axis skeleton to NetworkX spatial graph and Douglas-Peucker simplification | `skeleton_to_graph`, `trace_graph_branches`, `merge_contiguous_lines`, `vectorize_skeleton` | 1-pixel skeleton array, Affine transform | GeoDataFrame containing smoothed, simplified LineStrings | `networkx`, `geopandas`, `shapely.geometry`, `shapely.ops` | `IMPLEMENTED` |
| **`scripts/prepare_data.py`** | Master CLI data preparation runner orchestrating patch extraction | `main` | Raw 9 GeoTIFF tiles, BRK reference GeoPackage | 3,700 patches and metadata CSV catalogs | `gis.patch_extractor`, `yaml`, `argparse` | `IMPLEMENTED` |
| **`scripts/pipeline_parity_check.py`** | Bit-level parity verification between offline evaluation and WebGIS inference | `run_parity_check` | Production checkpoint, test patch image | Detailed numerical parity report comparing logits, masks, and lines | `torch`, `cv2`, `numpy`, `backend.services.*` | `IMPLEMENTED` |
| **`scripts/train_full.py`** | Production CLI runner for training full CadastreUNet on GPU | `main` | YAML configuration path (e.g. `full_512_fixed.yaml`) | Trained checkpoint file and evaluation JSON reports | `ai.trainer`, `ai.models.full_model`, `yaml` | `IMPLEMENTED` |
| **`scripts/run_ablation_experiments.py`** | Automated execution of multi-architecture ablation study | `main`, `run_experiment` | Baseline, PMG, DG, and Full model definitions | `ablation_results.json`, `ablation_summary.md`, comparison plots | `ai.models.*`, `ai.evaluate`, `pandas`, `tabulate` | `IMPLEMENTED` |
| **`requirements.txt`** | Complete pinned runtime and development Python dependencies | N/A | `pip install -r requirements.txt` | Installed environment packages | `torch`, `geopandas`, `fastapi`, etc. | `IMPLEMENTED` |

---

## 3. Detailed Component Deep-Dives

### 3.1 `ai/models/full_model.py`
- **Why it exists**: Defines the unified, flagship neural network (`CadastreUNetFull`) that addresses the primary challenge of cadastral boundary segmentation: detecting razor-thin parcel borders across heterogeneous urban and rural terrain without line fragmentation.
- **What it does**:
  - Encodes the RGB image through 4 hierarchical stages with residual skip connections.
  - Injects multiscale feature context using the **Progressive Multiscale Generator (PMG)**.
  - Applies **MixStyle domain generalization** across intermediate feature maps to prevent regional overfitting.
  - Feeds decoder features into **Dual Prediction Heads**: a pixel-wise boundary probability head and an 8-way directional connectivity head.
  - Fuses boundary and connectivity predictions through a convolutional refinement layer (`fusion_refine`).
- **Critical Hyperparameters**:
  - Encoder channels: `[32, 64, 128, 256, 512]`
  - PMG projection channels: `128`
  - Connectivity directions: `8` ($\{-1, 0, 1\}^2 \setminus \{(0,0)\}$)
  - Dropout: `0.1`
  - Total trainable parameters: **12,376,778**
- **Demo Criticality**: **MAXIMUM** — This is the core neural network loaded into the live WebGIS dashboard.

### 3.2 `backend/services/gis_service.py`
- **Why it exists**: Bridges the raw floating-point probability map produced by PyTorch with the discrete, survey-grade vector world of GIS systems (QGIS, ArcGIS, SVAMITVA, BRK).
- **What it does**:
  - Thresholds and morphologically closes boundary gaps.
  - Extracts 1-pixel medial axis skeletons.
  - Traces linear graphs in NetworkX and applies Douglas-Peucker simplification.
  - **Polygonizes closed property parcels**: Inverts the closed boundary mask, runs contour extraction, filters land areas between $20\text{ m}^2$ and $25,000\text{ m}^2$, calculates metric land area ($\text{m}^2$) and perimeter ($\text{m}$), and formats closed parcel polygons.
  - Reprojects all output coordinates from local metric projection (`EPSG:28992`) to standard WebGIS coordinates (`EPSG:4326`).
- **Demo Criticality**: **MAXIMUM** — Powers the vector boundary layer, parcel polygon shading, and survey attribute popups in the WebGIS interface.

### 3.3 `gis/topology.py`
- **Why it exists**: Land registry boundaries require strict geometric validity. Floating dangling segments or self-intersecting lines invalidate legal deeds. This engine acts as an automated quality auditor.
- **What it does**:
  - Implements an **STRtree** (Spatial Two-dimensional R-Tree) spatial index to perform rapid pairwise intersection checks across all generated LineStrings.
  - Audits for:
    1. OGC invalid geometries (`geom.is_valid`)
    2. Self-intersections (`geom.is_simple`)
    3. Duplicate overlapping lines (normalized coordinate tuple matching)
    4. Tiny noise fragments ($< 2.0\text{m}$)
    5. Dangling disconnected segments (endpoints not connected within $0.5\text{m}$)
    6. Interior crossings (non-nodal intersections)
- **Demo Criticality**: **HIGH** — Displayed live in the WebGIS right-hand dashboard to prove mathematical validity to hackathon judges.

---

## 4. Test Suite Map (`tests/`)

All 17 test files and 64 automated test cases pass with zero errors:

1. **`tests/test_model.py`**: Validates parameter counts and forward output tensor shapes for `CadastreUNetBaseline`.
2. **`tests/test_full_model.py`**: Verifies forward pass, dictionary outputs, parameter count (12.38M), and gradient flow for `CadastreUNetFull`.
3. **`tests/test_pmg.py`**: Tests multiscale dilated feature branches and coarse-to-fine top-down fusion.
4. **`tests/test_domain_generalization.py`**: Asserts that MixStyle perturbs feature statistics during training while acting as an identity during evaluation.
5. **`tests/test_connectivity.py`**: Validates directional neighbor offset shifts and masked affinity loss calculation.
6. **`tests/test_losses.py`**: Verifies numerical stability and behavior of Focal Loss, Dice Loss, and Combined Loss.
7. **`tests/test_metrics.py`**: Verifies confusion matrix tracking (TP, FP, FN, TN) and threshold sweeping.
8. **`tests/test_dataset.py`**: Checks PyTorch Dataset indexing, PIL-to-tensor conversion, and DataLoader batch shapes.
9. **`tests/test_patch_alignment.py`**: Confirms that pixel Affine transformation matrices strictly match bounding coordinates.
10. **`tests/test_splits_and_leakage.py`**: Mathematically verifies zero geographic overlap between train, validation, and test splits.
11. **`tests/test_rasterization_quality.py`**: Checks rasterized line burn-in thickness and boundary pixel density.
12. **`tests/test_data_integrity.py`**: Checks image file existence, RGB channels, and binary mask bounds.
13. **`tests/test_trainer_sanity.py`**: Verifies that 1 training step successfully decreases loss on a sample batch.
14. **`tests/test_vectorization.py`**: Tests NetworkX graph conversion, branch tracing, Douglas-Peucker simplification, and line merging.
15. **`tests/test_topology.py`**: Asserts correct detection of self-intersections, duplicates, and valid geometries.
16. **`tests/test_backend_api.py`**: Tests all FastAPI endpoints using `starlette.testclient.TestClient`.
17. **`tests/test_pipeline_parity.py`**: Asserts strict numerical parity between offline evaluation and live WebGIS inference.
