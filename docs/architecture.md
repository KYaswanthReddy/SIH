# SIH26012 System Architecture

## Overview
The **AI-Based Automated Urban Parcel Mapping and Cadastral Feature Extraction System** combines deep learning and computational GIS topology engines to extract high-fidelity cadastral property boundaries from 0.25m GSD aerial imagery.

```
Aerial / Drone Image (0.25m GSD, EPSG:28992)
        ↓
Data Preprocessing & Patching (512×512)
        ↓
Progressive Multiscale Generator (PMG)
        ↓
Domain Generalization (DG) via MixStyle Perturbation
        ↓
Cadastral Boundary Segmentation Backbone
        ↓
Dual Heads: Boundary Logits + 8-Way Connectivity Tensor
        ↓
Morphological Boundary Refinement & Component Filtering
        ↓
Guo-Hall / Medial Axis Skeletonization
        ↓
NetworkX Skeleton Graph Construction & Branch Tracing
        ↓
Douglas-Peucker Geometric Simplification
        ↓
Affine Spatial Restoration (EPSG:28992 → EPSG:4326)
        ↓
GIS Topology & Quality Audit Engine
        ↓
FastAPI Backend & Interactive Leaflet WebGIS
        ↓
Human-in-the-Loop Review & GIS Export (GeoJSON / Shapefile)
```

---

## 1. Neural Network Architecture

### Backbone: Lightweight Residual U-Net
- **Encoder**: 4 hierarchical downsampling stages ($[32, 64, 128, 256, 512]$ channels).
- **Residual Blocks**: Each stage incorporates double $3\times3$ convolutions with batch normalization, ReLU activation, and a $1\times1$ residual projection shortcut to mitigate vanishing gradients.

### Modular Research Components

#### A. Progressive Multiscale Generator (PMG) (`ai/models/pmg.py`)
- **Multi-Kernel Dilated Branches**: Each intermediate encoder stage is processed through parallel dilated convolutions (rates $d \in \{1, 2, 4\}$) capturing parcel line context at multiple spatial resolutions.
- **Top-Down Coarse-to-Fine Fusion**: Aggregates high-level spatial context progressively down to fine-scale boundary features.
- **Residual Adapter Injection**: Fuses multiscale context maps back into decoder skip connections.

#### B. Domain Generalization (DG) (`ai/models/domain_generalization.py`)
- **MixStyle Feature Statistics Perturbation**: Interpolates feature mean $\mu(x)$ and standard deviation $\sigma(x)$ across mini-batch instances during training with probability $p=0.5$ and $\text{Beta}(\alpha=0.2, \alpha=0.2)$ blending.
- **Geographic Robustness**: Prevents the network from overfitting to specific regional vegetation, roof types, or road paving textures.

#### C. Connectivity-Aware Dual-Head (`ai/models/connectivity.py` & `ai/models/full_model.py`)
- **Boundary Head**: Predicts continuous boundary pixel probabilities $P_{boundary} \in [0, 1]^{1 \times H \times W}$.
- **Connectivity Head**: Predicts an 8-neighborhood directional affinity tensor $P_{conn} \in [0, 1]^{8 \times H \times W}$ along directions $\{N, NE, E, SE, S, SW, W, NW\}$.
- **Topological Continuity Loss**: Masked Binary Cross Entropy loss penalizes boundary fragmentations along legitimate property lines.

---

## 2. Supporting GIS Layers vs Cadastral Boundaries
> **CRITICAL ARCHITECTURAL DISTINCTION**:
> Cadastral boundaries delineate legal land parcels and property ownership.
> Building footprints and road networks are **supporting context layers**, never substitutes for cadastral parcel boundaries.
