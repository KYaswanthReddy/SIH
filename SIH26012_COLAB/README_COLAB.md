# SIH26012 — Google Colab Pro Full Architecture Training Guide

This package contains everything required to train, evaluate, vectorize, and demonstrate the **Full Integrated SIH26012 Architecture** (Progressive Multiscale Generator + Domain Generalization + Connectivity Dual-Head + Residual U-Net Backbone) on Google Colab Pro.

---

## 1. Directory Structure

```
SIH26012_COLAB/
├── data/
│   └── processed/
│       ├── metadata/
│       │   ├── patches_train.csv     # 2,500 training patch catalog (5 tiles)
│       │   ├── patches_val.csv       # 600 validation patch catalog (2 tiles)
│       │   ├── patches_test.csv      # 600 test patch catalog (2 tiles)
│       │   └── dataset_summary.json
│       └── patches/
│           ├── train/ (images/, masks/)
│           ├── val/   (images/, masks/)
│           └── test/  (images/, masks/)
├── ai/
│   ├── models/
│   │   ├── full_model.py             # Integrated PMG + DG + Connectivity + U-Net
│   │   ├── pmg.py                    # Progressive Multiscale Generator
│   │   ├── domain_generalization.py  # MixStyle Feature Perturbation
│   │   ├── connectivity.py           # 8-Directional Affinity Head & Loss
│   │   └── unet_baseline.py          # Residual U-Net Backbone
│   ├── dataset.py
│   ├── transforms.py
│   ├── losses.py
│   ├── metrics.py
│   ├── visualizer.py
│   └── trainer.py
├── gis/
│   ├── pipeline.py                   # Complete Raster-to-Vector Pipeline
│   ├── vectorization.py              # Graph Tracing & Douglas-Peucker Simplification
│   ├── topology.py                   # STRtree Topology Audit Engine
│   ├── skeletonization.py
│   ├── refinement.py
│   ├── crs.py
│   └── export.py
├── backend/
│   ├── main.py                       # FastAPI Application
│   └── services/
├── frontend/
│   └── index.html                    # Interactive Leaflet + Tailwind WebGIS
├── configs/
│   └── full.yaml                     # Full Model Training Configuration
├── scripts/
│   ├── train_full.py                 # Full Model Training CLI Runner
│   └── run_ablation_experiments.py
├── notebooks/
│   └── train_full_colab.ipynb        # Master Colab Pro Training Notebook
├── requirements.txt
└── README_COLAB.md
```

---

## 2. Step-by-Step Google Colab Pro Instructions

### Step 1: Upload to Google Drive
1. Upload the entire folder `SIH26012_COLAB` to your Google Drive root directory:
   `My Drive / SIH26012_COLAB`

### Step 2: Open Notebook in Google Colab Pro
1. Navigate to Google Drive and open:
   `SIH26012_COLAB/notebooks/train_full_colab.ipynb` with **Google Colaboratory**.
2. Go to **Runtime** $\to$ **Change runtime type** $\to$ Select **GPU** (T4, V100, or A100).

### Step 3: Mount Google Drive & Set Working Directory
In the first Colab code cell, mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/SIH26012_COLAB')
print("Current Working Directory:", os.getcwd())
!ls -la
```

### Step 4: Install Dependencies
```bash
!pip install -q -r requirements.txt
```

### Step 5A: Ultra-Fast Sanity Verification (<1 Minute on GPU)
Before launching full training, run the ultra-fast sanity check (50 train, 10 val, 10 test patches, 256×256, 1 epoch) or open `SIH26012_COLAB/notebooks/train_sanity_full_colab.ipynb`:

```bash
!python scripts/train_full.py configs/sanity_full.yaml
```

### Step 5B: Start Full Model Training (Production 15 Epochs)
Once sanity check passes, run the complete production training script:

```bash
!python scripts/train_full.py configs/full.yaml
```

**Training Features Activated**:
- PyTorch AMP Mixed Precision (`torch.cuda.amp.autocast`)
- Cosine Annealing Learning Rate Schedule ($T_{\max}=15, \eta_{\min}=10^{-6}$)
- Combined Loss: $0.5 \times \text{Focal} + 0.5 \times \text{Dice} + 0.5 \times \text{Connectivity Loss}$
- Automatic Checkpoint Saving (`experiments/full/checkpoints/best_model.pth`)
- Early Stopping Patience: 4 epochs

### Step 6: Test Set Evaluation & Prediction Visualizations
The script automatically evaluates the best checkpoint on the 600 held-out test patches and saves 5-panel prediction figures (`Original | Ground Truth | Prediction | Error Overlay`) in `experiments/full/predictions/`.

### Step 7: End-to-End GIS Vectorization & Topology Check
Run in a Colab cell:

```python
import torch, os, json
import numpy as np
import pandas as pd
from PIL import Image
from ai.models.full_model import CadastreUNetFull
from gis.pipeline import RasterToVectorGISPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CadastreUNetFull().to(device)

ckpt = torch.load("experiments/full/checkpoints/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load test sample
df_test = pd.read_csv("data/processed/metadata/patches_test.csv")
sample_row = df_test.iloc[0]
img_pil = Image.open(os.path.join("data/processed", sample_row["image_path"])).convert("RGB")
img_tensor = torch.from_numpy(np.array(img_pil).transpose(2, 0, 1) / 255.0).unsqueeze(0).float().to(device)

with torch.no_grad():
    pred_out = model(img_tensor, return_dict=True)
    prob_map = pred_out["refined_prob"][0, 0].cpu().numpy()

# Run Raster to Vector Pipeline
pipeline = RasterToVectorGISPipeline(threshold=0.5, simplify_tolerance=0.5)
res = pipeline.run(prob_map, transform=(0.25, 0.0, 100000.0, 0.0, -0.25, 500000.0), output_geojson_path="experiments/full/predictions/demo_parcel.geojson")

print("=== GIS Quality & Topology Report ===")
print(f"Total Vector Lines: {res['total_lines']}")
print(f"Total Boundary Length: {res['total_length_m']} m")
print(f"Topology Status: {res['topology_report']['status']}")
print(f"Self-Intersections: {res['topology_report']['self_intersections_count']}")
print(f"Invalid Geometries: {res['topology_report']['invalid_geometries_count']}")
```

### Step 8: Run WebGIS Demonstration from Colab (Optional)
To run the interactive FastAPI WebGIS server with a public URL:

```python
!pip install -q pyngrok
from pyngrok import ngrok
import subprocess

# Open port 8000 tunnel
public_url = ngrok.connect(8000).public_url
print("WebGIS Live URL:", public_url)

# Launch FastAPI backend
!uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
