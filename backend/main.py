"""
SIH26012 FastAPI Backend Application.
Serves AI inference, GIS vectorization, real-time topology auditing, and WebGIS demo endpoints.
"""

import os
import io
import json
import base64
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from PIL import Image

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.services.inference_service import InferenceService
from backend.services.gis_service import GISService


app = FastAPI(
    title="SIH26012 Cadastral Feature Extraction WebGIS API",
    version="1.0.0",
    description="AI-Based Automated Urban Parcel Mapping and Cadastral Feature Extraction System"
)

# Enable CORS for local dev and frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Initialize Services
inference_service = InferenceService()
gis_service = GISService()

# Cache for latest prediction probability map per session/patch
PREDICTION_CACHE: Dict[str, np.ndarray] = {}


class InferenceRequest(BaseModel):
    patch_id: str
    model_name: str = "full"


class VectorizeRequest(BaseModel):
    patch_id: str
    threshold: float = 0.5
    simplify_tolerance: float = 0.5
    min_length_m: float = 3.5
    min_component_size: int = 50
    merge_contiguous: bool = True


class TopologyAuditRequest(BaseModel):
    geojson: Dict[str, Any]


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "system": "SIH26012 Cadastral Extraction System",
        "device": str(inference_service.device),
        "available_models": list(inference_service.models.keys()),
        "checkpoint_info": getattr(inference_service, "checkpoint_info", {}),
    }


@app.get("/api/demo-samples")
def get_demo_samples():
    """
    Returns curated demo samples (Benchmark Urban, High-Density Parcels, Rural, Validation, Test).
    """
    df = gis_service.metadata_df
    if df.empty:
        return {"samples": []}

    # Priority curated benchmark patches
    priority_ids = [
        "tile-190000-310000_r24576_c02048", # Test Vis 5: Residential parcel block (Held-out Test)
        "tile-100000-470000_r04608_c21504", # High-density urban parcels (Train)
        "tile-190000-310000_r01024_c37376", # Urban grid network (Test)
        "tile-190000-310000_r05120_c38912", # Residential neighborhood (Test)
        "tile-190000-310000_r09728_c34304", # SIH Benchmark Urban (Test)
        "tile-170000-460000_r08704_c13824", # Rural agricultural parcels (Test)
    ]
    priority_rows = df[df["patch_id"].isin(priority_ids)]

    # Additional representative samples
    test_top = df[df["split"] == "test"].sort_values("boundary_density", ascending=False).head(4)
    urban_samples = df[df["is_urban"] == True].head(3)
    rural_samples = df[(df["is_urban"] == False) & (df["boundary_pixels"] > 100)].head(3)

    curated = pd.concat([priority_rows, test_top, urban_samples, rural_samples]).drop_duplicates(subset=["patch_id"])
    
    samples_list = []
    for _, row in curated.iterrows():
        bounds = gis_service.get_patch_wgs84_bounds(row["patch_id"])
        samples_list.append({
            "patch_id": row["patch_id"],
            "tile_id": row["tile_id"],
            "split": row["split"],
            "is_urban": bool(row["is_urban"]),
            "boundary_pixels": int(row["boundary_pixels"]),
            "boundary_density": round(float(row["boundary_density"]), 4),
            "image_url": f"/api/sample-image/{row['patch_id']}",
            "mask_url": f"/api/sample-mask/{row['patch_id']}",
            "wgs84_bounds": bounds,
        })

    return {"samples": samples_list}


@app.get("/api/sample-image/{patch_id}")
def get_sample_image(patch_id: str):
    demo_path = f"demo_data/images/{patch_id}.png"
    if os.path.exists(demo_path):
        return FileResponse(demo_path, media_type="image/png")
    meta = gis_service.get_patch_metadata(patch_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Patch not found")
    img_path = os.path.join("data/processed", meta.get("image_path", ""))
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(img_path, media_type="image/png")


@app.get("/api/sample-mask/{patch_id}")
def get_sample_mask(patch_id: str):
    meta = gis_service.get_patch_metadata(patch_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Patch not found")
    mask_path = os.path.join("data/processed", meta.get("mask_path", ""))
    if not os.path.exists(mask_path):
        raise HTTPException(status_code=404, detail="Mask file not found")
    return FileResponse(mask_path, media_type="image/png")


@app.get("/api/ground-truth-vector/{patch_id}")
def get_ground_truth_vector(patch_id: str):
    """
    Returns authentic Ground-Truth cadastral boundaries as GeoJSON vector lines.
    """
    geojson = gis_service.get_ground_truth_vector(patch_id)
    return geojson


@app.post("/api/predict")
def predict_cadastral_boundaries(req: InferenceRequest):
    """
    Run neural network inference on a designated patch.
    Returns:
      - heatmap_base64: raw probability map (grayscale, prob*255)
      - skeleton_image_base64: skeletonized AI prediction as white lines on
        transparent PNG — matches the style of the offline GT mask visualization
        (Panel 2 in training result images)
    """
    demo_pred = f"demo_data/predictions/{req.patch_id}.npy"
    if os.path.exists(demo_pred):
        prob_map = np.load(demo_pred).astype(np.float32) / 255.0
    else:
        demo_img = f"demo_data/images/{req.patch_id}.png"
        if os.path.exists(demo_img):
            img = np.array(Image.open(demo_img).convert("RGB"))
        else:
            meta = gis_service.get_patch_metadata(req.patch_id)
            if not meta:
                raise HTTPException(status_code=404, detail="Patch metadata not found")
            img_path = os.path.join("data/processed", meta.get("image_path", ""))
            img = np.array(Image.open(img_path).convert("RGB"))

        inf_res = inference_service.predict(img, model_name=req.model_name)
        prob_map = inf_res["prob_map"]

    PREDICTION_CACHE[req.patch_id] = prob_map

    # --- 1. Raw probability heatmap (grayscale) ---------------------------------
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    prob_pil = Image.fromarray(prob_uint8)
    buf = io.BytesIO()
    prob_pil.save(buf, format="PNG")
    prob_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # --- 2. Skeletonized line image (white on transparent) ----------------------
    # Threshold at 0.35 (UI default) → binary → skeleton → RGBA PNG
    # This produces clean, thin, continuous lines identical to the GT mask
    # visualization (Panel 2 of training result images).
    try:
        import cv2
        from gis.skeletonization import fast_skeletonize
        DEFAULT_THRESHOLD = 0.35
        binary = (prob_map >= DEFAULT_THRESHOLD).astype(np.uint8)
        skel   = fast_skeletonize(binary)  # 1-pixel thin lines

        # Build two-layer RGBA image: dark halo + bright lime-green foreground
        # This produces the same clean "cadastral line" look as the GT mask panel
        # but stays visible over any aerial imagery background.
        h, w = skel.shape
        skel_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Layer 1: 5px black halo for contrast against any background
        kern_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        halo = cv2.dilate(skel.astype(np.uint8) * 255, kern_halo, iterations=1)
        halo_mask = halo > 0
        skel_rgba[halo_mask] = [0, 0, 0, 180]           # dark halo

        # Layer 2: 2px bright lime-green core
        kern_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        core = cv2.dilate(skel.astype(np.uint8) * 255, kern_core, iterations=1)
        core_mask = core > 0
        skel_rgba[core_mask] = [0, 255, 136, 245]        # bright #00FF88 green

        skel_img = Image.fromarray(skel_rgba, mode="RGBA")
        buf2 = io.BytesIO()
        skel_img.save(buf2, format="PNG")
        skel_b64 = "data:image/png;base64," + base64.b64encode(buf2.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"[predict] skeleton image generation failed: {e}")
        skel_b64 = None


    bounds = gis_service.get_patch_wgs84_bounds(req.patch_id)

    return {
        "patch_id": req.patch_id,
        "model_name": req.model_name,
        "mean_confidence": inf_res["mean_confidence"],
        "max_confidence": inf_res["max_confidence"],
        "heatmap_base64": f"data:image/png;base64,{prob_b64}",
        "skeleton_image_base64": skel_b64,
        "wgs84_bounds": bounds,
    }


@app.post("/api/vectorize")
def vectorize_prediction(req: VectorizeRequest):
    """
    Convert predicted probability map to GeoJSON and generate GIS topology report.
    """
    meta = gis_service.get_patch_metadata(req.patch_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Patch metadata not found")

    # Get cached or recomputed probability map
    if req.patch_id in PREDICTION_CACHE:
        prob_map = PREDICTION_CACHE[req.patch_id]
    else:
        # Compute default inference
        img_path = os.path.join("data/processed", meta["image_path"])
        img = np.array(Image.open(img_path).convert("RGB"))
        inf_res = inference_service.predict(img, model_name="full")
        prob_map = inf_res["prob_map"]
        PREDICTION_CACHE[req.patch_id] = prob_map

    transform = meta["transform"]
    gis_res = gis_service.vectorize_and_audit(
        prob_map=prob_map,
        transform=transform,
        threshold=req.threshold,
        simplify_tolerance=req.simplify_tolerance,
        min_length_m=req.min_length_m,
        min_component_size=req.min_component_size,
        merge_contiguous=req.merge_contiguous,
        source_crs=meta.get("crs", "EPSG:28992")
    )

    return {
        "patch_id": req.patch_id,
        "geojson": gis_res["geojson_wgs84"],
        "parcels_geojson": gis_res.get("parcels_geojson_wgs84"),
        "total_parcels": gis_res.get("total_parcels", 0),
        "topology_report": gis_res["topology_report"],
        "total_lines": gis_res["total_lines"],
        "total_length_m": gis_res["total_length_m"],
        "wgs84_bounds": gis_res.get("wgs84_bounds"),
    }


@app.post("/api/topology-audit")
def audit_topology(req: TopologyAuditRequest):
    """
    Validate topology of user-edited GeoJSON vectors in real-time.
    """
    report = gis_service.audit_edited_geojson(req.geojson)
    return {"topology_report": report}


@app.get("/api/supporting-layers/{layer_type}/{patch_id}")
def get_supporting_layer(layer_type: str, patch_id: str):
    """
    Returns supporting GIS context layers (Building footprints or Roads) clearly distinguished from cadastral boundaries.
    """
    if layer_type not in ["buildings", "roads"]:
        raise HTTPException(status_code=400, detail="Invalid layer type. Choose 'buildings' or 'roads'.")
    geojson = gis_service.get_supporting_layers(patch_id=patch_id, layer_type=layer_type)
    return geojson


@app.get("/presentation")
def get_presentation_slides():
    """
    Serves the official 16:9 widescreen SIH Idea Presentation slide deck.
    """
    html_path = os.path.abspath("docs/presentation_slides.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Presentation not found")


@app.get("/SIH26012_Idea_Presentation.pptx")
def download_presentation_pptx():
    """
    Downloads the editable PowerPoint presentation matching the official SIH template.
    """
    pptx_path = os.path.abspath("SIH26012_Idea_Presentation.pptx")
    if os.path.exists(pptx_path):
        return FileResponse(
            pptx_path,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename="SIH26012_Idea_Presentation.pptx"
        )
    raise HTTPException(status_code=404, detail="PPTX file not found")


# Mount static files and frontend
frontend_dir = os.path.abspath("frontend/dist") if os.path.exists("frontend/dist") else os.path.abspath("frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)

