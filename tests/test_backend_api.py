"""
Tests for FastAPI Backend API Endpoints.
Verifies health check, demo samples catalog, model prediction, raster-to-vector, topology auditing, and supporting layers.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app, gis_service

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "SIH26012" in data["system"]
    assert "available_models" in data
    assert "baseline" in data["available_models"]
    assert "full" in data["available_models"]


def test_demo_samples_endpoint():
    response = client.get("/api/demo-samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0
    sample = data["samples"][0]
    assert "patch_id" in sample
    assert "image_url" in sample
    assert "mask_url" in sample


def test_sample_image_and_mask_endpoints():
    # Fetch demo samples to get a real patch_id
    resp = client.get("/api/demo-samples")
    samples = resp.json()["samples"]
    if not samples:
        pytest.skip("No demo samples found")
    
    patch_id = samples[0]["patch_id"]

    # Test sample image
    img_resp = client.get(f"/api/sample-image/{patch_id}")
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"] == "image/png"

    # Test sample mask
    mask_resp = client.get(f"/api/sample-mask/{patch_id}")
    assert mask_resp.status_code == 200
    assert mask_resp.headers["content-type"] == "image/png"


def test_predict_endpoint():
    resp = client.get("/api/demo-samples")
    samples = resp.json()["samples"]
    if not samples:
        pytest.skip("No demo samples found")
    patch_id = samples[0]["patch_id"]

    # Test prediction with full model
    pred_resp = client.post("/api/predict", json={
        "patch_id": patch_id,
        "model_name": "full"
    })
    assert pred_resp.status_code == 200
    pred_data = pred_resp.json()
    assert pred_data["patch_id"] == patch_id
    assert pred_data["model_name"] == "full"
    assert "mean_confidence" in pred_data
    assert "heatmap_base64" in pred_data
    assert pred_data["heatmap_base64"].startswith("data:image/png;base64,")


def test_vectorize_endpoint():
    resp = client.get("/api/demo-samples")
    samples = resp.json()["samples"]
    if not samples:
        pytest.skip("No demo samples found")
    patch_id = samples[0]["patch_id"]

    vec_resp = client.post("/api/vectorize", json={
        "patch_id": patch_id,
        "threshold": 0.5,
        "simplify_tolerance": 0.5
    })
    assert vec_resp.status_code == 200
    vec_data = vec_resp.json()
    assert vec_data["patch_id"] == patch_id
    assert "geojson" in vec_data
    assert vec_data["geojson"]["type"] == "FeatureCollection"
    assert "topology_report" in vec_data
    assert "total_lines" in vec_data


def test_topology_audit_endpoint():
    # Two connected non-overlapping perpendicular lines
    valid_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[5.0, 52.0], [5.01, 52.0]]
                },
                "properties": {"id": 1}
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[5.01, 52.0], [5.01, 52.01]]
                },
                "properties": {"id": 2}
            }
        ]
    }
    audit_resp = client.post("/api/topology-audit", json={"geojson": valid_geojson})
    assert audit_resp.status_code == 200
    report = audit_resp.json()["topology_report"]
    assert report["is_valid"] is True
    assert report["invalid_geometries_count"] == 0

    # Self-intersecting LineString
    self_intersecting_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0, 0], [2, 2], [0, 2], [2, 0]]
                },
                "properties": {"id": 2}
            }
        ]
    }
    audit_resp2 = client.post("/api/topology-audit", json={"geojson": self_intersecting_geojson})
    assert audit_resp2.status_code == 200
    report2 = audit_resp2.json()["topology_report"]
    assert report2["self_intersections_count"] > 0


def test_supporting_layers_endpoint():
    resp = client.get("/api/demo-samples")
    samples = resp.json()["samples"]
    if not samples:
        pytest.skip("No demo samples found")
    patch_id = samples[0]["patch_id"]

    # Test buildings layer
    bldg_resp = client.get(f"/api/supporting-layers/buildings/{patch_id}")
    assert bldg_resp.status_code == 200
    bldg_data = bldg_resp.json()
    assert bldg_data["type"] == "FeatureCollection"
    assert "layer_info" in bldg_data

    # Test roads layer
    road_resp = client.get(f"/api/supporting-layers/roads/{patch_id}")
    assert road_resp.status_code == 200
    road_data = road_resp.json()
    assert road_data["type"] == "FeatureCollection"
    assert "layer_info" in road_data

    # Test invalid layer type
    err_resp = client.get(f"/api/supporting-layers/invalid_type/{patch_id}")
    assert err_resp.status_code == 400


def test_checkpoint_loaded_and_non_trivial_inference():
    """Verify trained quick_full_fixed checkpoint is loaded and produces non-trivial probabilities."""
    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    h_data = health_resp.json()
    assert "checkpoint_info" in h_data
    ckpt_info = h_data["checkpoint_info"]
    assert "best_model.pth" in ckpt_info.get("path", "")
    assert ckpt_info.get("epoch") is not None
    assert ckpt_info.get("val_f1") is not None

    # Predict on demo sample
    resp = client.get("/api/demo-samples")
    samples = resp.json()["samples"]
    if not samples:
        pytest.skip("No demo samples found")
    patch_id = samples[0]["patch_id"]

    pred_resp = client.post("/api/predict", json={
        "patch_id": patch_id,
        "model_name": "full"
    })
    assert pred_resp.status_code == 200
    p_data = pred_resp.json()
    assert p_data["max_confidence"] > 0.50, f"Expected non-trivial max confidence > 0.50, got {p_data['max_confidence']}"
    assert "wgs84_bounds" in p_data
    bounds = p_data["wgs84_bounds"]
    assert len(bounds) == 2 and len(bounds[0]) == 2 and len(bounds[1]) == 2
    # Verify coordinates are in Netherlands range (Lat 50-54, Lon 3-8)
    assert 50.0 <= bounds[0][0] <= 54.0
    assert 3.0 <= bounds[0][1] <= 8.0


def test_vectorize_produces_real_features():
    """Verify vectorize produces real GeoJSON vector features with the trained model."""
    resp = client.get("/api/demo-samples")
    samples = resp.json()["samples"]
    if not samples:
        pytest.skip("No demo samples found")
    
    # Pick a sample with boundaries
    patch_id = samples[0]["patch_id"]
    for s in samples:
        if s.get("is_urban") or s.get("boundary_pixels", 0) > 100:
            patch_id = s["patch_id"]
            break

    vec_resp = client.post("/api/vectorize", json={
        "patch_id": patch_id,
        "threshold": 0.40,
        "simplify_tolerance": 0.5
    })
    assert vec_resp.status_code == 200
    vec_data = vec_resp.json()
    assert len(vec_data["geojson"]["features"]) > 0, "Vectorization should produce real GeoJSON lines with trained model"
    assert vec_data["total_lines"] > 0
    assert vec_data["total_length_m"] > 0.0
    assert "wgs84_bounds" in vec_data


def test_vectorize_filtering_and_clean_geojson():
    """Verify that custom noise filtering parameters (min_length_m, min_component_size) clean the output."""
    resp = client.get("/api/demo-samples")
    samples = resp.json()["samples"]
    if not samples:
        pytest.skip("No demo samples found")
    
    patch_id = samples[0]["patch_id"]

    # 1. Raw / minimal filtering
    raw_resp = client.post("/api/vectorize", json={
        "patch_id": patch_id,
        "threshold": 0.50,
        "simplify_tolerance": 0.2,
        "min_length_m": 0.0,
        "min_component_size": 1,
        "merge_contiguous": False
    })
    assert raw_resp.status_code == 200
    raw_data = raw_resp.json()

    # 2. Clean / noise filtered
    clean_resp = client.post("/api/vectorize", json={
        "patch_id": patch_id,
        "threshold": 0.50,
        "simplify_tolerance": 0.5,
        "min_length_m": 3.5,
        "min_component_size": 50,
        "merge_contiguous": True
    })
    assert clean_resp.status_code == 200
    clean_data = clean_resp.json()

    # Verify noise filtering pruned false positive micro-fragments
    assert clean_data["total_lines"] <= raw_data["total_lines"]
    assert clean_data["topology_report"]["tiny_fragments_count"] <= raw_data["topology_report"]["tiny_fragments_count"]

    # Verify GeoJSON export structure and properties
    geojson = clean_data["geojson"]
    assert geojson["type"] == "FeatureCollection"
    if geojson["features"]:
        feat = geojson["features"][0]
        assert feat["type"] == "Feature"
        assert feat["geometry"]["type"] == "LineString"
        assert "line_id" in feat["properties"]
        assert "length_m" in feat["properties"]
        assert "mean_confidence" in feat["properties"]


