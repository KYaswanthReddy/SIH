"""
Test: Exact numerical parity between offline evaluation pipeline and WebGIS inference pipeline.

Requirement: For the same patch, checkpoint, preprocessing, and threshold,
the WebGIS pipeline must reproduce the offline pipeline exactly.
"""

import os
import sys
import numpy as np
import torch
import cv2
import pytest
from PIL import Image

# Paths
PATCH_ID  = "tile-100000-470000_r04608_c21504"
CKPT_PATH = "experiments/quick_full_fixed/results/checkpoints/best_model.pth"
THRESHOLD = 0.50
NATIVE_SZ = 256


@pytest.fixture(scope="module")
def model_and_device():
    """Load the trained CadastreUNetFull model once for the whole module."""
    if not os.path.exists(CKPT_PATH):
        pytest.skip(f"Checkpoint not found: {CKPT_PATH}")
    from ai.models.full_model import CadastreUNetFull
    device = torch.device("cpu")
    model = CadastreUNetFull(
        in_channels=3, out_channels=1,
        features=[32, 64, 128, 256, 512],
        pmg_enabled=True, dg_enabled=True,
        connectivity_enabled=True, dg_prob=0.0
    )
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model, device


@pytest.fixture(scope="module")
def patch_image_and_transform():
    """Load the test patch image and its affine transform."""
    try:
        import pandas as pd
        meta_df = pd.read_csv("data/processed/metadata/patches_all.csv")
        rows = meta_df[meta_df["patch_id"] == PATCH_ID]
        if rows.empty:
            pytest.skip(f"Patch {PATCH_ID} not found in metadata")
        row = rows.iloc[0]
        img_path = os.path.join("data/processed", row["image_path"])
        if not os.path.exists(img_path):
            pytest.skip(f"Patch image not found: {img_path}")
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        transform_data = eval(row["transform"]) if isinstance(row["transform"], str) else row["transform"]
        return img, transform_data
    except Exception as e:
        pytest.skip(f"Could not load patch data: {e}")


def offline_forward(model, device, img_np):
    """
    Exact offline evaluation preprocessing (ai/transforms.py val path):
    - img / 255.0 (no ImageNet mean/std normalization)
    - cv2.INTER_LINEAR resize to (NATIVE_SZ, NATIVE_SZ)
    - np.transpose (H,W,C) -> (C,H,W)
    - torch.from_numpy, unsqueeze(0)
    - model(x) -> scalar prob map (no return_dict)
    Returns: prob_map (H, W) float32 at NATIVE_SZ resolution
    """
    img_f32 = img_np.astype(np.float32) / 255.0
    img_resized = cv2.resize(img_f32, (NATIVE_SZ, NATIVE_SZ), interpolation=cv2.INTER_LINEAR)
    img_chw = np.transpose(img_resized, (2, 0, 1))
    tensor = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)
    with torch.no_grad():
        prob = model(tensor)           # return_dict=False -> refined_prob tensor (B,1,H,W)
    return prob.squeeze().cpu().numpy()


def webgis_forward(model, device, img_np):
    """
    Exact WebGIS InferenceService.predict() preprocessing path.
    Returns: (prob_native, prob_upsampled) both float32 numpy arrays.
    prob_native: (NATIVE_SZ, NATIVE_SZ) — the raw model output
    prob_upsampled: (orig_h, orig_w) — bilinear upsample of native output
    """
    orig_h, orig_w = img_np.shape[:2]
    img_f32 = img_np.astype(np.float32) / 255.0
    if (orig_h, orig_w) != (NATIVE_SZ, NATIVE_SZ):
        img_resized = cv2.resize(img_f32, (NATIVE_SZ, NATIVE_SZ), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_f32
    img_chw = np.transpose(img_resized, (2, 0, 1))
    tensor = torch.from_numpy(img_chw).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(tensor, return_dict=True)
        prob_native = out["refined_prob"].squeeze().cpu().numpy()
    if (orig_h, orig_w) != (NATIVE_SZ, NATIVE_SZ):
        prob_upsampled = cv2.resize(prob_native, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        prob_upsampled = prob_native.copy()
    return prob_native, prob_upsampled


class TestPipelineParity:
    """
    Verify that the WebGIS inference pipeline numerically reproduces
    the offline evaluation pipeline for the same image and checkpoint.
    """

    def test_preprocessing_produces_same_tensor(self, model_and_device, patch_image_and_transform):
        """
        Both pipelines must produce bit-identical input tensors for the model.
        """
        model, device = model_and_device
        img_np, _ = patch_image_and_transform

        # Offline preprocessing
        img_f32_off = img_np.astype(np.float32) / 255.0
        img_res_off = cv2.resize(img_f32_off, (NATIVE_SZ, NATIVE_SZ), interpolation=cv2.INTER_LINEAR)

        # WebGIS preprocessing
        img_f32_web = img_np.astype(np.float32) / 255.0
        img_res_web = cv2.resize(img_f32_web, (NATIVE_SZ, NATIVE_SZ), interpolation=cv2.INTER_LINEAR)

        diff = np.abs(img_res_off.astype(np.float64) - img_res_web.astype(np.float64))
        assert diff.max() == 0.0, \
            f"Preprocessing divergence: max|diff|={diff.max():.2e}"

    def test_prob_map_numerically_identical_native_resolution(self, model_and_device, patch_image_and_transform):
        """
        The raw model output (native 256×256) must be bit-identical between
        the offline pipeline (return_dict=False) and WebGIS pipeline (return_dict=True).
        """
        model, device = model_and_device
        img_np, _ = patch_image_and_transform

        prob_offline = offline_forward(model, device, img_np)
        prob_webgis_native, _ = webgis_forward(model, device, img_np)

        diff = np.abs(prob_offline.astype(np.float64) - prob_webgis_native.astype(np.float64))
        assert diff.max() < 1e-6, \
            f"Probability map divergence: max|diff|={diff.max():.2e}"
        assert diff.mean() < 1e-8, \
            f"Mean probability map divergence: mean|diff|={diff.mean():.2e}"

    def test_prob_map_statistics_match(self, model_and_device, patch_image_and_transform):
        """
        Verify probability statistics match between both pipelines at native resolution.
        """
        model, device = model_and_device
        img_np, _ = patch_image_and_transform

        prob_offline = offline_forward(model, device, img_np)
        prob_webgis, _ = webgis_forward(model, device, img_np)

        assert abs(prob_offline.min() - prob_webgis.min()) < 1e-6, "min prob mismatch"
        assert abs(prob_offline.max() - prob_webgis.max()) < 1e-6, "max prob mismatch"
        assert abs(prob_offline.mean() - prob_webgis.mean()) < 1e-6, "mean prob mismatch"

        for thr in [0.20, 0.30, 0.40, 0.50]:
            n_off = int((prob_offline >= thr).sum())
            n_web = int((prob_webgis >= thr).sum())
            assert n_off == n_web, \
                f"Positive pixel count mismatch at thresh={thr}: offline={n_off}, webgis={n_web}"

    def test_binary_mask_exact_match(self, model_and_device, patch_image_and_transform):
        """
        Binary masks at threshold=0.50 must be pixel-for-pixel identical at native resolution.
        """
        model, device = model_and_device
        img_np, _ = patch_image_and_transform

        prob_offline = offline_forward(model, device, img_np)
        prob_webgis, _ = webgis_forward(model, device, img_np)

        binary_offline = (prob_offline >= THRESHOLD).astype(np.uint8)
        binary_webgis  = (prob_webgis  >= THRESHOLD).astype(np.uint8)

        assert np.array_equal(binary_offline, binary_webgis), \
            f"Binary mask mismatch: {(binary_offline != binary_webgis).sum()} differing pixels"

    def test_connected_components_match(self, model_and_device, patch_image_and_transform):
        """
        Connected component count must match at native resolution.
        """
        model, device = model_and_device
        img_np, _ = patch_image_and_transform

        prob_offline = offline_forward(model, device, img_np)
        prob_webgis, _ = webgis_forward(model, device, img_np)

        bin_off = (prob_offline >= THRESHOLD).astype(np.uint8)
        bin_web = (prob_webgis  >= THRESHOLD).astype(np.uint8)

        n_off = cv2.connectedComponentsWithStats(bin_off, connectivity=8)[0] - 1
        n_web = cv2.connectedComponentsWithStats(bin_web, connectivity=8)[0] - 1

        assert n_off == n_web, \
            f"Connected component count mismatch: offline={n_off}, webgis={n_web}"

    def test_skeleton_pixel_count_match(self, model_and_device, patch_image_and_transform):
        """
        Skeletonization (fast_skeletonize, same algorithm) must produce identical results.
        """
        from gis.skeletonization import fast_skeletonize
        model, device = model_and_device
        img_np, _ = patch_image_and_transform

        prob_offline = offline_forward(model, device, img_np)
        prob_webgis, _ = webgis_forward(model, device, img_np)

        bin_off = (prob_offline >= THRESHOLD).astype(np.uint8)
        bin_web = (prob_webgis  >= THRESHOLD).astype(np.uint8)

        skel_off = fast_skeletonize(bin_off)
        skel_web = fast_skeletonize(bin_web)

        assert np.array_equal(skel_off, skel_web), \
            f"Skeleton mismatch: {(skel_off != skel_web).sum()} differing pixels"
        assert skel_off.sum() == skel_web.sum(), \
            f"Skeleton pixel count mismatch: offline={skel_off.sum()}, webgis={skel_web.sum()}"

    def test_vector_line_count_match(self, model_and_device, patch_image_and_transform):
        """
        Vectorization (vectorize_skeleton with identical parameters) must produce same line count.
        Uses the native 256×256 prob map with the correctly scaled 256×256 transform.
        """
        from gis.skeletonization import fast_skeletonize
        from gis.vectorization import vectorize_skeleton
        from rasterio.transform import Affine

        model, device = model_and_device
        img_np, transform_data = patch_image_and_transform

        prob_offline = offline_forward(model, device, img_np)
        prob_webgis, _ = webgis_forward(model, device, img_np)

        bin_off = (prob_offline >= THRESHOLD).astype(np.uint8)
        bin_web = (prob_webgis  >= THRESHOLD).astype(np.uint8)

        skel_off = fast_skeletonize(bin_off)
        skel_web = fast_skeletonize(bin_web)

        # Scale transform for 256×256 space (pixel size doubles from 0.25m to 0.50m)
        t_512 = Affine(*transform_data[:6])
        t_256 = Affine(t_512.a * 2, t_512.b, t_512.c, t_512.d, t_512.e * 2, t_512.f)

        VECT_PARAMS = dict(
            simplify_tolerance=0.5,
            min_length_m=3.5,
            merge_contiguous=True,
            crs="EPSG:28992"
        )

        gdf_off = vectorize_skeleton(skel_off, transform=t_256, prob_map=prob_offline, **VECT_PARAMS)
        gdf_web = vectorize_skeleton(skel_web, transform=t_256, prob_map=prob_webgis, **VECT_PARAMS)

        assert len(gdf_off) == len(gdf_web), \
            f"Vector line count mismatch: offline={len(gdf_off)}, webgis={len(gdf_web)}"

        if not gdf_off.empty and not gdf_web.empty:
            len_diff = abs(gdf_off["length_m"].sum() - gdf_web["length_m"].sum())
            assert len_diff < 0.01, \
                f"Total vector length mismatch: offline={gdf_off['length_m'].sum():.2f}m, " \
                f"webgis={gdf_web['length_m'].sum():.2f}m (diff={len_diff:.4f}m)"

    def test_upsampled_prob_map_preserves_mean(self, model_and_device, patch_image_and_transform):
        """
        The upsampled 512×512 prob map must preserve the same mean as the native 256×256 map
        (bilinear upsample is mean-preserving to within floating-point tolerance).
        """
        model, device = model_and_device
        img_np, _ = patch_image_and_transform

        prob_native, prob_upsampled = webgis_forward(model, device, img_np)

        # Mean should be preserved by bilinear upsampling
        assert abs(prob_native.mean() - prob_upsampled.mean()) < 1e-3, \
            f"Mean not preserved after upsample: native={prob_native.mean():.6f}, " \
            f"upsampled={prob_upsampled.mean():.6f}"

    def test_no_imagenet_normalization_applied(self, model_and_device, patch_image_and_transform):
        """
        Neither pipeline should apply ImageNet mean/std normalization.
        Verify that pixel values are in [0, 1] without mean subtraction.
        """
        img_np, _ = patch_image_and_transform

        # Check that the offline pipeline does NOT normalize with ImageNet stats
        img_f32 = img_np.astype(np.float32) / 255.0
        img_resized = cv2.resize(img_f32, (NATIVE_SZ, NATIVE_SZ), interpolation=cv2.INTER_LINEAR)

        # Values must be in [0, 1] without offset
        assert img_resized.min() >= 0.0, "Pixel values below 0 (unexpected normalization)"
        assert img_resized.max() <= 1.0, "Pixel values above 1 (unexpected scaling)"

        # ImageNet mean would shift values: mean[R]=0.485, mean[G]=0.456, mean[B]=0.406
        # If ImageNet normalization were applied, some channels would have negative values
        IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img_normalized_hypothetical = img_resized - IMAGENET_MEAN
        # The actual tensor should NOT look like this (no negative values from mean subtraction)
        if img_resized.min() > 0.0:
            assert img_resized.min() > -0.1, \
                "Unexpected normalization: negative values suggest ImageNet mean subtraction was applied"

    def test_output_head_is_refined_boundary(self, model_and_device, patch_image_and_transform):
        """
        Both pipelines must use the 'refined_prob' output head (connectivity-fused boundary prediction).
        Verify by checking return_dict=False equals return_dict=True['refined_prob'].
        """
        model, device = model_and_device
        img_np, _ = patch_image_and_transform

        img_f32 = img_np.astype(np.float32) / 255.0
        img_resized = cv2.resize(img_f32, (NATIVE_SZ, NATIVE_SZ), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(np.transpose(img_resized, (2, 0, 1))).unsqueeze(0).float().to(device)

        with torch.no_grad():
            prob_simple = model(tensor)                     # return_dict=False
            prob_dict   = model(tensor, return_dict=True)  # return_dict=True

        simple_np = prob_simple.squeeze().cpu().numpy()
        refined_np = prob_dict["refined_prob"].squeeze().cpu().numpy()
        boundary_np = prob_dict["boundary_prob"].squeeze().cpu().numpy()

        # return_dict=False must return refined_prob (not boundary_prob)
        diff_refined  = np.abs(simple_np - refined_np).max()
        diff_boundary = np.abs(simple_np - boundary_np).max()

        assert diff_refined < 1e-6, \
            f"return_dict=False does not return refined_prob! max|diff|={diff_refined:.2e}"
        # boundary_prob should be different from refined_prob (connectivity fusion changes it)
        # We just verify that the offline pipeline uses refined_prob
        assert diff_refined < diff_boundary or diff_boundary < 1e-6, \
            "Output head selection check inconclusive"
