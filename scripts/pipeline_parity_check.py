"""
Exact pipeline parity check: Offline Evaluation vs WebGIS Inference.

Runs BOTH pipelines on the exact same patch, checkpoint, and threshold.
Compares probability maps, binary masks, connected components, skeleton, and vector output numerically.
Prints a pass/fail parity report.
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────── CONFIGURATION ────────────────────────────────────────
PATCH_ID   = "tile-100000-470000_r04608_c21504"
CKPT_PATH  = "experiments/quick_full_fixed/results/checkpoints/best_model.pth"
THRESHOLD  = 0.50
NATIVE_SZ  = 256   # Model's training resolution
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model():
    from ai.models.full_model import CadastreUNetFull
    device = torch.device("cpu")
    model = CadastreUNetFull(
        in_channels=3, out_channels=1,
        features=[32, 64, 128, 256, 512],
        pmg_enabled=True, dg_enabled=True,
        connectivity_enabled=True, dg_prob=0.0
    )
    ckpt = torch.load(CKPT_PATH, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    print(f"[Parity] Checkpoint loaded: epoch={ckpt.get('epoch','N/A')}, "
          f"val_f1={ckpt.get('val_metrics',{}).get('f1','N/A')}")
    return model, device


def load_patch_image(meta):
    img_path = os.path.join("data/processed", meta["image_path"])
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    print(f"[Parity] Loaded image: {img.shape}, dtype={img.dtype}, "
          f"min={img.min()}, max={img.max()}")
    return img


def offline_pipeline(model, device, img_np):
    """
    Exact replica of the offline evaluation pipeline (ai/transforms.py val path):
      - img / 255.0 -> float32
      - resize to (NATIVE_SZ, NATIVE_SZ) with cv2.INTER_LINEAR
      - NO ImageNet normalization in offline eval (CadastreAugmentation does NOT apply mean/std)
      - CHW tensor, unsqueeze batch
      - model(x) -> refined_prob (return_dict=False)
    """
    print("\n" + "="*60)
    print(" PIPELINE A: Offline Evaluation (Gold Standard)")
    print("="*60)

    # Step 1: float32 [0,1]
    img_f32 = img_np.astype(np.float32) / 255.0

    # Step 2: resize to NATIVE_SZ x NATIVE_SZ with INTER_LINEAR
    img_resized = cv2.resize(img_f32, (NATIVE_SZ, NATIVE_SZ), interpolation=cv2.INTER_LINEAR)
    print(f"  After resize:  shape={img_resized.shape}, min={img_resized.min():.4f}, max={img_resized.max():.4f}")

    # Step 3: NO mean/std normalization (CadastreAugmentation does not apply ImageNet stats)
    # Transpose to CHW
    img_chw = np.transpose(img_resized, (2, 0, 1))  # (3, H, W)
    tensor = torch.from_numpy(img_chw).unsqueeze(0).float()  # (1, 3, H, W)
    print(f"  Tensor: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"min={tensor.min():.4f}, max={tensor.max():.4f}")

    # Step 4: Forward pass
    with torch.no_grad():
        prob_tensor = model(tensor.to(device))  # return_dict=False -> refined_prob
    prob = prob_tensor.squeeze().cpu().numpy()  # (NATIVE_SZ, NATIVE_SZ)
    print(f"  Raw prob shape: {prob.shape}")

    return prob


def webgis_pipeline(model, device, img_np):
    """
    Exact replica of the current WebGIS InferenceService.predict() path.
    """
    print("\n" + "="*60)
    print(" PIPELINE B: WebGIS InferenceService (Current Implementation)")
    print("="*60)

    orig_h, orig_w = img_np.shape[:2]
    target_size = (NATIVE_SZ, NATIVE_SZ)

    # Step 1+2: float32 [0,1] + resize
    if (orig_h, orig_w) != target_size:
        img_resized = cv2.resize(img_np.astype(np.float32) / 255.0,
                                 target_size, interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_np.astype(np.float32) / 255.0
    print(f"  After resize:  shape={img_resized.shape}, min={img_resized.min():.4f}, max={img_resized.max():.4f}")

    # Step 3: CHW tensor
    img_chw = np.transpose(img_resized, (2, 0, 1))
    tensor = torch.from_numpy(img_chw).unsqueeze(0).float()
    print(f"  Tensor: shape={tensor.shape}, dtype={tensor.dtype}, "
          f"min={tensor.min():.4f}, max={tensor.max():.4f}")

    # Step 4: Forward pass with return_dict=True -> refined_prob
    with torch.no_grad():
        out_dict = model(tensor.to(device), return_dict=True)
        prob_raw = out_dict["refined_prob"].squeeze().cpu().numpy()

    # Step 5: Upsample back to original size
    if (orig_h, orig_w) != (NATIVE_SZ, NATIVE_SZ):
        prob = cv2.resize(prob_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        prob = prob_raw

    print(f"  Prob map shape after upsample: {prob.shape}")
    return prob, prob_raw


def prob_stats(prob, name, threshold=THRESHOLD):
    n = prob.size
    print(f"\n  [{name}] prob shape={prob.shape}")
    print(f"    min   = {prob.min():.6f}")
    print(f"    max   = {prob.max():.6f}")
    print(f"    mean  = {prob.mean():.6f}")
    print(f"    ≥0.20 = {(prob >= 0.20).sum()} / {n}  ({100*(prob >= 0.20).mean():.2f}%)")
    print(f"    ≥0.30 = {(prob >= 0.30).sum()} / {n}  ({100*(prob >= 0.30).mean():.2f}%)")
    print(f"    ≥0.40 = {(prob >= 0.40).sum()} / {n}  ({100*(prob >= 0.40).mean():.2f}%)")
    print(f"    ≥0.50 = {(prob >= threshold).sum()} / {n}  ({100*(prob >= threshold).mean():.2f}%)")


def compare_probs(prob_a, prob_b, name_a="A", name_b="B"):
    diff = np.abs(prob_a.astype(np.float64) - prob_b.astype(np.float64))
    print(f"\n  [{name_a} vs {name_b}]")
    print(f"    max |diff| = {diff.max():.2e}")
    print(f"    mean |diff| = {diff.mean():.2e}")
    match = diff.max() < 1e-4
    print(f"    MATCH (tol 1e-4): {'✅ YES' if match else '❌ NO'}")
    return match


def component_stats(binary, name):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    n_comps = n_labels - 1  # exclude background
    sizes = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n_labels)]
    pos = binary.sum()
    print(f"\n  [{name}] binary mask stats:")
    print(f"    positive pixels = {pos}")
    print(f"    connected components = {n_comps}")
    if sizes:
        print(f"    avg component size = {np.mean(sizes):.1f}")
        print(f"    max component size = {max(sizes)}")
        print(f"    min component size = {min(sizes)}")
    return n_comps, pos, sizes


def skeleton_stats(binary, name):
    from gis.skeletonization import fast_skeletonize
    skel = fast_skeletonize(binary)
    skel_px = int(skel.sum())
    print(f"\n  [{name}] skeleton pixels = {skel_px}")
    return skel, skel_px


def vectorize_stats(skel, transform_data, prob_map, name):
    from gis.vectorization import vectorize_skeleton
    from rasterio.transform import Affine

    if isinstance(transform_data, (list, tuple)):
        transform = Affine(*transform_data[:6])
    else:
        transform = transform_data

    gdf = vectorize_skeleton(
        skeleton=skel,
        transform=transform,
        prob_map=prob_map,
        simplify_tolerance=0.5,
        min_length_m=3.5,
        merge_contiguous=True,
        crs="EPSG:28992"
    )
    n = len(gdf)
    total_len = gdf["length_m"].sum() if not gdf.empty else 0.0
    print(f"\n  [{name}] vectorization:")
    print(f"    vector lines = {n}")
    print(f"    total length = {total_len:.2f} m")
    if not gdf.empty:
        print(f"    min line = {gdf['length_m'].min():.2f} m")
        print(f"    max line = {gdf['length_m'].max():.2f} m")
        print(f"    mean conf = {gdf['mean_confidence'].mean():.4f}")
    return n, total_len


def main():
    print("\n" + "#"*70)
    print("#  PIPELINE PARITY CHECK: OFFLINE vs WEBGIS")
    print(f"#  Patch: {PATCH_ID}")
    print(f"#  Checkpoint: {CKPT_PATH}")
    print(f"#  Threshold: {THRESHOLD}")
    print("#"*70)

    # Load model (shared instance)
    model, device = load_model()

    # Load patch metadata
    import pandas as pd
    meta_df = pd.read_csv("data/processed/metadata/patches_all.csv")
    row = meta_df[meta_df["patch_id"] == PATCH_ID].iloc[0].to_dict()
    transform_data = eval(row["transform"]) if isinstance(row["transform"], str) else row["transform"]

    # Load original image
    img_np = load_patch_image(row)  # (H, W, 3) uint8, HxW = 512x512

    # ── PIPELINE A: Offline ──────────────────────────────────────────────────
    prob_a = offline_pipeline(model, device, img_np)  # (256, 256) native resolution
    prob_stats(prob_a, "A-Offline (256x256 native)")

    # ── PIPELINE B: WebGIS ───────────────────────────────────────────────────
    prob_b_512, prob_b_raw = webgis_pipeline(model, device, img_np)
    # prob_b_512 is upsampled to 512x512, prob_b_raw is 256x256 native output
    prob_stats(prob_b_raw, "B-WebGIS (256x256 before upsample)")
    prob_stats(prob_b_512, "B-WebGIS (512x512 after upsample)")

    # ── COMPARISON: Native 256x256 ────────────────────────────────────────────
    print("\n" + "="*60)
    print(" NUMERICAL PARITY: native 256x256 output")
    print("="*60)
    compare_probs(prob_a, prob_b_raw, "A(256x256)", "B(256x256 raw)")

    # ── BINARY MASKS ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" BINARY MASKS at threshold=0.50")
    print("="*60)
    binary_a = (prob_a >= THRESHOLD).astype(np.uint8)
    binary_b_raw = (prob_b_raw >= THRESHOLD).astype(np.uint8)
    binary_b_512 = (prob_b_512 >= THRESHOLD).astype(np.uint8)

    n_comps_a, pos_a, sizes_a = component_stats(binary_a, "A-Offline (256x256)")
    n_comps_b_raw, pos_b_raw, sizes_b_raw = component_stats(binary_b_raw, "B-WebGIS (256x256 raw)")
    n_comps_b_512, pos_b_512, sizes_b_512 = component_stats(binary_b_512, "B-WebGIS (512x512 upsampled)")

    # Mask agreement at native resolution
    mask_match = np.array_equal(binary_a, binary_b_raw)
    print(f"\n  Binary mask agreement A vs B(raw): {'✅ EXACT MATCH' if mask_match else '❌ MISMATCH'}")
    if not mask_match:
        disagree = (binary_a != binary_b_raw).sum()
        print(f"    Disagreeing pixels: {disagree} / {binary_a.size}")

    # ── SKELETONIZATION ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" SKELETONIZATION")
    print("="*60)
    skel_a, skel_px_a = skeleton_stats(binary_a, "A-Offline (256x256)")
    skel_b, skel_px_b = skeleton_stats(binary_b_raw, "B-WebGIS (256x256 raw)")

    skel_match = np.array_equal(skel_a, skel_b)
    print(f"\n  Skeleton agreement A vs B(raw): {'✅ EXACT MATCH' if skel_match else '❌ MISMATCH'}")
    if not skel_match:
        diff_px = (skel_a != skel_b).sum()
        print(f"    Disagreeing skeleton pixels: {diff_px}")

    # ── VECTORIZATION ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" VECTORIZATION (on native 256x256 space)")
    print("="*60)
    # For vectorization we need a transform at 256x256 resolution
    # Original transform is for 512x512; at 256x256 pixel size doubles
    from rasterio.transform import Affine
    t_orig = Affine(*transform_data[:6])
    # Scale pixel size by 2 for 256x256 space
    t_256 = Affine(t_orig.a * 2, t_orig.b, t_orig.c,
                   t_orig.d, t_orig.e * 2, t_orig.f)

    n_lines_a, len_a = vectorize_stats(skel_a, t_256, prob_a, "A-Offline (256x256)")
    n_lines_b, len_b = vectorize_stats(skel_b, t_256, prob_b_raw, "B-WebGIS (256x256 raw)")

    # ── FINAL PARITY REPORT ──────────────────────────────────────────────────
    print("\n" + "#"*70)
    print("# PARITY REPORT")
    print("#"*70)
    checks = {
        "Prob map (native 256x256) max|diff|<1e-4": compare_probs(prob_a, prob_b_raw, "A", "B"),
        "Binary mask exact match (native)": mask_match,
        "Skeleton pixel count match": skel_px_a == skel_px_b,
        "Connected component count match": n_comps_a == n_comps_b_raw,
        "Vector line count match": n_lines_a == n_lines_b,
    }
    all_pass = True
    for name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {name}")
        if not result:
            all_pass = False

    print()
    print(f"  Overall: {'✅ ALL CHECKS PASS — pipelines are numerically identical' if all_pass else '❌ FAILURES DETECTED — see above'}")
    print()

    # Summary table
    print("  Summary Table:")
    print(f"  {'Metric':<35} {'Offline A':>15} {'WebGIS B (raw)':>15}")
    print(f"  {'-'*65}")
    print(f"  {'Prob mean (256x256)':<35} {prob_a.mean():>15.6f} {prob_b_raw.mean():>15.6f}")
    print(f"  {'Positive pixels (thresh=0.50)':<35} {pos_a:>15} {pos_b_raw:>15}")
    print(f"  {'Connected components':<35} {n_comps_a:>15} {n_comps_b_raw:>15}")
    print(f"  {'Skeleton pixels':<35} {skel_px_a:>15} {skel_px_b:>15}")
    print(f"  {'Vector lines':<35} {n_lines_a:>15} {n_lines_b:>15}")
    print(f"  {'Total length (m)':<35} {len_a:>15.2f} {len_b:>15.2f}")

    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
