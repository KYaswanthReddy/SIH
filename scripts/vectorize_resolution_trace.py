"""
Check exactly what resolution the WebGIS vectorizes,
vs what the offline pipeline reference numbers came from.
Trace the vectorize_and_audit call with the actual GIS service parameters.
"""

import os, sys, json, numpy as np, torch, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PATCH_ID  = "tile-100000-470000_r04608_c21504"
CKPT_PATH = "experiments/quick_full_fixed/results/checkpoints/best_model.pth"
THRESHOLD = 0.50
NATIVE_SZ = 256

def get_prob_map():
    """Run inference exactly as InferenceService does, return 512x512 upsampled prob."""
    from ai.models.full_model import CadastreUNetFull
    device = torch.device("cpu")
    model = CadastreUNetFull(in_channels=3, out_channels=1,
        features=[32,64,128,256,512], pmg_enabled=True, dg_enabled=True,
        connectivity_enabled=True, dg_prob=0.0)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    import pandas as pd
    meta_df = pd.read_csv("data/processed/metadata/patches_all.csv")
    row = meta_df[meta_df["patch_id"] == PATCH_ID].iloc[0]
    img = np.array(Image.open(os.path.join("data/processed", row["image_path"])).convert("RGB"))
    transform_data = eval(row["transform"]) if isinstance(row["transform"], str) else row["transform"]

    # Replicate InferenceService.predict exactly
    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img.astype(np.float32)/255.0, (NATIVE_SZ, NATIVE_SZ), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.transpose(img_resized, (2,0,1))).unsqueeze(0).float()
    with torch.no_grad():
        out = model(tensor, return_dict=True)
        prob_raw = out["refined_prob"].squeeze().cpu().numpy()   # 256x256
    prob_512 = cv2.resize(prob_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)  # 512x512
    return prob_raw, prob_512, transform_data

def run_vectorize(prob_map, transform_data, label, min_length_m=3.5, min_component_size=50):
    """Run the exact GISService.vectorize_and_audit pipeline."""
    from gis.refinement import refine_boundary_mask
    from gis.skeletonization import fast_skeletonize
    from gis.vectorization import vectorize_skeleton
    from rasterio.transform import Affine
    import cv2

    if isinstance(transform_data, (list, tuple)):
        transform = Affine(*transform_data[:6])
    else:
        transform = transform_data

    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"  prob shape: {prob_map.shape}, pixel_size={abs(transform.a):.4f}m")
    print(f"{'='*60}")

    # Step 1: Morphological refinement
    refined = refine_boundary_mask(prob_map, threshold=THRESHOLD, min_component_size=min_component_size)
    pos = refined.sum()
    n_lab, _, stats, _ = cv2.connectedComponentsWithStats(refined.astype(np.uint8), connectivity=8)
    n_comps = n_lab - 1
    sizes = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n_lab)]
    print(f"  After refinement:")
    print(f"    positive pixels = {pos}")
    print(f"    components = {n_comps}")
    print(f"    avg size = {np.mean(sizes):.1f}" if sizes else "    (no components)")

    # Step 2: Skeletonization
    skel = fast_skeletonize(refined)
    skel_px = int(skel.sum())
    print(f"  Skeleton pixels = {skel_px}")

    # Step 3: Vectorization
    pixel_size = abs(transform.a)
    gdf = vectorize_skeleton(
        skeleton=skel,
        transform=transform,
        prob_map=prob_map,
        simplify_tolerance=0.5,
        min_length_m=min_length_m,
        merge_contiguous=True,
        crs="EPSG:28992"
    )
    n = len(gdf)
    total_len = gdf["length_m"].sum() if not gdf.empty else 0.0
    print(f"  Vector lines = {n}")
    print(f"  Total length = {total_len:.2f} m")
    if not gdf.empty:
        print(f"  min length = {gdf['length_m'].min():.2f} m")
        print(f"  max length = {gdf['length_m'].max():.2f} m")
    return n, total_len, skel_px, n_comps

print("\n" + "#"*70)
print("# VECTORIZATION RESOLUTION & PARAMETER COMPARISON")
print("#"*70)

prob_256, prob_512, tdata = get_prob_map()

# Get the transform for each resolution
from rasterio.transform import Affine
t_512 = Affine(*tdata[:6])  # original 512x512 transform (pixel_size=0.25m)
# For 256x256, pixel size doubles
t_256 = Affine(t_512.a * 2, t_512.b, t_512.c, t_512.d, t_512.e * 2, t_512.f)

print(f"\n  Transform for 512x512: pixel_size={abs(t_512.a):.4f}m")
print(f"  Transform for 256x256: pixel_size={abs(t_256.a):.4f}m")

# 1. What WebGIS currently does: vectorize on 512x512 upsampled map
n1, l1, s1, c1 = run_vectorize(prob_512, t_512, "WebGIS (512x512 upsampled, min_length=3.5m, min_comp=50)")

# 2. What offline reference shows: native 256x256
n2, l2, s2, c2 = run_vectorize(prob_256, t_256, "Offline native (256x256, min_length=3.5m, min_comp=50)")

# 3. WebGIS with 256x256 and correct 256x256 transform
n3, l3, s3, c3 = run_vectorize(prob_256, t_256, "WebGIS using 256x256 (matching offline exactly)")

# 4. Explore: what pipeline parameters give 67 lines, 12504 skel px, 369.4m?
print("\n" + "#"*70)
print("# SEARCHING FOR REFERENCE NUMBERS (67 lines, 12504 skel, 369.4m)")
print("#"*70)
# Try 512x512 with tighter min_length and smaller min_comp
for ml in [1.0, 2.0, 3.0, 3.5]:
    for mc in [10, 25, 50, 100]:
        n_, l_, s_, c_ = run_vectorize(prob_512, t_512,
            f"512x512 min_length={ml}m min_comp={mc}",
            min_length_m=ml, min_component_size=mc)
        if abs(n_ - 67) <= 10 or abs(s_ - 12504) <= 500:
            print(f"  *** NEAR MATCH: lines={n_}, skel={s_}, len={l_:.1f}m ***")
