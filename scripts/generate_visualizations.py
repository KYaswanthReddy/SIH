"""
Phase 1 Visual Verification Generator.
Creates 3-panel comparative visualizations (Original Aerial Image, Ground-Truth Mask, Overlay)
demonstrating spatial alignment across boundary-rich, low-boundary, and urban areas.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_config(config_path: str = "configs/dataset_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_overlay(image_np: np.ndarray, mask_np: np.ndarray, color=(255, 50, 50), alpha=0.6) -> np.ndarray:
    """
    Create a semi-transparent colored boundary overlay onto the aerial image.
    """
    overlay = image_np.copy()
    boundary_idx = mask_np > 0
    if np.any(boundary_idx):
        for c in range(3):
            overlay[boundary_idx, c] = (
                (1 - alpha) * image_np[boundary_idx, c] + alpha * color[c]
            ).astype(np.uint8)
    return overlay


def plot_triplet(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    overlay_np: np.ndarray,
    title: str,
    subtitle: str,
    output_path: str
):
    """
    Plot and save a 3-panel comparison figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), dpi=150)
    fig.patch.set_facecolor("#121212")

    # Panel A: Original Aerial Image
    axes[0].imshow(image_np)
    axes[0].set_title("A. Original Aerial Image (RGB)", color="#EEEEEE", fontsize=13, fontweight="bold", pad=12)
    axes[0].axis("off")

    # Panel B: Ground-Truth Boundary Mask
    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("B. Ground-Truth Cadastral Mask", color="#EEEEEE", fontsize=13, fontweight="bold", pad=12)
    axes[1].axis("off")

    # Panel C: Boundary Overlay
    axes[2].imshow(overlay_np)
    axes[2].set_title("C. Aerial Image + Cadastral Overlay", color="#EEEEEE", fontsize=13, fontweight="bold", pad=12)
    axes[2].axis("off")

    fig.suptitle(title, color="#FFFFFF", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.91, subtitle, color="#BBBBBB", fontsize=11, ha="center")
    plt.subplots_adjust(top=0.86, bottom=0.04, left=0.03, right=0.97, wspace=0.08)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def generate_visualizations():
    config = load_config()
    processed_dir = config["paths"]["processed_dir"]
    vis_dir = config["paths"]["visualizations_dir"]
    metadata_csv = os.path.join(config["paths"]["metadata_dir"], "patches_all.csv")

    if not os.path.exists(metadata_csv):
        raise FileNotFoundError(f"Metadata file not found: {metadata_csv}. Run prepare_data.py first.")

    os.makedirs(vis_dir, exist_ok=True)
    df = pd.read_csv(metadata_csv)

    print(f"Generating visual verification samples from {len(df)} total patches...")

    # Filter categories
    boundary_rich = df[df["boundary_density"] > df["boundary_density"].quantile(0.85)].sample(
        min(4, len(df)), random_state=42
    )
    low_boundary = df[(df["boundary_pixels"] > 0) & (df["boundary_density"] < df["boundary_density"].quantile(0.30))].sample(
        min(4, len(df)), random_state=42
    )
    urban = df[df["is_urban"] == True]
    if len(urban) > 4:
        urban = urban.sample(4, random_state=42)

    categories = [
        ("Boundary-Rich Cadastral Parcels", "boundary_rich", boundary_rich),
        ("Low-Boundary / Rural Parcels", "low_boundary", low_boundary),
        ("Urban Built-Up Area Parcels", "urban", urban),
    ]

    saved_images = []

    for cat_title, cat_prefix, subset_df in categories:
        for idx, (_, row) in enumerate(subset_df.iterrows()):
            img_full_path = os.path.join(processed_dir, row["image_path"])
            mask_full_path = os.path.join(processed_dir, row["mask_path"])

            if not (os.path.exists(img_full_path) and os.path.exists(mask_full_path)):
                continue

            img = np.array(Image.open(img_full_path).convert("RGB"))
            mask = np.array(Image.open(mask_full_path))

            # Semitransparent red/orange boundary overlay
            overlay = create_overlay(img, mask, color=(255, 60, 60), alpha=0.7)

            out_filename = f"{cat_prefix}_sample_{idx+1}_{row['patch_id']}.png"
            out_filepath = os.path.join(vis_dir, out_filename)

            subtitle = (
                f"Patch: {row['patch_id']} | Split: {row['split']} | "
                f"Lines: {row['num_lines']} | Boundary Pixels: {row['boundary_pixels']} ({row['boundary_density']*100:.2f}%) | "
                f"Urban: {row['is_urban']}"
            )

            plot_triplet(img, mask, overlay, f"{cat_title} (Sample {idx+1})", subtitle, out_filepath)
            saved_images.append(out_filepath)
            print(f"  Created: {out_filepath}")

    print(f"\nSuccessfully generated {len(saved_images)} visual verification images in {vis_dir}")


if __name__ == "__main__":
    generate_visualizations()
