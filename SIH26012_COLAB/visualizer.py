"""
Visualization Utilities for Cadastral Predictions and Training Progress.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch


def create_prediction_overlay(image_np: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Create a composite overlay on the aerial image:
    - Green = True Positive / Ground Truth
    - Red = False Positive / Prediction
    - Yellow = Overlap (True Positive prediction)
    """
    overlay = image_np.copy().astype(np.float32)
    
    gt_idx = gt_mask > 0.5
    pred_idx = pred_mask > 0.5
    
    tp_idx = gt_idx & pred_idx
    fp_idx = (~gt_idx) & pred_idx
    fn_idx = gt_idx & (~pred_idx)

    # TP: Yellow (255, 255, 0)
    if np.any(tp_idx):
        overlay[tp_idx] = (1 - alpha) * overlay[tp_idx] + alpha * np.array([255, 255, 0])
    # FP: Red (255, 50, 50)
    if np.any(fp_idx):
        overlay[fp_idx] = (1 - alpha) * overlay[fp_idx] + alpha * np.array([255, 50, 50])
    # FN: Cyan / Blue (50, 150, 255)
    if np.any(fn_idx):
        overlay[fn_idx] = (1 - alpha) * overlay[fn_idx] + alpha * np.array([50, 200, 50])

    return np.clip(overlay, 0, 255).astype(np.uint8)


def visualize_predictions(
    image_tensor: torch.Tensor,
    gt_tensor: torch.Tensor,
    pred_prob_tensor: torch.Tensor,
    patch_id: str,
    output_path: str,
    threshold: float = 0.5,
    metrics: Optional[Dict[str, float]] = None
):
    """
    Create a 5-panel comprehensive prediction evaluation figure:
    1. Original Aerial RGB
    2. Ground-Truth Boundary
    3. Predicted Probability Map
    4. Thresholded Binary Mask
    5. Multi-Class Error Overlay (Yellow=TP, Red=FP, Green=FN)
    """
    # Convert tensors to numpy
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    if gt_tensor.dim() == 4:
        gt_tensor = gt_tensor[0]
    if pred_prob_tensor.dim() == 4:
        pred_prob_tensor = pred_prob_tensor[0]

    img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    gt_np = gt_tensor.squeeze().cpu().numpy()
    pred_prob_np = pred_prob_tensor.squeeze().cpu().numpy()
    pred_bin_np = (pred_prob_np >= threshold).astype(np.uint8)

    overlay_np = create_prediction_overlay(img_np, gt_np, pred_bin_np)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), dpi=150)
    fig.patch.set_facecolor("#111111")

    # 1. Aerial RGB
    axes[0].imshow(img_np)
    axes[0].set_title("1. Original Aerial RGB", color="#EEEEEE", fontsize=12, fontweight="bold", pad=10)
    axes[0].axis("off")

    # 2. Ground-Truth
    axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("2. Ground-Truth Mask", color="#EEEEEE", fontsize=12, fontweight="bold", pad=10)
    axes[1].axis("off")

    # 3. Probability Heatmap
    im3 = axes[2].imshow(pred_prob_np, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("3. Predicted Probability", color="#EEEEEE", fontsize=12, fontweight="bold", pad=10)
    axes[2].axis("off")

    # 4. Thresholded Mask
    axes[3].imshow(pred_bin_np, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(f"4. Binary Prediction (th={threshold})", color="#EEEEEE", fontsize=12, fontweight="bold", pad=10)
    axes[3].axis("off")

    # 5. Overlay
    axes[4].imshow(overlay_np)
    axes[4].set_title("5. Error Overlay (Yellow=TP, Red=FP, Green=FN)", color="#EEEEEE", fontsize=12, fontweight="bold", pad=10)
    axes[4].axis("off")

    # Title & Subtitle
    subtitle = f"Patch: {patch_id}"
    if metrics:
        subtitle += f" | F1/Dice: {metrics.get('f1', 0.0):.3f} | IoU: {metrics.get('iou', 0.0):.3f} | Prec: {metrics.get('precision', 0.0):.3f} | Rec: {metrics.get('recall', 0.0):.3f}"
    
    fig.suptitle(f"Cadastral Boundary Extraction Baseline — {patch_id}", color="#FFFFFF", fontsize=15, fontweight="bold", y=0.98)
    fig.text(0.5, 0.90, subtitle, color="#AAAAAA", fontsize=11, ha="center")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.subplots_adjust(top=0.84, bottom=0.04, left=0.02, right=0.98, wspace=0.08)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def plot_training_history(history: Dict[str, List[float]], output_path: str):
    """
    Plot training and validation Loss, F1/Dice, and IoU curves over epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    fig.patch.set_facecolor("#181818")

    # 1. Loss
    axes[0].plot(epochs, history["train_loss"], "o-", color="#4CAF50", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "s--", color="#FF9800", label="Val Loss", linewidth=2)
    axes[0].set_title("Focal + Dice Loss", color="#EEEEEE", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", color="#CCCCCC")
    axes[0].set_ylabel("Loss", color="#CCCCCC")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()
    axes[0].tick_params(colors="#CCCCCC")

    # 2. F1 / Dice
    axes[1].plot(epochs, history["train_f1"], "o-", color="#2196F3", label="Train F1/Dice", linewidth=2)
    axes[1].plot(epochs, history["val_f1"], "s--", color="#E91E63", label="Val F1/Dice", linewidth=2)
    axes[1].set_title("F1-Score / Dice Coefficient", color="#EEEEEE", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", color="#CCCCCC")
    axes[1].set_ylabel("F1 Score", color="#CCCCCC")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()
    axes[1].tick_params(colors="#CCCCCC")

    # 3. IoU
    axes[2].plot(epochs, history["train_iou"], "o-", color="#9C27B0", label="Train IoU", linewidth=2)
    axes[2].plot(epochs, history["val_iou"], "s--", color="#00BCD4", label="Val IoU", linewidth=2)
    axes[2].set_title("Intersection over Union (IoU)", color="#EEEEEE", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch", color="#CCCCCC")
    axes[2].set_ylabel("IoU", color="#CCCCCC")
    axes[2].grid(True, linestyle="--", alpha=0.3)
    axes[2].legend()
    axes[2].tick_params(colors="#CCCCCC")

    fig.suptitle("SIH26012 Baseline Model Training History", color="#FFFFFF", fontsize=15, fontweight="bold", y=0.98)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
