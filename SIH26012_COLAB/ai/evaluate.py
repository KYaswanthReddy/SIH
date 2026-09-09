"""
Evaluation and Inference Utilities for Cadastral Boundary Segmentation.
"""

import os
import json
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import SegmentationMetricsTracker, compute_binary_metrics
from .visualizer import visualize_predictions


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
    default_threshold: float = 0.5,
    num_visualizations: int = 6,
    vis_output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate model across a dataset and generate visual prediction samples.
    """
    model.eval()
    model.to(device)
    tracker = SegmentationMetricsTracker(thresholds=thresholds)

    saved_vis_count = 0
    vis_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            patch_ids = batch["patch_id"]

            preds = model(images)
            tracker.update(preds, masks)

            # Generate sample visualizations
            if vis_output_dir is not None and saved_vis_count < num_visualizations:
                for i in range(images.size(0)):
                    if saved_vis_count >= num_visualizations:
                        break
                    
                    # Only visualize samples with boundaries or interesting features
                    gt_sum = masks[i].sum().item()
                    if gt_sum > 200 or saved_vis_count == num_visualizations - 1:
                        p_id = patch_ids[i]
                        out_path = os.path.join(vis_output_dir, f"pred_eval_{saved_vis_count+1}_{p_id}.png")
                        
                        sample_metrics = compute_binary_metrics(
                            preds[i], masks[i], threshold=default_threshold
                        )
                        
                        visualize_predictions(
                            image_tensor=images[i],
                            gt_tensor=masks[i],
                            pred_prob_tensor=preds[i],
                            patch_id=p_id,
                            output_path=out_path,
                            threshold=default_threshold,
                            metrics=sample_metrics
                        )
                        vis_paths.append(out_path)
                        saved_vis_count += 1

    metrics = tracker.get_metrics(default_threshold=default_threshold)
    metrics["saved_visualizations"] = vis_paths
    return metrics
