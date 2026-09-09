"""
Standalone Model Evaluation CLI for Cadastral Segmentation.
Evaluates best model checkpoint on Validation and Test splits with multi-threshold analysis and 5-panel visualizations.
"""

import os
import sys
import json
import yaml
import torch

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai.dataset import create_dataloaders
from ai.models.unet_baseline import build_baseline_model
from ai.trainer import CadastreTrainer
from ai.evaluate import evaluate_model


def load_config(config_path: str = "configs/baseline_train_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("==================================================")
    print(" SIH26012 — BASELINE MODEL EVALUATION")
    print("==================================================")

    config = load_config()
    
    # 1. Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Evaluation device: {device}")

    # 2. DataLoaders
    _, val_loader, test_loader = create_dataloaders(config, sanity=False)

    # 3. Model
    model = build_baseline_model(config)
    best_model_path = os.path.join(config["experiment"]["checkpoint_dir"], "best_model.pth")
    
    if not os.path.exists(best_model_path):
        # Fallback to latest or sanity checkpoint
        best_model_path = os.path.join("experiments/sanity/checkpoints", "best_model.pth")
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"No trained checkpoint found at {best_model_path}")
            
    print(f"Loading checkpoint: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 4. Evaluate Validation Set
    print("\n--- Evaluating Validation Set (600 patches) ---")
    val_metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        thresholds=config["evaluation"]["thresholds"],
        default_threshold=config["evaluation"]["default_threshold"],
        num_visualizations=4,
        vis_output_dir=os.path.join(config["experiment"]["predictions_dir"], "val")
    )
    val_out = os.path.join(config["experiment"]["metrics_dir"], "metrics_val.json")
    with open(val_out, "w") as f:
        json.dump(val_metrics, f, indent=2)
    print(f"Val F1: {val_metrics['f1']:.4f} | IoU: {val_metrics['iou']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}")

    # 5. Evaluate Test Set
    print("\n--- Evaluating Test Set (600 patches) ---")
    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        thresholds=config["evaluation"]["thresholds"],
        default_threshold=config["evaluation"]["default_threshold"],
        num_visualizations=4,
        vis_output_dir=os.path.join(config["experiment"]["predictions_dir"], "test")
    )
    test_out = os.path.join(config["experiment"]["metrics_dir"], "metrics_test.json")
    with open(test_out, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Test F1: {test_metrics['f1']:.4f} | IoU: {test_metrics['iou']:.4f} | Prec: {test_metrics['precision']:.4f} | Rec: {test_metrics['recall']:.4f}")

    print("\n==================================================")
    print(f"Validation metrics saved: {val_out}")
    print(f"Test metrics saved: {test_out}")
    print("==================================================")


if __name__ == "__main__":
    main()
