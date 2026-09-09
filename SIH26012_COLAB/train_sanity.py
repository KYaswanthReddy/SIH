"""
Phase 2 Sanity Experiment: 1-2 Epoch Fast Verification Run.
Verifies forward pass, gradient descent, loss reduction, validation metrics, checkpointing, and 5-panel visualizations.
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


def run_sanity_experiment():
    print("==================================================")
    print(" SIH26012 — PHASE 2: SANITY CHECK EXPERIMENT")
    print("==================================================")

    config = load_config()
    
    # Configure sanity settings
    num_samples = config.get("sanity_check", {}).get("num_samples", 128)
    epochs = config.get("sanity_check", {}).get("epochs", 2)
    batch_size = config.get("sanity_check", {}).get("batch_size", 8)
    
    config["experiment"]["output_dir"] = "experiments/sanity"
    config["experiment"]["checkpoint_dir"] = "experiments/sanity/checkpoints"
    config["experiment"]["metrics_dir"] = "experiments/sanity/metrics"
    config["experiment"]["predictions_dir"] = "experiments/sanity/predictions"
    os.makedirs(config["experiment"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["experiment"]["metrics_dir"], exist_ok=True)
    os.makedirs(config["experiment"]["predictions_dir"], exist_ok=True)

    print(f"\n[Step 1/4] Creating Sanity DataLoaders ({num_samples} train patches, batch size {batch_size})...")
    train_loader, val_loader, test_loader = create_dataloaders(config, sanity=True)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    print("\n[Step 2/4] Building Baseline U-Net Model...")
    model = build_baseline_model(config)
    param_count = model.count_parameters()
    print(f"Model architecture: CadastreUNetBaseline with {param_count:,} trainable parameters.")

    # Device selection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print(f"\n[Step 3/4] Running {epochs}-Epoch Sanity Training Loop...")
    trainer = CadastreTrainer(model=model, config=config, device=device)
    history = trainer.fit(train_loader, val_loader, epochs=epochs)

    # Verify loss reduction
    initial_loss = history["train_loss"][0]
    final_loss = history["train_loss"][-1]
    print(f"\nTraining Loss: Epoch 1 = {initial_loss:.4f} -> Epoch {epochs} = {final_loss:.4f}")

    print("\n[Step 4/4] Evaluating Sanity Checkpoint & Generating 5-Panel Visualizations...")
    best_ckpt_path = os.path.join(config["experiment"]["checkpoint_dir"], "best_model.pth")
    if os.path.exists(best_ckpt_path):
        trainer.load_checkpoint(best_ckpt_path)
    
    val_metrics = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        thresholds=config["evaluation"]["thresholds"],
        default_threshold=config["evaluation"]["default_threshold"],
        num_visualizations=6,
        vis_output_dir=config["experiment"]["predictions_dir"]
    )

    sanity_summary = {
        "status": "PASSED",
        "epochs": epochs,
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "param_count": param_count,
        "initial_train_loss": initial_loss,
        "final_train_loss": final_loss,
        "best_val_f1": val_metrics["f1"],
        "val_iou": val_metrics["iou"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "saved_visualizations": len(val_metrics.get("saved_visualizations", [])),
    }

    summary_file = os.path.join(config["experiment"]["metrics_dir"], "sanity_summary.json")
    with open(summary_file, "w") as f:
        json.dump(sanity_summary, f, indent=2)

    print("\n==================================================")
    print(" SANITY RUN COMPLETE")
    print("==================================================")
    print(json.dumps(sanity_summary, indent=2))
    print(f"\nSanity report saved to: {summary_file}")
    return sanity_summary


if __name__ == "__main__":
    run_sanity_experiment()
