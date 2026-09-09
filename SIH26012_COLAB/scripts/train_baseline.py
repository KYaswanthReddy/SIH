"""
Main Baseline Cadastral Segmentation Model Training CLI.
Executes full 15-20 epoch training across all 2,500 train patches and 600 validation patches.
"""

import os
import sys
import json
import time
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
    print(" SIH26012 — PHASE 2: BASELINE MODEL TRAINING")
    print("==================================================")
    start_time = time.time()
    
    config = load_config()
    
    # 1. DataLoaders
    print("\n[Step 1/4] Building Full DataLoaders (Train: 2,500, Val: 600, Test: 600)...")
    train_loader, val_loader, test_loader = create_dataloaders(config, sanity=False)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # 2. Model
    print("\n[Step 2/4] Initializing CadastreUNetBaseline Model...")
    model = build_baseline_model(config)
    print(f"Total trainable parameters: {model.count_parameters():,}")

    # 3. Training
    print(f"\n[Step 3/4] Launching {config['training']['epochs']} Epoch Training...")
    trainer = CadastreTrainer(model=model, config=config)
    history = trainer.fit(train_loader, val_loader)

    # 4. Evaluation on Test Set
    print("\n[Step 4/4] Evaluating Best Model on Held-Out Test Set (600 patches)...")
    best_model_path = os.path.join(config["experiment"]["checkpoint_dir"], "best_model.pth")
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)
    
    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=trainer.device,
        thresholds=config["evaluation"]["thresholds"],
        default_threshold=config["evaluation"]["default_threshold"],
        num_visualizations=8,
        vis_output_dir=config["experiment"]["predictions_dir"]
    )

    metrics_test_file = os.path.join(config["experiment"]["metrics_dir"], "metrics_test.json")
    with open(metrics_test_file, "w") as f:
        json.dump(test_metrics, f, indent=2)

    total_duration = time.time() - start_time
    print("\n" + "=" * 60)
    print(" PHASE 2 BASELINE TRAINING COMPLETE: FINAL SUMMARY")
    print("=" * 60)
    print(f"• Total Execution Time: {total_duration / 60:.2f} minutes ({total_duration:.1f}s)")
    print(f"• Best Epoch: {trainer.best_epoch} (Early stopping patience: {config['training']['early_stopping_patience']})")
    print(f"• Best Validation F1 / Dice: {trainer.best_val_f1:.4f}")
    print(f"• Checkpoint Location: {best_model_path}")
    print(f"• Latest Checkpoint: {os.path.join(config['experiment']['checkpoint_dir'], 'latest_checkpoint.pth')}")
    print(f"• Visual Predictions: {config['experiment']['predictions_dir']}")
    
    print("\n--- Epoch-by-Epoch Metric History ---")
    history_df = pd.DataFrame(history)
    cols = ["epoch", "train_loss", "val_loss", "val_f1", "val_dice", "val_iou", "val_precision", "val_recall", "lr", "time_seconds"]
    available_cols = [c for c in cols if c in history_df.columns]
    print(history_df[available_cols].to_string(index=False))

    print("\n--- Final Held-Out Test Set Evaluation (600 Patches) ---")
    print(f"• Test F1 Score:   {test_metrics['f1']:.4f}")
    print(f"• Test Dice Score: {test_metrics.get('dice', test_metrics['f1']):.4f}")
    print(f"• Test IoU:        {test_metrics['iou']:.4f}")
    print(f"• Test Precision:  {test_metrics['precision']:.4f}")
    print(f"• Test Recall:     {test_metrics['recall']:.4f}")
    print(f"• Test Accuracy:   {test_metrics.get('accuracy', 0.0):.4f}")
    print(f"• Test Metrics File: {metrics_test_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
