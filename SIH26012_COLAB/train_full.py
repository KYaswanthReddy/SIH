"""
Full CadastreVision Model Training CLI.
Executes end-to-end training of the integrated architecture:
PMG + Domain Generalization + Connectivity Dual-Head + Residual U-Net Backbone.
"""

import os
import sys
import json
import time
import yaml
import torch
from torch.utils.data import DataLoader

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
from ai.dataset import CadastreDataset
from ai.transforms import get_train_transforms, get_val_transforms
from ai.models.full_model import CadastreUNetFull, FullCadastreLoss
from ai.metrics import SegmentationMetricsTracker
from ai.visualizer import plot_training_history, plot_loss_curves, plot_metrics_progression, visualize_predictions
from gis.pipeline import RasterToVectorGISPipeline
from gis.topology import validate_cadastral_topology


def load_config(config_path: str = "configs/full.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_full_model(config_path: str = "configs/full.yaml"):
    config = load_config(config_path)
    start_time = time.time()

    print("=" * 60, flush=True)
    print(" SIH26012 — FULL ARCHITECTURE TRAINING", flush=True)
    print(" (PMG + Domain Generalization + Connectivity Dual-Head)", flush=True)
    print("=" * 60, flush=True)

    # 1. Device
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    print(f"[Hardware] Using Compute Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[Hardware] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[Hardware] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB", flush=True)

    # 2. Datasets & DataLoaders
    data_cfg = config["data"]
    train_cfg = config["training"]
    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 2) if device.type == "cuda" else 0
    pin_memory = train_cfg.get("pin_memory", True) and (device.type == "cuda")

    train_ds = CadastreDataset(
        data_cfg["train_metadata"],
        processed_dir=data_cfg["processed_dir"],
        transform=get_train_transforms(config),
        max_samples=data_cfg.get("max_train_samples", None),
    )
    val_ds = CadastreDataset(
        data_cfg["val_metadata"],
        processed_dir=data_cfg["processed_dir"],
        transform=get_val_transforms(config),
        max_samples=data_cfg.get("max_val_samples", None),
    )
    test_ds = CadastreDataset(
        data_cfg["test_metadata"],
        processed_dir=data_cfg["processed_dir"],
        transform=get_val_transforms(config),
        max_samples=data_cfg.get("max_test_samples", None),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"[Data] Train: {len(train_ds)} patches ({len(train_loader)} batches) | Val: {len(val_ds)} ({len(val_loader)} batches) | Test: {len(test_ds)} ({len(test_loader)} batches)", flush=True)

    # 3. Model & Loss
    model_cfg = config["model"]
    model = CadastreUNetFull(
        in_channels=data_cfg.get("in_channels", 3),
        out_channels=data_cfg.get("out_channels", 1),
        features=model_cfg.get("encoder_channels", [32, 64, 128, 256, 512]),
        pmg_channels=model_cfg.get("pmg_channels", 128),
        num_connectivity_directions=model_cfg.get("connectivity", {}).get("num_directions", 8),
        dropout=model_cfg.get("dropout", 0.1),
        pmg_enabled=model_cfg.get("pmg", {}).get("enabled", True),
        dg_enabled=model_cfg.get("domain_generalization", {}).get("enabled", True),
        connectivity_enabled=model_cfg.get("connectivity", {}).get("enabled", True),
        dg_prob=model_cfg.get("domain_generalization", {}).get("prob", 0.5)
    ).to(device)

    print(f"[Model] Total Trainable Parameters: {model.count_parameters():,}", flush=True)

    loss_cfg = config.get("loss", {})
    loss_fn = FullCadastreLoss(
        boundary_loss_weight=loss_cfg.get("boundary_loss_weight", 1.0),
        connectivity_loss_weight=loss_cfg.get("connectivity_loss_weight", 0.5),
        focal_weight=loss_cfg.get("focal_weight", 0.5),
        dice_weight=loss_cfg.get("dice_weight", 0.5),
        focal_alpha=loss_cfg.get("focal", {}).get("alpha", 0.25),
        focal_gamma=loss_cfg.get("focal", {}).get("gamma", 2.0),
        connectivity_pos_weight=loss_cfg.get("connectivity_pos_weight", 5.0),
    ).to(device)

    # 4. Optimizer & Scheduler
    lr = float(train_cfg.get("learning_rate", 1e-4))
    wd = float(train_cfg.get("weight_decay", 1e-4))
    epochs = train_cfg.get("epochs", 15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    use_amp = train_cfg.get("mixed_precision", True) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[Trainer] PyTorch AMP Mixed Precision enabled with GradScaler.", flush=True)

    # 5. Output directories
    exp_cfg = config["experiment"]
    output_dir = exp_cfg.get("output_dir", "experiments/full")
    ckpt_dir = exp_cfg.get("checkpoint_dir", os.path.join(output_dir, "checkpoints"))
    metrics_dir = exp_cfg.get("metrics_dir", os.path.join(output_dir, "metrics"))
    preds_dir = exp_cfg.get("predictions_dir", exp_cfg.get("visualizations_dir", os.path.join(output_dir, "visualizations")))
    summary_dir = exp_cfg.get("summary_dir", os.path.join(output_dir, "summary"))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    # 6. Training Loop
    best_val_f1 = -1.0
    best_val_metrics = {}
    best_val_threshold = 0.5
    best_epoch = 0
    patience = train_cfg.get("early_stopping_patience", 4)
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_dice": [],
        "val_iou": [],
        "val_precision": [],
        "val_recall": [],
        "val_threshold": [],
        "lr": [],
        "time_seconds": []
    }

    eval_cfg = config.get("evaluation", {})
    val_thresholds = eval_cfg.get("thresholds", [0.2, 0.3, 0.4, 0.5])
    print(f"\n[Training] Starting {epochs} Epochs Training Loop (Val Thresholds: {val_thresholds})...\n", flush=True)

    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        print(f"Epoch {epoch:02d}/{epochs:02d}", flush=True)

        # Train with tqdm
        model.train()
        train_loss_total = 0.0
        pbar_train = tqdm(train_loader, desc=f"Train", leave=True, file=sys.stdout)
        for batch in pbar_train:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    preds = model(images, return_dict=True)
                    loss_dict = loss_fn(preds, masks)
                    loss = loss_dict["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(images, return_dict=True)
                loss_dict = loss_fn(preds, masks)
                loss = loss_dict["loss"]
                loss.backward()
                optimizer.step()

            batch_loss = loss.item()
            train_loss_total += batch_loss * images.size(0)
            pbar_train.set_postfix({"loss": f"{batch_loss:.4f}"})

        train_loss_avg = train_loss_total / len(train_ds)

        # Validation with threshold sweep
        model.eval()
        val_loss_total = 0.0
        tracker = SegmentationMetricsTracker(thresholds=val_thresholds)
        pbar_val = tqdm(val_loader, desc=f"Val  ", leave=True, file=sys.stdout)
        with torch.no_grad():
            for batch in pbar_val:
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        preds = model(images, return_dict=True)
                        loss_dict = loss_fn(preds, masks)
                else:
                    preds = model(images, return_dict=True)
                    loss_dict = loss_fn(preds, masks)

                val_loss_total += loss_dict["loss"].item() * images.size(0)
                tracker.update(preds["refined_prob"], masks)

        val_loss_avg = val_loss_total / len(val_ds)
        val_metrics_all = tracker.get_metrics()
        opt_th = val_metrics_all["optimal_threshold"]
        val_metrics = tracker.get_metrics(default_threshold=opt_th)
        val_metrics["selected_threshold"] = opt_th
        val_metrics["optimal_f1"] = val_metrics_all["optimal_f1"]

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        ep_duration = time.time() - ep_start

        # Record metrics at best validation threshold
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss_avg, 4))
        history["val_loss"].append(round(val_loss_avg, 4))
        history["val_f1"].append(round(val_metrics["f1"], 4))
        history["val_dice"].append(round(val_metrics.get("dice", val_metrics["f1"]), 4))
        history["val_iou"].append(round(val_metrics["iou"], 4))
        history["val_precision"].append(round(val_metrics["precision"], 4))
        history["val_recall"].append(round(val_metrics["recall"], 4))
        history["val_threshold"].append(round(opt_th, 2))
        history["lr"].append(current_lr)
        history["time_seconds"].append(round(ep_duration, 2))

        # Print detailed epoch summary
        print(f"\n[Epoch {epoch:02d}] (Best Val Threshold: {opt_th:.2f})", flush=True)
        print(f"Train Loss: {train_loss_avg:.4f}", flush=True)
        print(f"Val Loss:   {val_loss_avg:.4f}", flush=True)
        print(f"F1 (th={opt_th:.2f}): {val_metrics['f1']:.4f}", flush=True)
        print(f"IoU:        {val_metrics['iou']:.4f}", flush=True)
        print(f"Precision:  {val_metrics['precision']:.4f}", flush=True)
        print(f"Recall:     {val_metrics['recall']:.4f}", flush=True)
        print(f"Accuracy:   {val_metrics['accuracy']:.4f}", flush=True)
        print(f"Dice:       {val_metrics.get('dice', val_metrics['f1']):.4f}", flush=True)
        print(f"LR:         {current_lr:.2e}", flush=True)
        print(f"Time:       {ep_duration:.1f}s", flush=True)
        if device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            print(f"Peak VRAM:  {peak_mb:.1f} MB", flush=True)
        elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
            try:
                mps_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
                print(f"MPS Memory: {mps_mb:.1f} MB", flush=True)
            except Exception:
                pass

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_metrics": val_metrics,
            "selected_threshold": opt_th,
            "config": config
        }
        torch.save(checkpoint, os.path.join(ckpt_dir, "latest_checkpoint.pth"))

        if val_metrics["f1"] > best_val_f1:
            diff = val_metrics["f1"] - best_val_f1
            best_val_f1 = val_metrics["f1"]
            best_val_metrics = val_metrics
            best_val_threshold = opt_th
            best_epoch = epoch
            patience_counter = 0
            best_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"\n[Checkpoint] ⭐ New best model saved: {best_path} (Val F1: {best_val_f1:.4f} @ th={opt_th:.2f}, +{diff:.4f})\n", flush=True)
        else:
            patience_counter += 1
            print(f"[Early Stopping] Patience: {patience_counter}/{patience}\n", flush=True)
            if patience_counter >= patience:
                print(f"\n[Early Stopping Triggered] No improvement for {patience} consecutive epochs. Stopping at epoch {epoch}.\n", flush=True)
                break

    # Save training history and best val metrics
    history_file = os.path.join(metrics_dir, "training_history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    val_metrics_file = os.path.join(metrics_dir, "metrics_val.json")
    with open(val_metrics_file, "w") as f:
        json.dump(best_val_metrics, f, indent=2)

    # Generate and save loss and metrics progress plots
    try:
        loss_plot_path = os.path.join(preds_dir, "training_loss.png")
        plot_loss_curves(history, loss_plot_path)
        metrics_plot_path = os.path.join(preds_dir, "metrics_progress.png")
        plot_metrics_progression(history, metrics_plot_path)
    except Exception as e:
        print(f"[Warning] Could not generate plots: {e}", flush=True)

    # 7. Test Evaluation
    print("\n" + "=" * 60)
    print(f" EVALUATING ON HELD-OUT TEST SET ({len(test_ds)} PATCHES)")
    print("=" * 60)

    best_path = os.path.join(ckpt_dir, "best_model.pth")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Eval] Loaded best model weights from {best_path} (Epoch {best_epoch})")

    model.eval()
    test_thresholds = eval_cfg.get("thresholds", [0.2, 0.3, 0.4, 0.5])
    test_tracker = SegmentationMetricsTracker(thresholds=test_thresholds)
    pipeline = RasterToVectorGISPipeline(threshold=best_val_threshold, simplify_tolerance=0.5)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            preds = model(images, return_dict=True)
            test_tracker.update(preds["refined_prob"], masks)

            # Generate sample visualizations for first 8 test samples
            if idx < 8:
                patch_id = batch["patch_id"][0]
                img_t = images[0]
                gt_t = masks[0]
                pred_t = preds["refined_prob"][0]
                vis_path = os.path.join(preds_dir, f"prediction_{idx+1}_{patch_id}.png")
                visualize_predictions(img_t, gt_t, pred_t, patch_id=patch_id, output_path=vis_path, threshold=best_val_threshold)
                # Keep legacy filename copy for backward compatibility if needed
                legacy_path = os.path.join(preds_dir, f"test_vis_{idx+1}_{patch_id}.png")
                if not os.path.exists(legacy_path):
                    import shutil
                    shutil.copyfile(vis_path, legacy_path)

    test_metrics = test_tracker.get_metrics(default_threshold=best_val_threshold)
    test_metrics["selected_validation_threshold"] = best_val_threshold
    test_metrics["selected_from_validation"] = True
    test_metrics_file = os.path.join(metrics_dir, "metrics_test.json")
    with open(test_metrics_file, "w") as f:
        json.dump(test_metrics, f, indent=2)

    total_time = time.time() - start_time

    # Save comprehensive experiment summary
    experiment_summary = {
        "experiment_name": exp_cfg.get("name", "full_cadastre_network"),
        "status": "COMPLETED",
        "total_epochs_trained": len(history["epoch"]),
        "best_epoch": best_epoch,
        "selected_validation_threshold": best_val_threshold,
        "best_val_metrics": best_val_metrics,
        "test_metrics_at_selected_validation_threshold": test_metrics,
        "total_training_duration_seconds": round(total_time, 2),
        "total_training_duration_minutes": round(total_time / 60, 2),
        "hardware_device": str(device),
        "model_parameters": model.count_parameters(),
        "config": {
            "training": train_cfg,
            "data": {
                "image_size": data_cfg.get("image_size"),
                "max_train_samples": data_cfg.get("max_train_samples"),
                "max_val_samples": data_cfg.get("max_val_samples"),
                "max_test_samples": data_cfg.get("max_test_samples")
            },
            "model": model_cfg,
            "loss": loss_cfg
        }
    }
    summary_file = os.path.join(summary_dir, "experiment_summary.json")
    with open(summary_file, "w") as f:
        json.dump(experiment_summary, f, indent=2)

    abs_results_dir = os.path.abspath(output_dir)

    print("\n" + "=" * 60)
    print(" TRAINING & EVALUATION COMPLETE")
    print("=" * 60)
    print(f"• Total Training Duration: {total_time / 60:.2f} minutes ({total_time:.1f}s)")
    print(f"• Best Epoch: {best_epoch} | Best Val F1: {best_val_f1:.4f} (Selected Val Threshold: {best_val_threshold:.2f})")
    print(f"• Test F1 (at Val th={best_val_threshold:.2f}): {test_metrics['f1']:.4f} | IoU: {test_metrics['iou']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"• Test Optimal F1 across all thresholds: {test_metrics.get('optimal_f1', test_metrics['f1']):.4f} (at th={test_metrics.get('optimal_threshold', best_val_threshold):.2f})")
    print(f"\n============================================================")
    print(f" ALL RESULTS SAVED IN DEDICATED DIRECTORY:")
    print(f" {abs_results_dir}")
    print(f"============================================================")
    print(f"├── metrics/")
    print(f"│   ├── training_history.json")
    print(f"│   ├── metrics_val.json")
    print(f"│   └── metrics_test.json")
    print(f"├── checkpoints/")
    print(f"│   └── best_model.pth")
    print(f"├── visualizations/")
    print(f"│   ├── prediction_*.png")
    print(f"│   ├── training_loss.png")
    print(f"│   └── metrics_progress.png")
    print(f"└── summary/")
    print(f"    └── experiment_summary.json\n")


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/full.yaml"
    train_full_model(config_file)
