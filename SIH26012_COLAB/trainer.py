"""
Training Engine for Cadastral Boundary Segmentation with AMP, Early Stopping, and Metrics Tracking.
"""

import os
import time
import json
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .losses import CombinedFocalDiceLoss, build_loss
from .metrics import SegmentationMetricsTracker
from .visualizer import plot_training_history, visualize_predictions


class CadastreTrainer:
    """
    Handles model training, validation, checkpointing, AMP scaling, and metric evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        self.config = config
        
        # 1. Device selection
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        print(f"[Trainer] Using compute device: {self.device} ({self.model.count_parameters():,} trainable parameters)")

        # 2. Loss function
        self.loss_fn = loss_fn if loss_fn is not None else build_loss(config)
        self.loss_fn.to(self.device)

        # 3. Optimizer
        train_cfg = config["training"]
        lr = float(train_cfg.get("learning_rate", 1e-4))
        wd = float(train_cfg.get("weight_decay", 1e-4))
        
        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd
        )

        # 4. Scheduler
        self.scheduler = scheduler
        self.epochs = train_cfg.get("epochs", 15)
        if self.scheduler is None and train_cfg.get("scheduler") == "CosineAnnealingLR":
            min_lr = float(train_cfg.get("min_lr", 1e-6))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=min_lr
            )

        # 5. Mixed Precision AMP
        self.use_amp = train_cfg.get("mixed_precision", True) and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            print("[Trainer] Mixed Precision (PyTorch AMP) enabled with GradScaler.")

        # 6. Paths & Early Stopping
        self.checkpoint_dir = config.get("experiment", {}).get("checkpoint_dir", "experiments/baseline/checkpoints")
        self.metrics_dir = config.get("experiment", {}).get("metrics_dir", "experiments/baseline/metrics")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.early_stopping_patience = train_cfg.get("early_stopping_patience", 3)
        self.best_val_f1 = -1.0
        self.best_epoch = 0
        self.patience_counter = 0

        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_f1": [],
            "train_dice": [],
            "train_iou": [],
            "val_loss": [],
            "val_f1": [],
            "val_dice": [],
            "val_iou": [],
            "val_precision": [],
            "val_recall": [],
            "lr": [],
            "time_seconds": [],
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, Dict[str, Any]]:
        self.model.train()
        total_loss = 0.0
        tracker = SegmentationMetricsTracker()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{self.epochs:02d} [Train]")
        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    preds = self.model(images)
                    loss = self.loss_fn(preds, masks)
                self.scaler.scale(loss).backward()
                if self.config["training"].get("gradient_clip", 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["gradient_clip"])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(images)
                loss = self.loss_fn(preds, masks)
                loss.backward()
                if self.config["training"].get("gradient_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["training"]["gradient_clip"])
                self.optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss * images.size(0)
            tracker.update(preds, masks)

            pbar.set_postfix({"loss": f"{batch_loss:.4f}"})

        avg_loss = total_loss / len(dataloader.dataset)
        metrics = tracker.get_metrics()
        return avg_loss, metrics

    def validate_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, Dict[str, Any]]:
        self.model.eval()
        total_loss = 0.0
        tracker = SegmentationMetricsTracker()

        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d}/{self.epochs:02d} [Val]")
            for batch in pbar:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        preds = self.model(images)
                        loss = self.loss_fn(preds, masks)
                else:
                    preds = self.model(images)
                    loss = self.loss_fn(preds, masks)

                total_loss += loss.item() * images.size(0)
                tracker.update(preds, masks)

        avg_loss = total_loss / len(dataloader.dataset)
        metrics = tracker.get_metrics()
        return avg_loss, metrics

    def save_checkpoint(self, filename: str, epoch: int, val_metrics: Dict[str, Any]):
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "val_metrics": val_metrics,
            "best_val_f1": self.best_val_f1,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)
        return filepath

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"[Trainer] Loaded checkpoint from {filepath} (Epoch {checkpoint.get('epoch', 0)})")
        return checkpoint

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: Optional[int] = None) -> Dict[str, Any]:
        total_epochs = epochs if epochs is not None else self.epochs
        print(f"\n==================================================")
        print(f" Starting Training: {total_epochs} Epochs on {len(train_loader.dataset)} Train / {len(val_loader.dataset)} Val Patches")
        print(f"==================================================")

        start_time = time.time()
        for epoch in range(1, total_epochs + 1):
            ep_start = time.time()
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Step scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()

            ep_duration = time.time() - ep_start

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["train_f1"].append(round(train_metrics["f1"], 4))
            self.history["train_dice"].append(round(train_metrics.get("dice", train_metrics["f1"]), 4))
            self.history["train_iou"].append(round(train_metrics["iou"], 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_f1"].append(round(val_metrics["f1"], 4))
            self.history["val_dice"].append(round(val_metrics.get("dice", val_metrics["f1"]), 4))
            self.history["val_iou"].append(round(val_metrics["iou"], 4))
            self.history["val_precision"].append(round(val_metrics["precision"], 4))
            self.history["val_recall"].append(round(val_metrics["recall"], 4))
            self.history["lr"].append(current_lr)
            self.history["time_seconds"].append(round(ep_duration, 2))

            print(
                f"Epoch {epoch:02d}/{total_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} (F1: {train_metrics['f1']:.4f}, Dice: {train_metrics.get('dice', train_metrics['f1']):.4f}) | "
                f"Val Loss: {val_loss:.4f} (F1: {val_metrics['f1']:.4f}, Dice: {val_metrics.get('dice', val_metrics['f1']):.4f}, IoU: {val_metrics['iou']:.4f}, Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}) | "
                f"LR: {current_lr:.2e} | Time: {ep_duration:.1f}s"
            )

            # Save latest checkpoint
            self.save_checkpoint("latest_checkpoint.pth", epoch, val_metrics)

            # Check best model
            if val_metrics["f1"] > self.best_val_f1:
                diff = val_metrics["f1"] - self.best_val_f1
                self.best_val_f1 = val_metrics["f1"]
                self.best_epoch = epoch
                self.patience_counter = 0
                best_path = self.save_checkpoint("best_model.pth", epoch, val_metrics)
                print(f"  >>> ⭐ New best model saved! (Val F1: {self.best_val_f1:.4f}, +{diff:.4f}) -> {best_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n[Early Stopping] No improvement in validation F1 for {self.early_stopping_patience} consecutive epochs. Stopping at epoch {epoch}.")
                    break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s. Best Epoch: {self.best_epoch} with Val F1: {self.best_val_f1:.4f}")

        # Save history
        history_json_path = os.path.join(self.metrics_dir, "training_history.json")
        with open(history_json_path, "w") as f:
            json.dump(self.history, f, indent=2)
            
        history_csv_path = os.path.join(self.metrics_dir, "training_history.csv")
        pd.DataFrame(self.history).to_csv(history_csv_path, index=False)

        # Plot training curves
        plot_path = os.path.join(self.metrics_dir, "training_curves.png")
        plot_training_history(self.history, plot_path)
        print(f"Saved training history to {history_json_path} and curves to {plot_path}")

        return self.history
