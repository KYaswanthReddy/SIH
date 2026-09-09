"""
Segmentation Evaluation Metrics: Precision, Recall, F1/Dice, IoU, Pixel Accuracy, and Boundary Analysis.
"""

from typing import Dict, Any, List, Union
import numpy as np
import torch


def compute_binary_metrics(
    pred_prob: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    eps: float = 1e-7
) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1 (Dice), IoU, and Accuracy given predicted probabilities and binary ground truth.
    """
    if isinstance(pred_prob, torch.Tensor):
        pred_prob = pred_prob.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Flatten arrays
    pred_bin = (pred_prob >= threshold).astype(np.uint8).ravel()
    gt_bin = (target > 0.5).astype(np.uint8).ravel()

    tp = np.sum((pred_bin == 1) & (gt_bin == 1))
    fp = np.sum((pred_bin == 1) & (gt_bin == 0))
    fn = np.sum((pred_bin == 0) & (gt_bin == 1))
    tn = np.sum((pred_bin == 0) & (gt_bin == 0))

    precision = float(tp / (tp + fp + eps))
    recall = float(tp / (tp + fn + eps))
    f1 = float(2.0 * tp / (2.0 * tp + fp + fn + eps))
    iou = float(tp / (tp + fp + fn + eps))
    accuracy = float((tp + tn) / (tp + tn + fp + fn + eps))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": f1,
        "iou": iou,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "threshold": threshold,
    }


class SegmentationMetricsTracker:
    """
    Accumulates true/false positives and negatives across batches for an entire validation/test run.
    """

    def __init__(self, thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7], eps: float = 1e-7):
        self.thresholds = thresholds
        self.eps = eps
        self.reset()

    def reset(self):
        self.stats = {
            th: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for th in self.thresholds
        }
        self.sample_count = 0

    def update(self, pred_prob: torch.Tensor, target: torch.Tensor):
        """
        Update tracker with batch predictions and targets.
        """
        pred_np = pred_prob.detach().cpu().numpy()
        gt_np = (target.detach().cpu().numpy() > 0.5).astype(np.uint8)

        self.sample_count += pred_np.shape[0]

        for th in self.thresholds:
            pred_bin = (pred_np >= th).astype(np.uint8)
            tp = int(np.sum((pred_bin == 1) & (gt_np == 1)))
            fp = int(np.sum((pred_bin == 1) & (gt_np == 0)))
            fn = int(np.sum((pred_bin == 0) & (gt_np == 1)))
            tn = int(np.sum((pred_bin == 0) & (gt_np == 0)))

            self.stats[th]["tp"] += tp
            self.stats[th]["fp"] += fp
            self.stats[th]["fn"] += fn
            self.stats[th]["tn"] += tn

    def get_metrics(self, default_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Compute aggregate metrics across the entire dataset.
        """
        threshold_results = {}
        best_f1 = -1.0
        best_th = default_threshold

        for th in self.thresholds:
            s = self.stats[th]
            tp, fp, fn, tn = s["tp"], s["fp"], s["fn"], s["tn"]
            
            prec = tp / (tp + fp + self.eps)
            rec = tp / (tp + fn + self.eps)
            f1 = 2.0 * tp / (2.0 * tp + fp + fn + self.eps)
            iou = tp / (tp + fp + fn + self.eps)
            acc = (tp + tn) / (tp + tn + fp + fn + self.eps)

            metrics_th = {
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "dice": round(float(f1), 4),
                "iou": round(float(iou), 4),
                "accuracy": round(float(acc), 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
            threshold_results[f"th_{th:.2f}"] = metrics_th
            
            if f1 > best_f1:
                best_f1 = f1
                best_th = th

        # Return primary metrics at default_threshold + best threshold summary
        def_s = self.stats.get(default_threshold, self.stats[self.thresholds[len(self.thresholds)//2]])
        def_tp, def_fp, def_fn, def_tn = def_s["tp"], def_s["fp"], def_s["fn"], def_s["tn"]

        primary = {
            "precision": round(float(def_tp / (def_tp + def_fp + self.eps)), 4),
            "recall": round(float(def_tp / (def_tp + def_fn + self.eps)), 4),
            "f1": round(float(2.0 * def_tp / (2.0 * def_tp + def_fp + def_fn + self.eps)), 4),
            "dice": round(float(2.0 * def_tp / (2.0 * def_tp + def_fp + def_fn + self.eps)), 4),
            "iou": round(float(def_tp / (def_tp + def_fp + def_fn + self.eps)), 4),
            "accuracy": round(float((def_tp + def_tn) / (def_tp + def_tn + def_fp + def_fn + self.eps)), 4),
            "tp": def_tp,
            "fp": def_fp,
            "fn": def_fn,
            "tn": def_tn,
            "optimal_threshold": best_th,
            "optimal_f1": round(float(best_f1), 4),
            "total_samples": self.sample_count,
            "threshold_sweep": threshold_results,
        }
        return primary


def compute_all_metrics(pred_prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    return compute_binary_metrics(pred_prob, target, threshold=threshold)
