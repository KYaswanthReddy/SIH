"""
Tests for Segmentation Metrics (Precision, Recall, F1/Dice, IoU).
"""

import pytest
import numpy as np
import torch
from ai.metrics import compute_binary_metrics, SegmentationMetricsTracker


def test_compute_binary_metrics():
    # 1. Perfect match
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[20:40, 20:40] = 1
    pred = gt.astype(np.float32)
    
    m = compute_binary_metrics(pred, gt, threshold=0.5)
    assert abs(m["f1"] - 1.0) < 1e-4
    assert abs(m["iou"] - 1.0) < 1e-4
    assert abs(m["precision"] - 1.0) < 1e-4
    assert abs(m["recall"] - 1.0) < 1e-4

    # 2. Complete mismatch
    pred_zeros = np.zeros((100, 100), dtype=np.float32)
    m2 = compute_binary_metrics(pred_zeros, gt, threshold=0.5)
    assert m2["f1"] < 1e-3
    assert m2["iou"] < 1e-3
    assert m2["recall"] < 1e-3


def test_metrics_tracker():
    tracker = SegmentationMetricsTracker(thresholds=[0.3, 0.5, 0.7])
    
    # Batch 1
    pred_batch = torch.zeros(2, 1, 32, 32)
    pred_batch[0, 0, 5:15, 5:15] = 0.8
    gt_batch = torch.zeros(2, 1, 32, 32)
    gt_batch[0, 0, 5:15, 5:15] = 1.0
    
    tracker.update(pred_batch, gt_batch)
    res = tracker.get_metrics(default_threshold=0.5)
    
    assert "f1" in res
    assert "iou" in res
    assert "threshold_sweep" in res
    assert res["total_samples"] == 2
    assert res["f1"] > 0.9
