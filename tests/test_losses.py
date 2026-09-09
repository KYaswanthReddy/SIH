"""
Tests for Focal Loss, Dice Loss, and Combined Focal+Dice Loss.
"""

import pytest
import torch
from ai.losses import FocalLoss, DiceLoss, CombinedFocalDiceLoss


def test_focal_loss():
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Perfect prediction
    target = torch.ones(2, 1, 64, 64)
    pred_perfect = torch.ones(2, 1, 64, 64) * 0.999
    loss_perfect = focal(pred_perfect, target)
    
    # Poor prediction
    pred_poor = torch.ones(2, 1, 64, 64) * 0.001
    loss_poor = focal(pred_poor, target)
    
    assert loss_perfect < loss_poor
    assert loss_perfect >= 0.0


def test_dice_loss():
    dice = DiceLoss(smooth=1.0)
    
    # Identical
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 10:30, 10:30] = 1.0
    pred_good = target.clone()
    loss_good = dice(pred_good, target)
    
    # Inverted
    pred_bad = 1.0 - target
    loss_bad = dice(pred_bad, target)
    
    assert loss_good < loss_bad
    assert abs(loss_good.item()) < 0.01


def test_combined_loss_backward():
    combined = CombinedFocalDiceLoss(focal_weight=0.5, dice_weight=0.5)
    
    # Raw logits leaf tensor requiring gradients
    raw_logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    pred = torch.sigmoid(raw_logits)
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 10:20, 10:20] = 1.0
    
    loss = combined(pred, target)
    assert loss.requires_grad
    loss.backward()
    assert raw_logits.grad is not None
