"""
Loss Functions for Cadastral Boundary Segmentation: Focal Loss, Dice Loss, and Combined Loss.
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for severe class imbalance in sparse boundary detection.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean", eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Probability map in [0, 1] or raw logits, shape (B, 1, H, W)
            target: Binary ground truth {0, 1}, shape (B, 1, H, W)
        """
        # Clamp predictions for numerical stability
        pred = torch.clamp(pred, min=self.eps, max=1.0 - self.eps)

        pt = torch.where(target == 1.0, pred, 1.0 - pred)
        alpha_t = torch.where(target == 1.0, self.alpha, 1.0 - self.alpha)

        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        loss = -focal_weight * torch.log(pt)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """
    Soft Dice Loss measuring boundary spatial overlap.
    Dice = 1 - (2 * |X cap Y| + smooth) / (|X| + |Y| + smooth)
    """

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities in [0, 1], shape (B, 1, H, W)
            target: Ground truth {0, 1}, shape (B, 1, H, W)
        """
        # Flatten spatial dimensions per batch item: (B, H*W)
        b = pred.size(0)
        pred_flat = pred.view(b, -1)
        target_flat = target.view(b, -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        cardinality = (pred_flat + target_flat).sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        dice_loss = 1.0 - dice_score

        return dice_loss.mean()


class CombinedFocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice Loss for balanced pixel-level and regional boundary optimization.
    Loss = w_focal * FocalLoss + w_dice * DiceLoss
    """

    def __init__(
        self,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        alpha: float = 0.25,
        gamma: float = 2.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f_loss = self.focal_loss(pred, target)
        d_loss = self.dice_loss(pred, target)
        return self.focal_weight * f_loss + self.dice_weight * d_loss


def build_loss(config: Dict[str, Any]) -> CombinedFocalDiceLoss:
    """Factory function for loss setup."""
    loss_cfg = config.get("loss", {})
    return CombinedFocalDiceLoss(
        focal_weight=loss_cfg.get("focal_weight", 0.5),
        dice_weight=loss_cfg.get("dice_weight", 0.5),
        alpha=loss_cfg.get("focal", {}).get("alpha", 0.25),
        gamma=loss_cfg.get("focal", {}).get("gamma", 2.0),
        smooth=loss_cfg.get("dice", {}).get("smooth", 1.0),
    )
