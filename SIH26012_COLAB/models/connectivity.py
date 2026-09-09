"""
Connectivity-Aware Cadastral Boundary Segmentation Module.
Predicts 8-neighborhood directional connectivity to enforce boundary continuity and prevent fragmentation.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_baseline import DoubleConv, Down, Up
from .pmg import PMGModule
from .domain_generalization import MixStyleDomainPerturbation


# 8-neighborhood directional offsets: [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]


def generate_connectivity_targets(boundary_mask: torch.Tensor) -> torch.Tensor:
    """
    Generate 8-channel ground truth connectivity target maps from a binary boundary mask.
    
    Args:
        boundary_mask: Tensor of shape (B, 1, H, W) in {0.0, 1.0}
        
    Returns:
        connectivity_targets: Tensor of shape (B, 8, H, W) in {0.0, 1.0}
    """
    B, _, H, W = boundary_mask.shape
    device = boundary_mask.device
    conn_targets = torch.zeros((B, 8, H, W), device=device, dtype=torch.float32)

    # Pad boundary mask by 1 pixel with reflection or zero
    padded = F.pad(boundary_mask, (1, 1, 1, 1), mode="constant", value=0.0)

    for k, (dr, dc) in enumerate(NEIGHBOR_OFFSETS):
        # Shifted slice corresponding to (r + dr, c + dc)
        r_start, r_end = 1 + dr, 1 + dr + H
        c_start, c_end = 1 + dc, 1 + dc + W
        shifted = padded[:, :, r_start:r_end, c_start:c_end]
        
        # Connectivity exists if BOTH current pixel and neighbor pixel are boundaries
        conn_targets[:, k:k+1, :, :] = boundary_mask * shifted

    return conn_targets


class ConnectivityHead(nn.Module):
    """
    Predicts 8-channel directional connectivity maps from decoder feature representations.
    """

    def __init__(self, in_channels: int = 32, num_directions: int = 8):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_directions, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 8-channel connectivity probabilities in [0.0, 1.0]."""
        return torch.sigmoid(self.conv(x))


class ConnectivityLoss(nn.Module):
    """
    Directional affinity loss weighted towards boundary junction and line continuity.
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred_conn: torch.Tensor, target_conn: torch.Tensor) -> torch.Tensor:
        """
        pred_conn: (B, 8, H, W)
        target_conn: (B, 8, H, W)
        """
        # Binary cross entropy with focus on non-empty connectivity channels
        pred_c = torch.clamp(pred_conn, self.eps, 1.0 - self.eps)
        bce = -(target_conn * torch.log(pred_c) + (1.0 - target_conn) * torch.log(1.0 - pred_c))
        
        # Weight positive connectivity edges
        pos_weight = torch.where(target_conn == 1.0, 5.0, 1.0)
        weighted_bce = (bce * pos_weight).mean()
        
        return weighted_bce
