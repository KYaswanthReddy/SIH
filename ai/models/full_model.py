"""
Full Cadastral Boundary Extraction Network (CadNet + PMG + DG + Connectivity).
Combines Multiscale Feature Pyramid, Domain Generalization, Dual-Head Boundary and Connectivity Prediction.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_baseline import DoubleConv, Down, Up
from .pmg import PMGModule
from .domain_generalization import MixStyleDomainPerturbation
from .connectivity import ConnectivityHead, ConnectivityLoss, generate_connectivity_targets
from ..losses import CombinedFocalDiceLoss


class CadastreUNetFull(nn.Module):
    """
    Full Architecture for SIH26012 Cadastral Feature Extraction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256, 512],
        pmg_channels: int = 128,
        num_connectivity_directions: int = 8,
        dropout: float = 0.1,
        pmg_enabled: bool = True,
        dg_enabled: bool = True,
        connectivity_enabled: bool = True,
        dg_prob: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pmg_enabled = pmg_enabled
        self.dg_enabled = dg_enabled
        self.connectivity_enabled = connectivity_enabled

        f = features
        # Encoder
        self.inc = DoubleConv(in_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        self.down4 = Down(f[3], f[4])

        # Domain Generalization Modules
        if self.dg_enabled:
            self.dg_stage1 = MixStyleDomainPerturbation(p=dg_prob, alpha=0.2)
            self.dg_stage2 = MixStyleDomainPerturbation(p=dg_prob, alpha=0.2)
        else:
            self.dg_stage1 = nn.Identity()
            self.dg_stage2 = nn.Identity()

        # PMG Module
        if self.pmg_enabled:
            self.pmg = PMGModule(in_channels_list=[f[1], f[2], f[3], f[4]], out_channels=pmg_channels)
            self.pmg_adapters = nn.ModuleList([
                nn.Conv2d(pmg_channels, f[i], kernel_size=1) for i in [1, 2, 3, 4]
            ])
        else:
            self.pmg = None

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Decoder
        self.up1 = Up(f[4], f[3])
        self.up2 = Up(f[3], f[2])
        self.up3 = Up(f[2], f[1])
        self.up4 = Up(f[1], f[0])

        # Primary Boundary Segmentation Head
        self.boundary_head = nn.Conv2d(f[0], out_channels, kernel_size=1)

        # Secondary 8-Neighborhood Connectivity Head
        if self.connectivity_enabled:
            self.connectivity_head = ConnectivityHead(in_channels=f[0], num_directions=num_connectivity_directions)
            # Connectivity-Refined Boundary Fusion Layer
            self.fusion_refine = nn.Sequential(
                nn.Conv2d(out_channels + num_connectivity_directions, f[0], kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(f[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(f[0], out_channels, kernel_size=1)
            )
        else:
            self.connectivity_head = None
            self.fusion_refine = None

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        Returns:
            If return_dict=False: refined boundary probability map (B, 1, H, W)
            If return_dict=True: dict with 'boundary_prob', 'connectivity_prob', 'refined_prob'
        """
        # 1. Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        if self.dg_enabled:
            x2 = self.dg_stage1(x2)
            
        x3 = self.down2(x2)
        if self.dg_enabled:
            x3 = self.dg_stage2(x3)

        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 2. PMG Module
        if self.pmg_enabled and self.pmg is not None:
            enhanced = self.pmg([x2, x3, x4, x5])
            x2 = x2 + self.pmg_adapters[0](enhanced[0])
            x3 = x3 + self.pmg_adapters[1](enhanced[1])
            x4 = x4 + self.pmg_adapters[2](enhanced[2])
            x5 = x5 + self.pmg_adapters[3](enhanced[3])

        x5 = self.drop(x5)

        # 3. Decoder
        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        # 4. Heads
        boundary_logits = self.boundary_head(d4)
        boundary_prob = torch.sigmoid(boundary_logits)

        if self.connectivity_enabled and self.connectivity_head is not None:
            conn_prob = self.connectivity_head(d4)
            # Refinement fusion
            fused_input = torch.cat([boundary_prob, conn_prob], dim=1)
            refined_logits = self.fusion_refine(fused_input)
            refined_prob = torch.sigmoid(refined_logits)
        else:
            conn_prob = None
            refined_prob = boundary_prob

        if return_dict:
            return {
                "boundary_prob": boundary_prob,
                "connectivity_prob": conn_prob,
                "refined_prob": refined_prob,
            }
        return refined_prob

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FullCadastreLoss(nn.Module):
    """
    Combined Loss for Full Model: Focal + Dice (Boundary) + Directional Affinity (Connectivity).
    """

    def __init__(
        self,
        boundary_loss_weight: float = 1.0,
        connectivity_loss_weight: float = 0.5,
        focal_weight: float = 0.3,
        dice_weight: float = 0.7,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        connectivity_pos_weight: float = 15.0,
    ):
        super().__init__()
        self.boundary_loss_weight = boundary_loss_weight
        self.connectivity_loss_weight = connectivity_loss_weight
        self.seg_loss = CombinedFocalDiceLoss(
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            alpha=focal_alpha,
            gamma=focal_gamma
        )
        self.conn_loss = ConnectivityLoss(pos_weight=connectivity_pos_weight)

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        gt_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        b_loss = self.seg_loss(preds["boundary_prob"], gt_mask)
        r_loss = self.seg_loss(preds["refined_prob"], gt_mask)
        seg_total = 0.5 * b_loss + 0.5 * r_loss

        if preds["connectivity_prob"] is not None:
            gt_conn = generate_connectivity_targets(gt_mask)
            c_loss = self.conn_loss(preds["connectivity_prob"], gt_conn)
            total = self.boundary_loss_weight * seg_total + self.connectivity_loss_weight * c_loss
        else:
            c_loss = torch.tensor(0.0, device=gt_mask.device)
            total = seg_total

        return {
            "loss": total,
            "boundary_loss": seg_total,
            "connectivity_loss": c_loss,
        }
