"""
Progressive Multiscale Generator (PMG) for Cadastral Feature Representation.
Captures fine-scale details, mid-scale parcel boundaries, and coarse spatial context.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_baseline import DoubleConv, Down, Up


class MultiscaleFeatureBranch(nn.Module):
    """
    Scale-specific branch with dilated residual convolutions for expanded receptive field without resolution loss.
    """

    def __init__(self, in_channels: int, out_channels: int, dilation_rates: List[int] = [1, 2, 4]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for d in dilation_rates
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * len(dilation_rates), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_feats = [conv(x) for conv in self.convs]
        fused = self.fusion(torch.cat(branch_feats, dim=1))
        return F.relu(fused + self.shortcut(x), inplace=True)


class PMGModule(nn.Module):
    """
    Progressive Multiscale Generator Module.
    Progressively aggregates features across Fine, Mid, and Coarse scales with top-down and lateral fusion.
    """

    def __init__(
        self,
        in_channels_list: List[int] = [64, 128, 256, 512],
        out_channels: int = 256
    ):
        super().__init__()
        self.num_scales = len(in_channels_list)
        
        # Scale-specific projection branches
        self.branches = nn.ModuleList([
            MultiscaleFeatureBranch(in_ch, out_channels, dilation_rates=[1, 2, 4] if idx >= 1 else [1, 2])
            for idx, in_ch in enumerate(in_channels_list)
        ])

        # Progressive Top-Down Refinement Layers
        self.refinement_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(self.num_scales - 1)
        ])

        # Global context pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels_list[-1], out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Final multi-scale fusion
        self.final_fusion = nn.Sequential(
            nn.Conv2d(out_channels * (self.num_scales + 1), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps [feat_scale1, feat_scale2, feat_scale3, feat_scale4]
                      from high resolution to low resolution.
        Returns:
            enhanced_features: List of multiscale enhanced features with same spatial dimensions.
        """
        assert len(features) == self.num_scales, f"Expected {self.num_scales} features, got {len(features)}"

        # 1. Process each scale through its multiscale branch
        projected = [self.branches[i](features[i]) for i in range(self.num_scales)]

        # 2. Progressive Top-Down Fusion (Coarse to Fine)
        # projected[-1] is coarsest, projected[0] is finest
        refined = [None] * self.num_scales
        refined[-1] = projected[-1]

        for i in range(self.num_scales - 2, -1, -1):
            # Upsample coarser level to match current level spatial dimensions
            coarser_up = F.interpolate(
                refined[i + 1],
                size=projected[i].shape[2:],
                mode="bilinear",
                align_corners=True
            )
            # Lateral connection + top-down fusion
            fused = projected[i] + coarser_up
            refined[i] = self.refinement_convs[i](fused)

        return refined


class CadastreUNetPMG(nn.Module):
    """
    Cadastral Segmentation Network integrated with Progressive Multiscale Generator (PMG).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256, 512],
        pmg_channels: int = 128,
        dropout: float = 0.1,
        pmg_enabled: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pmg_enabled = pmg_enabled

        f = features
        self.inc = DoubleConv(in_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        self.down4 = Down(f[3], f[4])

        if self.pmg_enabled:
            # PMG processes features from down1, down2, down3, down4
            self.pmg = PMGModule(in_channels_list=[f[1], f[2], f[3], f[4]], out_channels=pmg_channels)
            # Adapters back to decoder skip channels
            self.pmg_adapters = nn.ModuleList([
                nn.Conv2d(pmg_channels, f[i], kernel_size=1) for i in [1, 2, 3, 4]
            ])
        else:
            self.pmg = None

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.up1 = Up(f[4], f[3])
        self.up2 = Up(f[3], f[2])
        self.up3 = Up(f[2], f[1])
        self.up4 = Up(f[1], f[0])

        self.outc = nn.Conv2d(f[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.pmg_enabled and self.pmg is not None:
            # Apply PMG across multiscale encoder features
            enhanced = self.pmg([x2, x3, x4, x5])
            x2 = x2 + self.pmg_adapters[0](enhanced[0])
            x3 = x3 + self.pmg_adapters[1](enhanced[1])
            x4 = x4 + self.pmg_adapters[2](enhanced[2])
            x5 = x5 + self.pmg_adapters[3](enhanced[3])

        x5 = self.drop(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        if return_logits:
            return logits
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
