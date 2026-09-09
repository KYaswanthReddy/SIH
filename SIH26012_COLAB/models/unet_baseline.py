"""
Lightweight Residual U-Net Baseline for Cadastral Boundary Segmentation.
"""

from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2 with optional Residual shortcut"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, residual: bool = True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.residual = residual
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        
        self.shortcut = nn.Sequential()
        if residual and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.double_conv(x)
        if self.residual:
            out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int, residual: bool = True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, residual=residual)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False, residual: bool = True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, residual=residual)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, residual=residual)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle potential padding differences
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y > 0 or diff_x > 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CadastreUNetBaseline(nn.Module):
    """
    Lightweight Residual U-Net for 512x512 Cadastral Boundary Extraction.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256, 512],
        dropout: float = 0.1,
        bilinear: bool = False,
        residual: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        f = features
        self.inc = DoubleConv(in_channels, f[0], residual=residual)
        self.down1 = Down(f[0], f[1], residual=residual)
        self.down2 = Down(f[1], f[2], residual=residual)
        self.down3 = Down(f[2], f[3], residual=residual)
        self.down4 = Down(f[3], f[4], residual=residual)

        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.up1 = Up(f[4], f[3], bilinear=bilinear, residual=residual)
        self.up2 = Up(f[3], f[2], bilinear=bilinear, residual=residual)
        self.up3 = Up(f[2], f[1], bilinear=bilinear, residual=residual)
        self.up4 = Up(f[1], f[0], bilinear=bilinear, residual=residual)

        self.outc = nn.Conv2d(f[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: (B, 3, H, W)
            return_logits: if True, returns raw un-activated logits; if False, returns sigmoid probabilities in [0, 1]
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

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
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_baseline_model(config: Dict[str, Any]) -> CadastreUNetBaseline:
    """Factory function to build baseline model from config."""
    model_cfg = config.get("model", {})
    features = model_cfg.get("encoder_channels", [32, 64, 128, 256, 512])
    dropout = model_cfg.get("dropout", 0.1)
    residual = model_cfg.get("use_residual", True)

    model = CadastreUNetBaseline(
        in_channels=config.get("data", {}).get("in_channels", 3),
        out_channels=config.get("data", {}).get("out_channels", 1),
        features=features,
        dropout=dropout,
        residual=residual
    )
    return model
