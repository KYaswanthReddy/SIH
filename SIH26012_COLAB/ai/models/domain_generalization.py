"""
Domain Generalization (DG) Module for Robust Cross-Landscape Cadastral Segmentation.
Implements Feature Style Normalization Perturbation (MixStyle) and Domain-Invariant Alignment.
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_baseline import DoubleConv, Down, Up
from .pmg import PMGModule


class MixStyleDomainPerturbation(nn.Module):
    """
    MixStyle module for Domain Generalization in remote sensing.
    Perturbs feature statistics (mean and std) of intermediate representations
    during training to enforce domain-invariant cadastral feature extraction.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B = x.size(0)
        if B < 2:
            return x

        # Compute instance mean and std: (B, C, 1, 1)
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()

        # Normalize features
        x_normed = (x - mean) / sig

        # Sample permutation for cross-instance style mixing
        perm = torch.randperm(B)
        mean_perm = mean[perm]
        sig_perm = sig[perm]

        # Sample mix factor from Beta distribution
        gamma = torch.distributions.Beta(self.alpha, self.alpha).sample((B, 1, 1, 1)).to(x.device)

        # Mix statistics
        mixed_mean = gamma * mean + (1.0 - gamma) * mean_perm
        mixed_sig = gamma * sig + (1.0 - gamma) * sig_perm

        return x_normed * mixed_sig + mixed_mean


class DomainAwareFeatureAlignment(nn.Module):
    """
    Domain-Aware Feature Normalization with adaptive domain embedding.
    """

    def __init__(self, num_domains: int = 8, feature_dim: int = 256):
        super().__init__()
        self.domain_embed = nn.Embedding(num_domains, feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, feat: torch.Tensor, domain_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if domain_ids is None:
            return feat
        B, C, H, W = feat.shape
        d_emb = self.fc(self.domain_embed(domain_ids)).view(B, C, 1, 1)
        return feat * (1.0 + torch.sigmoid(d_emb))


class CadastreUNetDG(nn.Module):
    """
    Cadastral Segmentation Model with PMG and Domain Generalization (DG).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [32, 64, 128, 256, 512],
        pmg_channels: int = 128,
        dropout: float = 0.1,
        pmg_enabled: bool = True,
        dg_enabled: bool = True,
        dg_prob: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pmg_enabled = pmg_enabled
        self.dg_enabled = dg_enabled

        f = features
        self.inc = DoubleConv(in_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        self.down4 = Down(f[3], f[4])

        # Domain Generalization MixStyle modules on intermediate feature stages
        if self.dg_enabled:
            self.dg_stage1 = MixStyleDomainPerturbation(p=dg_prob, alpha=0.2)
            self.dg_stage2 = MixStyleDomainPerturbation(p=dg_prob, alpha=0.2)
        else:
            self.dg_stage1 = nn.Identity()
            self.dg_stage2 = nn.Identity()

        if self.pmg_enabled:
            self.pmg = PMGModule(in_channels_list=[f[1], f[2], f[3], f[4]], out_channels=pmg_channels)
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
        if self.dg_enabled:
            x2 = self.dg_stage1(x2)
            
        x3 = self.down2(x2)
        if self.dg_enabled:
            x3 = self.dg_stage2(x3)

        x4 = self.down3(x3)
        x5 = self.down4(x4)

        if self.pmg_enabled and self.pmg is not None:
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
