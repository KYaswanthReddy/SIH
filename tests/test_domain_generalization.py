"""
Tests for Domain Generalization (DG) Modules and Models.
"""

import pytest
import torch
from ai.models.domain_generalization import MixStyleDomainPerturbation, DomainAwareFeatureAlignment, CadastreUNetDG


def test_mixstyle_forward():
    mixstyle = MixStyleDomainPerturbation(p=1.0, alpha=0.2)
    mixstyle.train()

    # Batch of 4 samples, 32 channels, 64x64
    x = torch.randn(4, 32, 64, 64)
    out = mixstyle(x)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()

    # Eval mode should be identity
    mixstyle.eval()
    out_eval = mixstyle(x)
    assert torch.allclose(out_eval, x)


def test_domain_aware_feature_alignment():
    align = DomainAwareFeatureAlignment(num_domains=5, feature_dim=32)
    feat = torch.randn(4, 32, 16, 16)
    d_ids = torch.tensor([0, 1, 2, 3])

    out = align(feat, d_ids)
    assert out.shape == feat.shape
    assert not torch.isnan(out).any()


def test_cadastre_unet_dg_forward():
    model = CadastreUNetDG(
        features=[16, 32, 64, 128, 256],
        pmg_channels=64,
        pmg_enabled=True,
        dg_enabled=True
    )
    model.train()

    dummy_input = torch.randn(2, 3, 512, 512)
    out = model(dummy_input)

    assert out.shape == (2, 1, 512, 512)
    assert 0.0 <= out.min() <= out.max() <= 1.0
