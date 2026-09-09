"""
Tests for Progressive Multiscale Generator (PMG) Module and Model.
"""

import pytest
import torch
import yaml
from ai.models.pmg import PMGModule, CadastreUNetPMG


def test_pmg_module_forward():
    # Simulate 4 multiscale feature maps: 256x256, 128x128, 64x64, 32x32
    f1 = torch.randn(2, 64, 256, 256)
    f2 = torch.randn(2, 128, 128, 128)
    f3 = torch.randn(2, 256, 64, 64)
    f4 = torch.randn(2, 512, 32, 32)

    pmg = PMGModule(in_channels_list=[64, 128, 256, 512], out_channels=128)
    enhanced = pmg([f1, f2, f3, f4])

    assert len(enhanced) == 4
    assert enhanced[0].shape == (2, 128, 256, 256)
    assert enhanced[1].shape == (2, 128, 128, 128)
    assert enhanced[2].shape == (2, 128, 64, 64)
    assert enhanced[3].shape == (2, 128, 32, 32)


def test_cadastre_unet_pmg_forward():
    model = CadastreUNetPMG(
        in_channels=3,
        out_channels=1,
        features=[16, 32, 64, 128, 256],
        pmg_channels=64,
        pmg_enabled=True
    )
    model.eval()

    dummy_input = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy_input)
        assert out.shape == (2, 1, 512, 512)
        assert 0.0 <= out.min() <= out.max() <= 1.0


def test_pmg_toggle():
    # Test model with PMG disabled vs enabled
    model_disabled = CadastreUNetPMG(features=[16, 32, 64, 128, 256], pmg_enabled=False)
    model_enabled = CadastreUNetPMG(features=[16, 32, 64, 128, 256], pmg_enabled=True)

    p_dis = model_disabled.count_parameters()
    p_en = model_enabled.count_parameters()

    assert p_en > p_dis, "PMG enabled model should have additional parameters"
