"""
Tests for Baseline U-Net Architecture and Forward Passes.
"""

import pytest
import torch
import yaml
from ai.models.unet_baseline import CadastreUNetBaseline, build_baseline_model


@pytest.fixture
def config():
    with open("configs/baseline_train_config.yaml", "r") as f:
        return yaml.safe_load(f)


def test_model_build_and_parameters(config):
    model = build_baseline_model(config)
    params = model.count_parameters()
    
    assert isinstance(model, CadastreUNetBaseline)
    assert 5_000_000 < params < 15_000_000, f"Unexpected parameter count: {params}"


def test_model_forward_pass():
    model = CadastreUNetBaseline(
        in_channels=3,
        out_channels=1,
        features=[16, 32, 64, 128, 256], # Smaller channels for fast unit test
        dropout=0.0
    )
    model.eval()

    # Input batch: 2 samples of shape (3, 512, 512)
    dummy_input = torch.randn(2, 3, 512, 512)
    
    with torch.no_grad():
        # Probability output
        prob_output = model(dummy_input, return_logits=False)
        assert prob_output.shape == (2, 1, 512, 512)
        assert 0.0 <= prob_output.min() <= prob_output.max() <= 1.0

        # Logits output
        logits_output = model(dummy_input, return_logits=True)
        assert logits_output.shape == (2, 1, 512, 512)
