"""
Tests for Full Integrated Model (CadastreUNetFull) and Loss.
"""

import pytest
import torch
from ai.models.full_model import CadastreUNetFull, FullCadastreLoss


def test_full_model_forward():
    model = CadastreUNetFull(
        features=[16, 32, 64, 128, 256],
        pmg_channels=64,
        pmg_enabled=True,
        dg_enabled=True,
        connectivity_enabled=True
    )
    model.train()

    dummy_input = torch.randn(2, 3, 512, 512)

    # Standard forward
    prob = model(dummy_input, return_dict=False)
    assert prob.shape == (2, 1, 512, 512)
    assert 0.0 <= prob.min() <= prob.max() <= 1.0

    # Dict forward
    out_dict = model(dummy_input, return_dict=True)
    assert "boundary_prob" in out_dict
    assert "connectivity_prob" in out_dict
    assert "refined_prob" in out_dict
    assert out_dict["connectivity_prob"].shape == (2, 8, 512, 512)


def test_full_loss_backward():
    model = CadastreUNetFull(
        features=[8, 16, 32, 64, 128],
        pmg_channels=32,
        pmg_enabled=True,
        dg_enabled=True,
        connectivity_enabled=True
    )
    loss_fn = FullCadastreLoss()

    dummy_input = torch.randn(2, 3, 64, 64)
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, 10:20, 10:20] = 1.0

    preds = model(dummy_input, return_dict=True)
    loss_dict = loss_fn(preds, target)

    assert "loss" in loss_dict
    assert "boundary_loss" in loss_dict
    assert "connectivity_loss" in loss_dict

    loss_dict["loss"].backward()


def test_full_model_256_sanity():
    import yaml
    with open("configs/sanity_full.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    assert cfg["data"]["max_train_samples"] == 50
    assert cfg["data"]["max_val_samples"] == 10
    assert cfg["data"]["max_test_samples"] == 10
    assert cfg["data"]["image_size"] == [256, 256]
    assert cfg["training"]["epochs"] == 1
    assert cfg["model"]["pmg"]["enabled"] is True
    assert cfg["model"]["domain_generalization"]["enabled"] is True
    assert cfg["model"]["connectivity"]["enabled"] is True

    model = CadastreUNetFull(
        features=[8, 16, 32, 64, 128],
        pmg_channels=32,
        pmg_enabled=True,
        dg_enabled=True,
        connectivity_enabled=True
    )
    dummy_input = torch.randn(2, 3, 256, 256)
    out_dict = model(dummy_input, return_dict=True)
    assert out_dict["refined_prob"].shape == (2, 1, 256, 256)
    assert out_dict["connectivity_prob"].shape == (2, 8, 256, 256)


def test_quick_full_config_validation():
    import yaml
    with open("configs/quick_full.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg["data"]["max_train_samples"] == 500
    assert cfg["data"]["max_val_samples"] == 100
    assert cfg["data"]["max_test_samples"] == 100
    assert cfg["data"]["image_size"] == [256, 256]
    assert cfg["training"]["epochs"] == 5
    assert cfg["training"]["batch_size"] == 8
    assert cfg["model"]["architecture"] == "CadastreUNetFull"
    assert cfg["model"]["pmg"]["enabled"] is True
    assert cfg["model"]["domain_generalization"]["enabled"] is True
    assert cfg["model"]["connectivity"]["enabled"] is True
    assert cfg["loss"]["boundary_loss_weight"] == 1.0
    assert cfg["loss"]["connectivity_loss_weight"] == 0.5


def test_quick_full_15epoch_config_validation():
    import yaml
    with open("configs/quick_full_15epoch.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg["data"]["max_train_samples"] == 500
    assert cfg["data"]["max_val_samples"] == 100
    assert cfg["data"]["max_test_samples"] == 100
    assert cfg["data"]["image_size"] == [256, 256]
    assert cfg["training"]["epochs"] == 15
    assert cfg["training"]["batch_size"] == 8
    assert cfg["training"]["early_stopping_patience"] == 5
    assert cfg["model"]["architecture"] == "CadastreUNetFull"
    assert cfg["model"]["pmg"]["enabled"] is True
    assert cfg["model"]["connectivity"]["enabled"] is True


def test_quick_full_fixed_config_validation():
    import yaml
    with open("configs/quick_full_fixed.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg["data"]["max_train_samples"] == 500
    assert cfg["data"]["max_val_samples"] == 100
    assert cfg["data"]["max_test_samples"] == 100
    assert cfg["data"]["image_size"] == [256, 256]
    assert cfg["training"]["epochs"] == 10
    assert cfg["training"]["batch_size"] == 8
    assert cfg["loss"]["focal"]["alpha"] == 0.75
    assert cfg["loss"]["focal_weight"] == 0.3
    assert cfg["loss"]["dice_weight"] == 0.7
    assert cfg["loss"]["connectivity_pos_weight"] == 15.0
    assert cfg["evaluation"]["thresholds"] == [0.2, 0.3, 0.4, 0.5]
    assert cfg["model"]["architecture"] == "CadastreUNetFull"
    assert cfg["model"]["pmg"]["enabled"] is True
    assert cfg["model"]["domain_generalization"]["enabled"] is True
    assert cfg["model"]["connectivity"]["enabled"] is True




