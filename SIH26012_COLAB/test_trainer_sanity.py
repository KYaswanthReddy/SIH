"""
Integration Test: End-to-End Sanity Training and Evaluation.
"""

import os
import pytest
import torch
import yaml
from ai.dataset import create_dataloaders
from ai.models.unet_baseline import CadastreUNetBaseline
from ai.trainer import CadastreTrainer


@pytest.fixture
def config():
    with open("configs/baseline_train_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg["sanity_check"]["num_samples"] = 16
    cfg["sanity_check"]["batch_size"] = 4
    cfg["training"]["epochs"] = 1
    cfg["experiment"]["checkpoint_dir"] = "experiments/test_sanity/checkpoints"
    cfg["experiment"]["metrics_dir"] = "experiments/test_sanity/metrics"
    return cfg


def test_trainer_sanity_fit(config):
    # Lightweight model for test speed
    model = CadastreUNetBaseline(
        in_channels=3,
        out_channels=1,
        features=[8, 16, 32, 64, 128],
        dropout=0.0
    )
    
    train_loader, val_loader, _ = create_dataloaders(config, sanity=True)
    device = torch.device("cpu")
    
    trainer = CadastreTrainer(model=model, config=config, device=device)
    history = trainer.fit(train_loader, val_loader, epochs=1)
    
    assert len(history["train_loss"]) == 1
    assert os.path.exists(os.path.join(config["experiment"]["checkpoint_dir"], "best_model.pth"))
