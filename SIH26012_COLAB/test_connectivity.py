"""
Tests for Connectivity Target Generation, Connectivity Head, and Loss.
"""

import pytest
import torch
from ai.models.connectivity import generate_connectivity_targets, ConnectivityHead, ConnectivityLoss


def test_generate_connectivity_targets():
    # Create synthetic binary mask with a 5x5 square in batch size 2
    mask = torch.zeros(2, 1, 32, 32)
    mask[:, :, 10:15, 10:15] = 1.0

    conn = generate_connectivity_targets(mask)

    assert conn.shape == (2, 8, 32, 32)
    assert set(torch.unique(conn).tolist()).issubset({0.0, 1.0})
    # For a single batch item, interior pixel (12, 12) has all 8 neighbors
    assert conn[0, :, 12, 12].sum() == 8.0
    assert conn[1, :, 12, 12].sum() == 8.0


def test_connectivity_head_forward():
    head = ConnectivityHead(in_channels=32, num_directions=8)
    dummy_feat = torch.randn(2, 32, 64, 64)

    out = head(dummy_feat)
    assert out.shape == (2, 8, 64, 64)
    assert 0.0 <= out.min() <= out.max() <= 1.0


def test_connectivity_loss():
    loss_fn = ConnectivityLoss()
    raw_logits = torch.randn(2, 8, 32, 32, requires_grad=True)
    pred_conn = torch.sigmoid(raw_logits)
    target_conn = torch.zeros(2, 8, 32, 32)
    target_conn[:, :, 10:15, 10:15] = 1.0

    loss = loss_fn(pred_conn, target_conn)
    assert loss.requires_grad
    loss.backward()
    assert raw_logits.grad is not None
