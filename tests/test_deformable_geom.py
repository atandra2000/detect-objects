"""Deformable-attention geometry test: sampling stays inside image bounds."""
from __future__ import annotations

import torch

from models.deformable_attention import MultiScaleDeformableAttention


def test_sampling_inside_padded_image():
    B, Nq, C = 2, 10, 256
    n_levels, n_heads, n_points = 3, 8, 4
    attn = MultiScaleDeformableAttention(C, n_levels, n_heads, n_points)
    attn.eval()

    features = [
        torch.randn(B, C, 64, 64),
        torch.randn(B, C, 32, 32),
        torch.randn(B, C, 16, 16),
    ]
    spatial_shapes = [(64, 64), (32, 32), (16, 16)]
    query = torch.randn(B, Nq, C)
    ref = torch.rand(B, Nq, n_levels, 2)

    with torch.no_grad():
        out = attn(query, ref, features, spatial_shapes)
    assert out.shape == (B, Nq, C)
    assert torch.isfinite(out).all(), "deformable output has NaN/Inf"
