"""Encoder forward path runs under autocast and keeps FP32 LayerNorm."""
from __future__ import annotations

import torch

from models.set_decoder import SetDecoder


def _tiny_cfg():
    return {
        "num_classes": 80,
        "model": {
            "encoder": {"d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.0},
            "decoder": {
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 2,
                "dropout": 0.0,
                "num_queries": 10,
                "n_feature_levels": 3,
                "n_sampling_points": 2,
                "iterative_refinement": True,
            },
        },
    }


def test_encoder_autocast_fp32_ln():
    cfg = _tiny_cfg()
    model = SetDecoder(cfg)
    model.eval()
    features = [
        torch.randn(1, 64, 32, 32),
        torch.randn(1, 64, 16, 16),
        torch.randn(1, 64, 8, 8),
    ]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=True):
        classes, boxes = model(features)
    assert classes.shape[1] == 1
    assert boxes.shape[-1] == 4
    assert (boxes >= 0).all() and (boxes <= 1).all()
    # verify LayerNorm remains FP32
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            assert m.weight.dtype == torch.float32
