"""End-to-end inference test: synthetic image → annotated output + JSON."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from PIL import Image

from infer import detect
from models import DetectObjectsModel
from safetensors.torch import save_file
from utils.config import load_config


def _tiny_cfg():
    return {
        "project": "detect-objects",
        "num_classes": 80,
        "num_queries": 10,
        "model": {
            "backbone": {"name": "resnet50", "out_indices": [1, 2, 3], "channels_last": False},
            "fpn": {"in_channels": [512, 1024, 2048], "out_channels": 64, "num_levels": 3},
            "encoder": {"d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.0},
            "decoder": {
                "d_model": 64, "n_heads": 4, "n_layers": 2, "dropout": 0.0,
                "num_queries": 10, "n_feature_levels": 3, "n_sampling_points": 2,
                "iterative_refinement": True,
            },
            "matcher": {"class_cost": 1.0, "bbox_cost": 5.0, "giou_cost": 2.0},
            "losses": {"class_weight": 1.0, "bbox_weight": 5.0, "giou_weight": 2.0, "focal_alpha": 0.25, "focal_gamma": 2.0},
        },
    }


def test_infer_cli(tmp_path: Path = None) -> None:
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    cfg = _tiny_cfg()
    model = DetectObjectsModel(cfg)
    sd = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    ckpt = tmp_path / "ckpt.safetensors"
    save_file(sd, str(ckpt))

    img_path = tmp_path / "in.jpg"
    Image.new("RGB", (640, 480), color=(128, 128, 128)).save(img_path)
    out_path = tmp_path / "det.jpg"
    json_path = tmp_path / "det.json"

    detections = detect(cfg, str(ckpt), str(img_path), str(out_path), threshold=0.0, json_path=str(json_path), device="cpu")
    assert out_path.exists(), "annotated image not written"
    assert json_path.exists(), "detections JSON not written"
    with open(json_path) as f:
        loaded = json.load(f)
    assert isinstance(loaded, list)
    assert len(loaded) == len(detections)
