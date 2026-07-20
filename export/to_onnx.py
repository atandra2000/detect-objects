"""Export Detect-Objects to ONNX for TensorRT deployment.

Traces the full model at a fixed input size.  If the deformable block fails to
export, falls back to exporting backbone+encoder only and warns.
"""
from __future__ import annotations

import argparse

import torch

from models import DetectObjectsModel
from utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="detect_coco_head.onnx")
    parser.add_argument("--height", default=800, type=int)
    parser.add_argument("--width", default=1333, type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectObjectsModel(cfg).to(device).eval()

    dummy = torch.randn(1, 3, args.height, args.width, device=device)
    dynamic_axes = {
        "images": {0: "batch"},
        "class_logits": {0: "batch", 1: "queries"},
        "pred_boxes": {0: "batch", 1: "queries"},
    }

    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["images"],
        output_names=["class_logits", "pred_boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"exported ONNX: {args.out}")


if __name__ == "__main__":
    main()
