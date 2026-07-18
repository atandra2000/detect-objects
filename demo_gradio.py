"""Gradio demo for Detect-Objects.

Upload an image, adjust the confidence threshold, and see annotated boxes.
"""
from __future__ import annotations

import argparse

import gradio as gr
import torch

from infer import COCO_CLASSES, _draw_detections, _load_model, _postprocess, _preprocess
from utils.config import load_config


def make_demo(cfg: dict, ckpt_path: str, device: str):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg, ckpt_path, device)

    def predict(image, threshold):
        if image is None:
            return None
        import numpy as np
        from PIL import Image
        img = Image.fromarray(image).convert("RGB")
        orig_size = img.size[::-1]
        tensor, _ = _preprocess(img)
        tensor = tensor.to(device)
        with torch.no_grad():
            class_logits, pred_boxes = model(tensor)
        detections = _postprocess(class_logits[-1][0], pred_boxes[-1][0], orig_size, threshold)
        return _draw_detections(img, detections)

    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="numpy", label="Input image"),
            gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Confidence threshold"),
        ],
        outputs=gr.Image(type="pil", label="Detections"),
        title="Detect-Objects",
        description="Upload an image to see COCO object detections.",
        examples=None,
    )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    demo = make_demo(cfg, args.ckpt, args.device)
    demo.launch(server_name="0.0.0.0", server_port=args.port)


if __name__ == "__main__":
    main()
