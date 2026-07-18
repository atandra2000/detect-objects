"""Detect-Objects inference CLI.

Loads the shipped safetensors checkpoint, runs the detector, draws boxes +
labels on the image, and writes the annotated image plus an optional JSON of
detections.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

from models import build_model
from utils.config import load_config
from utils.stability import EMA

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_model(cfg: dict, ckpt_path: str, device: torch.device):
    model = build_model(cfg).to(device)
    # try to load state directly; supports both raw state_dicts and our safetensors
    from safetensors.torch import load_file
    sd = load_file(ckpt_path, device=str(device))
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def _preprocess(img: Image.Image, short: int = 900, max_size: int = 1333) -> tuple[torch.Tensor, tuple[int, int]]:
    w, h = img.size
    scale = short / min(h, w)
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    tensor = T.ToTensor()(img)
    tensor = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(tensor)
    return tensor.unsqueeze(0), (new_h, new_w)


def _postprocess(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    orig_size: tuple[int, int],
    threshold: float,
) -> list[dict]:
    scores, labels = logits.softmax(-1).max(-1)
    keep = scores >= threshold
    detections = []
    h, w = orig_size
    for i in range(logits.shape[0]):
        if not keep[i]:
            continue
        cx, cy, bw, bh = boxes[i]
        x1 = float((cx - 0.5 * bw) * w)
        y1 = float((cy - 0.5 * bh) * h)
        x2 = float((cx + 0.5 * bw) * w)
        y2 = float((cy + 0.5 * bh) * h)
        detections.append(
            {
                "label": COCO_CLASSES[int(labels[i])],
                "class_id": int(labels[i]),
                "score": float(scores[i]),
                "box": [x1, y1, x2, y2],
            }
        )
    return detections


def _draw_detections(
    img: Image.Image,
    detections: list[dict],
    font: ImageFont.ImageFont | None = None,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        text = f"{det['label']} {det['score']:.2f}"
        draw.text((x1, y1 - 12), text, fill="red", font=font)
    return img


def detect(
    cfg: dict,
    ckpt_path: str,
    image_path: str,
    output_path: str,
    threshold: float = 0.5,
    json_path: str | None = None,
    device: str = "cuda",
) -> list[dict]:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg, ckpt_path, device)

    img = Image.open(image_path).convert("RGB")
    orig_size = img.size[::-1]
    tensor, _ = _preprocess(img)
    tensor = tensor.to(device)

    with torch.no_grad():
        class_logits, pred_boxes = model(tensor)
    detections = _postprocess(class_logits[-1][0], pred_boxes[-1][0], orig_size, threshold)

    annotated = _draw_detections(img.copy(), detections)
    annotated.save(output_path)

    if json_path:
        with open(json_path, "w") as f:
            json.dump(detections, f, indent=2)

    return detections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="path to YAML config")
    parser.add_argument("--ckpt", required=True, help="path to .safetensors checkpoint")
    parser.add_argument("--in", "--input", dest="input", required=True, help="input image path")
    parser.add_argument("--out", required=True, help="output annotated image path")
    parser.add_argument("--thr", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--json", default=None, help="optional detections JSON output")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = load_config(args.config)
    detections = detect(cfg, args.ckpt, args.input, args.out, args.thr, args.json, args.device)
    print(f"wrote {args.out} with {len(detections)} detections")


if __name__ == "__main__":
    main()
