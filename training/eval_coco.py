"""COCO evaluation: run model on val and report mAP via pycocotools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from data.coco_dataset import build_val_dataset
from data.detection_transforms import collate_detection
from models import build_model
from utils.checkpoint import CheckpointManager
from utils.config import load_config
from utils.logging import setup_logger
from utils.memory import apply_channels_last
from utils.stability import EMA


coco_id_to_class = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15, 18: 16, 19: 17, 20: 18,
    21: 19, 22: 20, 23: 21, 24: 22, 25: 23, 27: 24, 28: 25, 31: 26, 32: 27,
    33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 40: 35, 41: 36,
    42: 37, 43: 38, 44: 39, 46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45,
    52: 46, 53: 47, 54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54,
    61: 55, 62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63,
    74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71, 82: 72,
    84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}
class_to_coco_id = {v: k for k, v in coco_id_to_class.items()}


def _box_cxcywh_to_xyxy(boxes: torch.Tensor, h: int, w: int) -> list[list[float]]:
    boxes = boxes.clone()
    boxes[:, [0, 2]] *= w
    boxes[:, [1, 3]] *= h
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = cx - 0.5 * bw
    y1 = cy - 0.5 * bh
    x2 = cx + 0.5 * bw
    y2 = cy + 0.5 * bh
    return torch.stack([x1, y1, x2, y2], dim=-1).tolist()


def evaluate(cfg: dict, ckpt_path: str | None = None, device: str = "cuda") -> dict:
    logger = setup_logger("detect-objects.eval")
    model = build_model(cfg).to(device)
    if cfg["train"].get("channels_last", True) and torch.cuda.is_available():
        model = apply_channels_last(model)

    ema = EMA(model, decay=cfg["train"]["ema_decay"], device=device)
    if ckpt_path:
        ckpt_mgr = CheckpointManager(Path(ckpt_path).parent)
        ckpt_mgr.load(model, optimizer=None, scheduler=None, ema=ema, device=device)
        ema.apply_shadow(model)

    model.eval()
    val_ds = build_val_dataset(cfg)
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
        collate_fn=collate_detection,
    )

    results = []
    img_ids = []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="eval"):
            images = images.to(device)
            class_logits, pred_boxes = model(images)
            # use last decoder layer
            pred_logits = class_logits[-1]
            pred_boxes = pred_boxes[-1]
            scores, labels = pred_logits.softmax(-1).max(-1)
            keep = scores > 0.3
            for b in range(images.shape[0]):
                h, w = targets[b].get("orig_size", (images.shape[2], images.shape[3]))
                boxes_xyxy = _box_cxcywh_to_xyxy(pred_boxes[b][keep[b]], h, w)
                for box, score, label in zip(boxes_xyxy, scores[b][keep[b]], labels[b][keep[b]]):
                    if int(label) >= cfg["num_classes"]:
                        continue
                    results.append(
                        {
                            "image_id": int(targets[b]["image_id"]),
                            "category_id": class_to_coco_id[int(label)],
                            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                            "score": float(score),
                        }
                    )
                img_ids.append(int(targets[b]["image_id"]))

    if ema:
        ema.restore(model, {})

    ann_file = cfg["data"]["val_ann"]
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = list(set(img_ids))
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = {
        "mAP": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
        "mAP_75": float(coco_eval.stats[2]),
        "mAP_S": float(coco_eval.stats[3]),
        "mAP_M": float(coco_eval.stats[4]),
        "mAP_L": float(coco_eval.stats[5]),
    }
    logger.info(f"COCO metrics: {stats}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt", required=True, help="safetensors checkpoint path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, args.ckpt, args.device)


if __name__ == "__main__":
    main()
