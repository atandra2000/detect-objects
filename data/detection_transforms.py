"""Box-aware COCO transforms and collate helpers for Detect-Objects."""
from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize_short_edge(
    img: Image.Image,
    target_short: int,
    max_size: int,
    boxes: np.ndarray | None = None,
) -> tuple[Image.Image, np.ndarray | None, tuple[int, int]]:
    """Resize so the short edge == target_short, capped at max_size."""
    w, h = img.size
    scale = target_short / min(h, w)
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img = TF.resize(img, (new_h, new_w))
    if boxes is not None:
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= new_w / w
        boxes[:, [1, 3]] *= new_h / h
    return img, boxes, (new_h, new_w)


def _random_crop_with_boxes(
    img: Image.Image,
    boxes: np.ndarray,
    labels: torch.Tensor,
    min_size: int = 384,
    max_size: int = 640,
) -> tuple[Image.Image, np.ndarray, torch.Tensor]:
    """Random crop that preserves at least one GT box (DETR-style)."""
    w, h = img.size
    if boxes.shape[0] == 0:
        crop_w = crop_h = min(max_size, min_size)
        x = random.randint(0, max(0, w - crop_w))
        y = random.randint(0, max(0, h - crop_h))
        img = TF.crop(img, y, x, crop_h, crop_w)
        return img, boxes, labels

    for _ in range(50):
        crop_w = random.randint(min_size, min(w, max_size))
        crop_h = random.randint(min_size, min(h, max_size))
        if crop_w / crop_h < 0.5 or crop_h / crop_w < 0.5:
            continue
        x = random.randint(0, max(0, w - crop_w))
        y = random.randint(0, max(0, h - crop_h))
        crop_box = np.array([x, y, x + crop_w, y + crop_h], dtype=np.float32)

        # check intersection with GT boxes
        inter_x1 = np.maximum(boxes[:, 0], crop_box[0])
        inter_y1 = np.maximum(boxes[:, 1], crop_box[1])
        inter_x2 = np.minimum(boxes[:, 2], crop_box[2])
        inter_y2 = np.minimum(boxes[:, 3], crop_box[3])
        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = (inter_area >= 0.5 * box_area) & (inter_area > 0)
        if keep.any():
            img = TF.crop(img, y, x, crop_h, crop_w)
            boxes = boxes[keep]
            labels = labels[keep]
            boxes[:, [0, 2]] -= x
            boxes[:, [1, 3]] -= y
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, crop_w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, crop_h)
            return img, boxes, labels
    # fallback: no crop
    return img, boxes, labels


def _horizontal_flip(img: Image.Image, boxes: np.ndarray, w: int) -> tuple[Image.Image, np.ndarray]:
    """Mirror image and boxes horizontally."""
    img = TF.hflip(img)
    boxes = boxes.copy()
    x1 = w - boxes[:, 2]
    x2 = w - boxes[:, 0]
    boxes[:, 0] = x1
    boxes[:, 2] = x2
    return img, boxes


def xyxy_to_cxcywh(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """Convert absolute XYXY boxes to normalized CXCYWH."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] /= w
    boxes[:, [1, 3]] /= h
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    return np.stack([cx, cy, bw, bh], axis=1)


class COCODetectionTransform:
    """Train-time COCO transform with random resize, crop, flip, normalize."""

    def __init__(
        self,
        short_sides: list[int],
        max_size: int,
        crop_min: int = 384,
        crop_max: int = 640,
        flip_prob: float = 0.5,
    ):
        self.short_sides = short_sides
        self.max_size = max_size
        self.crop_min = crop_min
        self.crop_max = crop_max
        self.flip_prob = flip_prob
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(
        self, img: Image.Image, boxes: np.ndarray, labels: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)

        # random resize short edge
        short = random.choice(self.short_sides)
        img, boxes, (h, w) = _resize_short_edge(img, short, self.max_size, boxes)

        # random crop preserving boxes
        img, boxes, labels = _random_crop_with_boxes(
            img, boxes, labels, min_size=self.crop_min, max_size=self.crop_max
        )
        h, w = img.size[1], img.size[0]

        # random horizontal flip
        if random.random() < self.flip_prob:
            img, boxes = _horizontal_flip(img, boxes, w)

        # to tensor and normalize
        img = TF.to_tensor(img)
        img = self.normalize(img)

        boxes = xyxy_to_cxcywh(boxes, h, w)
        # clamp boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, 1)
        # remove degenerate boxes
        keep = (boxes[:, 2] > 1e-4) & (boxes[:, 3] > 1e-4)
        boxes = boxes[keep]
        labels = labels[keep]

        target = {"boxes": torch.from_numpy(boxes).float(), "labels": labels}
        return img, target


class COCODetectionValTransform:
    """Validation transform: fixed short edge, no crop/flip."""

    def __init__(self, short_side: int, max_size: int):
        self.short_side = short_side
        self.max_size = max_size
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(
        self, img: Image.Image, boxes: np.ndarray, labels: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        img, boxes, (h, w) = _resize_short_edge(
            img, self.short_side, self.max_size, boxes
        )
        img = TF.to_tensor(img)
        img = self.normalize(img)
        boxes = xyxy_to_cxcywh(boxes, h, w)
        target = {"boxes": torch.from_numpy(boxes).float(), "labels": labels}
        return img, target


def collate_detection(batch: list[tuple[torch.Tensor, dict]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Batch images (pad with 0) and keep target list."""
    images, targets = zip(*batch)
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    batched = torch.zeros(len(images), 3, max_h, max_w, dtype=images[0].dtype)
    for i, img in enumerate(images):
        _, h, w = img.shape
        batched[i, :, :h, :w] = img
    return batched, list(targets)
