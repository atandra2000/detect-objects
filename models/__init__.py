"""Detect-Objects model factory and build helpers."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.backbone_resnet import build_resnet50_fpn
from models.set_decoder import build_set_decoder


class DetectObjectsModel(nn.Module):
    """End-to-end set-prediction detector: backbone + FPN + encoder/decoder."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_resnet50_fpn(cfg)
        self.decoder = build_set_decoder(cfg)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: images (B, 3, H, W). Returns (L, B, Nq, C+1), (L, B, Nq, 4)."""
        features = self.backbone(images)
        return self.decoder(features)

    def compute_loss(
        self,
        images: torch.Tensor,
        targets: list[dict[str, torch.Tensor]],
        matcher,
        criterion,
    ) -> dict[str, torch.Tensor]:
        class_logits, pred_boxes = self(images)
        outputs = {"pred_logits": class_logits[-1], "pred_boxes": pred_boxes[-1]}
        indices = matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float32, device=images.device)
        return criterion(outputs, targets, indices, num_boxes.item())


def build_model(cfg: dict) -> DetectObjectsModel:
    return DetectObjectsModel(cfg)
