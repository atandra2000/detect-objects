"""Detection losses: sigmoid focal classification + L1 box + GIoU."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.matcher import box_cxcywh_to_xyxy, generalized_box_iou


class DetectionLoss(nn.Module):
    """Combined classification + box regression loss for set prediction."""

    def __init__(
        self,
        num_classes: int,
        class_weight: float = 1.0,
        bbox_weight: float = 5.0,
        giou_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.bbox_weight = bbox_weight
        self.giou_weight = giou_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> dict[str, torch.Tensor]:
        """Compute losses for the matched predictions.

        Args:
            outputs: dict with "pred_logits" (B, N, C+1) and "pred_boxes" (B, N, 4).
            targets: GT targets per sample.
            indices: Hungarian match per sample from matcher.
            num_boxes: normalisation factor (total matched boxes across batch).

        Returns:
            Dict of scalar loss tensors.
        """
        src_logits = outputs["pred_logits"]
        src_boxes = outputs["pred_boxes"]

        # gather matched src/tgt indices
        src_idx = torch.cat([src for src, _ in indices])
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for _, tgt in indices])

        # classification targets: matched = class, unmatched = background (num_classes)
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.long, device=src_logits.device
        )
        target_classes[batch_idx, src_idx] = torch.cat(
            [t["labels"][tgt] for t, (_, tgt) in zip(targets, indices)]
        )

        # focal loss on all queries
        loss_cls = self._focal_loss(src_logits, target_classes, num_boxes)

        # box losses only on matched queries
        pred_boxes = src_boxes[batch_idx, src_idx, :]
        tgt_boxes = torch.cat([t["boxes"][tgt] for t, (_, tgt) in zip(targets, indices)])
        loss_bbox = F.l1_loss(pred_boxes, tgt_boxes, reduction="sum") / num_boxes
        loss_giou = (1 - torch.diag(generalized_box_iou(pred_boxes, tgt_boxes))).sum() / num_boxes

        losses = {
            "loss_cls": self.class_weight * loss_cls,
            "loss_bbox": self.bbox_weight * loss_bbox,
            "loss_giou": self.giou_weight * loss_giou,
        }
        losses["loss_total"] = sum(losses.values())
        return losses

    def _focal_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, num_boxes: float
    ) -> torch.Tensor:
        """Sigmoid focal loss over (B, N, C+1) logits and long targets."""
        B, N, C1 = logits.shape
        # one-hot
        target_one_hot = torch.zeros(B * N, C1, device=logits.device, dtype=logits.dtype)
        target_one_hot.scatter_(1, targets.view(-1, 1), 1.0)
        probs = logits.sigmoid().view(-1, C1)
        ce = F.binary_cross_entropy_with_logits(
            logits.view(-1, C1), target_one_hot, reduction="none"
        )
        p_t = probs * target_one_hot + (1 - probs) * (1 - target_one_hot)
        focal_weight = self.focal_alpha * (1 - p_t) ** self.focal_gamma
        loss = (focal_weight * ce).sum() / max(num_boxes, 1.0)
        return loss


def build_loss(cfg: dict) -> DetectionLoss:
    l = cfg["model"]["losses"]
    return DetectionLoss(
        num_classes=cfg["num_classes"],
        class_weight=l["class_weight"],
        bbox_weight=l["bbox_weight"],
        giou_weight=l["giou_weight"],
        focal_alpha=l["focal_alpha"],
        focal_gamma=l["focal_gamma"],
    )
