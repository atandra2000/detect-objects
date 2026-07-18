"""Hungarian matcher for set-prediction detection.

Matches N predictions to M ground-truth boxes using scipy's
linear_sum_assignment on a cost = cls + L1 + GIoU.  Runs on CPU because the
assignment graph is tiny (≤ 300 × ~10).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """Hungarian matcher with configurable cost weights."""

    def __init__(
        self,
        class_cost: float = 1.0,
        bbox_cost: float = 5.0,
        giou_cost: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Match last-layer predictions to GT.

        Args:
            outputs: dict with keys "pred_logits" (B, N, C+1) and
                "pred_boxes" (B, N, 4) in [0,1] cxcywh.
            targets: list of length B, each dict with "labels" (M,) and
                "boxes" (M, 4) in [0,1] cxcywh.

        Returns:
            List of (src_idx, tgt_idx) tensors per sample.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # flatten batch and queries for batched cost computation
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # (B*N, C+1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)             # (B*N, 4)

        sizes = [len(t["labels"]) for t in targets]
        # build target indices in flat tensor
        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])

        # classification cost: focal-weighted negative log prob of target class
        neg_prob = 1 - out_prob[:, tgt_ids]
        focal_weight = self.focal_alpha * (neg_prob ** self.focal_gamma)
        cost_class = -focal_weight * torch.log(out_prob[:, tgt_ids] + 1e-8)

        # L1 box cost
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # GIoU cost
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        C = self.bbox_cost * cost_bbox + self.class_cost * cost_class + self.giou_cost * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        indices = []
        start = 0
        for i, size in enumerate(sizes):
            c = C[i, :, start : start + size]
            src_i, tgt_i = linear_sum_assignment(c.numpy())
            indices.append((torch.as_tensor(src_i), torch.as_tensor(tgt_i)))
            start += size
        return indices


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise GIoU between cxcywh boxes.

    Returns:
        (N, M) tensor of GIoU values.
    """
    b1 = box_cxcywh_to_xyxy(boxes1)
    b2 = box_cxcywh_to_xyxy(boxes2)

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])

    inter_x1 = torch.maximum(b1[:, None, 0], b2[None, :, 0])
    inter_y1 = torch.maximum(b1[:, None, 1], b2[None, :, 1])
    inter_x2 = torch.minimum(b1[:, None, 2], b2[None, :, 2])
    inter_y2 = torch.minimum(b1[:, None, 3], b2[None, :, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / (union + 1e-8)

    # enclosing box
    enc_x1 = torch.minimum(b1[:, None, 0], b2[None, :, 0])
    enc_y1 = torch.minimum(b1[:, None, 1], b2[None, :, 1])
    enc_x2 = torch.maximum(b1[:, None, 2], b2[None, :, 2])
    enc_y2 = torch.maximum(b1[:, None, 3], b2[None, :, 3])
    enc_area = (enc_x2 - enc_x1).clamp(min=0) * (enc_y2 - enc_y1).clamp(min=0)

    giou = iou - (enc_area - union) / (enc_area + 1e-8)
    return giou


def build_matcher(cfg: dict) -> HungarianMatcher:
    m_cfg = cfg["model"]["matcher"]
    loss_cfg = cfg["model"]["losses"]
    return HungarianMatcher(
        class_cost=m_cfg["class_cost"],
        bbox_cost=m_cfg["bbox_cost"],
        giou_cost=m_cfg["giou_cost"],
        focal_alpha=loss_cfg["focal_alpha"],
        focal_gamma=loss_cfg["focal_gamma"],
    )
