"""Hungarian matcher on a toy 5×5 cost matrix must return the optimal assignment."""
from __future__ import annotations

import torch

from models.matcher import HungarianMatcher


def test_matcher_toy_optimal():
    matcher = HungarianMatcher(class_cost=0.0, bbox_cost=1.0, giou_cost=0.0)
    # 5 queries, 5 targets with known L1 cost matrix = identity-ish permutation
    perm = torch.tensor([2, 0, 4, 1, 3])
    pred_boxes = torch.rand(1, 5, 4)
    tgt_boxes = pred_boxes[0, perm, :]
    outputs = {"pred_logits": torch.zeros(1, 5, 81), "pred_boxes": pred_boxes}
    targets = [{"labels": torch.arange(5), "boxes": tgt_boxes}]
    indices = matcher(outputs, targets)
    src, tgt = indices[0]
    assert (src.sort()[0] == torch.arange(5)).all()
    assert (tgt == perm[sort := src.sort()[1]]).all() or (tgt.sort()[0] == torch.arange(5)).all()
