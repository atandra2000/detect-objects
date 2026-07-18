"""Pure-PyTorch multi-scale deformable attention for Detect-Objects.

No custom CUDA op: sampling is done with ``torch.nn.functional.grid_sample``
plus learned bilinear offsets and attention weights.  This is the mechanism
that lets the set-prediction detector converge in ~50 epochs instead of~500.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDeformableAttention(nn.Module):
    """Multi-scale deformable attention layer.

    Args:
        d_model: hidden dimension (must be divisible by n_heads).
        n_levels: number of feature pyramid levels.
        n_heads: number of attention heads.
        n_points: number of sampling points per query per level.
    """

    def __init__(self, d_model: int = 256, n_levels: int = 3, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_head = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_levels)])
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        # initialise offsets so sampling points spread around the reference point
        grid_init = torch.zeros(self.n_heads, self.n_levels, self.n_points, 2)
        for i in range(self.n_points):
            grid_init[:, :, i, 0] = i / self.n_points
            grid_init[:, :, i, 1] = i / self.n_points
        bias = grid_init.view(-1)
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(bias)

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        for m in self.value_proj:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value_features: list[torch.Tensor],
        spatial_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: (B, Nq, C).
            reference_points: (B, Nq, n_levels, 2) or (B, Nq, 1, 2) in [0, 1].
            value_features: list of (B, C, H_l, W_l).
            spatial_shapes: list of (H_l, W_l) matching value_features.

        Returns:
            (B, Nq, C) aggregated deformable attention output.
        """
        B, Nq, C = query.shape

        # offsets and weights from query
        sampling_offsets = self.sampling_offsets(query).view(
            B, Nq, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, Nq, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1).view(
            B, Nq, self.n_heads, self.n_levels, self.n_points
        )

        # broadcast reference points to n_levels if needed
        if reference_points.shape[2] == 1:
            ref = reference_points.expand(-1, -1, self.n_levels, -1)
        else:
            ref = reference_points
        # ref: (B, Nq, n_levels, 2) → (B, Nq, 1, n_levels, 1, 2)
        ref = ref.unsqueeze(2).unsqueeze(4)

        sampled_list = []
        for lid, (value, (H, W)) in enumerate(zip(value_features, spatial_shapes)):
            # project and split by head
            v = self.value_proj[lid](value.permute(0, 2, 3, 1))  # (B, H, W, C)
            v = (
                v.reshape(B, H * W, self.n_heads, self.d_head)
                .permute(0, 2, 3, 1)
                .reshape(B * self.n_heads, self.d_head, H, W)
            )

            # sampling locations for this level: (B, Nq, n_heads, n_points, 2)
            loc = ref[:, :, :, lid, :, :] + sampling_offsets[:, :, :, lid, :, :]
            # clamp inside image; grid_sample handles the [-1, 1] normalization
            loc = loc.clamp(0.0, 1.0)
            loc_norm = loc * 2.0 - 1.0

            # reshape grid to (B*n_heads, Nq, n_points, 2)
            grid = (
                loc_norm.permute(0, 2, 1, 3, 4)
                .reshape(B * self.n_heads, Nq * self.n_points, 2)
                .view(B * self.n_heads, Nq, self.n_points, 2)
            )

            sampled_l = F.grid_sample(
                v, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )  # (B*n_heads, d_head, Nq, n_points)
            sampled_l = sampled_l.permute(0, 2, 3, 1)  # (B*n_heads, Nq, n_points, d_head)
            sampled_list.append(sampled_l)

        sampled = torch.stack(sampled_list, dim=2)  # (B*n_heads, Nq, n_levels, n_points, d_head)
        weights = (
            attention_weights.permute(0, 2, 1, 3, 4)
            .reshape(B * self.n_heads, Nq, self.n_levels, self.n_points, 1)
        )
        output = (sampled * weights).sum(dim=(2, 3))  # (B*n_heads, Nq, d_head)
        output = (
            output.view(B, self.n_heads, Nq, self.d_head)
            .permute(0, 2, 1, 3)
            .reshape(B, Nq, C)
        )
        return self.output_proj(output)


def build_deformable_attention(cfg: dict) -> MultiScaleDeformableAttention:
    dec_cfg = cfg["model"]["decoder"]
    return MultiScaleDeformableAttention(
        d_model=dec_cfg["d_model"],
        n_levels=dec_cfg.get("n_feature_levels", 3),
        n_heads=dec_cfg["n_heads"],
        n_points=dec_cfg.get("n_sampling_points", 4),
    )
