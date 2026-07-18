"""Set decoder: FA2/SDPA encoder + two-stage proposals + deformable decoder.

The encoder refines flattened FPN tokens.  The top-300 encoder tokens become
initial reference points for the decoder, which iteratively refines boxes
through 6 deformable cross-attention layers.  No anchors, no NMS.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deformable_attention import MultiScaleDeformableAttention


def _pos_embed_sine(
    h: int, w: int, d_model: int, temperature: float = 10000.0, device=None
) -> torch.Tensor:
    """Standard 2-D sine positional embedding, shape (1, h*w, d_model)."""
    y_embed = torch.arange(h, dtype=torch.float32, device=device).unsqueeze(1).repeat(1, w)
    x_embed = torch.arange(w, dtype=torch.float32, device=device).unsqueeze(0).repeat(h, 1)
    dim_t = torch.arange(d_model // 2, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / (d_model // 2))
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).view(h * w, d_model).unsqueeze(0)
    return pos


class MLP(nn.Module):
    """ReLU MLP with configurable depth; used for box/class heads."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer with self-attention + FFN."""

    def __init__(self, d_model: int = 256, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_head = d_model // n_heads
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, src: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        # Prefer SDPA/flash backend on Blackwell; MHA is the portable API.
        q = (src + pos) if pos is not None else src
        k = q
        v = src
        attn_out, _ = self.self_attn(q, k, v, need_weights=False)
        src = self.norm1(src + attn_out)
        ffn_out = self.ffn(src)
        return self.norm2(src + ffn_out)


class DecoderLayer(nn.Module):
    """Pre-norm decoder layer: self-attn + deformable cross-attn + FFN."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 3,
        n_points: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MultiScaleDeformableAttention(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        spatial_shapes: list[tuple[int, int]],
        reference_points: torch.Tensor,
    ) -> torch.Tensor:
        # self-attention
        q = tgt + query_pos
        self_out, _ = self.self_attn(q, q, tgt, need_weights=False)
        tgt = self.norm1(tgt + self_out)

        # cross-attention (deformable)
        q = tgt + query_pos
        # memory is (B, N_mem, C). Reshape to feature maps.
        cross_out = self.cross_attn(q, reference_points, memory_features(memory, spatial_shapes), spatial_shapes)
        tgt = self.norm2(tgt + cross_out)

        ffn_out = self.ffn(tgt)
        return self.norm3(tgt + ffn_out)


def memory_features(memory: torch.Tensor, spatial_shapes: list[tuple[int, int]]) -> list[torch.Tensor]:
    """Reshape flat memory (B, sum_l H_l*W_l, C) back to level feature maps."""
    B, _, C = memory.shape
    feats = []
    start = 0
    for H, W in spatial_shapes:
        n = H * W
        feats.append(memory[:, start : start + n, :].transpose(1, 2).view(B, C, H, W))
        start += n
    return feats


class SetDecoder(nn.Module):
    """Full encoder/decoder for set-prediction object detection."""

    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg["model"]
        enc_cfg = m["encoder"]
        dec_cfg = m["decoder"]
        self.d_model = enc_cfg["d_model"]
        self.n_levels = dec_cfg.get("n_feature_levels", 3)
        self.n_queries = dec_cfg["num_queries"]
        self.num_classes = cfg["num_classes"]
        self.iterative_refinement = dec_cfg.get("iterative_refinement", True)
        self.look_forward_twice = dec_cfg.get("look_forward_twice", True)

        # level embeddings added to positional encodings
        self.level_embed = nn.Parameter(torch.zeros(self.n_levels, self.d_model))

        # encoder
        enc_layer = EncoderLayer(
            self.d_model, enc_cfg["n_heads"], enc_cfg.get("dropout", 0.0)
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=enc_cfg["n_heads"],
                dim_feedforward=self.d_model * 4,
                dropout=enc_cfg.get("dropout", 0.0),
                activation="relu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=enc_cfg["n_layers"],
            enable_nested_tensor=False,
        )

        # proposal heads (two-stage)
        self.enc_score_head = nn.Linear(self.d_model, 1)
        self.enc_bbox_head = MLP(self.d_model, self.d_model, 4, 3)

        # decoder
        self.query_embed = nn.Embedding(self.n_queries, self.d_model * 2)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    self.d_model,
                    dec_cfg["n_heads"],
                    self.n_levels,
                    dec_cfg.get("n_sampling_points", 4),
                    dec_cfg.get("dropout", 0.0),
                )
                for _ in range(dec_cfg["n_layers"])
            ]
        )
        self.num_dec_layers = dec_cfg["n_layers"]

        # output heads
        self.class_embed = nn.Linear(self.d_model, self.num_classes + 1)
        self.bbox_embed = MLP(self.d_model, self.d_model, 4, 3)
        self.ref_point_head = MLP(self.d_model, self.d_model, 2, 2)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.level_embed)
        nn.init.normal_(self.query_embed.weight)

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Args: list of (B, C, H_l, W_l). Returns class logits, boxes (both layers)."""
        B = features[0].shape[0]
        device = features[0].device

        # flatten FPN features and build level-aware positional embeddings
        srcs, pos_embeds, spatial_shapes = [], [], []
        for lvl, feat in enumerate(features):
            _, C, H, W = feat.shape
            spatial_shapes.append((H, W))
            feat_flat = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            pos = _pos_embed_sine(H, W, self.d_model, device=device)
            pos = pos + self.level_embed[lvl].view(1, 1, C)
            srcs.append(feat_flat)
            pos_embeds.append(pos.expand(B, -1, -1))

        src_flat = torch.cat(srcs, dim=1)  # (B, N_mem, C)
        pos_flat = torch.cat(pos_embeds, dim=1)

        # encoder (pre-norm TransformerEncoder expects src + optional src_key_padding_mask)
        memory = self.encoder(src_flat + pos_flat)

        # two-stage proposals: top-300 encoder outputs
        n_mem = memory.shape[1]
        topk = min(self.n_queries, n_mem)
        enc_scores = self.enc_score_head(memory).squeeze(-1)  # (B, N_mem)
        topk_indices = torch.topk(enc_scores, topk, dim=1).indices  # (B, topk)
        enc_boxes = self.enc_bbox_head(memory).sigmoid()  # (B, N_mem, 4)
        topk_boxes = torch.gather(
            enc_boxes, 1, topk_indices.unsqueeze(-1).expand(-1, -1, 4)
        )
        # initial reference points are box centres of top-k proposals
        init_reference = topk_boxes[:, :, :2]  # (B, topk, 2)
        # pad to full query count with learned content queries
        if topk < self.n_queries:
            pad = self.n_queries - topk
            pad_ref = init_reference[:, :1, :].expand(-1, pad, -1)
            init_reference = torch.cat([init_reference, pad_ref], dim=1)

        # decoder queries: split learned embed into content + positional parts
        query_embed = self.query_embed.weight  # (Nq, 2*C)
        tgt = query_embed[:, : self.d_model].unsqueeze(0).expand(B, -1, -1)
        query_pos = query_embed[:, self.d_model :].unsqueeze(0).expand(B, -1, -1)

        # optionally seed decoder content from proposal features
        if topk == self.n_queries:
            proposal_features = torch.gather(
                memory, 1, topk_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
            )
            tgt = proposal_features + tgt

        # iterative refinement through decoder layers
        reference_points = init_reference.unsqueeze(2)  # (B, Nq, 1, 2)
        intermediate_classes, intermediate_boxes = [], []
        for lid, layer in enumerate(self.decoder_layers):
            tgt = layer(tgt, query_pos, memory, pos_flat, spatial_shapes, reference_points)
            outputs_class = self.class_embed(tgt)
            tmp = self.bbox_embed(tgt)
            if self.iterative_refinement:
                if lid == 0:
                    tmp[..., :2] += init_reference
                else:
                    tmp[..., :2] += reference_points.squeeze(2)
            outputs_coord = tmp.sigmoid()
            intermediate_classes.append(outputs_class)
            intermediate_boxes.append(outputs_coord)
            if self.iterative_refinement and lid < self.num_dec_layers - 1:
                # next layer's reference is current box centre
                reference_points = outputs_coord[:, :, :2].detach().unsqueeze(2)

        return torch.stack(intermediate_classes), torch.stack(intermediate_boxes)


def build_set_decoder(cfg: dict) -> SetDecoder:
    return SetDecoder(cfg)
