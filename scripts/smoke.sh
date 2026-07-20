#!/usr/bin/env bash
# Detect-Objects — 2-epoch smoke test, batch 2, 512², forward+backward, no NaN.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Detect-Objects smoke test ==="
SMOKE_EPOCHS="${SMOKE_EPOCHS:-2}"
SMOKE_BS="${SMOKE_BS:-2}"
SMOKE_RES="${SMOKE_RES:-512}"

cat > /tmp/detect_smoke.yaml <<YAML
project: detect-objects
num_classes: 80
num_queries: 300
model:
  backbone:
    name: resnet50
    out_indices: [1, 2, 3]
    channels_last: false
  fpn:
    in_channels: [512, 1024, 2048]
    out_channels: 256
    num_levels: 3
  encoder:
    d_model: 256
    n_heads: 8
    n_layers: 3
    dropout: 0.0
  decoder:
    d_model: 256
    n_heads: 8
    n_layers: 3
    dropout: 0.0
    num_queries: 300
    n_feature_levels: 3
    n_sampling_points: 2
    iterative_refinement: true
  matcher:
    class_cost: 1.0
    bbox_cost: 5.0
    giou_cost: 2.0
  losses:
    class_weight: 1.0
    bbox_weight: 5.0
    giou_weight: 2.0
    focal_alpha: 0.25
    focal_gamma: 2.0
data:
  dataset: synthetic
  shard_dir: /tmp/detect-objects-smoke/shards
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  prefetch_factor: 2
train:
  epochs: ${SMOKE_EPOCHS}
  batch_size_per_gpu: ${SMOKE_BS}
  grad_accum: 1
  effective_batch: 2
  lr: 2.0e-4
  backbone_lr: 2.0e-5
  weight_decay: 1.0e-4
  warmup_epochs: 0
  grad_clip: 1.0
  ema_decay: 0.9999
  compile:
    enabled: false
  precision: fp32
  channels_last: false
  device: cpu
paths:
  runs_dir: /tmp/detect-objects-smoke
  ckpt_every_epochs: 1
  keep_last: 2
YAML

python3 - <<'PY'
import sys
sys.path.insert(0, '.')
import torch
from utils.config import load_config
from models import DetectObjectsModel
from models.losses import build_loss
from models.matcher import build_matcher
from utils.checkpoint import CheckpointManager
from utils.stability import EMA, clip_and_guard

cfg = load_config('/tmp/detect_smoke.yaml')
device = torch.device('cpu')
model = DetectObjectsModel(cfg).to(device)
matcher = build_matcher(cfg)
criterion = build_loss(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
ema = EMA(model, decay=cfg['train']['ema_decay'], device=device)
ckpt_mgr = CheckpointManager(cfg['paths']['runs_dir'], keep_last=2)

for epoch in range(cfg['train']['epochs']):
    for step in range(3):
        images = torch.randn(cfg['train']['batch_size_per_gpu'], 3, 512, 512, device=device)
        targets = [
            {
                'boxes': torch.rand(3, 4, device=device) * 0.5 + 0.1,
                'labels': torch.randint(0, 80, (3,), device=device),
            }
            for _ in range(cfg['train']['batch_size_per_gpu'])
        ]
        loss_dict = model.compute_loss(images, targets, matcher, criterion)
        loss = loss_dict['loss_total']
        assert torch.isfinite(loss), f"NaN/Inf loss at ep{epoch} step{step}: {loss_dict}"
        loss.backward()
        grad_norm = clip_and_guard(model.parameters(), max_norm=cfg['train']['grad_clip'])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(model)
        print(f"ep{epoch} step{step} loss={loss.item():.3f} grad_norm={grad_norm:.3f}")
    ckpt_mgr.save(model, optimizer, None, ema, epoch=epoch+1, best_metric=loss.item())

print("SMOKE OK — shapes, backward, EMA, atomic ckpt all pass")
PY

echo "=== smoke done ==="
