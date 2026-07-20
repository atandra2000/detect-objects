# AGENTS.md — detect-objects

> Read root `AGENTS.md` and `self.md` first. Workspace rules are authoritative;
> this file adds **project-specific** rules only. **This file wins over root
> for `Vision/detect-objects/` only.** Design authority:
> `vision-research/DESIGN-detect-objects.md` +
> `vision-research/EXECUTION-PLAN-detect-objects.md` +
> `vision-research/candidates/03-detect-objects.md`.

> **Project:** `Vision/detect-objects/` · **Type:** object-detection **product**
> (ResNet-50 + FPN + FA2 encoder + pure-PyTorch multi-scale deformable
> attention + two-stage decoder) · **Hardware:** 2× RTX 5090 32 GB
> (Blackwell sm_120) DDP, no time cap · **Stack:** PyTorch 2.12, BF16,
> FA2/SDPA, channels_last, torch.compile, EMA, atomic safetensors ckpts.

This file is the **developer reference** for the project. The design doc holds
the *what*; this file holds the *where-it-lives + how-to-touch-it*.

---

## 0. Operating principles (non-negotiable — from root `AGENTS.md §1`)

- **Raw PyTorch first.** No HF Trainer / Lightning. The training loop is
  hand-written DDP in `training/train.py`.
- **First-principles debugging.** Trace NaN / OOM / slow-throughput to math,
  memory-layout, or architecture — never superficial patches. See §6 below.
- **BF16 over FP16** (Ampere/Blackwell) → **no `GradScaler`**.
- **`channels_last`** on backbone convs before first forward
  (`utils/memory.apply_channels_last`).
- **Atomic checkpoints:** `safetensors` weights + `torch.save` state → `.tmp`
  → `os.replace`; full RNG + optim + sched + EMA state. **No `pickle` for
  weights.** (`utils/checkpoint.py`.)
- **No magic numbers.** Architectural constants live in
  `configs/coco_r50_detect_2x5090.yaml` and are named + documented.
- **Prefer `tools/` and `scripts/`** over one-off shells.

---

## 1. Quick commands

```bash
# ── environment (RunPod 2× 5090) ──────────────────────────────────────────
cd Vision/detect-objects
bash scripts/env_check.sh                      # verify deps + sm_120 + kernels
pip install -r requirements.txt                # torch, safetensors, pycocotools, scipy, gradio, onnx
pip install flash-attn                         # Blackwell sm_120 build

# ── data ────────────────────────────────────────────────────────────────────
python3 data/build_coco.py --out /workspace/data/coco

# ── smoke (local Mac OK — CPU fallback, no real data) ─────────────────────
bash scripts/smoke.sh                          # 2-ep, batch 2, 512², no NaN

# ── tests ─────────────────────────────────────────────────────────────────
PYTHONPATH=. python3 -m pytest tests/ -q

# ── full training (RunPod 2× 5090, ~5–7 days, 50 ep) ───────────────────────
bash scripts/launch_2x5090.sh

# ── eval + product surface ────────────────────────────────────────────────
python3 -m training.eval_coco --ckpt /workspace/runs/detect-objects/detect_latest.safetensors
python3 infer.py --ckpt detect_coco_final.safetensors --in img.jpg --out det.jpg --thr 0.5
python3 demo_gradio.py --ckpt detect_coco_final.safetensors

# ── ship ──────────────────────────────────────────────────────────────────
python3 export/to_safetensors.py --ckpt-dir /workspace/runs/detect-objects --out detect_coco_final.safetensors
python3 export/to_onnx.py --out detect_coco_head.onnx
bash scripts/dump_dets.sh detect_coco_final.safetensors /workspace/data/coco/val2017 samples
```

---

## 2. Source tree → responsibility

```
Vision/detect-objects/
├── configs/coco_r50_detect_2x5090.yaml   # single source of truth for shapes/recipe
├── data/
│   ├── build_coco.py                      # COCO 2017 → shards
│   ├── coco_dataset.py                    # tar-shard COCO loader
│   └── detection_transforms.py          # box-aware aug + collate
├── models/
│   ├── backbone_resnet.py               # ★ from-scratch R50 + FPN (channels_last)
│   ├── deformable_attention.py          # ★ 4-pt multi-scale deformable attn (grid_sample)
│   ├── set_decoder.py                   # ★ FA2/SDPA encoder + 2-stage decoder + iterative refine
│   ├── matcher.py                       # ★ Hungarian (scipy)
│   ├── losses.py                        # ★ focal + L1 + GIoU
│   └── __init__.py                      # DetectObjectsModel factory
├── training/
│   ├── train.py                         # DDP main loop (BF16, EMA, grad-clip, atomic ckpt)
│   └── eval_coco.py                     # pycocotools mAP
├── infer.py                             # ★ product CLI
├── demo_gradio.py                       # ★ browser demo with confidence slider
├── export/
│   ├── to_safetensors.py                # ship detect_coco_final.safetensors (EMA-swapped)
│   └── to_onnx.py                       # full-model ONNX export
├── utils/{config,checkpoint,logging,memory,schedule,stability}.py
├── scripts/{env_check,smoke,launch_2x5090,dump_dets}.sh
└── tests/{test_deformable_geom,test_matcher,test_fa2_encoder,test_infer_cli}.py
```

---

## 3. Architecture at a glance

**Set-prediction detector (DETR/Deformable-DETR/DINO-style).** A from-scratch
**ResNet-50 + FPN** backbone produces multi-scale features (1/8, 1/16, 1/32).
A **6-layer encoder** refines flattened FPN tokens with self-attention
(SDPA/flash backend). The top-300 encoder outputs seed **two-stage proposals**,
which a **6-layer decoder** iteratively refines using **multi-scale
deformable cross-attention** (pure PyTorch `grid_sample`, 4 reference points
per level). Class + box heads predict 80 COCO classes + 1 no-object.
No anchors, no NMS.

**Training pair generator.** Full COCO 2017 (118 K train, 5 K val, 80 classes)
loaded via tar shards. Images are resized (short side 480–800), randomly
cropped preserving GT boxes, mirrored, and normalized.

**Matching + loss.** `scipy.optimize.linear_sum_assignment` matches 300
predictions to GT per image on a cost of focal class + 5·L1 + 2·GIoU. Loss is
sigmoid focal classification + 5·L1 box + 2·GIoU on the matched set.

---

## 4. Config → knob map (single source of truth is YAML)

| Knob | YAML path | Default | Notes |
|------|-----------|---------|-------|
| Backbone | `model.backbone.name` | resnet50 | from-scratch, no IN-1K pretrain |
| FPN width | `model.fpn.out_channels` | 256 | matches encoder/decoder |
| Encoder layers/heads | `model.encoder.{n_layers,n_heads}` | 6 / 8 | SDPA/flash self-attn |
| Decoder layers/heads | `model.decoder.{n_layers,n_heads}` | 6 / 8 | deformable cross-attn |
| Queries | `model.decoder.num_queries` | 300 | two-stage proposals |
| Sampling points | `model.decoder.n_sampling_points` | 4 | per level per query |
| Matcher cost | `model.matcher.{class,bbox,giou}_cost` | 1 / 5 / 2 | Hungarian weights |
| Loss weights | `model.losses.{class,bbox,giou}_weight` | 1 / 5 / 2 | loss scaling |
| Focal α/γ | `model.losses.focal_{alpha,gamma}` | 0.25 / 2.0 | focal classification |
| Effective batch | `train.batch_size_per_gpu × grad_accum × world` | 4×4×2 = 32 | as spec |
| LR backbone | `train.backbone_lr` | 2e-5 | 10× lower than detector head |
| Schedule | `train.{epochs,warmup_epochs,drop_epoch}` | 50 / 1 / 40 | cosine + drop |
| EMA decay | `train.ema_decay` | 0.9999 | with warmup ramp |
| Grad-clip | `train.grad_clip` | 1.0 | NaN guard zeros grads if non-finite |
| Compile | `train.compile.{enabled,mode}` | true / max-autotune | eager fallback on failure |
| Ckpt every | `paths.ckpt_every_epochs` | 5 | + best-loss |

> **Never hard-code a shape or LR in Python.** Read it from the config via
> `utils/config.load_config()` and the `build_*` helpers in `models/__init__.py`.

---

## 5. Precision, kernels, memory-format (the stability contract)

- **BF16 autocast** on model forward; **FP32 master weights**
  **FP32 LayerNorm/GroupNorm/BN** (stable
  under BF16). **No `GradScaler`** (BF16, not FP16).
- **FA2 / SDPA flash backend:** encoder self-attention uses
  `torch.nn.MultiheadAttention`, which dispatches to the fastest SDPA
  backend (flash on sm_120 when available). Deformable cross-attention is pure
  PyTorch `F.grid_sample`.
- **`channels_last`** on the backbone before the first forward
  (`utils.memory.apply_channels_last`) — Blackwell sm_120 mandate.
- **`torch.compile(mode="max-autotune")`** on the full model where stable; if
  the deformable block recompiles, disable it with `compile.enabled: false`.
- **EMA** shadow on the full model; eval/inference swaps EMA in.
- **Grad-clip 1.0** + **NaN guard** (`utils.stability.clip_and_guard`).

---

## 6. Failure-mode triage (first-principles — DESIGN §8)

| Symptom | Check first |
|---------|-------------|
| FA2/flash backend unavailable | SDPA automatically falls back to math/mem-eff; check `torch.backends.cuda.flash_sdp_enabled()` |
| mAP low at 50 ep (from-scratch R50) | expected ~42 without IN-1K pretrain; extend schedule (no time cap) or raise augmentation |
| OOM at batch 4/GPU, 800×1333 | raise `grad_accum` to 4/8, or fall back to 640×800 |
| Deformable sampling outside image | `grid_sample` padding is "zeros"; verify reference-point clamp in `deformable_attention.py` |
| Hungarian matcher slow | it runs on CPU (scipy) for ≤300×~10 matrices — should be <5% of step time |
| `torch.compile` recompiles on deformable block | set `train.compile.enabled: false` and compile only backbone+encoder if needed |
| ONNX export of deformable attention fails | export full model with fixed input size; on failure, export backbone+encoder only |
| NaN loss | `clip_and_guard` zeros grads; localise with FP32-only step or inspect deformable offsets for overflow |

---

## 7. Subagent routing (when to spawn which)

Spawn via the Agent tool when the trigger matches (root `AGENTS.md §2`):

| Agent | Trigger for this project |
|-------|--------------------------|
| `pytorch-debugger` | NaN loss, OOM, slow throughput, grad issues, ckpt corruption in `training/train.py` or models |
| `data-pipeline-engineer` | COCO shard build, tar-shard loader, box-aware augmentation bugs |
| `ml-benchmarks` | imgs/sec, memory profiling, ≥95% util probe, mAP tracking |
| `project-finder` | "which file owns the deformable attention?", "where is the matcher?" |
| `research-engineer-dev` | (only if extending the platform itself — not this project) |

> For architecture / attention / memory-layout questions, prefer the
> `.agents/skills/{computer-vision-multimodal, llm-architecture}` skills.

---

## 8. Hard don'ts (from root `AGENTS.md §6`)

- Don't bypass this file for `Vision/detect-objects/` questions.
- Don't hard-code shapes / LRs / schedule constants in Python — read the YAML.
- Don't `pickle` checkpoints — `safetensors` weights + `torch.save` state.
- Don't rewrite in JAX/TF.
- Don't add HF Trainer / Lightning — hand-written DDP only.
- Don't ship a checkpoint without the EMA swap (`export/to_safetensors.py`).

---

## 9. Vault sync (root `AGENTS.md §8`)

`Vision/detect-objects/` lives under `CoreProjects/` → the workspace Stop hook
auto-mirrors every new/modified `.md` into `~/Documents/obsidian`.
**Create new `.md` here, never in the vault.** Never hand-edit mirror files.
