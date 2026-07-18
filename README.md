# Detect-Objects

A trained **object-detection product** built from scratch for the
`vision-research` portfolio. Give it any image and it returns bounding boxes +
class labels + confidence scores across the 80 COCO classes.

> **Architecture:** ResNet-50 + FPN → FA2/SDPA encoder → two-stage deformable
> decoder (RT-DETR/DINO-style). No anchors, no NMS.
>
> **Hardware target:** 2× NVIDIA RTX 5090 32 GB (Blackwell sm_120) DDP.
>
> **Training data:** full COCO 2017 (118 K train, 5 K val).
>
> **Product surface:** `infer.py` CLI + `demo_gradio.py` web demo + safetensors
> / ONNX export for TensorRT deployment.

---

## 1. Headline capability (target)

"Detect-Objects: a from-scratch multi-scale deformable set-prediction detector
trained on full COCO 2017 in ~5–7 days on 2× RTX 5090 32 GB, reaching ≈ 42
mAP on COCO val. Ships an inference CLI, a Gradio demo, and safetensors + ONNX
export."

---

## 2. Quick start

```bash
cd Vision/detect-objects
pip install -r requirements.txt

# verify environment (RunPod 2× 5090)
bash scripts/env_check.sh

# run local smoke test (CPU OK)
bash scripts/smoke.sh

# run unit tests
PYTHONPATH=. python3 -m pytest tests/ -q
```

### Inference on a single image

```bash
python3 infer.py \
    --ckpt detect_coco_final.safetensors \
    --in photo.jpg \
    --out det.jpg \
    --thr 0.5 \
    --json detections.json
```

### Gradio demo

```bash
python3 demo_gradio.py --ckpt detect_coco_final.safetensors --port 7860
# → open http://localhost:7860, upload an image, adjust the confidence slider
```

---

## 3. Training

### Prepare data

```bash
python3 data/build_coco.py --out /workspace/data/coco
```

### Launch full DDP run

```bash
CKPT_DIR=/workspace/runs/detect-objects bash scripts/launch_2x5090.sh
```

Training uses BF16 autocast, FP32 master weights, FP32 LayerNorm, EMA,
gradient clipping, atomic safetensors checkpoints, and `channels_last` on the
backbone. Resume is automatic from `detect_latest.safetensors`.

### Evaluate

```bash
python3 -m training.eval_coco \
    --ckpt /workspace/runs/detect-objects/detect_latest.safetensors
```

---

## 4. Results

| Metric | Value |
|--------|-------|
| COCO mAP | *pending 50-epoch run on 2× RTX 5090* |
| mAP-50 | *pending* |
| mAP-75 | *pending* |
| mAP-S | *pending* |
| mAP-M | *pending* |
| mAP-L | *pending* |
| GPU util | target ≥ 95% |
| Wall time | ~5–7 days on 2× RTX 5090 32 GB |

### Local verification (CPU)

```bash
bash scripts/smoke.sh              # 2-ep, batch 2, 512², forward+backward, no NaN
PYTHONPATH=. python3 -m pytest tests/ -q
```

Smoke test and all unit tests pass locally:
- `test_deformable_geom.py` — 4-pt sampling stays inside image
- `test_matcher.py` — Hungarian finds optimal toy assignment
- `test_fa2_encoder.py` — encoder path under BF16 autocast + FP32 LayerNorm
- `test_infer_cli.py` — synthetic image → annotated JPEG + JSON

Sample detections will be added to this README after the GPU training run.

---

## 5. Architecture details

| Component | Details | Params |
|-----------|---------|:------:|
| Backbone | ResNet-50 from scratch + FPN (3 levels) | ~25 M |
| Encoder | 6 layers, d=256, 8 heads, FA2/SDPA self-attention | ~10 M |
| Decoder | 6 layers, 300 queries, multi-scale deformable cross-attention | ~15 M |
| **Total** | | **~50 M** |

- **Backbone:** `models/backbone_resnet.py` — channels_last convs, no ImageNet pretrain.
- **Deformable attention:** `models/deformable_attention.py` — pure PyTorch `grid_sample`.
- **Encoder/decoder:** `models/set_decoder.py` — FA2 encoder + two-stage proposals + iterative refinement.
- **Matching/loss:** `models/matcher.py` + `models/losses.py` — Hungarian + focal/L1/GIoU.
- **Training:** `training/train.py` — hand-written DDP loop.

---

## 6. Export

```bash
# ship EMA-swapped safetensors
python3 export/to_safetensors.py --ckpt-dir /workspace/runs/detect-objects --out detect_coco_final.safetensors

# ONNX export (TensorRT-ready)
python3 export/to_onnx.py --out detect_coco_head.onnx
```

---

## 7. Project files

- `AGENTS.md` — developer reference and routing
- `SKILLS.md` — smoke/train/eval/ship/debug playbooks
- `configs/coco_r50_detect_2x5090.yaml` — single source of truth

---

*Author: Atandra Bharati · Part of the CoreProjects vision-research portfolio.*
