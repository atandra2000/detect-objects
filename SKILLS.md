# SKILLS.md — detect-objects

> Read root `AGENTS.md` and `self.md` first, then `Vision/detect-objects/AGENTS.md`.
> Workspace rules are authoritative; this file adds **project-specific
> workflows** — the interactive playbooks that aren't obvious from a single
> `AGENTS.md` read. Design authority: `vision-research/DESIGN-detect-objects.md`.

This is the companion to `AGENTS.md` (this folder): that file holds the
architecture / source-tree / config-knob reference; **this file holds the
hands-on workflows** (smoke, train, eval, ship, debug).

---

## Skill 1 — Local smoke-test on a Mac (no CUDA)

The whole detector runs on CPU via fallback kernels (SDPA attention,
pure-PyTorch deformable `grid_sample`). This is the gate before any GPU run.

```bash
cd Vision/detect-objects
bash scripts/smoke.sh                  # 2-ep, batch 2, 512², no NaN
PYTHONPATH=. python3 -m pytest tests/ -q
```

**What it proves:** backbone/FPN shapes, deformable sampling stays inside the
image, Hungarian matcher finds optimal assignments, encoder path runs under
BF16 autocast + FP32 LayerNorm, inference CLI round-trips a synthetic image
to an annotated JPEG + JSON, and atomic checkpoints write
`detect_epoch_*.safetensors` + `.state.pt` with no NaN.
**Exit code 0 = green; do not push to the GPU pod without it.**

---

## Skill 2 — Launch the full 50-epoch training run (RunPod 2× 5090)

```bash
# Phase 0 + 1: pod env + data shards (one-time)
cd Vision/detect-objects
bash scripts/env_check.sh
python3 data/build_coco.py --out /workspace/data/coco

# Phase 4: the run (~5–7 days, ≥95% util target, ckpt every 5 ep)
CKPT_DIR=/workspace/runs/detect-objects bash scripts/launch_2x5090.sh
```

`launch_2x5090.sh` runs `torchrun --nproc-per-node=2`. Watch:

- `loss_total`, `loss_cls`, `loss_bbox`, `loss_giou` decreasing
- `imgs/sec` and GPU util (target ≥95% via `nvidia-smi`)
- **Epoch 20 early signal:** COCO mAP should be ≥ 30. If not, triage aug/lr/loss weights.
- If util < 85% → check `compile.enabled`; consider raising `batch_size_per_gpu` or enabling CUDA graphs.

**Resume from crash:** the loop auto-resumes from `detect_latest.*` (atomic
state + RNG). Just re-launch the same command.

---

## Skill 3 — Evaluate on COCO val (the product benchmark)

```bash
python3 -m training.eval_coco \
    --ckpt /workspace/runs/detect-objects/detect_latest.safetensors \
    --device cuda
```

This swaps the **EMA weights** in and runs COCO-API `pycocotools` bbox
evaluation on the full 5 K val set, printing mAP / mAP-50 / mAP-75 /
mAP-S / M / L. Acceptance target: **COCO mAP ≥ 40 (~42 expected).**
Log the numbers into `README.md` §Results.

---

## Skill 4 — Ship the product (safetensors + ONNX + samples)

```bash
# 1. swap EMA weights in → ship weights
python3 export/to_safetensors.py \
    --ckpt-dir /workspace/runs/detect-objects \
    --out detect_coco_final.safetensors

# 2. ONNX export (+ optional onnxruntime round-trip check)
python3 export/to_onnx.py --out detect_coco_head.onnx

# 3. 8 annotated detection PNGs for the README
bash scripts/dump_dets.sh detect_coco_final.safetensors /workspace/data/coco/val2017 samples
```

**Verify the CLI end-to-end** (the product contract):

```bash
python3 infer.py --ckpt detect_coco_final.safetensors \
    --in samples/000000000001.jpg --out samples/det.jpg --thr 0.5 --json dets.json
python3 demo_gradio.py --ckpt detect_coco_final.safetensors
```

---

## Skill 5 — Debug NaN / OOM / slow throughput (first-principles)

| Symptom | First-principles move |
|---------|------------------------|
| **NaN loss** | `clip_and_guard` already zeros grads. Drop to FP32 master-only for one step to localise. Inspect deformable offset gradients for overflow. |
| **OOM @ 32 GB** | raise `grad_accum` to 4/8; fall back to 640×800. |
| **util < 85%** | confirm `channels_last` applied *before* first forward; confirm SDPA flash path; raise `num_workers`; disable compile on deformable block if recompiling. |
| **slow throughput** | largest batch that fits; persistent workers; `pin_memory`; compile only backbone+encoder if deformable block causes recompiles. |
| **ckpt corruption** | atomic write is `.tmp → os.replace`; if a `.tmp` lingers, the prior write was killed mid-flight — delete it and resume from `detect_latest`. |
| **low mAP from-scratch R50** | expected ~42 at 50 ep without IN-1K pretrain. Extend schedule (no time cap) or raise augmentation. |

**Rule:** trace to math / memory-layout / architecture. Never paper over with a
`try/except` or a `.clamp()` you can't justify.

---

## Skill 6 — Reuse from the portfolio

- **Atomic ckpt / `channels_last` / logging / EMA** patterns — mirror
  `upscale-sr/utils/` and `LLM/LLaMA-3-Lite`.
- **DDP launch patterns** — mirror `Vision/StableDiffusion/` and `upscale-sr/`.
- **COCO eval scaffolding** — reuse conventions from `Vision/ActionRecognition/`
  where present.

---

## 7. Cross-references

- **Design:** `vision-research/DESIGN-detect-objects.md`
- **Execution plan:** `vision-research/EXECUTION-PLAN-detect-objects.md`
- **Candidate / fit:** `vision-research/candidates/03-detect-objects.md`
- **Workspace rules:** root `AGENTS.md`
- **Personal tone:** `self.md`
- **Sibling skills:** `.agents/skills/{computer-vision-multimodal, llm-architecture}/SKILL.md`
