#!/usr/bin/env bash
# Detect-Objects — launch full 50-epoch DDP training on RunPod 2× RTX 5090.
set -euo pipefail

cd "$(dirname "$0")/.."

export CKPT_DIR="${CKPT_DIR:-/workspace/runs/detect-objects}"
export PYTHONPATH="${PYTHONPATH:-}:."

torchrun \
    --nnodes=1 \
    --nproc-per-node=2 \
    training/train.py \
    --world-size 2 \
    --log-file "${CKPT_DIR}/train.log" \
    "$@"
