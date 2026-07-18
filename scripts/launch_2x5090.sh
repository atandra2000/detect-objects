#!/usr/bin/env bash
# Detect-Objects — launch full 50-epoch DDP training on RunPod 2× RTX 5090.
set -euo pipefail

cd "$(dirname "$0")/.."

export CKPT_DIR="${CKPT_DIR:-/workspace/runs/detect-objects}"
export PYTHONPATH="${PYTHONPATH:-}:."

# Background GPU util probe
python3 training/profiler.py --out "${CKPT_DIR}/util.log" &
PROFILER_PID=$!
trap 'kill $PROFILER_PID 2>/dev/null || true' EXIT

torchrun \
    --nnodes=1 \
    --nproc-per-node=2 \
    training/train.py \
    --world-size 2 \
    --log-file "${CKPT_DIR}/train.log" \
    "$@"
