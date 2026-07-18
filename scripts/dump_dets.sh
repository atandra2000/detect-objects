#!/usr/bin/env bash
# Detect-Objects — generate 8 annotated detection PNGs for README.
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT="${1:-detect_coco_final.safetensors}"
IMG_DIR="${2:-/workspace/data/coco/val2017}"
OUT_DIR="${3:-samples}"
mkdir -p "${OUT_DIR}"

for img in $(find "${IMG_DIR}" -type f -name '*.jpg' | head -8); do
    base=$(basename "${img}" .jpg)
    python3 infer.py --ckpt "${CKPT}" --in "${img}" --out "${OUT_DIR}/${base}_det.jpg" --thr 0.5
done

echo "wrote annotated samples to ${OUT_DIR}"
