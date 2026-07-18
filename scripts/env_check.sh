#!/usr/bin/env bash
# Detect-Objects — verify RunPod 2× RTX 5090 environment.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Detect-Objects environment check ==="
python3 - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda {torch.version.cuda}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {p.name} | {p.total_memory/1e9:.1f} GB | sm_{p.major}{p.minor}")
        assert (p.major, p.minor) >= (8, 9), "Blackwell/Ampere required"
else:
    print("WARNING: CUDA unavailable — this is a CPU-only environment.")

for mod in ["flash_attn", "scipy", "pycocotools", "safetensors", "yaml", "PIL", "gradio"]:
    try:
        __import__(mod)
        print(f"ok: {mod}")
    except Exception as e:
        print(f"MISSING: {mod} ({e})")
PY

echo "=== env_check done ==="
