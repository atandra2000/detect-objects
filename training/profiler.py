"""GPU utilisation probe used by launch_2x5090.sh to prove ≥95% util."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def _gpu_util(idx: int = 0) -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={idx}", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        return float(out.splitlines()[0]) if out else 0.0
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--interval", default=5, type=float)
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        while True:
            util0 = _gpu_util(0)
            util1 = _gpu_util(1)
            line = f"{time.time():.0f} gpu0={util0:.1f} gpu1={util1:.1f}\n"
            f.write(line)
            f.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
