"""Ship the final product checkpoint as EMA-swapped safetensors."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models import DetectObjectsModel
from utils.checkpoint import CheckpointManager
from utils.config import load_config
from utils.stability import EMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt-dir", required=True, help="directory containing detect_latest.safetensors")
    parser.add_argument("--out", default="detect_coco_final.safetensors")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectObjectsModel(cfg).to(device)
    ema = EMA(model, decay=cfg["train"]["ema_decay"], device=device)

    ckpt_mgr = CheckpointManager(args.ckpt_dir)
    ckpt_mgr.load_latest(model, optimizer=None, scheduler=None, ema=ema, device=str(device))
    backup = ema.apply_shadow(model)

    from safetensors.torch import save_file
    sd = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, args.out)
    ema.restore(model, backup)
    print(f"shipped {args.out}")


if __name__ == "__main__":
    main()
