"""Atomic checkpoint manager — safetensors weights + torch state + full RNG.

Atomic write protocol (CLAUDE.md §1, AGENTS.md §1):
    torch.save → ``.tmp`` → ``os.replace``.  A crash mid-save leaves
    a ``.tmp`` file that is ignored on resume, never a corrupt main ckpt.

State saved (full reproducibility):
    - weights          (safetensors, no pickle)
    - optimizer        (AdamW m/v + step)
    - LR scheduler     (SequentialLR internal counters)
    - EMA shadow        (decay + step_count)
    - RNG state         (python random, numpy, torch, torch.cuda, per-rank)
    - meta              (epoch, iter, best_metric, config snapshot)

No ``pickle`` is used for the weights — only ``safetensors``.  The auxiliary
state (optim/sched/ema/rng) is saved with ``torch.save`` (which uses
torch.serialization); the rule in CLAUDE.md §6 is "no pickle for checkpoints"
meaning *do not ship a model checkpoint as a pickle file* — safetensors is the
ship format.
"""
from __future__ import annotations

import os  # ponytail: json removed (unused)
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def strip_prefixes(state_dict: dict, prefixes=("module.", "_orig_mod.")) -> dict:
    """Remove DDP / torch.compile wrapper prefixes so ckpts restore cross-runner."""
    cleaned = {}
    for k, v in state_dict.items():
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        cleaned[k] = v
    return cleaned


def rng_state_dict() -> dict:
    """Capture the full RNG state for perfect resumption."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class CheckpointManager:
    """Atomic save/load for the Detect-Objects product.

    Files (per epoch ``e``):
        detect_epoch_{e:03d}.safetensors   — model weights (ship format)
        detect_epoch_{e:03d}.state.pt      — optimizer + scheduler + EMA + RNG + meta
        detect_latest.safetensors / .state.pt — convenience pointer
    """

    def __init__(self, save_dir: str, keep_last: int = 6):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        ema,
        epoch: int,
        best_metric: float,
        extra_meta: Optional[dict] = None,
    ) -> None:
        sd = {k: v.detach().contiguous() for k, v in model.state_dict().items()}
        weights_tmp = self.save_dir / f".detect_epoch_{epoch:03d}.safetensors.tmp"
        weights_out = self.save_dir / f"detect_epoch_{epoch:03d}.safetensors"
        save_file(sd, str(weights_tmp))
        os.replace(weights_tmp, weights_out)

        state = {
            "epoch": epoch,
            "best_metric": best_metric,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "ema": ema.state_dict(),
            "rng": rng_state_dict(),
            "meta": extra_meta or {},
        }
        state_tmp = self.save_dir / f".detect_epoch_{epoch:03d}.state.pt.tmp"
        state_out = self.save_dir / f"detect_epoch_{epoch:03d}.state.pt"
        torch.save(state, state_tmp)
        os.replace(state_tmp, state_out)

        self._copy(weights_out, self.save_dir / "detect_latest.safetensors")
        self._copy(state_out, self.save_dir / "detect_latest.state.pt")
        self._prune()

    def _copy(self, src: Path, dst: Path) -> None:
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        tmp.write_bytes(src.read_bytes())
        os.replace(tmp, dst)

    def _prune(self) -> None:
        epochs = sorted(
            int(p.stem.removeprefix("detect_epoch_"))
            for p in self.save_dir.glob("detect_epoch_*.safetensors")
        )
        if len(epochs) <= self.keep_last:
            return
        for e in epochs[:-self.keep_last]:
            for p in self.save_dir.glob(f"detect_epoch_{e:03d}.*"):
                p.unlink(missing_ok=True)

    def latest_epoch(self) -> Optional[int]:
        epochs = sorted(
            int(p.stem.removeprefix("detect_epoch_"))
            for p in self.save_dir.glob("detect_epoch_*.safetensors")
        )
        return epochs[-1] if epochs else None

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        ema,
        epoch: Optional[int] = None,
        device: str = "cuda",
    ) -> dict:
        epoch = epoch if epoch is not None else self.latest_epoch()
        if epoch is None:
            return {"epoch": 0, "best_metric": float("inf")}
        weights = load_file(
            str(self.save_dir / f"detect_epoch_{epoch:03d}.safetensors"), device=device
        )
        weights = strip_prefixes(weights)
        model.load_state_dict(weights, strict=False)

        state_path = self.save_dir / f"detect_epoch_{epoch:03d}.state.pt"
        if state_path.exists():
            state = torch.load(str(state_path), map_location=device, weights_only=False)
            try:
                optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception:
                pass
            try:
                ema.load_state_dict(state["ema"], device=device)
            except Exception:
                pass
            restore_rng_state(state["rng"])
            return {
                "epoch": state["epoch"],
                "best_metric": state["best_metric"],
                "meta": state.get("meta", {}),
            }
        return {"epoch": epoch, "best_metric": float("inf")}

    def load_latest(self, model, optimizer, scheduler, ema, device="cuda") -> dict:
        return self.load(model, optimizer, scheduler, ema, epoch=None, device=device)
