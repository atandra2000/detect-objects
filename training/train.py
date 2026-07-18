"""DDP training loop for Detect-Objects.

Raw PyTorch, no HF Trainer.  BF16 autocast, FP32 master weights, FP32
LayerNorm, EMA, grad-clip, atomic safetensors checkpoints, channels_last on
Blackwell, torch.compile where stable.
"""
from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from data.coco_dataset import build_train_dataset, build_val_dataset
from data.detection_transforms import collate_detection
from models import build_model
from models.losses import build_loss
from models.matcher import build_matcher
from utils.checkpoint import CheckpointManager
from utils.config import load_config
from utils.logging import is_main_process, log_env_summary, rank, setup_logger, world_size
from utils.memory import apply_channels_last
from utils.schedule import build_schedule
from utils.stability import EMA, FP32Master, clip_and_guard


def _set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataloaders(cfg: dict, world_size: int, rank: int):
    train_ds = build_train_dataset(cfg)
    val_ds = build_val_dataset(cfg)
    data_cfg = cfg["data"]
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        sampler=train_sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        persistent_workers=data_cfg.get("persistent_workers", True) and data_cfg["num_workers"] > 0,
        prefetch_factor=data_cfg.get("prefetch_factor", 2),
        collate_fn=collate_detection,
        drop_last=True,
    )
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        sampler=val_sampler,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        persistent_workers=data_cfg.get("persistent_workers", True) and data_cfg["num_workers"] > 0,
        collate_fn=collate_detection,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler


def _build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    lr = train_cfg["lr"]
    backbone_lr = train_cfg.get("backbone_lr", lr)
    wd = train_cfg["weight_decay"]
    params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr_now = backbone_lr if "backbone" in n else lr
        params.append({"params": [p], "lr": lr_now, "weight_decay": wd})
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, fused=True)


def _setup_device() -> torch.device:
    if dist.is_available() and dist.is_initialized():
        return torch.device(f"cuda:{dist.get_rank()}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_worker(rank_id: int, world_size: int, cfg: dict, args) -> None:
    _set_seed(cfg.get("seed", 42) + rank_id)
    if world_size > 1:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank_id, world_size=world_size)

    device = _setup_device()
    logger = setup_logger("detect-objects.train", log_file=args.log_file)
    if is_main_process():
        log_env_summary(logger)

    model = build_model(cfg).to(device)
    if cfg["train"].get("channels_last", True) and torch.cuda.is_available():
        model = apply_channels_last(model)

    matcher = build_matcher(cfg)
    criterion = build_loss(cfg)

    optimizer = _build_optimizer(model, cfg)
    train_loader, val_loader, train_sampler = _build_dataloaders(cfg, world_size, rank())

    iters_per_epoch = len(train_loader)
    total_iters = cfg["train"]["epochs"] * iters_per_epoch
    warmup_iters = cfg["train"]["warmup_epochs"] * iters_per_epoch
    scheduler = build_schedule(optimizer, total_iters, warmup_iters)

    master = FP32Master(model, device)
    ema = EMA(model, decay=cfg["train"]["ema_decay"], device=device)

    ckpt_dir = Path(cfg["paths"]["runs_dir"])
    ckpt_mgr = CheckpointManager(ckpt_dir, keep_last=cfg["paths"]["keep_last"])
    resume = ckpt_mgr.load_latest(model, optimizer, scheduler, ema, device=str(device))
    start_epoch = resume.get("epoch", 0)

    if world_size > 1:
        model = DDP(model, device_ids=[rank_id], find_unused_parameters=False)

    compile_cfg = cfg["train"].get("compile", {})
    if compile_cfg.get("enabled", True) and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode=compile_cfg.get("mode", "max-autotune"))
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}; continuing eager")

    grad_accum = cfg["train"]["grad_accum"]
    grad_clip = cfg["train"]["grad_clip"]
    use_bf16 = cfg["train"].get("precision", "bf16") == "bf16" and torch.cuda.is_available()

    best_metric = resume.get("best_metric", float("inf"))
    scaler = None  # no GradScaler for BF16

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
                loss_dict = model.module.compute_loss(images, targets, matcher, criterion) if isinstance(model, DDP) else model.compute_loss(images, targets, matcher, criterion)
                loss = loss_dict["loss_total"]

            loss_scaled = loss / grad_accum
            loss_scaled.backward()

            if (i + 1) % grad_accum == 0:
                master.copy_grad_to_master(model.module if isinstance(model, DDP) else model)
                grad_norm = clip_and_guard(master.master.values(), max_norm=grad_clip)
                optimizer.step()
                master.copy_master_to_model(model.module if isinstance(model, DDP) else model)
                optimizer.zero_grad(set_to_none=True)
                ema.update(model.module if isinstance(model, DDP) else model)
                scheduler.step()

            epoch_loss += float(loss.detach())
            if is_main_process() and i % 50 == 0:
                logger.info(
                    f"ep {epoch} iter {i}/{len(train_loader)} | loss={loss.item():.3f} "
                    f"cls={loss_dict['loss_cls'].item():.3f} bbox={loss_dict['loss_bbox'].item():.3f} "
                    f"giou={loss_dict['loss_giou'].item():.3f} lr={optimizer.param_groups[0]['lr']:.2e}"
                )

        elapsed = time.time() - t0
        avg_loss = epoch_loss / max(len(train_loader), 1)
        if is_main_process():
            logger.info(f"ep {epoch} done | avg_loss={avg_loss:.4f} | elapsed={elapsed:.1f}s")

        # checkpoint every N epochs + best
        save_epoch = (epoch + 1) % cfg["paths"]["ckpt_every_epochs"] == 0
        if avg_loss < best_metric:
            best_metric = avg_loss
            save_epoch = True
        if is_main_process() and save_epoch:
            unwrapped = model.module if isinstance(model, DDP) else model
            ckpt_mgr.save(
                unwrapped,
                optimizer,
                scheduler,
                ema,
                epoch=epoch + 1,
                best_metric=best_metric,
                extra_meta={"avg_loss": avg_loss},
            )
            logger.info(f"saved checkpoint at epoch {epoch + 1}")

    if world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="path to YAML config")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--log-file", default="/workspace/runs/detect-objects/train.log")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.world_size <= 1:
        train_worker(0, 1, cfg, args)
    else:
        mp.spawn(train_worker, args=(args.world_size, cfg, args), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
