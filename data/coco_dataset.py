"""COCO dataset loaders: from webdataset shards or from annotation JSON."""
from __future__ import annotations

import json
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.detection_transforms import (
    COCODetectionTransform,
    COCODetectionValTransform,
)


def _load_image_bytes(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


class COCOTarDataset(Dataset):
    """Read COCO samples from webdataset-style tar shards."""

    def __init__(
        self,
        shard_dir: Path | str,
        split: str,
        transform: COCODetectionTransform | COCODetectionValTransform | None = None,
    ):
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.transform = transform
        self.shards = sorted(self.shard_dir.glob(f"{split}-*.tar"))
        self.samples = []
        for shard_path in self.shards:
            with tarfile.open(shard_path, "r") as tar:
                members = tar.getmembers()
                keys = {m.name.split(".")[0] for m in members if m.name.endswith(".jpg")}
                self.samples.extend([(shard_path, k) for k in sorted(keys)])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        shard_path, key = self.samples[idx]
        with tarfile.open(shard_path, "r") as tar:
            jpg_member = tar.getmember(f"{key}.jpg")
            json_member = tar.getmember(f"{key}.json")
            img = _load_image_bytes(tar.extractfile(jpg_member).read())
            meta = json.loads(tar.extractfile(json_member).read().decode("utf-8"))
        boxes = np.array(meta["boxes"], dtype=np.float32)
        labels = meta["labels"]
        if self.transform is not None:
            img, target = self.transform(img, boxes, labels)
            target["image_id"] = meta.get("image_id", 0)
            target["orig_size"] = (meta.get("height", img.size[1]), meta.get("width", img.size[0]))
            return img, target
        return img, boxes, labels


class COCOAnnotationDataset(Dataset):
    """Load from COCO annotation JSON (used when shards are not yet built)."""

    def __init__(
        self,
        ann_file: Path | str,
        img_dir: Path | str,
        transform=None,
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform
        with open(ann_file, "r") as f:
            coco = json.load(f)
        self.images = coco["images"]
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(coco["categories"])}
        self.anns_by_img: dict[int, list] = {}
        for ann in coco["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_info = self.images[idx]
        img_path = self.img_dir / img_info["file_name"]
        img = Image.open(img_path).convert("RGB")
        anns = self.anns_by_img.get(img_info["id"], [])
        boxes = np.array(
            [
                [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in anns
            ],
            dtype=np.float32,
        )
        labels = [self.cat_id_to_idx[a["category_id"]] for a in anns]
        if self.transform is not None:
            img, target = self.transform(img, boxes, labels)
            target["image_id"] = img_info["id"]
            target["orig_size"] = (img_info["height"], img_info["width"])
            return img, target
        return img, boxes, labels


def build_train_dataset(cfg: dict):
    data_cfg = cfg["data"]
    transform = COCODetectionTransform(
        short_sides=data_cfg["train_short_side"],
        max_size=data_cfg["train_max_size"],
    )
    shard_dir = Path(data_cfg["shard_dir"])
    if any(shard_dir.glob("train-*.tar")):
        return COCOTarDataset(shard_dir, "train", transform)
    return COCOAnnotationDataset(
        data_cfg["train_ann"], data_cfg["train_img_dir"], transform
    )


def build_val_dataset(cfg: dict):
    data_cfg = cfg["data"]
    transform = COCODetectionValTransform(
        short_side=data_cfg["val_short_side"], max_size=data_cfg["val_max_size"]
    )
    shard_dir = Path(data_cfg["shard_dir"])
    if any(shard_dir.glob("val-*.tar")):
        return COCOTarDataset(shard_dir, "val", transform)
    return COCOAnnotationDataset(
        data_cfg["val_ann"], data_cfg["val_img_dir"], transform
    )
