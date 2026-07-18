"""Download COCO 2017 and shard train/val into webdataset tar files.

Intended for the RunPod persistent volume at /workspace/data/coco.  The
vision-research plan calls for 118 K train + 5 K val images decoded on the fly
from shards.
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def _download(url: str, out: Path) -> None:
    """Best-effort download using urllib (no external CLI dependency)."""
    import urllib.request

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"{out} exists — skip download")
        return
    print(f"Downloading {url} → {out}")
    urllib.request.urlretrieve(url, out)


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    import zipfile

    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def _make_shards(
    img_dir: Path,
    ann_path: Path,
    shard_dir: Path,
    split: str,
    samples_per_shard: int = 2048,
) -> int:
    shard_dir.mkdir(parents=True, exist_ok=True)
    with open(ann_path, "r") as f:
        coco = json.load(f)

    # group annotations by image id
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    images = coco["images"]
    total = len(images)
    n_shards = (total + samples_per_shard - 1) // samples_per_shard

    cat_id_to_idx = {c["id"]: i for i, c in enumerate(coco["categories"])}

    shard_idx = 0
    buffer = []
    for i, img_info in enumerate(tqdm(images, desc=f"sharding {split}")):
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                img_bytes = BytesIO()
                im.save(img_bytes, format="JPEG", quality=95)
        except Exception as e:
            print(f"skip {img_path}: {e}")
            continue

        anns = anns_by_img.get(img_info["id"], [])
        sample = {
            "__key__": f"{split}_{i:08d}",
            "jpg": img_bytes.getvalue(),
            "json": {
                "image_id": img_info["id"],
                "width": img_info["width"],
                "height": img_info["height"],
                "boxes": [
                    [
                        a["bbox"][0],
                        a["bbox"][1],
                        a["bbox"][0] + a["bbox"][2],
                        a["bbox"][1] + a["bbox"][3],
                    ]
                    for a in anns
                ],
                "labels": [cat_id_to_idx[a["category_id"]] for a in anns],
            },
        }
        buffer.append(sample)

        if len(buffer) >= samples_per_shard or i == total - 1:
            shard_name = shard_dir / f"{split}-{shard_idx:05d}.tar"
            with tarfile.open(shard_name, "w") as tar:
                for s in buffer:
                    jpg_data = BytesIO(s["jpg"])
                    jpg_info = tarfile.TarInfo(name=f"{s['__key__']}.jpg")
                    jpg_info.size = len(s["jpg"])
                    tar.addfile(jpg_info, jpg_data)
                    json_data = json.dumps(s["json"]).encode("utf-8")
                    json_info = tarfile.TarInfo(name=f"{s['__key__']}.json")
                    json_info.size = len(json_data)
                    tar.addfile(json_info, BytesIO(json_data))
            buffer.clear()
            shard_idx += 1

    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/workspace/data/coco", type=Path)
    parser.add_argument("--samples-per-shard", default=2048, type=int)
    args = parser.parse_args()

    root = args.out
    root.mkdir(parents=True, exist_ok=True)

    urls = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    }

    zip_dir = root / "zips"
    zip_dir.mkdir(exist_ok=True)
    for name, url in urls.items():
        _download(url, zip_dir / f"{name}.zip")
        _extract_zip(zip_dir / f"{name}.zip", root)

    # train shards
    n_train = _make_shards(
        root / "train2017",
        root / "annotations" / "instances_train2017.json",
        root / "shards",
        "train",
        args.samples_per_shard,
    )
    n_val = _make_shards(
        root / "val2017",
        root / "annotations" / "instances_val2017.json",
        root / "shards",
        "val",
        args.samples_per_shard,
    )

    print(f"COCO sharded: train={n_train}, val={n_val}")
    assert n_train > 110_000 and n_val > 4_000, "unexpected COCO split size"


if __name__ == "__main__":
    main()
