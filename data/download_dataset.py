"""
Download the 51 ISIC 2018 images this project uses, plus their manual
segmentation masks, and lay them out the way melanoma_detector.py expects.

Two official sources, verified against the live endpoints:
  - Images:  ISIC Archive API v2 (per-image, full resolution)
  - Masks:   ISIC 2018 Task 1 Training Ground Truth archive (one ZIP,
             ~26MB, containing masks for the whole 2018 challenge - only
             the 51 this project needs are extracted)

Run once, from the repo root:
    pip install requests
    python data/download_dataset.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent
DATASET_DIR = DATA_DIR / "dataset"
LIST_CSV = DATA_DIR / "list.csv"

API_IMAGE = "https://api.isic-archive.com/api/v2/images/{id}/"
GROUND_TRUTH_ZIP = (
    "https://isic-challenge-data.s3.amazonaws.com/2018/"
    "ISIC2018_Task1_Training_GroundTruth.zip"
)


def image_ids() -> list[str]:
    with open(LIST_CSV) as f:
        return [line.split(",")[0].strip() for line in f if line.strip()]


def download_images(ids: list[str]) -> None:
    out_dir = DATASET_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, image_id in enumerate(ids, 1):
        dest = out_dir / f"{image_id}.jpg"
        if dest.exists():
            continue
        print(f"  [{i}/{len(ids)}] {image_id}.jpg")
        meta = requests.get(API_IMAGE.format(id=image_id), timeout=30).json()
        url = meta["files"]["full"]["url"]
        dest.write_bytes(requests.get(url, timeout=60).content)


def download_masks(ids: list[str]) -> None:
    out_dir = DATASET_DIR / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    needed = {image_id for image_id in ids if not (out_dir / f"{image_id}_Segmentation.png").exists()}
    if not needed:
        return

    print(f"  Downloading ground-truth archive (~26MB, covers the full 2018 challenge)...")
    resp = requests.get(GROUND_TRUTH_ZIP, timeout=120)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for image_id in needed:
            # The official archive names masks in lowercase; melanoma_detector.py
            # expects the capitalized form used in the original class handout.
            src_name = f"ISIC2018_Task1_Training_GroundTruth/{image_id}_segmentation.png"
            (out_dir / f"{image_id}_Segmentation.png").write_bytes(zf.read(src_name))


def main() -> None:
    ids = image_ids()
    print(f"Fetching {len(ids)} images...")
    download_images(ids)
    print(f"Fetching {len(ids)} masks...")
    download_masks(ids)
    print(f"\nDone. Dataset ready at {DATASET_DIR}/")


if __name__ == "__main__":
    main()
