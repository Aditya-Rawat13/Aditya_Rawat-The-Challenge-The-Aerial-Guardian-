#!/usr/bin/env python3
"""
Download YOLOv8n weights and optionally the VisDrone validation set.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --visdrone  # also download dataset
"""

import os
import argparse
import urllib.request
from pathlib import Path


YOLOV8N_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
YOLOV8N_VISDRONE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
# Note: For the VisDrone fine-tuned model, train using scripts/train_visdrone.py
# or download a community checkpoint from Hugging Face.


def download_file(url: str, dest: Path):
    print(f"Downloading {url} → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visdrone", action="store_true", help="Download VisDrone val set")
    p.add_argument("--weights-dir", default="weights", help="Weights directory")
    args = p.parse_args()

    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(exist_ok=True)

    # Download base YOLOv8n
    base_pt = weights_dir / "yolov8n.pt"
    if not base_pt.exists():
        download_file(YOLOV8N_URL, base_pt)
        print(f"✓ Downloaded yolov8n.pt ({base_pt.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"✓ yolov8n.pt already exists")

    # Check if VisDrone fine-tuned weights exist
    visdrone_pt = weights_dir / "yolov8n_visdrone.pt"
    if not visdrone_pt.exists():
        print(f"\n⚠  No VisDrone fine-tuned weights found at {visdrone_pt}")
        print("   Options:")
        print("   1. Run: python scripts/train_visdrone.py  (trains on VisDrone)")
        print("   2. Symlink base weights: ln -s yolov8n.pt weights/yolov8n_visdrone.pt")
        print("   Using base weights for now...")
        import shutil
        shutil.copy(base_pt, visdrone_pt)
        print(f"   Copied yolov8n.pt → yolov8n_visdrone.pt")
    else:
        print(f"✓ yolov8n_visdrone.pt exists")

    if args.visdrone:
        print("\nVisDrone dataset download:")
        print("  Please download the Task 4 MOT Validation set manually from:")
        print("  https://github.com/VisDrone/VisDrone-Dataset")
        print("  or use your Google Drive link and place in data/VisDrone/")

    print("\n✓ Setup complete. Run:")
    print("  python src/pipeline.py --input data/VisDrone/sequences/uav0000086_00000_v --output outputs/result.mp4")


if __name__ == "__main__":
    main()
