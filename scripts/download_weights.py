#!/usr/bin/env python3
import os
import argparse
import urllib.request
from pathlib import Path
import shutil

YOL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
YOL_VIS = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"


def dwn_fil(url: str, dst: Path):
    print(f"Downloading {url} → {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst, reporthook=_prg)
    print()


def _prg(blk_num, blk_siz, tot_siz):
    dwn = blk_num * blk_siz
    if tot_siz > 0:
        pct = min(100, dwn * 100 / tot_siz)
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:5.1f}%", end="", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visdrone", action="store_true")
    p.add_argument("--weights-dir", default="weights")
    arg = p.parse_args()

    wei_dir = Path(arg.weights_dir)
    wei_dir.mkdir(exist_ok=True)

    bas_pt = wei_dir / "yolov8n.pt"
    if not bas_pt.exists():
        dwn_fil(YOL_URL, bas_pt)
        print(f"✓ Downloaded yolov8n.pt ({bas_pt.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"✓ yolov8n.pt already exists")

    vis_pt = wei_dir / "yolov8n_visdrone.pt"
    if not vis_pt.exists():
        print(f"\n⚠  No VisDrone weights found at {vis_pt}")
        print("   1. Run: python scripts/train_visdrone.py")
        print("   2. Symlink: ln -s yolov8n.pt weights/yolov8n_visdrone.pt")
        print("   Using base weights for now...")
        shutil.copy(bas_pt, vis_pt)
        print(f"   Copied yolov8n.pt → yolov8n_visdrone.pt")
    else:
        print(f"✓ yolov8n_visdrone.pt exists")

    if arg.visdrone:
        print("\nVisDrone dataset download:")
        print("  https://github.com/VisDrone/VisDrone-Dataset")
        print("  place in data/VisDrone/")

    print("\n✓ Setup complete. Run:")
    print("  python src/pipeline.py --input data/VisDrone/sequences/uav0000086_00000_v --output outputs/result.mp4")


if __name__ == "__main__":
    main()
