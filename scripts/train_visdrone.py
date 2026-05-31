#!/usr/bin/env python3
import argparse
from pathlib import Path
from ultralytics import YOLO

VIS_YML = """
path: ./data/VisDrone
train: images/train
val:   images/val
test:  images/test
nc: 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""

PER_YML = """
path: ./data/VisDrone_persons
train: images/train
val:   images/val
nc: 1
names: [person]
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    default="configs/visdrone_person.yaml")
    p.add_argument("--model",   default="weights/yolov8n.pt")
    p.add_argument("--epo",     type=int, default=100)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--bat",     type=int, default=16)
    p.add_argument("--dev",     default="0")
    p.add_argument("--pro",     default="runs/train")
    p.add_argument("--nam",     default="visdrone_person_yolov8n")
    arg = p.parse_args()

    cfg_pth = Path(arg.data)
    cfg_pth.parent.mkdir(exist_ok=True)
    if not cfg_pth.exists():
        cfg_pth.write_text(PER_YML)
        print(f"Wrote dataset config → {cfg_pth}")

    mod = YOLO(arg.model)

    res = mod.train(
        data=arg.data,
        epochs=arg.epo,
        imgsz=arg.imgsz,
        batch=arg.bat,
        device=arg.dev,
        project=arg.pro,
        name=arg.nam,
        multi_scale=True,
        mosaic=1.0,
        copy_paste=0.3,
        close_mosaic=10,
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        save=True,
        save_period=10,
        val=True,
    )

    bes = Path(arg.pro) / arg.nam / "weights" / "best.pt"
    print(f"\n✓ Training complete. Best weights: {bes}")
    print(f"  cp {bes} weights/yolov8n_visdrone.pt")


if __name__ == "__main__":
    main()
