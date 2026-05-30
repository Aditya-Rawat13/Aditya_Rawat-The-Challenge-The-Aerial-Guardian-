"""
Fine-tune YOLOv8n on VisDrone dataset for improved small-object detection.

VisDrone classes:
  0: pedestrian  1: people     2: bicycle     3: car
  4: van         5: truck      6: tricycle    7: awning-tricycle
  8: bus         9: motor

We remap to COCO-style: person = class 0,1 combined.

Usage:
    python scripts/train_visdrone.py --data data/visdrone.yaml --epochs 50
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


VISDRONE_YAML = """
# VisDrone Dataset — Person-only subset for Aerial Guardian
path: ./data/VisDrone
train: images/train
val:   images/val
test:  images/test

# Classes (VisDrone original, we filter to persons in training)
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

PERSON_ONLY_YAML = """
# VisDrone — Person only (merged pedestrian + people → person)
path: ./data/VisDrone_persons
train: images/train
val:   images/val

nc: 1
names: [person]
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",     default="configs/visdrone_person.yaml")
    p.add_argument("--model",    default="weights/yolov8n.pt")
    p.add_argument("--epochs",   type=int, default=100)
    p.add_argument("--imgsz",    type=int, default=640)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--device",   default="0")  # GPU 0, or 'cpu'
    p.add_argument("--project",  default="runs/train")
    p.add_argument("--name",     default="visdrone_person_yolov8n")
    args = p.parse_args()

    # Write config
    cfg_path = Path(args.data)
    cfg_path.parent.mkdir(exist_ok=True)
    if not cfg_path.exists():
        cfg_path.write_text(PERSON_ONLY_YAML)
        print(f"Wrote dataset config → {cfg_path}")

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,

        # ── Small object optimizations ──────────────────────────────────
        # Multi-scale training: randomly resize images during training
        # so the model learns objects at many scales
        multi_scale=True,

        # Mosaic augmentation: stitch 4 images together
        # Crucial for small objects — increases small-object occurrences per batch
        mosaic=1.0,

        # Copy-paste: paste additional objects into scenes
        # Boosts recall for crowded drone scenes
        copy_paste=0.3,

        # Close mosaic in last N epochs (stabilizes training)
        close_mosaic=10,

        # Anchor-free detection means no manual anchor tuning needed
        # YOLOv8 uses decoupled head with DFL (Distribution Focal Loss)

        # Learning rate schedule
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=3,

        # Augmentations for drone imagery
        hsv_h=0.015,    # hue variation
        hsv_s=0.7,      # saturation variation (drone lighting changes)
        hsv_v=0.4,      # brightness variation (shadows, reflections)
        degrees=5.0,    # slight rotation (drone tilt)
        translate=0.1,
        scale=0.5,      # scale augmentation (altitude variation)
        fliplr=0.5,
        flipud=0.0,     # drones don't fly upside down

        # Save best checkpoint
        save=True,
        save_period=10,
        val=True,
    )

    best = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n✓ Training complete. Best weights: {best}")
    print(f"  Copy to weights/: cp {best} weights/yolov8n_visdrone.pt")


if __name__ == "__main__":
    main()
