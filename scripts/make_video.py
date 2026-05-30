"""
Convert annotated image sequence → output video.
Also assembles VisDrone sequence directories into videos.

Usage:
    python scripts/make_video.py --frames outputs/seq001 --output outputs/seq001.mp4 --fps 30
"""

import cv2
import argparse
from pathlib import Path
import numpy as np


def frames_to_video(frames_dir: str, output_path: str, fps: float = 30.0):
    frames_dir = Path(frames_dir)
    frames = sorted(list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png")))

    if not frames:
        print(f"No frames found in {frames_dir}")
        return

    sample = cv2.imread(str(frames[0]))
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for fp in frames:
        frame = cv2.imread(str(fp))
        if frame is not None:
            writer.write(frame)

    writer.release()
    print(f"✓ Written {len(frames)} frames → {output_path}  ({w}x{h} @ {fps}fps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=float, default=30.0)
    args = p.parse_args()
    frames_to_video(args.frames, args.output, args.fps)
