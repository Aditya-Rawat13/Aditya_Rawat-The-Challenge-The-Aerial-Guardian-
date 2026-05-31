#!/usr/bin/env python3
import cv2
import argparse
from pathlib import Path


def frm_vid(frm_dir: str, oup_pth: str, fps: float = 30.0):
    frm_dir = Path(frm_dir)
    frms = sorted(list(frm_dir.glob("*.jpg")) + list(frm_dir.glob("*.png")))

    if not frms:
        print(f"No frames found in {frm_dir}")
        return

    smp = cv2.imread(str(frms[0]))
    h, w = smp.shape[:2]

    fcc = cv2.VideoWriter_fourcc(*"mp4v")
    wri = cv2.VideoWriter(oup_pth, fcc, fps, (w, h))

    for fp in frms:
        frm = cv2.imread(str(fp))
        if frm is not None:
            wri.write(frm)

    wri.release()
    print(f"✓ Written {len(frms)} frames → {oup_pth}  ({w}x{h} @ {fps}fps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fps", type=float, default=30.0)
    arg = p.parse_args()
    frm_vid(arg.frames, arg.output, arg.fps)
