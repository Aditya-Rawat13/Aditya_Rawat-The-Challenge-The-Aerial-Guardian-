import cv2
import numpy as np
import torch
import time
import argparse
import logging
from pathlib import Path
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image

from tracker.bytetrack import BYTETracker
from utils.gmc import GlobalMotionCompensation
from utils.visualization import Visualizer
from utils.metrics import FPSCounter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class AerialGuardianPipeline:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Using device: {self.dev}")

        self.mod = YOLO(cfg["model_path"])

        self.sah = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=cfg["model_path"],
            confidence_threshold=cfg["conf_threshold"],
            device=self.dev,
        )

        self.trk = BYTETracker(cfg["tracker"])
        self.gmc = GlobalMotionCompensation(met=cfg.get("gmc_method", "orb"), dsc=2.0)
        self.viz = Visualizer(tai_len=cfg.get("tail_length", 30), col_sed=42)
        self.fps = FPSCounter(win=30)
        self.fid = 0
        self.his: Dict[int, deque] = defaultdict(lambda: deque(maxlen=cfg.get("tail_length", 30)))

        log.info("Pipeline initialized successfully.")

    def det(self, frm: np.ndarray) -> np.ndarray:
        use = self.cfg.get("use_sahi", True)

        if use:
            res = get_sliced_prediction(
                frm,
                self.sah,
                slice_height=self.cfg.get("slice_height", 320),
                slice_width=self.cfg.get("slice_width", 320),
                overlap_height_ratio=self.cfg.get("overlap_ratio", 0.2),
                overlap_width_ratio=self.cfg.get("overlap_ratio", 0.2),
                perform_standard_pred=True,
                postprocess_type="NMM",
                postprocess_match_threshold=0.5,
            )
            det_list = []
            for obj in res.object_prediction_list:
                if obj.category.id == 0:
                    bb = obj.bbox
                    det_list.append([bb.minx, bb.miny, bb.maxx, bb.maxy, obj.score.value, 0])
            return np.array(det_list, dtype=np.float32) if det_list else np.empty((0, 6), dtype=np.float32)
        else:
            res = self.mod(frm, classes=[0], conf=self.cfg["conf_threshold"], verbose=False)
            box = res[0].boxes
            if box is None or len(box) == 0:
                return np.empty((0, 6), dtype=np.float32)
            xyxy = box.xyxy.cpu().numpy()
            con = box.conf.cpu().numpy().reshape(-1, 1)
            cls = box.cls.cpu().numpy().reshape(-1, 1)
            return np.hstack([xyxy, con, cls]).astype(np.float32)

    def pro(self, frm: np.ndarray) -> Tuple[np.ndarray, List]:
        t0 = time.perf_counter()
        self.fid += 1
        h, w = frm.shape[:2]

        det_arr = self.det(frm)

        war = self.gmc.apply(frm)
        if war is not None and len(self.trk.tracked_stracks) > 0:
            self.trk.compensate_camera_motion(war)

        det_trk = det_arr[:, :5] if len(det_arr) > 0 else np.empty((0, 5), dtype=np.float32)
        onl = self.trk.update(det_trk, [h, w], [h, w])

        out = []
        for t in onl:
            tid = t.tid
            x1, y1, x2, y2 = t.tlbr
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            self.his[tid].append((cx, cy))
            out.append({
                "id": tid,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": (cx, cy),
                "score": t.sco,
                "tail": list(self.his[tid]),
            })

        t1 = time.perf_counter()
        self.fps.update(t1 - t0)
        ann = self.viz.draw(frm, out, self.fps.fps, self.fid)

        return ann, out

    def run_vid(self, inp: str, oup: str):
        cap = cv2.VideoCapture(inp)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {inp}")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        wri = cv2.VideoWriter(oup, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))
        log.info(f"Processing {tot} frames ({W}x{H} @ {fps_in:.1f}fps)...")

        ftm = []
        while True:
            ret, frm = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            ann, _ = self.pro(frm)
            dt = time.perf_counter() - t0
            ftm.append(dt)
            wri.write(ann)

            if self.fid % 50 == 0:
                avg = 1.0 / np.mean(ftm[-50:])
                log.info(f"Frame {self.fid}/{tot} | Pipeline FPS: {avg:.1f}")

        cap.release()
        wri.release()

        avg = 1.0 / np.mean(ftm)
        log.info(f"\nDone! Output: {oup}")
        log.info(f"Average FPS: {avg:.2f} on {self.dev.upper()}")
        return avg

    def run_seq(self, img_dir: str, oup_dir: str):
        img_dir = Path(img_dir)
        oup_dir = Path(oup_dir)
        oup_dir.mkdir(parents=True, exist_ok=True)

        frms = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        log.info(f"Found {len(frms)} frames in {img_dir}")

        ftm = []
        for i, fp in enumerate(frms):
            frm = cv2.imread(str(fp))
            if frm is None:
                continue

            t0 = time.perf_counter()
            ann, _ = self.pro(frm)
            ftm.append(time.perf_counter() - t0)

            cv2.imwrite(str(oup_dir / fp.name), ann)

            if (i + 1) % 50 == 0:
                cur = 1.0 / np.mean(ftm[-50:])
                log.info(f"  {i+1}/{len(frms)} | FPS: {cur:.1f}")

        avg = 1.0 / np.mean(ftm) if ftm else 0
        log.info(f"Average FPS: {avg:.2f}")
        return avg


def par_arg():
    p = argparse.ArgumentParser(description="Aerial Guardian — Drone Person Tracker")
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--no-sahi", action="store_true")
    p.add_argument("--model",  default="weights/yolov8n.pt")
    return p.parse_args()


if __name__ == "__main__":
    import yaml
    arg = par_arg()

    with open(arg.config) as f:
        cfg = yaml.safe_load(f)

    if arg.no_sahi:
        cfg["use_sahi"] = False
    if arg.model:
        cfg["model_path"] = arg.model

    pip = AerialGuardianPipeline(cfg)

    inp = Path(arg.input)
    if inp.is_dir():
        pip.run_seq(str(inp), arg.output)
    else:
        pip.run_vid(str(inp), arg.output)
