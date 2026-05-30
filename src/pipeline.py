"""
Aerial Guardian - Drone Person Detection & Tracking Pipeline
Architecture: YOLOv8n + SAHI tiling + ByteTrack + GMC (camera motion compensation)
"""

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
logger = logging.getLogger(__name__)


class AerialGuardianPipeline:
    """
    End-to-end drone person detection and tracking pipeline.

    Key design decisions:
    1. YOLOv8n (~6MB) for lightweight inference
    2. SAHI slicing for small object detection (persons at high altitude)
    3. ByteTrack for robust multi-object tracking
    4. Global Motion Compensation to handle drone ego-motion
    5. Trajectory tails for visual history
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # --- Detector ---
        logger.info("Loading YOLOv8n detector...")
        self.model = YOLO(config["model_path"])

        # SAHI wrapper for sliced inference (small object detection)
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=config["model_path"],
            confidence_threshold=config["conf_threshold"],
            device=self.device,
        )

        # --- Tracker ---
        self.tracker = BYTETracker(config["tracker"])

        # --- Camera Motion Compensator ---
        self.gmc = GlobalMotionCompensation(method=config.get("gmc_method", "orb"))

        # --- Visualization ---
        self.viz = Visualizer(
            tail_length=config.get("tail_length", 30),
            colors_seed=42,
        )

        # --- FPS Counter ---
        self.fps_counter = FPSCounter(window=30)

        # State
        self.frame_id = 0
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=config.get("tail_length", 30)))

        logger.info("Pipeline initialized successfully.")

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection using SAHI sliced inference.
        Returns detections array: [x1, y1, x2, y2, conf, class_id]

        SAHI splits the frame into overlapping tiles and merges results via NMS,
        dramatically improving recall for small (drone-altitude) persons.
        """
        use_sahi = self.config.get("use_sahi", True)

        if use_sahi:
            result = get_sliced_prediction(
                frame,
                self.sahi_model,
                slice_height=self.config.get("slice_height", 320),
                slice_width=self.config.get("slice_width", 320),
                overlap_height_ratio=self.config.get("overlap_ratio", 0.2),
                overlap_width_ratio=self.config.get("overlap_ratio", 0.2),
                perform_standard_pred=True,   # also run full-frame pass
                postprocess_type="NMM",        # Non-maximum merging (better than NMS for SAHI)
                postprocess_match_threshold=0.5,
            )
            detections = []
            for obj in result.object_prediction_list:
                # Filter to person class only (COCO class 0)
                if obj.category.id == 0:
                    bb = obj.bbox
                    detections.append([bb.minx, bb.miny, bb.maxx, bb.maxy, obj.score.value, 0])
            return np.array(detections, dtype=np.float32) if detections else np.empty((0, 6), dtype=np.float32)
        else:
            # Standard full-frame inference (faster, less accurate for small objects)
            results = self.model(frame, classes=[0], conf=self.config["conf_threshold"], verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return np.empty((0, 6), dtype=np.float32)
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy().reshape(-1, 1)
            cls  = boxes.cls.cpu().numpy().reshape(-1, 1)
            return np.hstack([xyxy, conf, cls]).astype(np.float32)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Full pipeline for a single frame:
        1. Detect persons
        2. Compensate for camera motion (GMC)
        3. Update tracker
        4. Draw visualizations
        Returns annotated frame and list of active tracks.
        """
        t0 = time.perf_counter()
        self.frame_id += 1
        h, w = frame.shape[:2]

        # Step 1: Detect
        detections = self.detect(frame)

        # Step 2: Global Motion Compensation
        # Estimate camera's own motion so tracker doesn't confuse ego-motion with object motion
        warp_matrix = self.gmc.apply(frame)
        if warp_matrix is not None and len(self.tracker.tracked_stracks) > 0:
            self.tracker.compensate_camera_motion(warp_matrix)

        # Step 3: Track
        # ByteTrack expects [x1,y1,x2,y2,conf] — strip class column
        dets_for_tracker = detections[:, :5] if len(detections) > 0 else np.empty((0, 5), dtype=np.float32)
        online_targets = self.tracker.update(dets_for_tracker, [h, w], [h, w])

        # Step 4: Update trajectory history
        tracks_out = []
        for t in online_targets:
            tid = t.track_id
            x1, y1, x2, y2 = t.tlbr
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            self.track_history[tid].append((cx, cy))
            tracks_out.append({
                "id": tid,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": (cx, cy),
                "score": t.score,
                "tail": list(self.track_history[tid]),
            })

        # Step 5: Visualize
        t1 = time.perf_counter()
        self.fps_counter.update(t1 - t0)
        annotated = self.viz.draw(frame, tracks_out, self.fps_counter.fps, self.frame_id)

        return annotated, tracks_out

    def run_video(self, input_path: str, output_path: str):
        """Process a video file end-to-end."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {input_path}")

        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))

        logger.info(f"Processing {total_frames} frames ({W}x{H} @ {fps_in:.1f}fps)...")

        frame_times = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            annotated, _ = self.process_frame(frame)
            dt = time.perf_counter() - t0
            frame_times.append(dt)

            out.write(annotated)

            if self.frame_id % 50 == 0:
                avg_fps = 1.0 / np.mean(frame_times[-50:])
                logger.info(f"Frame {self.frame_id}/{total_frames} | Pipeline FPS: {avg_fps:.1f}")

        cap.release()
        out.release()

        avg_fps = 1.0 / np.mean(frame_times)
        logger.info(f"\nDone! Output: {output_path}")
        logger.info(f"Average FPS: {avg_fps:.2f} on {self.device.upper()}")
        return avg_fps

    def run_image_sequence(self, img_dir: str, output_dir: str):
        """Process VisDrone-style image sequence (folder of JPGs)."""
        img_dir = Path(img_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        logger.info(f"Found {len(frames)} frames in {img_dir}")

        frame_times = []
        for i, fp in enumerate(frames):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue

            t0 = time.perf_counter()
            annotated, _ = self.process_frame(frame)
            frame_times.append(time.perf_counter() - t0)

            out_fp = output_dir / fp.name
            cv2.imwrite(str(out_fp), annotated)

            if (i + 1) % 50 == 0:
                fps = 1.0 / np.mean(frame_times[-50:])
                logger.info(f"  {i+1}/{len(frames)} | FPS: {fps:.1f}")

        avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
        logger.info(f"Average FPS: {avg_fps:.2f}")
        return avg_fps


def parse_args():
    p = argparse.ArgumentParser(description="Aerial Guardian — Drone Person Tracker")
    p.add_argument("--input",  required=True, help="Input video or image sequence directory")
    p.add_argument("--output", required=True, help="Output video path or directory")
    p.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    p.add_argument("--no-sahi", action="store_true", help="Disable SAHI slicing (faster)")
    p.add_argument("--model",  default="weights/yolov8n.pt", help="YOLOv8 model path")
    return p.parse_args()


if __name__ == "__main__":
    import yaml
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.no_sahi:
        config["use_sahi"] = False
    if args.model:
        config["model_path"] = args.model

    pipeline = AerialGuardianPipeline(config)

    inp = Path(args.input)
    if inp.is_dir():
        pipeline.run_image_sequence(str(inp), args.output)
    else:
        pipeline.run_video(str(inp), args.output)
