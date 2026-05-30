"""
Visualization utilities.
Renders bounding boxes, unique ID labels, trajectory tails, FPS overlay,
and detection count on each frame.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import colorsys


class Visualizer:
    """
    Draws tracking annotations on frames:
    - Colored bounding boxes (unique color per track ID)
    - ID label with score
    - Trajectory "tail" — fading polyline of recent centers
    - HUD overlay: FPS, frame count, active tracks
    """

    def __init__(self, tail_length: int = 30, colors_seed: int = 42):
        self.tail_length = tail_length
        self._color_cache: Dict[int, Tuple[int, int, int]] = {}
        self._seed = colors_seed
        np.random.seed(colors_seed)

        # Pre-generate 512 visually distinct colors using HSV spacing
        n = 512
        self._palette = []
        for i in range(n):
            h = (i * 0.618033988749895) % 1.0  # golden ratio spacing
            s = 0.7 + 0.3 * (i % 3) / 2
            v = 0.85 + 0.15 * (i % 2)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            self._palette.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        if track_id not in self._color_cache:
            self._color_cache[track_id] = self._palette[track_id % len(self._palette)]
        return self._color_cache[track_id]

    def draw(self, frame: np.ndarray, tracks: List[dict], fps: float, frame_id: int) -> np.ndarray:
        """
        Render all annotations onto frame (in-place copy).

        Args:
            frame:    BGR image
            tracks:   List of dicts with keys: id, bbox, center, score, tail
            fps:      Current pipeline FPS
            frame_id: Current frame number

        Returns:
            Annotated BGR image
        """
        out = frame.copy()

        for track in tracks:
            tid   = track["id"]
            x1, y1, x2, y2 = track["bbox"]
            score = track["score"]
            tail  = track["tail"]
            color = self.get_color(tid)

            # ── Trajectory tail ──────────────────────────────────────────────
            if len(tail) > 1:
                self._draw_tail(out, tail, color)

            # ── Bounding box ─────────────────────────────────────────────────
            thickness = max(1, int((x2 - x1) / 60))  # thicker for larger boxes
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # ── ID label ─────────────────────────────────────────────────────
            label = f"#{tid}  {score:.2f}"
            self._draw_label(out, label, (x1, y1), color)

        # ── HUD Overlay ───────────────────────────────────────────────────────
        self._draw_hud(out, fps, frame_id, len(tracks))

        return out

    def _draw_tail(self, frame: np.ndarray, tail: List[Tuple[int,int]], color: Tuple):
        """Draw fading polyline trajectory."""
        n = len(tail)
        for i in range(1, n):
            alpha = i / n                          # fade from 0 (oldest) to 1 (newest)
            thickness = max(1, int(alpha * 3))
            faded_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, tail[i-1], tail[i], faded_color, thickness, cv2.LINE_AA)

        # Dot at current position
        cv2.circle(frame, tail[-1], 3, color, -1, cv2.LINE_AA)

    def _draw_label(self, frame: np.ndarray, label: str, pos: Tuple[int,int], color: Tuple):
        """Draw label with dark background for readability."""
        x, y = pos
        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = 0.45
        thickness = 1

        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        pad = 3

        # Dark background pill
        y_top  = max(0, y - th - baseline - pad * 2)
        y_bot  = y
        cv2.rectangle(frame, (x, y_top), (x + tw + pad * 2, y_bot), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y_top), (x + tw + pad * 2, y_bot), color, 1)

        cv2.putText(frame, label, (x + pad, y - baseline - pad),
                    font, scale, color, thickness, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, fps: float, frame_id: int, n_tracks: int):
        """Top-left HUD with pipeline stats."""
        h, w = frame.shape[:2]

        # Semi-transparent dark background panel
        panel_h = 75
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (220, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        color = (200, 220, 255)

        cv2.putText(frame, f"FPS:    {fps:5.1f}",      (8, 20), font, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Frame:  {frame_id}",       (8, 40), font, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Tracks: {n_tracks}",       (8, 60), font, 0.5, color, 1, cv2.LINE_AA)

        # Aerial Guardian branding
        cv2.putText(frame, "AERIAL GUARDIAN",
                    (w - 160, 18), font, 0.45, (100, 200, 100), 1, cv2.LINE_AA)
