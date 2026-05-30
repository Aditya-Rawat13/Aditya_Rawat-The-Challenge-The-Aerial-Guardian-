"""
Performance metrics utilities.
"""

import time
import numpy as np
from collections import deque


class FPSCounter:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30):
        self._times = deque(maxlen=window)
        self.fps = 0.0

    def update(self, dt: float):
        self._times.append(dt)
        if len(self._times) > 1:
            self.fps = 1.0 / np.mean(self._times)

    def reset(self):
        self._times.clear()
        self.fps = 0.0


class MOTMetrics:
    """
    Lightweight MOT metrics.
    Computes basic MOTA/MOTP if ground-truth is available.
    """

    def __init__(self):
        self.fp   = 0   # False positives
        self.fn   = 0   # False negatives
        self.id_sw = 0  # ID switches
        self.gt_count = 0
        self.matched_iou_sum = 0.0
        self.matched_count   = 0

    def update(self, gt_boxes, pred_boxes, prev_ids=None, curr_ids=None):
        pass  # Extend for full evaluation

    @property
    def mota(self):
        if self.gt_count == 0:
            return 0.0
        return 1.0 - (self.fp + self.fn + self.id_sw) / self.gt_count

    @property
    def motp(self):
        if self.matched_count == 0:
            return 0.0
        return self.matched_iou_sum / self.matched_count
