import time
import numpy as np
from collections import deque


class FPSCounter:

    def __init__(self, win: int = 30):
        self._tim = deque(maxlen=win)
        self.fps = 0.0

    def update(self, dt: float):
        self._tim.append(dt)
        if len(self._tim) > 1:
            self.fps = 1.0 / np.mean(self._tim)

    def reset(self):
        self._tim.clear()
        self.fps = 0.0


class MOTMetrics:

    def __init__(self):
        self.fp  = 0
        self.fn  = 0
        self.ids = 0
        self.gt_cnt = 0
        self.mat_iou_sum = 0.0
        self.mat_cnt = 0

    def update(self, gt_box, pre_box, prv_ids=None, cur_ids=None):
        pass

    @property
    def mota(self):
        if self.gt_cnt == 0:
            return 0.0
        return 1.0 - (self.fp + self.fn + self.ids) / self.gt_cnt

    @property
    def motp(self):
        if self.mat_cnt == 0:
            return 0.0
        return self.mat_iou_sum / self.mat_cnt
