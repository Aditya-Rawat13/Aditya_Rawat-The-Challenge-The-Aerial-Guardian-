"""
Matching utilities for ByteTrack association.
"""

import numpy as np
from typing import List


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU between two sets of bounding boxes.
    Both in [x1, y1, x2, y2] format.
    """
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    inter_x1 = np.maximum(bboxes_a[:, None, 0], bboxes_b[None, :, 0])
    inter_y1 = np.maximum(bboxes_a[:, None, 1], bboxes_b[None, :, 1])
    inter_x2 = np.minimum(bboxes_a[:, None, 2], bboxes_b[None, :, 2])
    inter_y2 = np.minimum(bboxes_a[:, None, 3], bboxes_b[None, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    union = area_a[:, None] + area_b[None, :] - inter
    iou   = inter / (union + 1e-6)
    return iou


def iou_distance(atracks, btracks) -> np.ndarray:
    """
    Cost matrix based on 1 - IoU between track predicted boxes and detections.
    Lower cost = better match.
    """
    if len(atracks) == 0 or len(btracks) == 0:
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)

    atlbr = np.array([t.tlbr for t in atracks])
    btlbr = np.array([t.tlbr for t in btracks])

    ious = iou_batch(atlbr, btlbr)
    return 1.0 - ious


def fuse_score(cost_matrix: np.ndarray, detections) -> np.ndarray:
    """
    Fuse IoU cost with detection confidence score.
    Penalises low-confidence matches to reduce false associations.

    fused_cost = 1 - (1 - cost) * score
    """
    if cost_matrix.size == 0:
        return cost_matrix
    scores = np.array([d.score for d in detections])
    iou_sim  = 1 - cost_matrix          # [0,1] where 1 = perfect overlap
    fused_sim = iou_sim * scores[None, :]
    return 1 - fused_sim
