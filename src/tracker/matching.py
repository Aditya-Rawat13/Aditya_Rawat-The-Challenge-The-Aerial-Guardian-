import numpy as np
from typing import List


def iou_bat(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    are_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    are_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

    int_x1 = np.maximum(box_a[:, None, 0], box_b[None, :, 0])
    int_y1 = np.maximum(box_a[:, None, 1], box_b[None, :, 1])
    int_x2 = np.minimum(box_a[:, None, 2], box_b[None, :, 2])
    int_y2 = np.minimum(box_a[:, None, 3], box_b[None, :, 3])

    int_w = np.maximum(0, int_x2 - int_x1)
    int_h = np.maximum(0, int_y2 - int_y1)
    int_are = int_w * int_h

    uni = are_a[:, None] + are_b[None, :] - int_are
    iou = int_are / (uni + 1e-6)
    return iou


def iou_distance(atr, btr) -> np.ndarray:
    if len(atr) == 0 or len(btr) == 0:
        return np.zeros((len(atr), len(btr)), dtype=np.float32)

    atl = np.array([t.tlbr for t in atr])
    btl = np.array([t.tlbr for t in btr])

    iou = iou_bat(atl, btl)
    return 1.0 - iou


def fuse_score(cos_mat: np.ndarray, det) -> np.ndarray:
    if cos_mat.size == 0:
        return cos_mat
    sco = np.array([d.sco for d in det])
    iou_sim = 1 - cos_mat
    fus_sim = iou_sim * sco[None, :]
    return 1 - fus_sim
