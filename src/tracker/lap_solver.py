import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cos_mat: np.ndarray, thresh: float):
    if cos_mat.size == 0:
        return [], list(range(cos_mat.shape[0])), list(range(cos_mat.shape[1]))

    row, col = linear_sum_assignment(cos_mat)

    mat = []
    unm_a = list(range(cos_mat.shape[0]))
    unm_b = list(range(cos_mat.shape[1]))

    for r, c in zip(row, col):
        if cos_mat[r, c] <= thresh:
            mat.append((r, c))
            if r in unm_a: unm_a.remove(r)
            if c in unm_b: unm_b.remove(c)

    return mat, unm_a, unm_b
