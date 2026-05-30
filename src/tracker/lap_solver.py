"""
Linear Assignment Problem solver.
Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """
    Solve assignment problem with threshold gating.

    Args:
        cost_matrix: [N_tracks, N_dets] cost matrix
        thresh: max cost to accept a match

    Returns:
        matches:      list of (track_idx, det_idx) pairs
        unmatched_a:  unmatched track indices
        unmatched_b:  unmatched detection indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches     = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            if r in unmatched_a: unmatched_a.remove(r)
            if c in unmatched_b: unmatched_b.remove(c)

    return matches, unmatched_a, unmatched_b
