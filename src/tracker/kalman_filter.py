"""
Kalman Filter for bounding box state estimation.

State: [cx, cy, aspect_ratio, height, vcx, vcy, va, vh]
       (center x/y, w/h aspect ratio, height, and their velocities)

Using a constant velocity model — good enough for drone footage
where targets move relatively slowly compared to frame rate.
"""

import numpy as np
from scipy.linalg import block_diag


class KalmanFilter:
    """
    Standard Kalman Filter with constant-velocity motion model.

    Observation: [cx, cy, aspect, h]   (4-dim)
    State:       [cx, cy, aspect, h, vcx, vcy, va, vh]  (8-dim)
    """

    def __init__(self):
        ndim, dt = 4, 1.0

        # State transition matrix F
        self.F = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = dt  # position += velocity * dt

        # Observation matrix H  (maps state → observation)
        self.H = np.eye(ndim, 2 * ndim)

        # Process noise covariance — how much the motion model can drift
        self._std_weight_pos = 1.0 / 20
        self._std_weight_vel = 1.0 / 160

    def initiate(self, measurement: np.ndarray):
        """
        Create initial state from first measurement.
        Returns (mean, covariance).
        """
        mean_pos = measurement  # [cx, cy, a, h]
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        std = [
            2 * self._std_weight_pos * measurement[3],
            2 * self._std_weight_pos * measurement[3],
            1e-2,
            2 * self._std_weight_pos * measurement[3],
            10 * self._std_weight_vel * measurement[3],
            10 * self._std_weight_vel * measurement[3],
            1e-5,
            10 * self._std_weight_vel * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, state):
        """Predict next state using motion model."""
        mean, covariance = state
        h = mean[3]

        std = [
            self._std_weight_pos * h,
            self._std_weight_pos * h,
            1e-2,
            self._std_weight_pos * h,
            self._std_weight_vel * h,
            self._std_weight_vel * h,
            1e-5,
            self._std_weight_vel * h,
        ]
        Q = np.diag(np.square(std))

        mean_pred = self.F @ mean
        cov_pred  = self.F @ covariance @ self.F.T + Q
        return mean_pred, cov_pred

    def update(self, state, measurement: np.ndarray):
        """Update state with new measurement using Kalman gain."""
        mean, covariance = state
        h = mean[3]

        std = [
            self._std_weight_pos * h,
            self._std_weight_pos * h,
            0.1,
            self._std_weight_pos * h,
        ]
        R = np.diag(np.square(std))

        # Innovation covariance
        S = self.H @ covariance @ self.H.T + R

        # Kalman gain
        K = covariance @ self.H.T @ np.linalg.inv(S)

        # Update
        innovation = measurement - self.H @ mean
        mean_new = mean + K @ innovation
        cov_new  = (np.eye(len(mean)) - K @ self.H) @ covariance

        return mean_new, cov_new

    def gating_distance(self, state, measurements: np.ndarray) -> np.ndarray:
        """
        Mahalanobis distance between predicted state and measurements.
        Used for association gating — reject matches that are too far.
        """
        mean, covariance = state
        projected_mean = self.H @ mean
        projected_cov  = self.H @ covariance @ self.H.T

        diff = measurements - projected_mean
        # Solve: chol(S)^-1 * diff for numerical stability
        try:
            L = np.linalg.cholesky(projected_cov)
            z = np.linalg.solve(L, diff.T)
            sq_maha = np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            sq_maha = np.full(len(measurements), np.inf)

        return sq_maha
