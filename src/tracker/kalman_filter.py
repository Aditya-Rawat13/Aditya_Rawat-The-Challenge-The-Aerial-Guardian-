import numpy as np
from scipy.linalg import block_diag


class KalmanFilter:

    def __init__(self):
        ndm, dt = 4, 1.0

        self.F = np.eye(2 * ndm, 2 * ndm)
        for i in range(ndm):
            self.F[i, ndm + i] = dt

        self.H = np.eye(ndm, 2 * ndm)

        self._swp = 1.0 / 20
        self._swv = 1.0 / 160

    def initiate(self, mea: np.ndarray):
        mea_pos = mea
        mea_vel = np.zeros_like(mea_pos)
        mea_all = np.concatenate([mea_pos, mea_vel])

        std = [
            2 * self._swp * mea[3],
            2 * self._swp * mea[3],
            1e-2,
            2 * self._swp * mea[3],
            10 * self._swv * mea[3],
            10 * self._swv * mea[3],
            1e-5,
            10 * self._swv * mea[3],
        ]
        cov = np.diag(np.square(std))
        return mea_all, cov

    def predict(self, sta):
        mea, cov = sta
        h = mea[3]

        std = [
            self._swp * h,
            self._swp * h,
            1e-2,
            self._swp * h,
            self._swv * h,
            self._swv * h,
            1e-5,
            self._swv * h,
        ]
        Q = np.diag(np.square(std))

        mea_pre = self.F @ mea
        cov_pre = self.F @ cov @ self.F.T + Q
        return mea_pre, cov_pre

    def update(self, sta, mea: np.ndarray):
        mea_cur, cov = sta
        h = mea_cur[3]

        std = [
            self._swp * h,
            self._swp * h,
            0.1,
            self._swp * h,
        ]
        R = np.diag(np.square(std))

        S = self.H @ cov @ self.H.T + R
        K = cov @ self.H.T @ np.linalg.inv(S)

        inn = mea - self.H @ mea_cur
        mea_new = mea_cur + K @ inn
        cov_new = (np.eye(len(mea_cur)) - K @ self.H) @ cov

        return mea_new, cov_new

    def gat_dis(self, sta, meas: np.ndarray) -> np.ndarray:
        mea, cov = sta
        prj_mea = self.H @ mea
        prj_cov = self.H @ cov @ self.H.T

        dif = meas - prj_mea
        try:
            L = np.linalg.cholesky(prj_cov)
            z = np.linalg.solve(L, dif.T)
            sq_mah = np.sum(z * z, axis=0)
        except np.linalg.LinAlgError:
            sq_mah = np.full(len(meas), np.inf)

        return sq_mah
