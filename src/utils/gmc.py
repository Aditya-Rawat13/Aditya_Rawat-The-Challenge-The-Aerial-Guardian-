import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)


class GlobalMotionCompensation:

    def __init__(self, met: str = "orb", dsc: float = 2.0):
        self.met = met.lower()
        self.dsc = dsc
        self.prv_gry = None
        self.prv_kps = None
        self.prv_des = None

        if self.met == "orb":
            self.det = cv2.ORB_create(
                nfeatures=1000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
            )
            self.mat = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        elif self.met == "sift":
            self.det = cv2.SIFT_create(nfeatures=500)
            self.mat = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        elif self.met in ("ecc", "optflow"):
            pass
        else:
            raise ValueError(f"Unknown GMC method: {met}")

    def apply(self, frm: np.ndarray):
        gry = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

        if self.dsc > 1.0:
            h, w = gry.shape
            gry_sml = cv2.resize(gry, (int(w / self.dsc), int(h / self.dsc)))
        else:
            gry_sml = gry

        war = None

        if self.prv_gry is not None:
            if self.met in ("orb", "sift"):
                war = self._fet_mat(gry_sml)
            elif self.met == "ecc":
                war = self._ecc(gry_sml)
            elif self.met == "optflow":
                war = self._opt(gry_sml)

        self.prv_gry = gry_sml
        return war

    def _fet_mat(self, gry: np.ndarray):
        kps, des = self.det.detectAndCompute(gry, None)
        war = None

        if self.prv_kps is not None and des is not None and len(des) > 10:
            try:
                mat = self.mat.knnMatch(self.prv_des, des, k=2)

                god = []
                for m_lst in mat:
                    if len(m_lst) == 2:
                        m, n = m_lst
                        if m.distance < 0.7 * n.distance:
                            god.append(m)

                if len(god) >= 8:
                    src_pts = np.float32([self.prv_kps[m.queryIdx].pt for m in god])
                    dst_pts = np.float32([kps[m.trainIdx].pt for m in god])

                    M, msk = cv2.estimateAffinePartial2D(
                        src_pts, dst_pts,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        maxIters=500,
                        confidence=0.99,
                    )

                    if M is not None:
                        M[0, 2] *= self.dsc
                        M[1, 2] *= self.dsc
                        war = M

            except cv2.error as e:
                log.debug(f"GMC feature matching error: {e}")

        self.prv_kps = kps
        self.prv_des = des
        return war

    def _ecc(self, gry: np.ndarray):
        if self.prv_gry is None:
            return None

        war_ini = np.eye(2, 3, dtype=np.float32)
        cri = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)

        try:
            _, M = cv2.findTransformECC(
                self.prv_gry, gry,
                war_ini, cv2.MOTION_EUCLIDEAN,
                cri, None, 1
            )
            M[0, 2] *= self.dsc
            M[1, 2] *= self.dsc
            return M
        except cv2.error:
            return None

    def _opt(self, gry: np.ndarray):
        if self.prv_gry is None:
            return None

        prv_pts = cv2.goodFeaturesToTrack(
            self.prv_gry, maxCorners=200, qualityLevel=0.01,
            minDistance=10, blockSize=7
        )
        if prv_pts is None:
            return None

        cur_pts, sta, _ = cv2.calcOpticalFlowPyrLK(
            self.prv_gry, gry, prv_pts, None,
            winSize=(15, 15), maxLevel=3
        )

        if cur_pts is None or sta is None:
            return None

        god_prv = prv_pts[sta.ravel() == 1]
        god_cur = cur_pts[sta.ravel() == 1]

        if len(god_prv) < 6:
            return None

        try:
            M, _ = cv2.estimateAffinePartial2D(
                god_prv, god_cur,
                method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if M is not None:
                M[0, 2] *= self.dsc
                M[1, 2] *= self.dsc
            return M
        except cv2.error:
            return None
