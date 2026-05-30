"""
Global Motion Compensation (GMC)
=================================
The core problem with drone tracking: the camera itself is moving.
When the drone pans/tilts/translates, every pixel shifts — making it
look like all tracked objects moved, even stationary ones.

Solution: Estimate the camera's affine motion between frames using
background feature matching, then subtract that motion from track positions
before running the association step.

Methods supported:
  - ORB: Fast feature detector, CPU-friendly, good for real-time
  - SIFT: More accurate but slower
  - ECC (Enhanced Correlation Coefficient): Template-based, robust to lighting
  - OptFlow (Sparse Optical Flow): Lucas-Kanade on Good Features to Track
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GlobalMotionCompensation:
    """
    Estimates per-frame affine warp matrix representing camera ego-motion.

    The warp M is a 2x3 affine matrix such that:
        [x', y'] = M * [x, y, 1]^T

    This maps a point in the previous frame to where it should appear
    in the current frame if the camera moved but the world stayed still.
    We use this to pre-warp tracked Kalman states before association.
    """

    def __init__(self, method: str = "orb", downscale: float = 2.0):
        """
        Args:
            method:    Feature matching method: 'orb', 'sift', 'ecc', 'optflow'
            downscale: Resize factor for speed (2.0 = half resolution)
        """
        self.method = method.lower()
        self.downscale = downscale
        self.prev_frame_gray = None
        self.prev_keypoints  = None
        self.prev_descriptors = None

        if self.method == "orb":
            # ORB: Binary descriptor, very fast, ~6ms on CPU
            self.detector = cv2.ORB_create(
                nfeatures=1000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        elif self.method == "sift":
            self.detector = cv2.SIFT_create(nfeatures=500)
            self.matcher  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        elif self.method in ("ecc", "optflow"):
            pass  # No detector needed
        else:
            raise ValueError(f"Unknown GMC method: {method}")

    def apply(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Estimate affine warp between previous and current frame.
        Returns 2x3 warp matrix, or None if estimation fails / first frame.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downscale for speed
        if self.downscale > 1.0:
            h, w = gray.shape
            gray_small = cv2.resize(gray, (int(w / self.downscale), int(h / self.downscale)))
        else:
            gray_small = gray

        warp = None

        if self.prev_frame_gray is not None:
            if self.method in ("orb", "sift"):
                warp = self._feature_matching(gray_small)
            elif self.method == "ecc":
                warp = self._ecc(gray_small)
            elif self.method == "optflow":
                warp = self._optflow(gray_small)

        self.prev_frame_gray = gray_small
        return warp

    def _feature_matching(self, gray: np.ndarray) -> np.ndarray | None:
        """ORB/SIFT feature matching → estimate affine transform."""
        kps, descs = self.detector.detectAndCompute(gray, None)
        warp = None

        if self.prev_keypoints is not None and descs is not None and len(descs) > 10:
            try:
                matches = self.matcher.knnMatch(self.prev_descriptors, descs, k=2)

                # Lowe's ratio test: keep only unambiguous matches
                good = []
                for m_list in matches:
                    if len(m_list) == 2:
                        m, n = m_list
                        if m.distance < 0.7 * n.distance:
                            good.append(m)

                if len(good) >= 8:
                    src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good])
                    dst_pts = np.float32([kps[m.trainIdx].pt for m in good])

                    # RANSAC robust estimation of affine transform
                    M, mask = cv2.estimateAffinePartial2D(
                        src_pts, dst_pts,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=3.0,
                        maxIters=500,
                        confidence=0.99,
                    )

                    if M is not None:
                        # Scale translation back to full resolution
                        M[0, 2] *= self.downscale
                        M[1, 2] *= self.downscale
                        warp = M

            except cv2.error as e:
                logger.debug(f"GMC feature matching error: {e}")

        self.prev_keypoints   = kps
        self.prev_descriptors = descs
        return warp

    def _ecc(self, gray: np.ndarray) -> np.ndarray | None:
        """
        Enhanced Correlation Coefficient maximisation.
        More robust to illumination changes than feature matching.
        Slower but very accurate for small motions (hover/slow pan).
        """
        if self.prev_frame_gray is None:
            return None

        warp_init = np.eye(2, 3, dtype=np.float32)
        criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)

        try:
            _, M = cv2.findTransformECC(
                self.prev_frame_gray, gray,
                warp_init, cv2.MOTION_EUCLIDEAN,
                criteria, None, 1
            )
            M[0, 2] *= self.downscale
            M[1, 2] *= self.downscale
            return M
        except cv2.error:
            return None

    def _optflow(self, gray: np.ndarray) -> np.ndarray | None:
        """
        Sparse optical flow (Lucas-Kanade) on detected corners.
        Fast, lightweight, good for moderate motions.
        """
        if self.prev_frame_gray is None:
            return None

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_frame_gray, maxCorners=200, qualityLevel=0.01,
            minDistance=10, blockSize=7
        )
        if prev_pts is None:
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_frame_gray, gray, prev_pts, None,
            winSize=(15, 15), maxLevel=3
        )

        if curr_pts is None or status is None:
            return None

        good_prev = prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) < 6:
            return None

        try:
            M, _ = cv2.estimateAffinePartial2D(
                good_prev, good_curr,
                method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            if M is not None:
                M[0, 2] *= self.downscale
                M[1, 2] *= self.downscale
            return M
        except cv2.error:
            return None
