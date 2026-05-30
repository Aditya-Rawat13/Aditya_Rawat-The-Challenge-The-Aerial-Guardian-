"""
ByteTrack — Multi-Object Tracker
Based on: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (Zhang et al., 2022)

Key insight: Use BOTH high-confidence AND low-confidence detections.
- High-conf detections → associate with existing tracks (standard)
- Low-conf detections → try to recover lost/occluded tracks (ByteTrack's innovation)

This significantly reduces ID switches in crowded/occluded drone footage.
"""

import numpy as np
from typing import List, Tuple
from collections import OrderedDict

from .kalman_filter import KalmanFilter
from .lap_solver import linear_assignment
from .matching import iou_distance, fuse_score


class TrackState:
    New      = 0
    Tracked  = 1
    Lost     = 2
    Removed  = 3


class STrack:
    """
    Single object track — maintains Kalman state, track ID, and status.
    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    """
    shared_kalman = KalmanFilter()
    _id_count = 0

    def __init__(self, tlwh: np.ndarray, score: float):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_state = None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.state = TrackState.New
        self.frame_id = 0
        self.start_frame = 0
        self.track_id = 0

    @staticmethod
    def next_id():
        STrack._id_count += 1
        return STrack._id_count

    def activate(self, frame_id: int):
        self.track_id = self.next_id()
        self.kalman_state = self.shared_kalman.initiate(self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False):
        self.kalman_state = self.shared_kalman.update(
            self.kalman_state, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.score = new_track.score
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track: "STrack", frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.kalman_state = self.shared_kalman.update(
            self.kalman_state, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

    def predict(self):
        if self.kalman_state is not None:
            self.kalman_state = self.shared_kalman.predict(self.kalman_state)

    def apply_camera_motion(self, warp_matrix: np.ndarray):
        """
        Adjust track position based on estimated camera (drone) motion.
        Transforms track center by the affine warp matrix so the tracker
        doesn't confuse camera pan/tilt with actual object movement.
        """
        if self.kalman_state is None:
            return
        # Extract center from state
        mean = self.kalman_state[0].copy()
        cx, cy = mean[0], mean[1]
        # Apply 2x3 affine warp
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        warped = cv2.transform(pt, warp_matrix)[0][0]
        # Update mean position in-place
        self.kalman_state[0][0] = warped[0]
        self.kalman_state[0][1] = warped[1]

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def tlwh(self) -> np.ndarray:
        """Top-left-width-height from Kalman state."""
        if self.kalman_state is None:
            return self._tlwh.copy()
        mean = self.kalman_state[0].copy()
        mean[2] *= mean[3]           # w = aspect * h
        mean[:2] -= mean[2:4] / 2   # top-left = center - half wh
        return mean[:4]

    @property
    def tlbr(self) -> np.ndarray:
        """Top-left-bottom-right."""
        tlwh = self.tlwh
        return np.array([tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]])

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert [x,y,w,h] → [cx,cy,aspect,h]."""
        x, y, w, h = tlwh
        return np.array([x + w/2, y + h/2, w / (h + 1e-6), h], dtype=np.float32)

    def __repr__(self):
        return f"OT_{self.track_id}({self.start_frame}-{self.frame_id})"


class BYTETracker:
    """
    ByteTrack tracker.

    Two-round association:
    Round 1: High-confidence detections ↔ active tracks (IoU matching)
    Round 2: Low-confidence detections  ↔ lost tracks   (IoU matching)

    This second round is the key innovation — it allows recovery of
    temporarily occluded persons before they get a new ID.
    """

    def __init__(self, config: dict):
        self.track_thresh    = config.get("track_thresh", 0.45)
        self.match_thresh    = config.get("match_thresh", 0.8)
        self.track_buffer    = config.get("track_buffer", 30)
        self.min_box_area    = config.get("min_box_area", 10)
        self.frame_rate      = config.get("frame_rate", 30)

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks:    List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.max_time_lost = int(self.frame_rate / 30.0 * self.track_buffer)

        # Reset global ID counter for clean runs
        STrack._id_count = 0

    def update(self, dets: np.ndarray, img_size: List[int], ori_img_size: List[int]) -> List[STrack]:
        """
        Args:
            dets: [N, 5] array of [x1, y1, x2, y2, score]
            img_size: [H, W] of processed frame
            ori_img_size: [H, W] of original frame

        Returns:
            List of active STrack objects for this frame
        """
        self.frame_id += 1

        # Split detections into high/low confidence
        if len(dets) > 0:
            scores = dets[:, 4]
            bboxes = dets[:, :4]
            # Convert xyxy → tlwh
            tlwhs = np.stack([
                bboxes[:, 0],
                bboxes[:, 1],
                bboxes[:, 2] - bboxes[:, 0],
                bboxes[:, 3] - bboxes[:, 1],
            ], axis=1)

            hi_mask = scores >= self.track_thresh
            lo_mask = (~hi_mask) & (scores > 0.1)

            dets_hi = [STrack(tlwhs[i], scores[i]) for i in np.where(hi_mask)[0]]
            dets_lo = [STrack(tlwhs[i], scores[i]) for i in np.where(lo_mask)[0]]
        else:
            dets_hi, dets_lo = [], []

        # Classify current tracks
        unconfirmed = []
        tracked_stracks = []
        for t in self.tracked_stracks:
            if not t.is_activated:
                unconfirmed.append(t)
            else:
                tracked_stracks.append(t)

        # Predict new positions via Kalman filter
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        for st in strack_pool:
            st.predict()

        # ── Round 1: High-confidence dets ↔ active tracks ──────────────────
        dists = iou_distance(strack_pool, dets_hi)
        dists = fuse_score(dists, dets_hi)
        matches, u_track, u_det_hi = linear_assignment(dists, thresh=self.match_thresh)

        activated, refound = [], []
        for itrack, idet in matches:
            track = strack_pool[itrack]
            det   = dets_hi[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refound.append(track)

        # ── Round 2: Low-confidence dets ↔ unmatched lost tracks ───────────
        # This is ByteTrack's key contribution: recovering occluded objects
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists2 = iou_distance(r_tracked, dets_lo)
        matches2, u_track2, _ = linear_assignment(dists2, thresh=0.5)

        for itrack, idet in matches2:
            track = r_tracked[itrack]
            det   = dets_lo[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refound.append(track)

        # Mark remaining unmatched active tracks as lost
        lost = []
        for i in u_track2:
            track = r_tracked[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # ── Unconfirmed tracks ↔ remaining high-conf dets ───────────────────
        u_dets_hi = [dets_hi[i] for i in u_det_hi]
        dists3 = iou_distance(unconfirmed, u_dets_hi)
        matches3, u_unconf, u_det_hi2 = linear_assignment(dists3, thresh=0.7)

        for itrack, idet in matches3:
            unconfirmed[itrack].update(u_dets_hi[idet], self.frame_id)
            activated.append(unconfirmed[itrack])

        for i in u_unconf:
            unconfirmed[i].mark_removed()

        # Initialise new tracks from unmatched high-conf dets
        new_tracks = []
        for i in u_det_hi2:
            det = u_dets_hi[i]
            if det.score >= self.track_thresh:
                det.activate(self.frame_id)
                new_tracks.append(det)

        # Remove stale lost tracks
        removed = []
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        # Update state pools
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks  = joint_stracks(self.tracked_stracks, activated)
        self.tracked_stracks  = joint_stracks(self.tracked_stracks, refound)
        self.lost_stracks     = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks     = sub_stracks(self.lost_stracks, removed)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        output = [t for t in self.tracked_stracks if t.is_activated]
        return output

    def compensate_camera_motion(self, warp_matrix: np.ndarray):
        """Apply GMC warp to all active track positions."""
        import cv2
        for track in self.tracked_stracks + self.lost_stracks:
            track.apply_camera_motion(warp_matrix)


# ── Utility functions ──────────────────────────────────────────────────────────

def joint_stracks(a: List[STrack], b: List[STrack]) -> List[STrack]:
    exists = {t.track_id: t for t in a}
    return a + [t for t in b if t.track_id not in exists]

def sub_stracks(a: List[STrack], b: List[STrack]) -> List[STrack]:
    ids = {t.track_id for t in b}
    return [t for t in a if t.track_id not in ids]

def remove_duplicate_stracks(stracksa, stracksb):
    dists = iou_distance(stracksa, stracksb)
    pairs = np.where(dists < 0.15)
    dupa, dupb = set(), set()
    for p, q in zip(*pairs):
        ta = stracksa[p].frame_id - stracksa[p].start_frame
        tb = stracksb[q].frame_id - stracksb[q].start_frame
        if ta > tb: dupb.add(q)
        else:       dupa.add(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
