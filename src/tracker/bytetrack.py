import numpy as np
from typing import List
from collections import OrderedDict

from .kalman_filter import KalmanFilter
from .lap_solver import linear_assignment
from .matching import iou_distance, fuse_score


class TrkSta:
    New     = 0
    Tracked = 1
    Lost    = 2
    Removed = 3


class STrack:
    sha_kal = KalmanFilter()
    _id_cnt = 0

    def __init__(self, tlwh: np.ndarray, sco: float):
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kal_sta = None
        self.is_act = False
        self.sco = sco
        self.trk_len = 0
        self.sta = TrkSta.New
        self.fid = 0
        self.sta_fid = 0
        self.tid = 0

    @staticmethod
    def nxt_id():
        STrack._id_cnt += 1
        return STrack._id_cnt

    def act(self, fid: int):
        self.tid = self.nxt_id()
        self.kal_sta = self.sha_kal.initiate(self.tlwh_to_xyah(self._tlwh))
        self.trk_len = 0
        self.sta = TrkSta.Tracked
        self.is_act = True
        self.fid = fid
        self.sta_fid = fid

    def re_act(self, new_trk: "STrack", fid: int, new_id: bool = False):
        self.kal_sta = self.sha_kal.update(
            self.kal_sta, self.tlwh_to_xyah(new_trk.tlwh)
        )
        self.trk_len = 0
        self.sta = TrkSta.Tracked
        self.is_act = True
        self.fid = fid
        self.sco = new_trk.sco
        if new_id:
            self.tid = self.nxt_id()

    def upd(self, new_trk: "STrack", fid: int):
        self.fid = fid
        self.trk_len += 1
        self.kal_sta = self.sha_kal.update(
            self.kal_sta, self.tlwh_to_xyah(new_trk.tlwh)
        )
        self.sta = TrkSta.Tracked
        self.is_act = True
        self.sco = new_trk.sco

    def pre(self):
        if self.kal_sta is not None:
            self.kal_sta = self.sha_kal.predict(self.kal_sta)

    def apl_cam(self, war: np.ndarray):
        if self.kal_sta is None:
            return
        mea = self.kal_sta[0].copy()
        cx, cy = mea[0], mea[1]
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        wpt = cv2.transform(pt, war)[0][0]
        self.kal_sta[0][0] = wpt[0]
        self.kal_sta[0][1] = wpt[1]

    def mrk_los(self):
        self.sta = TrkSta.Lost

    def mrk_rem(self):
        self.sta = TrkSta.Removed

    @property
    def tlwh(self) -> np.ndarray:
        if self.kal_sta is None:
            return self._tlwh.copy()
        mea = self.kal_sta[0].copy()
        mea[2] *= mea[3]
        mea[:2] -= mea[2:4] / 2
        return mea[:4]

    @property
    def tlbr(self) -> np.ndarray:
        tlwh = self.tlwh
        return np.array([tlwh[0], tlwh[1], tlwh[0]+tlwh[2], tlwh[1]+tlwh[3]])

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        x, y, w, h = tlwh
        return np.array([x + w/2, y + h/2, w / (h + 1e-6), h], dtype=np.float32)

    def __repr__(self):
        return f"OT_{self.tid}({self.sta_fid}-{self.fid})"


class BYTETracker:

    def __init__(self, cfg: dict):
        self.trk_thr = cfg.get("track_thresh", 0.45)
        self.mat_thr = cfg.get("match_thresh", 0.8)
        self.trk_buf = cfg.get("track_buffer", 30)
        self.min_box = cfg.get("min_box_area", 10)
        self.frm_rat = cfg.get("frame_rate", 30)

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks:    List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.fid = 0
        self.max_los = int(self.frm_rat / 30.0 * self.trk_buf)

        STrack._id_cnt = 0

    def update(self, det: np.ndarray, img_siz: List[int], ori_siz: List[int]) -> List[STrack]:
        self.fid += 1

        if len(det) > 0:
            sco = det[:, 4]
            box = det[:, :4]
            tlw = np.stack([
                box[:, 0],
                box[:, 1],
                box[:, 2] - box[:, 0],
                box[:, 3] - box[:, 1],
            ], axis=1)

            hi_msk = sco >= self.trk_thr
            lo_msk = (~hi_msk) & (sco > 0.1)

            det_hi = [STrack(tlw[i], sco[i]) for i in np.where(hi_msk)[0]]
            det_lo = [STrack(tlw[i], sco[i]) for i in np.where(lo_msk)[0]]
        else:
            det_hi, det_lo = [], []

        unc = []
        trk_act = []
        for t in self.tracked_stracks:
            if not t.is_act:
                unc.append(t)
            else:
                trk_act.append(t)

        pol = jnt_str(trk_act, self.lost_stracks)
        for st in pol:
            st.pre()

        dis = iou_distance(pol, det_hi)
        dis = fuse_score(dis, det_hi)
        mat, u_trk, u_det_hi = linear_assignment(dis, thresh=self.mat_thr)

        act, ref = [], []
        for itr, idet in mat:
            trk = pol[itr]
            d   = det_hi[idet]
            if trk.sta == TrkSta.Tracked:
                trk.upd(d, self.fid)
                act.append(trk)
            else:
                trk.re_act(d, self.fid)
                ref.append(trk)

        r_trk = [pol[i] for i in u_trk if pol[i].sta == TrkSta.Tracked]
        dis2 = iou_distance(r_trk, det_lo)
        mat2, u_trk2, _ = linear_assignment(dis2, thresh=0.5)

        for itr, idet in mat2:
            trk = r_trk[itr]
            d   = det_lo[idet]
            if trk.sta == TrkSta.Tracked:
                trk.upd(d, self.fid)
                act.append(trk)
            else:
                trk.re_act(d, self.fid)
                ref.append(trk)

        los = []
        for i in u_trk2:
            trk = r_trk[i]
            if trk.sta != TrkSta.Lost:
                trk.mrk_los()
                los.append(trk)

        u_det_hi2_lst = [det_hi[i] for i in u_det_hi]
        dis3 = iou_distance(unc, u_det_hi2_lst)
        mat3, u_unc, u_det_hi2 = linear_assignment(dis3, thresh=0.7)

        for itr, idet in mat3:
            unc[itr].upd(u_det_hi2_lst[idet], self.fid)
            act.append(unc[itr])

        for i in u_unc:
            unc[i].mrk_rem()

        new_trk = []
        for i in u_det_hi2:
            d = u_det_hi2_lst[i]
            if d.sco >= self.trk_thr:
                d.act(self.fid)
                new_trk.append(d)

        rem = []
        for trk in self.lost_stracks:
            if self.fid - trk.fid > self.max_los:
                trk.mrk_rem()
                rem.append(trk)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.sta == TrkSta.Tracked]
        self.tracked_stracks = jnt_str(self.tracked_stracks, act)
        self.tracked_stracks = jnt_str(self.tracked_stracks, ref)
        self.lost_stracks    = sub_str(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(los)
        self.lost_stracks    = sub_str(self.lost_stracks, rem)
        self.removed_stracks.extend(rem)
        self.tracked_stracks, self.lost_stracks = rem_dup(
            self.tracked_stracks, self.lost_stracks
        )

        return [t for t in self.tracked_stracks if t.is_act]

    def compensate_camera_motion(self, war: np.ndarray):
        import cv2
        for trk in self.tracked_stracks + self.lost_stracks:
            trk.apl_cam(war)


def jnt_str(a: List[STrack], b: List[STrack]) -> List[STrack]:
    exi = {t.tid: t for t in a}
    return a + [t for t in b if t.tid not in exi]

def sub_str(a: List[STrack], b: List[STrack]) -> List[STrack]:
    ids = {t.tid for t in b}
    return [t for t in a if t.tid not in ids]

def rem_dup(str_a, str_b):
    dis = iou_distance(str_a, str_b)
    pai = np.where(dis < 0.15)
    dup_a, dup_b = set(), set()
    for p, q in zip(*pai):
        ta = str_a[p].fid - str_a[p].sta_fid
        tb = str_b[q].fid - str_b[q].sta_fid
        if ta > tb: dup_b.add(q)
        else:       dup_a.add(p)
    res_a = [t for i, t in enumerate(str_a) if i not in dup_a]
    res_b = [t for i, t in enumerate(str_b) if i not in dup_b]
    return res_a, res_b
