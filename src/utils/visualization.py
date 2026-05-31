import cv2
import numpy as np
from typing import List, Dict, Tuple
import colorsys


class Visualizer:

    def __init__(self, tai_len: int = 30, col_sed: int = 42):
        self.tai_len = tai_len
        self._col_cac: Dict[int, Tuple[int, int, int]] = {}
        self._sed = col_sed
        np.random.seed(col_sed)

        n = 512
        self._pal = []
        for i in range(n):
            h = (i * 0.618033988749895) % 1.0
            s = 0.7 + 0.3 * (i % 3) / 2
            v = 0.85 + 0.15 * (i % 2)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            self._pal.append((int(b * 255), int(g * 255), int(r * 255)))

    def get_col(self, tid: int) -> Tuple[int, int, int]:
        if tid not in self._col_cac:
            self._col_cac[tid] = self._pal[tid % len(self._pal)]
        return self._col_cac[tid]

    def draw(self, frm: np.ndarray, trks: List[dict], fps: float, fid: int) -> np.ndarray:
        out = frm.copy()

        for trk in trks:
            tid = trk["id"]
            x1, y1, x2, y2 = trk["bbox"]
            sco = trk["score"]
            tai = trk["tail"]
            col = self.get_col(tid)

            if len(tai) > 1:
                self._drw_tai(out, tai, col)

            thk = max(1, int((x2 - x1) / 60))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, thk)

            lbl = f"#{tid}  {sco:.2f}"
            self._drw_lbl(out, lbl, (x1, y1), col)

        self._drw_hud(out, fps, fid, len(trks))
        return out

    def _drw_tai(self, frm: np.ndarray, tai: List[Tuple[int,int]], col: Tuple):
        n = len(tai)
        for i in range(1, n):
            alp = i / n
            thk = max(1, int(alp * 3))
            fad_col = tuple(int(c * alp) for c in col)
            cv2.line(frm, tai[i-1], tai[i], fad_col, thk, cv2.LINE_AA)
        cv2.circle(frm, tai[-1], 3, col, -1, cv2.LINE_AA)

    def _drw_lbl(self, frm: np.ndarray, lbl: str, pos: Tuple[int,int], col: Tuple):
        x, y = pos
        fnt = cv2.FONT_HERSHEY_SIMPLEX
        scl = 0.45
        thk = 1

        (tw, th), bas = cv2.getTextSize(lbl, fnt, scl, thk)
        pad = 3

        y_top = max(0, y - th - bas - pad * 2)
        y_bot = y
        cv2.rectangle(frm, (x, y_top), (x + tw + pad * 2, y_bot), (0, 0, 0), -1)
        cv2.rectangle(frm, (x, y_top), (x + tw + pad * 2, y_bot), col, 1)
        cv2.putText(frm, lbl, (x + pad, y - bas - pad), fnt, scl, col, thk, cv2.LINE_AA)

    def _drw_hud(self, frm: np.ndarray, fps: float, fid: int, n_trk: int):
        h, w = frm.shape[:2]
        ovl = frm.copy()
        cv2.rectangle(ovl, (0, 0), (220, 75), (0, 0, 0), -1)
        cv2.addWeighted(ovl, 0.5, frm, 0.5, 0, frm)

        fnt = cv2.FONT_HERSHEY_SIMPLEX
        col = (200, 220, 255)

        cv2.putText(frm, f"FPS:    {fps:5.1f}", (8, 20), fnt, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(frm, f"Frame:  {fid}",       (8, 40), fnt, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(frm, f"Tracks: {n_trk}",     (8, 60), fnt, 0.5, col, 1, cv2.LINE_AA)
        cv2.putText(frm, "AERIAL GUARDIAN", (w - 160, 18), fnt, 0.45, (100, 200, 100), 1, cv2.LINE_AA)
