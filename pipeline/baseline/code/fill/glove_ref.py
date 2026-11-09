"""Glove references of one mid1min take, in the head camera with the take's refined calibration (tracker -> wrist ->
head camera, as yuzhen's skel_compare / mano_render): the Wuji SDK skeleton and yuzhen's MANO fit (setup/export_mano_glove.py)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

TB = Path("/root/autodl-tmp/tactile_baseline")
os.environ.setdefault("MANO_PKL", str(TB / "VITRA/weights/mano/MANO_RIGHT.pkl"))
sys.path.insert(0, str(TB / "hltc_handfit/handfit"))
from common import interp_tracker, load_meta, load_npz  # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
FCOL = [(255, 255, 255), (60, 60, 255), (60, 255, 60), (255, 160, 60), (60, 220, 255)]   # BGR: palm, thumb, index, middle, ring+pinky


def T44(R, p):
    T = np.tile(np.eye(4), (len(R), 1, 1)); T[:, :3, :3] = R; T[:, :3, 3] = p
    return T


def pose7(v):
    T = np.eye(4); T[:3, :3] = Rot.from_quat(np.asarray(v[3:7], float)).as_matrix(); T[:3, 3] = v[:3]
    return T


class Take:
    def __init__(self, ID, hfo=None):
        self.ID, self.take = ID, TB / "data/hilotah/mid1min" / ID
        self.hfo = Path(hfo) if hfo else TB / "results/handfit" / ID.replace("/", "_")
        self.meta = load_meta(str(self.take))
        self.st, ext = self.meta["config"]["station"], self.meta["config"]["camcal"]["extrinsics"]
        self.t = np.asarray(np.load(self.take / "cameras/ego/frames.npz")["t_us"], np.int64)
        self.N = len(self.t)
        ci = json.loads((self.take / "cameras/ego/intrinsics.json").read_text())["color"]
        self.K = np.array([[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]], [0, 0, 1.0]])
        self.W, self.H = ci["width"], ci["height"]
        Rh, ph, self.okh = interp_tracker(*load_npz("%s/station/tracker_%s.npz" % (self.take, self.st["trackers"]["head"])), self.t)
        self.Tcw = np.linalg.inv(T44(Rh, ph) @ np.asarray(ext["ego"]["T_tracker_cam"], float))
        self._z, self._kin = {}, {}

    def A(self, side):
        """(N, 4, 4) head camera <- SDK wrist, and validity."""
        Rt, pt, okt = interp_tracker(*load_npz("%s/station/tracker_%s.npz" % (self.take, self.st["trackers"][side])), self.t)
        return self.Tcw @ T44(Rt, pt) @ pose7(self.st["tracker_to_wrist"][side]), okt & self.okh

    def sdk(self, side):
        ts, D = load_npz("%s/gloves/%s/skeleton.npz" % (self.take, side))
        i = np.searchsorted(ts, self.t, "right") - 1
        ok = (i >= 0) & (self.t - ts[np.clip(i, 0, None)] <= 100_000)
        X = np.full((self.N, 21, 3), np.nan); X[ok] = D[i[ok], :21, :3]
        X[~(np.abs(np.nan_to_num(X)).sum((1, 2)) > 0)] = np.nan
        return X

    def _mano_idx(self, side):
        if side not in self._z:
            z = dict(np.load(self.hfo / ("glove_mano_%s.npz" % side)))
            j = np.clip(np.searchsorted(z["t_us"], self.t), 0, len(z["t_us"]) - 1)
            z["_j"], z["_ok"] = j, (z["t_us"][j] == self.t) & z["ok"][j]
            self._z[side] = z
        return self._z[side]

    def mano(self, side):
        z = self._mano_idx(side)
        X = np.full((self.N, 21, 3), np.nan); X[z["_ok"]] = z["data"][z["_j"][z["_ok"]], :, :3]
        return X

    def cam(self, G, side):
        A, ok = self.A(side)
        C = np.einsum("nij,nkj->nki", A[:, :3, :3], G) + A[:, None, :3, 3]
        C[~ok] = np.nan
        return C

    def proj(self, C, K=None):
        K = self.K if K is None else K
        return np.stack([K[0, 0] * C[..., 0] / C[..., 2] + K[0, 2], K[1, 1] * C[..., 1] / C[..., 2] + K[1, 2]], -1)

    def mano_mesh_cam(self, side, f, A_cam):
        """MANO vertices (778, 3) in the head camera at clip frame f (None if no fit there) and the faces."""
        import mano_kin as MK
        from mano_fit import _mano
        z = self._mano_idx(side)
        if not z["_ok"][f]:
            return None, None
        if side not in self._kin:
            self._kin[side] = MK.ManoKin(_mano(), side)
        kin, j = self._kin[side], z["_j"][f]
        V = kin.mano_vertices(z["beta"], {fg: z["q_" + fg][j].astype(float) for fg in MK.FING})
        V = V @ z["A"][j][:3, :3].T.astype(float) + z["A"][j][:3, 3]          # SDK wrist frame
        V = V @ A_cam[:3, :3].T + A_cam[:3, 3]                               # head camera
        Fc = kin.m.f if side == "right" else kin.m.f[:, [0, 2, 1]]
        return V, Fc
