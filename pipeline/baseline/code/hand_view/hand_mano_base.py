"""Shared MANO bits for the Touch panel's hand view: a right MANO hand, fingers slightly spread and curled, in a palm-view frame
(x to the viewer's right, y toward the fingertips, z out of the palm toward the viewer), subdivided once, with per-vertex part labels
(palm, thumb, index, middle, ring, little from the skinning weights) and a z-buffer of the palm-side surface."""
import pickle
from pathlib import Path
import numpy as np

TB = Path("/root/autodl-tmp/tactile_baseline")
PARTS = ["palm", "thumb", "index", "middle", "ring", "little"]
GROUP = {"palm": [0], "index": [1, 2, 3], "middle": [4, 5, 6], "little": [7, 8, 9], "ring": [10, 11, 12], "thumb": [13, 14, 15]}
TIPV = {"thumb": 745, "index": 317, "middle": 444, "ring": 556, "little": 673}
CHAIN = {"thumb": [13, 14, 15], "index": [1, 2, 3], "middle": [4, 5, 6], "ring": [10, 11, 12], "little": [7, 8, 9]}


def rod(r):
    th = np.linalg.norm(r, axis=1, keepdims=True) + 1e-12; k = r / th
    K = np.zeros((len(r), 3, 3)); K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]; K[:, 1, 0] = k[:, 2]; K[:, 1, 2] = -k[:, 0]; K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    s = np.sin(th)[:, :, None]; c = np.cos(th)[:, :, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


class Mano:
    def __init__(self):
        m = pickle.load(open(TB / "VITRA/weights/mano/MANO_RIGHT.pkl", "rb"), encoding="latin1")
        self.vt = np.asarray(m["v_template"], np.float64); self.f = np.asarray(m["f"], np.int64)
        Jr = m["J_regressor"]; self.Jr = Jr.toarray() if hasattr(Jr, "toarray") else np.asarray(Jr)
        self.W = np.asarray(m["weights"], np.float64); self.par = np.asarray(m["kintree_table"])[0].astype(int); self.par[0] = -1
        self.PD = np.asarray(m["posedirs"], np.float64)

    def lbs(self, pose):
        J = self.Jr @ self.vt; R = rod(pose)
        vp = self.vt + self.PD @ (R[1:] - np.eye(3)).reshape(-1)
        G = np.zeros((16, 4, 4))
        for i in range(16):
            T = np.eye(4); T[:3, :3] = R[i]; T[:3, 3] = J[i] - (J[self.par[i]] if i else 0)
            G[i] = T if i == 0 else G[self.par[i]] @ T
        A = G.copy(); A[:, :3, 3] -= np.einsum("jab,jb->ja", G[:, :3, :3], J)
        Tv = np.einsum("vj,jab->vab", self.W, A)
        return np.einsum("vab,vb->va", Tv[:, :3, :3], vp) + Tv[:, :3, 3], G[:, :3, 3]


def subdiv(v, f, attrs):
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    ue, inv = np.unique(e, axis=0, return_inverse=True); inv = np.asarray(inv).reshape(-1)
    mid = len(v) + inv.reshape(3, -1).T
    nv = np.vstack([v, 0.5 * (v[ue[:, 0]] + v[ue[:, 1]])])
    na = [np.vstack([a, 0.5 * (a[ue[:, 0]] + a[ue[:, 1]])]) for a in attrs]
    a_, b_, c_ = f.T; m01, m12, m20 = mid.T
    nf = np.concatenate([np.stack([a_, m01, m20], 1), np.stack([m01, b_, m12], 1), np.stack([m20, m12, c_], 1), np.stack([m01, m12, m20], 1)])
    return nv, nf, na


def vnorm(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def build_hand(spread=(0, 0, 0, 0), curl=0.0, thumb_abd=0.0, levels=1):
    M = Mano()
    v0, J0 = M.lbs(np.zeros((16, 3)))
    u = J0[4] - J0[0]; u /= np.linalg.norm(u)
    r = J0[1] - J0[7]; r -= u * (r @ u); r /= np.linalg.norm(r)
    x = r; y = u; z = np.cross(x, y)                         # right hand seen from the palm: thumb on the viewer's right, z out of the palm
    pose = np.zeros((16, 3)); d = np.pi / 180
    for j, a, sg in zip((1, 4, 10, 7), spread, (-1, -1, 1, 1)):   # spread: degrees away from the middle finger (index to the right, ring and little to the left)
        pose[j] += sg * a * d * z
    for fing in ("index", "middle", "ring", "little"):
        for j in CHAIN[fing]:
            pose[j] += curl * d * x
    pose[13] += thumb_abd * d * z
    v, J = M.lbs(pose)
    n0 = vnorm(v, M.f)
    if np.mean(np.sum((v - v.mean(0)) * n0, 1)) < 0:
        M.f = M.f[:, ::-1].copy(); n0 = -n0
    Rv = np.stack([x, y, z]); o = J[[0, 1, 4, 7, 10]].mean(0)
    P = (v - o) @ Rv.T; N0 = n0 @ Rv.T; JP = (J - o) @ Rv.T
    TP = {k: (v[i] - o) @ Rv.T for k, i in TIPV.items()}
    gw = np.stack([M.W[:, GROUP[p]].sum(1) for p in PARTS], 1)
    P1, F1, N1, GW1 = P, M.f, N0, gw
    for _ in range(levels):
        P1, F1, (N1, GW1) = subdiv(P1, F1, [N1, GW1])
        N1 /= np.linalg.norm(N1, axis=1, keepdims=True)
    lab = np.argmax(GW1, 1)
    return dict(P=P1, F=F1, N=N1, lab=lab, J=JP, tips=TP, P0=P, F0=M.f)


def raster(P, F, lab, res=0.25e-3):
    lo = P[:, :2].min(0) - 3e-3; hi = P[:, :2].max(0) + 3e-3
    nx, ny = np.ceil((hi - lo) / res).astype(int)
    zb = np.full((ny, nx), -np.inf); tid = np.full((ny, nx), -1, np.int64); bary = np.zeros((ny, nx, 3), np.float32)
    Q = (P[:, :2] - lo) / res
    for t, (a, b, c) in enumerate(F):
        A, B, C = Q[a], Q[b], Q[c]
        d = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        if abs(d) < 1e-9:
            continue
        x0, y0 = np.floor(np.minimum(np.minimum(A, B), C)).astype(int); x1, y1 = np.ceil(np.maximum(np.maximum(A, B), C)).astype(int)
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        l1 = ((B[1] - C[1]) * (xs - C[0]) + (C[0] - B[0]) * (ys - C[1])) / d
        l2 = ((C[1] - A[1]) * (xs - C[0]) + (A[0] - C[0]) * (ys - C[1])) / d
        l3 = 1 - l1 - l2
        ins = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not ins.any():
            continue
        zz = l1 * P[a, 2] + l2 * P[b, 2] + l3 * P[c, 2]
        iy, ix = np.nonzero(ins); gy, gx = iy + y0, ix + x0
        ok = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
        iy, ix, gy, gx = iy[ok], ix[ok], gy[ok], gx[ok]
        bt = zz[iy, ix] > zb[gy, gx]
        iy, ix, gy, gx = iy[bt], ix[bt], gy[bt], gx[bt]
        zb[gy, gx] = zz[iy, ix]; tid[gy, gx] = t; bary[gy, gx] = np.stack([l1[iy, ix], l2[iy, ix], l3[iy, ix]], 1)
    flab = np.full(tid.shape, -1)
    m = tid >= 0
    tl = lab[F[tid[m]]]                                       # the part most of the triangle's corners belong to
    flab[m] = np.where(tl[:, 0] == tl[:, 1], tl[:, 0], np.where(tl[:, 1] == tl[:, 2], tl[:, 1], tl[:, 0]))
    return dict(lo=lo, res=res, zb=zb, tid=tid, bary=bary, lab=flab)


def surf(R, P, F, pts2):
    g = np.floor((np.asarray(pts2) - R["lo"]) / R["res"]).astype(int)
    ny, nx = R["tid"].shape
    inb = (g[:, 0] >= 0) & (g[:, 0] < nx) & (g[:, 1] >= 0) & (g[:, 1] < ny)
    gx = np.clip(g[:, 0], 0, nx - 1); gy = np.clip(g[:, 1], 0, ny - 1)
    t = np.where(inb, R["tid"][gy, gx], -1); b = R["bary"][gy, gx].astype(np.float64)
    X = np.einsum("nk,nkd->nd", b, P[F[np.maximum(t, 0)]]); X[t < 0] = np.nan
    return X, t, np.where(t >= 0, R["lab"][gy, gx], -1)
