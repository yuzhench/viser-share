"""Approximate placement of the glove's 24 x 31 taxel grid on a MANO hand, for the hand view of the Touch panel.
The grid is a picture of a right hand seen from the palm (fingertips up, thumb held along the palm): columns 0-4 little, 5-10 ring,
13-18 middle, 21-25 index, 26-30 thumb, rows 10-23 palm, column 0 rows 14-23 a strip on the side of the hand below the little finger.
The same for both gloves (checked on the 11-07 no-contact take: each finger strip follows its own finger's bend, tmp/nc_blocks.py), so
the left glove is drawn on the mirrored hand.
Each finger strip is laid along its MANO finger (tip row at the fingertip, base row at the knuckle; the thumb's down to the middle of
its long bone), columns across the finger; the palm block onto the palm with a thin-plate spline through the knuckles, the palm's
edges (the ball of the thumb included below the thumb strip) and the wrist; the side strip on the hand's little-finger side.
Every triangle of the (twice subdivided) MANO mesh is given the region of the nearest placed point; each of its corners gets the
grid position of that region's nearest placed point and a coverage that fades over 2-4.5 mm. The page draws, per pixel, a Gaussian
mix (sigma 0.45 taxel) of the working taxels around the interpolated grid position.
Approximate: the glove SDK knows the true taxel positions (tactile_point_cloud), but they are not recorded.
   envs/vitra/bin/python setup/hand_taxel_map.py   -> results/tactile_hand/hand.json + hand.bin (+ hand_debug.npz for tmp/hand_check.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hand_mano_base import PARTS, TB, build_hand, raster, surf  # noqa: E402

OUT = TB / "results/tactile_hand"
OUT.mkdir(parents=True, exist_ok=True)
H, W = 24, 31
POSE = dict(spread=(0, 0, 0, -6), curl=6.0, thumb_abd=10.0, levels=2)
THUMB_BASE = 0.5                                             # thumb strip's base edge: this far from the base joint to the knuckle
STRIP = {"middle": (-0.5, 9.5, 13, 18), "ring": (0.5, 9.5, 5, 10), "index": (0.5, 9.5, 21, 25), "little": (4.5, 13.5, 0, 4), "thumb": (8.5, 18.5, 26, 30)}
PI = {p: i for i, p in enumerate(PARTS)}
INSET = 0.85

Hd = build_hand(**POSE); P, F, N, lab, J, TP = Hd["P"], Hd["F"], Hd["N"], Hd["lab"], Hd["J"][:, :2], {k: v[:2] for k, v in Hd["tips"].items()}
R = raster(P, F, lab, res=0.3e-3)
LI = R["lab"]; ny, nx = LI.shape
print("mesh: %d vertices, %d triangles" % (len(P), len(F)), flush=True)


def lab_at(pts):
    g = np.floor((np.asarray(pts) - R["lo"]) / R["res"]).astype(int)
    ok = (g[:, 0] >= 0) & (g[:, 0] < nx) & (g[:, 1] >= 0) & (g[:, 1] < ny)
    out = np.full(len(g), -1); out[ok] = LI[g[ok, 1], g[ok, 0]]
    return out


def polyline(pts):
    pts = np.asarray(pts, float); seg = np.diff(pts, axis=0); L = np.linalg.norm(seg, axis=1); cum = np.r_[0, np.cumsum(L)]
    def at(s):
        s = np.clip(s, 0, cum[-1]); i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(L) - 1)
        w = (s - cum[i]) / L[i]
        return pts[i] + w[:, None] * seg[i], seg[i] / L[i][:, None]
    return at, cum[-1]


def side_normal(fg, A, D):
    nrm = np.stack([D[:, 1], -D[:, 0]], 1)                  # the axis turned clockwise: toward +x (thumb side) for an upward finger
    if fg == "thumb" and np.mean(np.sum((A - J[1]) * nrm, 1)) < 0:
        nrm = -nrm                                           # thumb: + = away from the index finger
    return nrm


CH = {"thumb": [13, 14, 15], "index": [1, 2, 3], "middle": [4, 5, 6], "ring": [10, 11, 12], "little": [7, 8, 9]}
FW = {}
for fg, ch in CH.items():
    pts = ([J[ch[0]] + THUMB_BASE * (J[ch[1]] - J[ch[0]])] + [J[c] for c in ch[1:]] if fg == "thumb" else [J[c] for c in ch]) + [TP[fg]]
    at, L = polyline(pts)
    s = np.linspace(0, L, 200); A, D = at(s); nrm = side_normal(fg, A, D)
    off = np.arange(-25e-3, 25e-3, 0.25e-3); c = len(off) // 2
    wpos = np.full(len(s), np.nan); wneg = np.full(len(s), np.nan)
    for i in range(len(s)):
        lb = lab_at(A[i] + off[:, None] * nrm[i]) == PI[fg]
        if not lb[c]:
            continue
        hi = c
        while hi + 1 < len(off) and lb[hi + 1]:
            hi += 1
        lo = c
        while lo - 1 >= 0 and lb[lo - 1]:
            lo -= 1
        wpos[i] = off[hi]; wneg[i] = -off[lo]
    good = np.isfinite(wpos); mid = (s > 0.25 * L) & (s < 0.6 * L)
    for arr in (wpos, wneg):
        rv = np.nanmedian(arr[mid])
        arr[~good] = np.interp(s[~good], s[good], arr[good])
        np.minimum(arr, 1.15 * rv, out=arr)
    k = 9; sm = lambda a: np.convolve(np.pad(a, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    FW[fg] = dict(at=at, L=L, s=s, wpos=sm(wpos), wneg=sm(wneg))
    print("%-6s axis %.0f mm, half-widths at 40%%: +%.1f / -%.1f mm (thumb side +)" % (fg, L * 1e3, np.interp(0.4 * L, s, FW[fg]["wpos"]) * 1e3,
          np.interp(0.4 * L, s, FW[fg]["wneg"]) * 1e3), flush=True)


def strip_map(fg, rows, cols):
    t0, t1, c0, c1 = STRIP[fg]; f = FW[fg]
    arc = (t1 - rows) / (t1 - t0) * (f["L"] - 1.2e-3)          # base edge at the knuckle, tip edge 1.2 mm short of the fingertip
    A, D = f["at"](arc); nrm = side_normal(fg, A, D)
    lat = (cols - (c0 - 0.5)) / (c1 - c0 + 1) - 0.5            # -0.5 .. +0.5 across the strip; higher columns are toward the thumb
    wp = np.interp(arc, f["s"], f["wpos"]); wn = np.interp(arc, f["s"], f["wneg"])
    o = np.where(lat >= 0, lat * 2 * wp, lat * 2 * wn) * INSET
    return A + o[:, None] * nrm


def palm_edge(y, side, parts=("palm",)):
    iy = int(np.clip(np.floor((y - R["lo"][1]) / R["res"]), 0, ny - 1)); xs = np.nonzero(np.isin(LI[iy], [PI[q] for q in parts]))[0]
    return R["lo"][0] + (xs.max() + 0.5 if side > 0 else xs.min() + 0.5) * R["res"]


ex = np.array([1.0, 0.0]); IN = 2.0e-3
yb = P[:, 1].min() + 6e-3                                    # the heel of the palm, just above the mesh's cut at the wrist
yr = lambda row: J[1][1] - (row - 9.5) / 14 * (J[1][1] - yb)  # height of a palm row along the thumb side
ctrl = [((9.5, 23.0), J[1]), ((9.5, 15.5), J[4]), ((9.5, 7.5), J[10]), ((9.5, 25.5), J[1] + 0.8 * np.interp(0, FW["index"]["s"], FW["index"]["wpos"]) * ex),
        ((9.5, 4.5), J[10] + 0.35 * (J[7] - J[10])), ((13.5, 4.5), J[7] + 4e-3 * ex),
        ((12.0, 25.5), np.array([palm_edge(yr(12.0), +1) - IN, yr(12.0)])),
        ((14.5, 24.5), np.array([palm_edge(yr(14.5), +1) - IN, yr(14.5)])),                                  # the web between thumb and index
        ((18.5, 25.5), np.array([palm_edge(yr(18.5), +1, ("palm", "thumb")) - 2 * IN, yr(18.5)])),           # the ball of the thumb
        ((20.5, 25.5), np.array([palm_edge(yr(20.5), +1, ("palm", "thumb")) - 2 * IN, yr(20.5)])),
        ((23.5, 22.5), np.array([palm_edge(yb, +1, ("palm", "thumb")) - 3 * IN, yb])),
        ((18.5, 4.5), np.array([palm_edge(J[7][1] - 0.4 * (J[7][1] - yb), -1) + IN, J[7][1] - 0.4 * (J[7][1] - yb)])),
        ((23.5, 4.5), np.array([palm_edge(yb, -1) + IN, yb])),
        ((16.5, 15.5), np.array([J[4][0] - 0.1 * (J[4][0] - J[0][0]), yr(16.5)])), ((23.5, 15.5), np.array([J[0][0], yb]))]
cg = np.array([c[0] for c in ctrl], float); cp = np.array([c[1] for c in ctrl], float)
assert np.isfinite(cp).all(), cp
tps = RBFInterpolator(cg, cp, kernel="thin_plate_spline", smoothing=0.0)
print("palm control points (row, col -> x, y mm):", [(tuple(g), tuple(np.round(p * 1e3, 1))) for g, p in zip(cg, cp)], flush=True)


def palm_map(rows, cols):
    return tps(np.stack([rows, cols], 1))


def edge_3d(rows, cols):                                     # column 0-1, rows 14-23: on the side of the hand below the little finger
    ys = palm_map(rows, np.full(len(rows), 4.5))[:, 1]; out = np.zeros((len(rows), 3))
    for i, (y, c) in enumerate(zip(ys, cols)):
        sel = np.abs(P[:, 1] - y) < 1.5e-3
        k = np.argmin(np.where(sel, P[:, 0], np.inf))
        out[i] = P[k] + c * 2.5e-3 * np.array([0.35, 0.0, 0.94])
    return out


REG = {"middle": (-0.5, 9.5, 12.5, 18.5), "ring": (0.5, 9.5, 4.5, 10.5), "index": (0.5, 9.5, 20.5, 25.5), "little": (4.5, 13.5, -0.5, 4.5),
       "thumb": (8.5, 18.5, 25.5, 30.5), "palm": (9.5, 23.5, 4.5, 25.5), "edge": (13.5, 23.5, -0.5, 1.5)}
RN = list(REG)


def region_of(r, c):
    if r >= 14 and c <= 1:
        return "edge"
    if c <= 4 and 5 <= r <= 13:
        return "little"
    if c >= 26 and 9 <= r <= 18:
        return "thumb"
    if r >= 10 and 2 <= c <= 25:
        return "palm"
    for fg in ("ring", "middle", "index"):
        if STRIP[fg][2] <= c <= STRIP[fg][3] and r <= 9:
            return fg
    return None


def place(reg, rows, cols):                                  # grid positions -> points on the hand (3D) or nan
    if reg == "edge":
        return edge_3d(rows, cols)
    xy = palm_map(rows, cols) if reg == "palm" else strip_map(reg, rows, cols)
    return surf(R, P, F, xy)[0]


S_xyz, S_g, S_reg = [], [], []
for reg, (r0, r1, c0, c1) in REG.items():
    rr, cc = np.meshgrid(np.arange(r0, r1 + 1e-9, 0.1), np.arange(c0, c1 + 1e-9, 0.1), indexing="ij"); rr = rr.ravel(); cc = cc.ravel()
    X = place(reg, rr, cc); ok = np.isfinite(X[:, 0])
    S_xyz.append(X[ok]); S_g.append(np.stack([cc[ok], rr[ok]], 1)); S_reg.append(np.full(ok.sum(), RN.index(reg)))
    print("%-6s %6d samples, %4.1f%% off the hand" % (reg, len(rr), 100 * (1 - ok.mean())), flush=True)
S_xyz = np.concatenate(S_xyz); S_g = np.concatenate(S_g); S_reg = np.concatenate(S_reg)
trees = {i: cKDTree(S_xyz[S_reg == i]) for i in range(len(RN))}
sidx = {i: np.nonzero(S_reg == i)[0] for i in range(len(RN))}

# each triangle -> one region (the nearest placed point among the regions its MANO part may carry), then a majority clean-up
ALLOW = {"palm": RN, "thumb": ["thumb", "palm"], "index": ["index", "palm"], "middle": ["middle", "palm"], "ring": ["ring", "palm"],
         "little": ["little", "palm", "edge"]}
cen = P[F].mean(1); tl = lab[F]
maj = np.where(tl[:, 0] == tl[:, 1], tl[:, 0], np.where(tl[:, 1] == tl[:, 2], tl[:, 1], tl[:, 0]))
bd = np.full(len(F), np.inf); treg = np.full(len(F), -1)
for part, regs in ALLOW.items():
    ti = np.nonzero(maj == PI[part])[0]
    for rg in regs:
        i = RN.index(rg); d, _ = trees[i].query(cen[ti]); b = d < bd[ti]; bd[ti[b]] = d[b]; treg[ti[b]] = i
E = np.concatenate([np.sort(F[:, [0, 1]], 1), np.sort(F[:, [1, 2]], 1), np.sort(F[:, [2, 0]], 1)])
ue, inv = np.unique(E, axis=0, return_inverse=True); inv = np.asarray(inv).reshape(-1); fid = np.tile(np.arange(len(F)), 3)
order = np.argsort(inv, kind="stable"); inv_s = inv[order]; f_s = fid[order]
pair = inv_s[1:] == inv_s[:-1]
a_, b_ = f_s[:-1][pair], f_s[1:][pair]
nb = [[] for _ in range(len(F))]
for x, y in zip(a_, b_):
    nb[x].append(y); nb[y].append(x)
for _ in range(3):
    new = treg.copy()
    for t in range(len(F)):
        rs = [treg[q] for q in nb[t]]
        if rs and treg[t] not in rs:
            vals, cnt = np.unique(rs, return_counts=True); new[t] = vals[np.argmax(cnt)]
        elif len(rs) == 3 and rs.count(treg[t]) == 1 and rs[0] != treg[t] and rs.count(rs[0]) == 2:
            new[t] = rs[0]
    treg = new

# corners: one vertex per (MANO vertex, region)
NR = len(RN)
keys = F * NR + treg[:, None]
uk, inv2 = np.unique(keys.ravel(), return_inverse=True); inv2 = np.asarray(inv2).reshape(-1)
vid = uk // NR; vreg = uk % NR; F2 = inv2.reshape(-1, 3)
G2 = np.zeros((len(uk), 2)); D2 = np.zeros(len(uk))
for i in range(NR):
    sel = np.nonzero(vreg == i)[0]
    if not len(sel):
        continue
    d, k = trees[i].query(P[vid[sel]]); G2[sel] = S_g[sidx[i][k]]; D2[sel] = d
COV = np.clip((4.5e-3 - D2) / 2.5e-3, 0, 1)
P2, N2 = P[vid], N[vid]
print("triangles per region:", {RN[i]: int((treg == i).sum()) for i in range(NR)}, "| vertices %d -> %d" % (len(P), len(P2)), flush=True)
print("vertices with coverage > 0.5: %d (%.0f%%)" % ((COV > 0.5).sum(), 100 * (COV > 0.5).mean()), flush=True)

SLOT = {}
for r in range(H):
    for c in range(W):
        rg = region_of(r, c)
        if rg:
            X = place(rg, np.array([float(r)]), np.array([float(c)]))[0]; SLOT[r * W + c] = (rg, X)

# write (little-endian): pos int16 n*3 (metres x 250000), normal int8 n*3 (x 127), grid uint16 n*3 ((column + 1) x 2000, (row + 1) x 2000,
# coverage x 65535), faces uint16 m*3; each block starts on a 4-byte boundary
QP = 250000.0
assert np.abs(P2).max() * QP < 32767 and len(P2) < 65536
blocks = [np.round(P2 * QP).astype("<i2"), np.round(N2 * 127).astype("i1"),
          np.stack([np.round((G2[:, 0] + 1) * 2000), np.round((G2[:, 1] + 1) * 2000), np.round(COV * 65535)], 1).astype("<u2"), F2.astype("<u2")]
offs, blob = [], b""
for bk in blocks:
    blob += b"\0" * (-len(blob) % 4); offs.append(len(blob)); blob += bk.tobytes()
(OUT / "hand.bin").write_bytes(blob)
meta = dict(version=2, nV=len(P2), nF=len(F2), offsets=dict(pos=offs[0], normal=offs[1], grid=offs[2], faces=offs[3]), qpos=QP, qgrid=2000,
            bbox=[P2.min(0).round(4).tolist(), P2.max(0).round(4).tolist()], grid=[H, W], sigma=0.45,
            frame="right MANO hand from the palm: x to the viewer's right (thumb side), y to the fingertips, z out of the palm; metres",
            pose={k: v for k, v in POSE.items()}, thumb_base=THUMB_BASE,
            note="approximate taxel placement (finger strips along the MANO fingers, palm block by a thin-plate spline); left glove = mirrored hand")
(OUT / "hand.json").write_text(json.dumps(meta))
np.savez(OUT / "hand_debug.npz", P=P2, F=F2, N=N2, G=G2, COV=COV, lab=lab[vid], treg=treg, RN=np.array(RN),
         slot_ids=np.array(list(SLOT)), slot_xyz=np.array([v[1] for v in SLOT.values()]), slot_reg=np.array([RN.index(v[0]) for v in SLOT.values()]))
print("wrote %s (%.0f kB) and hand.json" % (OUT / "hand.bin", len(blob) / 1e3))
print("HAND_MAP_DONE")
