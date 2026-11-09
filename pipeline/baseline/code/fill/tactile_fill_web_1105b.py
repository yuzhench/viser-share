"""Dead-taxel fill for the touch panel of the ours vs VITRA page (take 2026-11-05/20-32-48). Each glove's dead taxels
(tactile/meta.json 'dead': never above 0.01 in the whole take) are filled from the SAME glove's working taxels, on the
filtered contact value the panel shows (results/tactile_1105b/filtered_<side>.npz).
  model    : per dead taxel, its K = 12 nearest working taxels by path length on the slot grid (paths only through slots that
             carry a sensor, so the next finger across a gap is not 'near'; at most 5 steps); one ridge without intercept from
             those 12 relative positions to the centre, learned at every OTHER place of this glove where the centre and all 12
             offsets work (translation-invariant, as round 3), over the whole take at 30 Hz. Fewer learning places than 6 ->
             8, 6, 4 offsets.
  check    : the learning places split into the left / right half of the grid (checkerboard if a half is too small), fit on
             one half, scored on the other (correlation over all frames), both ways; also the plain mean of the same
             neighbours, for comparison
  keep     : a dead taxel is filled only if its check reaches r >= 0.3; the others stay empty (on this take: the left glove's
             three dead rows, where neither the fit nor the plain mean predicts anything)
  output   : filled value = max(0, prediction), 0 below 0.01 (the filter's off level); measured taxels keep their filtered value
Writes <out>/part<k>_filled.bin (left filled, right filled; each n x 526 uint8 = value x 255, frames as the export),
meta.json['filled'] and results/tactile_fill/web_1105b_check.json (per dead taxel: offsets used, learning places, r).
    python setup/tactile_fill_web_1025.py [--out dir]"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

TB = Path("/root/autodl-tmp/tactile_baseline")
sys.path.insert(0, str(TB / "setup"))
from glove_ref import Take  # noqa: E402

OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else TB / "results/site_preview/pipeline/baseline_1105b/tactile"
meta = json.loads((OUT / "meta.json").read_text())
H, W, FPS = meta["H"], meta["W"], meta["fps"]
slots = np.array(meta["slots"])
T = Take("2026-11-05/20-32-48"); src0 = int(T.meta["excerpt"]["source_t_start_us"])
TAKE = TB / "data/hilotah/full/2026-11-05/20-32-48"
RAD, MINQ, RMIN = 5, 6, 0.3
STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def geodesic(valid2d, p):
    D = np.full((H, W), -1, int); r0, c0 = divmod(int(p), W); D[r0, c0] = 0; dq = deque([(r0, c0)])
    while dq:
        r, c = dq.popleft()
        if D[r, c] >= RAD:
            continue
        for dr, dc in STEPS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and valid2d[rr, cc] and D[rr, cc] < 0:
                D[rr, cc] = D[r, c] + 1; dq.append((rr, cc))
    return D


def places(live2d, O):
    return [(r, c) for r in range(H) for c in range(W)
            if live2d[r, c] and all(0 <= r + dr < H and 0 <= c + dc < W and live2d[r + dr, c + dc] for dr, dc in O)]


def fit(G, Q, O):
    K = len(O); A = np.zeros((K, K)); b = np.zeros(K)
    for r, c in Q:
        idx = [(r + dr) * W + (c + dc) for dr, dc in O]
        A += G[np.ix_(idx, idx)]; b += G[idx, r * W + c]
    lam = 1e-3 * np.trace(A) / K + 1e-12
    return np.linalg.solve(A + lam * np.eye(K), b)


def score(G, s, n, Q, O, w):
    if len(Q) < 3:
        return float("nan")
    spy = spp = syy = sp = sy = 0.0
    for r, c in Q:
        q = r * W + c; idx = [(r + dr) * W + (c + dc) for dr, dc in O]
        spy += w @ G[idx, q]; spp += w @ G[np.ix_(idx, idx)] @ w; syy += G[q, q]; sp += w @ s[idx]; sy += s[q]
    N = n * len(Q)
    cov = spy / N - sp / N * sy / N; vp = spp / N - (sp / N) ** 2; vy = syy / N - (sy / N) ** 2
    return float(cov / np.sqrt(vp * vy)) if vp > 1e-12 and vy > 1e-12 else float("nan")


def q8(a):
    return np.round(np.clip(a.astype(np.float32), 0, 1) * 255).astype(np.uint8)


val, ts, M, info, UNF, CHECK = {}, {}, {}, {}, {}, []
for side in ("left", "right"):
    f = np.load(TB / "results/tactile_1105b" / ("filtered_%s.npz" % side))
    valid = f["valid"].astype(bool); dead = np.zeros(744, bool); dead[meta["dead"][side]] = True; live = valid & ~dead
    val[side] = f["value"]
    ts[side] = (np.load(TAKE / "gloves" / side / "tactile.npz")["t_us"] - src0) / 1e6
    assert len(ts[side]) == len(val[side])
    V = val[side][::4].astype(np.float32); V[:, ~live] = 0
    G = V.T.astype(np.float64) @ V.astype(np.float64); s = V.sum(0).astype(np.float64); n = len(V)
    valid2d, live2d = valid.reshape(H, W), live.reshape(H, W)
    models, unf, kept_r = {}, [], []
    for p in np.flatnonzero(dead):
        D = geodesic(valid2d, p); r0, c0 = divmod(int(p), W)
        cand = sorted((D[r, c], (r - r0) ** 2 + (c - c0) ** 2, r - r0, c - c0) for r in range(H) for c in range(W) if D[r, c] > 0 and live2d[r, c])
        row = {"side": side, "slot": int(p), "row": r0, "col": c0}
        for K in (12, 8, 6, 4):
            if len(cand) < K:
                continue
            O = [(dr, dc) for _, _, dr, dc in cand[:K]]
            Q = places(live2d, O)
            if len(Q) < MINQ:
                continue
            A_ = [x for x in Q if x[1] < W // 2]; B_ = [x for x in Q if x[1] >= W // 2]
            if min(len(A_), len(B_)) < 3:
                A_ = [x for x in Q if (x[0] + x[1]) % 2 == 0]; B_ = [x for x in Q if (x[0] + x[1]) % 2 == 1]
            rr = np.nanmean([score(G, s, n, B_, O, fit(G, A_, O)), score(G, s, n, A_, O, fit(G, B_, O))])
            wm = np.ones(K) / K
            rm = np.nanmean([score(G, s, n, B_, O, wm), score(G, s, n, A_, O, wm)])
            row.update({"K": K, "places": len(Q), "r_fit": round(float(rr), 3), "r_plain_mean": round(float(rm), 3), "offsets": O})
            if rr >= RMIN:
                models[int(p)] = ([(r0 + dr) * W + (c0 + dc) for dr, dc in O], fit(G, Q, O)); kept_r.append(rr)
            break
        if int(p) not in models:
            unf.append(int(p))
        row["filled"] = int(p) in models
        CHECK.append(row)
    M[side], UNF[side] = models, unf
    info[side] = {"dead": int(dead.sum()), "filled": len(models), "left_empty": len(unf),
                  "heldout_r_median_filled": round(float(np.median(kept_r)), 2) if kept_r else None,
                  "heldout_r_range_filled": [round(float(min(kept_r)), 2), round(float(max(kept_r)), 2)] if kept_r else None}
    print(side, info[side], flush=True)

files = {}
for k in range(1, 6):
    P = meta["parts"]["part%d" % k]
    tt = np.load(TB / "results/viser_baseline_1105b/prep" / ("part%d.npz" % k))["t_take_s"].astype(np.float64)
    n3 = len(tt); n = int(np.ceil(n3 * 0.2 * FPS)) + 1
    assert n == P["n"], (n, P["n"])
    tau = np.arange(n) / FPS; take_t = tt[0] + tau * (tt[-1] - tt[0]) / (0.2 * (n3 - 1))
    chunks, rep = [], []
    for side in ("left", "right"):
        t = ts[side]; i = np.clip(np.searchsorted(t, take_t), 1, len(t) - 1)
        i = np.where(np.abs(t[i - 1] - take_t) <= np.abs(t[i] - take_t), i - 1, i)
        F = val[side][i].astype(np.float32); out = F.copy()
        dl = np.array(meta["dead"][side], int); out[:, dl] = 0
        fl = np.array(sorted(M[side].keys()), int)
        for p, (idx, w) in M[side].items():
            y = F[:, idx] @ w
            out[:, p] = np.where(y >= 0.01, np.clip(y, 0, 1), 0)
        chunks.append(q8(out[:, slots]))
        on = (q8(out[:, fl]) > 0).sum(1) if len(fl) else np.zeros(n, int)
        rep.append("%s: filled in contact per frame median %d max %d, max filled %.2f" % (side, np.median(on), on.max(), out[:, fl].max() if len(fl) else 0))
    fn = "part%d_filled.bin" % k
    (OUT / fn).write_bytes(np.concatenate([c.reshape(-1) for c in chunks]).tobytes())
    files["part%d" % k] = fn
    print("part%d n=%d | %s" % (k, n, " | ".join(rep)), flush=True)
meta["filled"] = {"order": ["left/filled", "right/filled"], "files": files, "info": info, "unfilled": UNF, "r_min": RMIN,
                  "rule": "dead taxels filled from the same glove's working neighbours: per dead taxel a ridge from its 12 nearest working taxels (path length on the slot grid), learned at every other place of this glove where the same offsets work, whole take at 30 Hz; kept only where a held-out check reaches r >= 0.3; measured taxels unchanged"}
(OUT / "meta.json").write_text(json.dumps(meta))
(TB / "results/tactile_fill/web_1105b_check.json").write_text(json.dumps(CHECK))
print("FILL_WEB_DONE")
