"""Round 3 of the dead-taxel fill: left glove WG1JA06260822006 (2026-11-04 .. 11-07), whose real holes are two columns wide
(cols 15-16 through the middle finger and the palm, cols 11-12 through the palm); the right glove WG1KA06260725553 of the same days has
no dead taxel, and both gloves use the same 24 x 31 grid (same row and column = same spot). Is a fill correct and reasonable?
  correct    : hide two WORKING columns of the left glove (ring finger 7-8, index finger 22-23, palm 7-8 and 19-20), fill them, score
               against what they really read (test takes only)
  reasonable : fill the REAL holes (cols 15-16 and 11-12) on a test take and look at them next to the right glove
Fills (the left glove's own values never come from the hidden columns):
  interp           linear across the hole from the nearest working column on each side (left glove only)
  ridge            one shared ridge from the 2 columns on each side, rows r-1..r+1 (12 values), learned on the left glove at OTHER
                   two-column windows of the training takes (left glove only; works for columns that never worked)
  copy             the right glove's value at the same spot
  copy_scaled      copy x (left / right sum over the same working taxels of that finger or of the palm, this frame)
  interp_x_right   interp x the right glove's own ratio at that spot and frame (value in the hole / its interpolation across the hole):
  ridge_x_right    the right hand says whether the hole is higher or lower than its sides, the left glove sets the level
Signal: raw minus each taxel's floor (5th percentile over the take), 30 Hz. Gate = frames where the two hands do the same thing there:
both hands' side columns active (mean > 0.05) and their side patterns correlated (> 0.5).
    envs/vitra/bin/python setup/tactile_fill_round3.py      -> results/tactile_fill/round3/"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402

TB = Path("/root/autodl-tmp/tactile_baseline"); D = TB / "data/hilotah/full"; OUT = TB / "results/tactile_fill/round3"; OUT.mkdir(parents=True, exist_ok=True)
H, W, STEP, MAXF = 24, 31, 4, 40000
TRAIN = ["2026-11-04/22-47-08", "2026-11-04/23-33-45", "2026-11-05/20-06-26"]
TEST = ["2026-11-06/19-31-29", "2026-11-07/21-44-54", "2026-11-07/23-12-44"]
HOLES = {"ring finger 7-8": (7, 8, range(1, 10)), "index finger 22-23": (22, 23, range(1, 10)),
         "palm 7-8": (7, 8, range(14, 24)), "palm 19-20": (19, 20, range(10, 24))}
REALDEAD = {11, 12, 15, 16}
METHODS = ["interp", "ridge", "copy", "copy_scaled", "interp_x_right", "ridge_x_right"]
FING = [(0, 5), (5, 11), (13, 19), (21, 26), (26, 31)]


def region(r, c):
    if r >= 10 and 5 <= c <= 25:
        return [(rr, cc) for rr in range(10, 24) for cc in range(5, 26)]
    for a, b in FING:
        if a <= c < b:
            return [(rr, cc) for rr in range(0, 24 if a in (0, 26) else 10) for cc in range(a, b)]
    return []


def prep(t):
    zl, zr = np.load(D / t / "gloves/left/tactile.npz"), np.load(D / t / "gloves/right/tactile.npz")
    tl, tr = zl["t_us"], zr["t_us"]
    j = np.clip(np.searchsorted(tr, tl), 1, len(tr) - 1); j = np.where(np.abs(tr[j - 1] - tl) < np.abs(tr[j] - tl), j - 1, j)
    idx = np.arange(0, len(tl), STEP)
    if len(idx) > MAXF:
        idx = idx[np.linspace(0, len(idx) - 1, MAXF).astype(int)]
    res = []
    for R in (zl["data"][idx].astype(np.float32), zr["data"][j[idx]].astype(np.float32)):
        valid = (R != -1).all(0); alive = valid & ((R > 0).mean(0) >= 0.01)
        fl = np.percentile(R[:, valid], 5, axis=0); X = np.zeros((len(R), 744), np.float32); X[:, valid] = np.clip(R[:, valid] - fl, 0, None)
        res.append((X, alive))
    (XL, aL), (XR, aR) = res
    return XL, XR, aL, aR, (tl[idx] - tl[0]) / 1e6


def cid(r, c):
    return r * W + c


def feats(X, alive, h1, h2, r, near_left):
    cols = [h1 - 1, h1 - 2, h2 + 1, h2 + 2] if near_left else [h2 + 1, h2 + 2, h1 - 1, h1 - 2]
    F = np.zeros((len(X), 12), np.float32); k = 0
    for rr in (r - 1, r, r + 1):
        for cc in cols:
            if 0 <= rr < H and 0 <= cc < W and alive[cid(rr, cc)]:
                F[:, k] = X[:, cid(rr, cc)]
            k += 1
    return F


def fill_hole(XL, XR, aL, aR, h1, h2, rows, w, need_truth):
    """Every method's fill of the hole columns h1, h2 at the given rows. Returns {method: (frames, n_cells)}, truth or None, cells, gate."""
    cells, out, truth, gate = [], {m: [] for m in METHODS}, [], []
    for r in rows:
        sl, sr = cid(r, h1 - 1), cid(r, h2 + 1)
        if not (0 <= h1 - 1 and h2 + 1 < W and aL[sl] and aL[sr] and aR[sl] and aR[sr] and aR[cid(r, h1)] and aR[cid(r, h2)]):
            continue
        if need_truth and not (aL[cid(r, h1)] and aL[cid(r, h2)]):
            continue
        reg = [cid(rr, cc) for rr, cc in region(r, h1) if cc not in (h1, h2) and aL[cid(rr, cc)] and aR[cid(rr, cc)]]
        sL, sR = XL[:, reg].sum(1), XR[:, reg].sum(1)
        scale = np.where(sR > 0.05 * max(len(reg), 1), sL / np.maximum(sR, 1e-6), np.nan)
        side = [cid(rr, cc) for rr in (r - 1, r, r + 1) for cc in (h1 - 2, h1 - 1, h2 + 1, h2 + 2)
                if 0 <= rr < H and 0 <= cc < W and aL[cid(rr, cc)] and aR[cid(rr, cc)]]
        A, B = XL[:, side], XR[:, side]
        act = (A.mean(1) > 0.05) & (B.mean(1) > 0.05)
        Ac, Bc = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
        cor = (Ac * Bc).sum(1) / np.maximum(np.sqrt((Ac ** 2).sum(1) * (Bc ** 2).sum(1)), 1e-9)
        g = act & (cor > 0.5)
        for h, near_left in ((h1, True), (h2, False)):
            wl, wr = (2 / 3, 1 / 3) if near_left else (1 / 3, 2 / 3)
            iL = wl * XL[:, sl] + wr * XL[:, sr]; iR = wl * XR[:, sl] + wr * XR[:, sr]
            rid = np.clip(feats(XL, aL, h1, h2, r, near_left) @ w, 0, None)
            ratio = np.where(iR > 0.02, np.clip(XR[:, cid(r, h)] / np.maximum(iR, 1e-6), 0, 4), 1.0)
            cp = XR[:, cid(r, h)]
            out["interp"].append(iL); out["ridge"].append(rid); out["copy"].append(cp)
            out["copy_scaled"].append(np.where(np.isfinite(scale), cp * np.nan_to_num(scale), iL))
            out["interp_x_right"].append(iL * ratio); out["ridge_x_right"].append(rid * ratio)
            truth.append(XL[:, cid(r, h)]); cells.append((r, h)); gate.append(g)
    if not cells:
        return None
    return {m: np.stack(v, 1) for m, v in out.items()}, np.stack(truth, 1), cells, np.stack(gate, 1)


def metrics(Yt, Yf):
    t1, f1 = Yt > 0.10, Yf > 0.10; tp = (t1 & f1).sum(); pr = tp / max(1, f1.sum()); rc = tp / max(1, t1.sum())
    c = Yt > 0.05; quiet = Yt < 0.05
    return {"r": float(np.corrcoef(Yt.ravel(), Yf.ravel())[0, 1]) if Yf.std() > 0 else 0.0, "F1@0.10": float(2 * pr * rc / max(pr + rc, 1e-9)),
            "precision@0.10": float(pr), "recall@0.10": float(rc),
            "MAE@contact": float(np.abs(Yt[c] - Yf[c]).mean()) if c.any() else None,
            "false_contact": float((Yf[quiet] > 0.10).mean()) if quiet.any() else None,
            "missed_contact": float((Yf[Yt > 0.10] < 0.05).mean()) if (Yt > 0.10).any() else None, "n": int(Yt.size)}


# 1) the shared ridge, learned on the left glove at two-column windows away from the test holes and the real holes
test_cells = {(r, c) for h1, h2, rows in HOLES.values() for r in rows for c in (h1, h2)}
FA, YA = [], []
rng = np.random.default_rng(0)
for t in TRAIN:
    XL, XR, aL, aR, _ = prep(t)
    act = XL.sum(1); pick = np.unique(np.concatenate([rng.choice(len(XL), min(1500, len(XL)), replace=False), np.argsort(-act)[:1500]]))
    Xs = XL[pick]
    for c0 in range(0, W - 5):
        h1, h2 = c0 + 2, c0 + 3
        if REALDEAD & set(range(c0, c0 + 6)):
            continue
        for r in range(H):
            if (r, h1) in test_cells or (r, h2) in test_cells:
                continue
            if not all(aL[cid(r, c)] for c in (h1 - 1, h1, h2, h2 + 1)):
                continue
            for h, nl in ((h1, True), (h2, False)):
                FA.append(feats(Xs, aL, h1, h2, r, nl)); YA.append(Xs[:, cid(r, h)])
FA, YA = np.concatenate(FA), np.concatenate(YA)
G = FA.T @ FA; w = np.linalg.solve(G + 1e-2 * np.trace(G) / 12 * np.eye(12), FA.T @ YA)
print("shared ridge: %d samples from %d training takes; weights (near1 near2 far1 far2 for rows r-1, r, r+1):" % (len(YA), len(TRAIN)))
print("   " + " | ".join(" ".join("%+.2f" % x for x in w[k:k + 4]) for k in (0, 4, 8)), flush=True)
# 2) scores on the test takes
res, per_hole, pooled = {}, {}, {m: [] for m in METHODS}
pooled_t, pooled_g = [], []
for t in TEST:
    XL, XR, aL, aR, ts = prep(t)
    for name, (h1, h2, rows) in HOLES.items():
        o = fill_hole(XL, XR, aL, aR, h1, h2, rows, w, True)
        if o is None:
            continue
        F, Yt, cells, g = o
        for m in METHODS:
            pooled[m].append(F[m].ravel())
        pooled_t.append(Yt.ravel()); pooled_g.append(g.ravel())
        per_hole.setdefault(name, {"t": [], "g": [], **{m: [] for m in METHODS}})
        per_hole[name]["t"].append(Yt.ravel()); per_hole[name]["g"].append(g.ravel())
        for m in METHODS:
            per_hole[name][m].append(F[m].ravel())
    print("test take %s done" % t, flush=True)
Yt, Gt = np.concatenate(pooled_t), np.concatenate(pooled_g)
out = {"note": __doc__.split("    envs")[0].strip(), "train": TRAIN, "test": TEST, "ridge_weights": w.tolist(),
       "gate_open_share": float(Gt.mean()), "gate_open_share_when_truth_in_contact": float(Gt[Yt > 0.10].mean()) if (Yt > 0.10).any() else None,
       "all": {}, "gate_open": {}, "per_hole": {}}
print("hidden working taxels, test takes: %d values; truth above 0.10 in %.1f%%; gate open in %.1f%% of them (%.1f%% of the in-contact ones)"
      % (Yt.size, 100 * (Yt > 0.10).mean(), 100 * Gt.mean(), 100 * out["gate_open_share_when_truth_in_contact"]))
for m in METHODS:
    Yf = np.concatenate(pooled[m])
    out["all"][m] = metrics(Yt, Yf); out["gate_open"][m] = metrics(Yt[Gt], Yf[Gt]) if Gt.any() else None
for name, d in per_hole.items():
    yt = np.concatenate(d["t"]); out["per_hole"][name] = {m: metrics(yt, np.concatenate(d[m])) for m in METHODS}
for tag in ("all", "gate_open"):
    print("== %s" % tag)
    for m in METHODS:
        q = out[tag][m]
        if q:
            print("   %-15s r %.2f  F1@0.10 %.2f (P %.2f R %.2f)  MAE@contact %.3f  false contact %4.1f%%  missed contact %4.1f%%"
                  % (m, q["r"], q["F1@0.10"], q["precision@0.10"], q["recall@0.10"], q["MAE@contact"] or 0, 100 * (q["false_contact"] or 0), 100 * (q["missed_contact"] or 0)))
for name in out["per_hole"]:
    print("   %-20s " % name + "  ".join("%s r %.2f F1 %.2f" % (m, out["per_hole"][name][m]["r"], out["per_hole"][name][m]["F1@0.10"]) for m in ("interp", "ridge", "copy", "interp_x_right")))
(OUT / "metrics.json").write_text(json.dumps(out, indent=1))
np.save(OUT / "ridge_weights.npy", w)
print("ROUND3_SCORES_DONE", flush=True)
