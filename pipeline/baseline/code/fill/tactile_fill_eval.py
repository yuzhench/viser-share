"""Round 1 of the dead-taxel fill (team plan DEAD_TAXEL_FILL_PLAN.md, section 4): fill the RIGHT glove at the LEFT glove's dead
taxels of the same take, then compare with the right glove's true readings. Gloves of 2026-11-04 .. 11-07 (left WG1JA06260822006,
right WG1KA06260725553), six takes on the server.
Signal: raw tactile minus each taxel's floor (its 5th percentile over the take), negatives -> 0; 30 Hz, at most 40000 frames a take;
frames where the right glove's column 10 (rows 2-10) jumps as one are left out. Dead (left): readings > 0 in < 1 % of the take's frames.
Methods (leave one take out: whatever is learned, is learned on the other five takes, with the test take's dead mask):
  zero      no fill
  side_mean nearest alive taxel on each side across the dead line (dead row: up / down in the column; dead column: left / right in the
            row, across a 2-wide group; crossings and isolated dead taxels: all four directions), mean of the sides found
  side_max  the same, the larger side
  inpaint   harmonic fill: 30 rounds of the 3 x 3 mean over alive and already filled cells
  ridge     per dead position, ridge regression without intercept from the alive taxels of its 5 x 5 window (learned on the right glove)
  knn_mean  per dead position, the 16 training frames with the closest 5 x 5 alive pattern; their mean
  knn_draw  the same 16, one drawn at random per frame (keeps the noise)
Metrics at the dead positions: contact (true > tau vs filled > tau) precision / recall / F1 for tau = 0.03, 0.05, 0.10; MAE over all
and over true > 0.05; Pearson r; temporal roughness = mean |frame-to-frame change| of the fill / of the truth.
Writes results/tactile_fill/round1/metrics.json and examples.npz.     python setup/tactile_fill_eval.py"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve
from sklearn.neighbors import NearestNeighbors

TB = Path("/root/autodl-tmp/tactile_baseline"); D = TB / "data/hilotah/full"; OUT = TB / "results/tactile_fill/round1"; OUT.mkdir(parents=True, exist_ok=True)
TAKES = ["2026-11-04/22-47-08", "2026-11-04/23-33-45", "2026-11-05/20-06-26", "2026-11-06/19-31-29", "2026-11-07/21-44-54", "2026-11-07/23-12-44"]
H, W, STEP, MAXF, K, KNN_TRAIN, KNN_TEST, KNN_POOL = 24, 31, 4, 40000, 16, 60000, 8000, 12000
TAUS = (0.03, 0.05, 0.10); rng = np.random.default_rng(0)


def load(t):
    L = np.load(D / t / "gloves/left/tactile.npz")["data"]
    lv = (L != -1).all(0); act = (L[:, lv] > 0).mean(0); dead = np.zeros(744, bool); dead[np.flatnonzero(lv)[act < 0.01]] = True
    R = np.load(D / t / "gloves/right/tactile.npz")["data"]; rv = (R != -1).all(0)
    R = R[::STEP].astype(np.float32)
    if len(R) > MAXF:
        R = R[np.linspace(0, len(R) - 1, MAXF).astype(int)]
    fl = np.percentile(R[:, rv], 5, axis=0); X = np.zeros((len(R), 744), np.float32); X[:, rv] = np.clip(R[:, rv] - fl, 0, None)
    g = X.reshape(-1, H, W); c10 = g[:, 2:11, 10]; nb = np.concatenate([g[:, 2:11, 9], g[:, 2:11, 11]], 1)
    bad = ((c10 > 0.35).sum(1) >= 7) & (np.median(nb, 1) < 0.2)
    return {"take": t, "X": X[~bad], "rv": rv, "dead": dead & rv, "n_bad": int(bad.sum())}


def side_neighbours(dead, valid):
    g, v = dead.reshape(H, W), valid.reshape(H, W)
    drow = {r for r in range(H) if v[r].sum() and g[r][v[r]].mean() >= 0.8}; dcol = {c for c in range(W) if v[:, c].sum() and g[:, c][v[:, c]].mean() >= 0.8}
    out = {}
    for p in np.flatnonzero(dead):
        r, c = divmod(int(p), W)
        dirs = [(-1, 0), (1, 0)] if (r in drow and c not in dcol) else [(0, -1), (0, 1)] if (c in dcol and r not in drow) else [(-1, 0), (1, 0), (0, -1), (0, 1)]
        nb = []
        for dr, dc in dirs:
            rr, cc = r + dr, c + dc
            for _ in range(4):
                if not (0 <= rr < H and 0 <= cc < W) or not v[rr, cc]:
                    break
                if not g[rr, cc]:
                    nb.append(rr * W + cc); break
                rr, cc = rr + dr, cc + dc
        out[int(p)] = nb
    return out


def window(p, alive):
    r, c = divmod(int(p), W)
    return [rr * W + cc for rr in range(r - 2, r + 3) for cc in range(c - 2, c + 3) if 0 <= rr < H and 0 <= cc < W and (rr, cc) != (r, c) and alive[rr * W + cc]]


def inpaint(X, alive, valid, iters=30):
    G = X.reshape(-1, H, W).copy(); hole = (valid & ~alive).reshape(H, W); G[:, hole] = 0
    known = alive.reshape(H, W).astype(np.float32); K3 = np.ones((3, 3), np.float32)
    for _ in range(iters):
        den = convolve(known, K3, mode="constant"); num = convolve(G * known[None], K3[None], mode="constant")
        fill = hole & (den > 0)
        G[:, fill] = num[:, fill] / den[fill]
        known = np.where(fill, 1.0, known).astype(np.float32)
    return G.reshape(-1, 744)


def metrics(Yt, Yf):
    m = {}
    for tau in TAUS:
        t, f = Yt > tau, Yf > tau; tp = (t & f).sum(); fp = (~t & f).sum(); fn = (t & ~f).sum()
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        m["P@%.2f" % tau], m["R@%.2f" % tau], m["F1@%.2f" % tau] = float(pr), float(rc), float(2 * pr * rc / max(1e-9, pr + rc))
    m["MAE"] = float(np.abs(Yt - Yf).mean()); c = Yt > 0.05; m["MAE@contact"] = float(np.abs(Yt[c] - Yf[c]).mean()) if c.any() else None
    m["r"] = float(np.corrcoef(Yt.ravel(), Yf.ravel())[0, 1]) if Yf.std() > 0 else 0.0
    dt, df = np.abs(np.diff(Yt, axis=0)).mean(), np.abs(np.diff(Yf, axis=0)).mean(); m["rough"] = float(df / max(dt, 1e-9))
    return m


t0 = time.time(); data = [load(t) for t in TAKES]
print("loaded", [(d["take"], len(d["X"]), int(d["dead"].sum()), d["n_bad"]) for d in data], "%.0f s" % (time.time() - t0), flush=True)
allm, examples = {}, {}
for ti, te in enumerate(data):
    dead, valid = te["dead"], te["rv"]; alive = valid & ~dead; P = np.flatnonzero(dead)
    train = [d for j, d in enumerate(data) if j != ti]; Xtr = np.concatenate([d["X"] for d in train]); Xte = te["X"]
    Xm = Xte.copy(); Xm[:, dead] = 0
    Yt = Xte[:, P]; fills = {"zero": np.zeros_like(Yt)}
    sn = side_neighbours(dead, valid); sm = np.zeros_like(Yt); sx = np.zeros_like(Yt)
    IP = inpaint(Xm, alive, valid)[:, P]
    for k, p in enumerate(P):
        nb = sn[int(p)]
        if nb:
            sm[:, k] = Xm[:, nb].mean(1); sx[:, k] = Xm[:, nb].max(1)
        else:
            sm[:, k] = sx[:, k] = IP[:, k]
    fills["side_mean"], fills["side_max"], fills["inpaint"] = sm, sx, IP
    rg = np.zeros_like(Yt); km = np.zeros_like(Yt); kd = np.zeros_like(Yt)
    sel_tr = rng.choice(len(Xtr), min(KNN_TRAIN, len(Xtr)), replace=False); sel_te = np.arange(len(Xte)) if len(Xte) <= KNN_TEST else np.linspace(0, len(Xte) - 1, KNN_TEST).astype(int)
    for k, p in enumerate(P):
        f = window(p, alive)
        if not f:
            rg[:, k] = km[:, k] = kd[:, k] = IP[:, k]; continue
        A, y = Xtr[:, f], Xtr[:, p]
        w = np.linalg.solve(A.T @ A + 0.1 * np.eye(len(f)), A.T @ y); rg[:, k] = np.clip(Xm[:, f] @ w, 0, None)
        At, yt = A[sel_tr], y[sel_tr]; act = At.max(1) > 0.02
        q = Xm[sel_te][:, f]; qa = q.max(1) > 0.02
        if act.sum() > K and qa.any():
            Aa, ya = At[act], yt[act]
            if len(Aa) > KNN_POOL:
                keep = rng.choice(len(Aa), KNN_POOL, replace=False); Aa, ya = Aa[keep], ya[keep]
            _, nn = NearestNeighbors(n_neighbors=K, algorithm="brute").fit(Aa).kneighbors(q[qa]); Ya = ya[nn]
            v1 = np.zeros(len(sel_te), np.float32); v2 = np.zeros(len(sel_te), np.float32)
            v1[qa] = Ya.mean(1); v2[qa] = Ya[np.arange(len(Ya)), rng.integers(0, K, len(Ya))]
            km[:, k] = np.interp(np.arange(len(Xte)), sel_te, v1); kd[:, k] = v2[np.clip(np.searchsorted(sel_te, np.arange(len(Xte))), 0, len(sel_te) - 1)]
    fills["ridge"], fills["knn_mean"], fills["knn_draw"] = rg, km, kd
    allm[te["take"]] = {"n_frames": int(len(Xte)), "n_dead": int(len(P)), "base_rate": {"%.2f" % t: float((Yt > t).mean()) for t in TAUS},
                        "methods": {k: metrics(Yt, v) for k, v in fills.items()}}
    print("== test %s | frames %d, dead %d | true > 0.05 at dead positions: %.1f%%" % (te["take"], len(Xte), len(P), 100 * (Yt > 0.05).mean()), flush=True)
    for k, m in allm[te["take"]]["methods"].items():
        print("   %-9s F1@.03 %.2f  F1@.05 %.2f (P %.2f R %.2f)  F1@.10 %.2f | MAE %.4f  MAE@contact %s  r %.2f  rough %.2f" % (
            k, m["F1@0.03"], m["F1@0.05"], m["P@0.05"], m["R@0.05"], m["F1@0.10"], m["MAE"], "%.3f" % m["MAE@contact"] if m["MAE@contact"] is not None else "-", m["r"], m["rough"]), flush=True)
    if te["take"] == "2026-11-07/21-44-54":
        busy = np.argsort(-(Yt > 0.05).sum(1))[:400:40]
        examples = {"take": te["take"], "frames": busy, "true": Xte[busy], "masked": Xm[busy], "dead": dead, "valid": valid, "P": P,
                    **{"fill_" + k: v[busy] for k, v in fills.items()}}
(OUT / "metrics.json").write_text(json.dumps(allm, indent=1))
np.savez_compressed(OUT / "examples.npz", **examples)
print("FILL_EVAL_DONE %.0f s" % (time.time() - t0))
