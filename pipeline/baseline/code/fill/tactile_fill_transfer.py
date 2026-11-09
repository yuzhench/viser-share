"""Round 2 of the dead-taxel fill: does a fill learned on the RIGHT glove carry over to the LEFT glove, whose hand does different things?
Test on the left glove itself, where the truth is known: in take 2026-11-04/22-47-08 the left glove's rows 6 and 14 still work (they are
dead in every later take), so we hide them (the take's real dead columns 11, 12, 15, 16 stay hidden too) and fill them, scored on the
second half of the take. Signal: raw minus each taxel's floor (5th percentile over the take), 30 Hz.
  zero, side_mean, inpaint      left glove only, nothing learned
  ridge_right                   per position, ridge from the 5 x 5 alive window, learned on the right glove (all six takes, same mask)
  ridge_right_norm              the same on per-taxel normalised readings (each taxel / its 99th percentile in the take; the hidden
                                taxel's scale on the left taken as the mean scale of its alive window), so glove sensitivity cancels
  ridge_left                    per position, learned on the left glove itself, on the first half of the same take (an upper bound:
                                only possible for lines that worked at some point)
Also: how the two gloves' readings compare at the same positions (99th percentile per taxel).
    python setup/tactile_fill_transfer.py"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve

TB = Path("/root/autodl-tmp/tactile_baseline"); D = TB / "data/hilotah/full"; OUT = TB / "results/tactile_fill/round2"; OUT.mkdir(parents=True, exist_ok=True)
TAKES = ["2026-11-04/22-47-08", "2026-11-04/23-33-45", "2026-11-05/20-06-26", "2026-11-06/19-31-29", "2026-11-07/21-44-54", "2026-11-07/23-12-44"]
HIDE = [int(x) for x in sys.argv[1:]] or [6, 14]                  # rows to hide on the left (e.g. 17 19: healthy palm rows)
TEST = TAKES[0]; H, W, STEP, MAXF = 24, 31, 4, 40000; TAUS = (0.03, 0.05, 0.10)


def prep(t, side):
    R = np.load(D / t / "gloves" / side / "tactile.npz")["data"]; valid = (R != -1).all(0)
    dead = np.zeros(744, bool); dead[np.flatnonzero(valid)[(R[:, valid] > 0).mean(0) < 0.01]] = True
    R = R[::STEP].astype(np.float32)
    if len(R) > MAXF:
        R = R[np.linspace(0, len(R) - 1, MAXF).astype(int)]
    fl = np.percentile(R[:, valid], 5, axis=0); X = np.zeros((len(R), 744), np.float32); X[:, valid] = np.clip(R[:, valid] - fl, 0, None)
    sc = np.ones(744, np.float32); sc[valid] = np.maximum(np.percentile(X[:, valid], 99, axis=0), 0.02)
    return X, valid, dead, sc


def window(p, alive):
    r, c = divmod(int(p), W)
    return [rr * W + cc for rr in range(r - 2, r + 3) for cc in range(c - 2, c + 3) if 0 <= rr < H and 0 <= cc < W and (rr, cc) != (r, c) and alive[rr * W + cc]]


def ridge(A, y):
    return np.linalg.solve(A.T @ A + 0.1 * np.eye(A.shape[1]), A.T @ y)


def inpaint(X, alive, valid, iters=30):
    G = X.reshape(-1, H, W).copy(); hole = (valid & ~alive).reshape(H, W); G[:, hole] = 0
    known = alive.reshape(H, W).astype(np.float32); K3 = np.ones((3, 3), np.float32)
    for _ in range(iters):
        den = convolve(known, K3, mode="constant"); num = convolve(G * known[None], K3[None], mode="constant"); fill = hole & (den > 0)
        G[:, fill] = num[:, fill] / den[fill]; known = np.where(fill, 1.0, known).astype(np.float32)
    return G.reshape(-1, 744)


def metrics(Yt, Yf):
    m = {}
    for tau in TAUS:
        t, f = Yt > tau, Yf > tau; tp = (t & f).sum(); fp = (~t & f).sum(); fn = (t & ~f).sum(); pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn)
        m["F1@%.2f" % tau] = float(2 * pr * rc / max(1e-9, pr + rc)); m["P@%.2f" % tau] = float(pr); m["R@%.2f" % tau] = float(rc)
    m["MAE"] = float(np.abs(Yt - Yf).mean()); c = Yt > 0.05; m["MAE@contact"] = float(np.abs(Yt[c] - Yf[c]).mean()) if c.any() else None
    m["r"] = float(np.corrcoef(Yt.ravel(), Yf.ravel())[0, 1]) if Yf.std() > 0 else 0.0
    m["rough"] = float(np.abs(np.diff(Yf, axis=0)).mean() / max(np.abs(np.diff(Yt, axis=0)).mean(), 1e-9)); return m


XL, vL, dL, scL = prep(TEST, "left")
hide = np.zeros(744, bool)
for r in HIDE:
    hide[r * W:(r + 1) * W] = True
hide &= vL & ~dL                                   # the working taxels of rows 6 and 14 (the truth to score)
mask = dL | hide; alive = vL & ~mask; P = np.flatnonzero(hide)
half = len(XL) // 2; Xa, Xb = XL[:half], XL[half:]
Xm = Xb.copy(); Xm[:, mask] = 0; Yt = Xb[:, P]
fills = {"zero": np.zeros_like(Yt)}
IP = inpaint(Xm, alive, vL)[:, P]; fills["inpaint"] = IP
sm = np.zeros_like(Yt)
for k, p in enumerate(P):
    r, c = divmod(int(p), W); nb = []
    for dr in (-1, 1):
        rr = r + dr
        while 0 <= rr < H and vL[rr * W + c] and mask[rr * W + c]:
            rr += dr
        if 0 <= rr < H and vL[rr * W + c] and not mask[rr * W + c]:
            nb.append(rr * W + c)
    sm[:, k] = Xm[:, nb].mean(1) if nb else IP[:, k]
fills["side_mean"] = sm
R = [prep(t, "right") for t in TAKES]
XR = np.concatenate([x[0] for x in R]); scR = np.mean([x[3] for x in R], 0)
XRn = np.concatenate([x[0] / x[3][None] for x in R])
rr_, rn_, rl_ = np.zeros_like(Yt), np.zeros_like(Yt), np.zeros_like(Yt)
for k, p in enumerate(P):
    f = window(p, alive & R[0][1])
    if not f:
        rr_[:, k] = rn_[:, k] = rl_[:, k] = IP[:, k]; continue
    rr_[:, k] = np.clip(Xm[:, f] @ ridge(XR[:, f], XR[:, p]), 0, None)
    tgt_scale = scL[f].mean()                                    # the hidden taxel's scale on the left: its window's mean scale
    rn_[:, k] = np.clip((Xm[:, f] / scL[f][None]) @ ridge(XRn[:, f], XRn[:, p]), 0, None) * tgt_scale
    rl_[:, k] = np.clip(Xm[:, f] @ ridge(Xa[:, f], Xa[:, p]), 0, None)
fills["ridge_right"], fills["ridge_right_norm"], fills["ridge_left"] = rr_, rn_, rl_
res = {k: metrics(Yt, v) for k, v in fills.items()}
both = vL & R[0][1] & ~dL
ratio = R[0][3][both] / scL[both]
print("left take %s, hidden rows %s: %d taxels, scored on %d frames (second half); true > 0.05 there: %.1f%%" % (TEST, " and ".join(map(str, HIDE)), len(P), len(Xb), 100 * (Yt > 0.05).mean()))
print("gloves at the same positions, 99th percentile of the reading (this take): right / left median %.2f (p10 %.2f, p90 %.2f)" % (np.median(ratio), np.percentile(ratio, 10), np.percentile(ratio, 90)))
for k, m in res.items():
    print("   %-16s F1@.03 %.2f  F1@.05 %.2f  F1@.10 %.2f (P %.2f R %.2f) | MAE %.4f  MAE@contact %s  r %.2f  rough %.2f" % (
        k, m["F1@0.03"], m["F1@0.05"], m["F1@0.10"], m["P@0.10"], m["R@0.10"], m["MAE"], "%.3f" % m["MAE@contact"] if m["MAE@contact"] is not None else "-", m["r"], m["rough"]))
(OUT / ("metrics_rows_%s.json" % "_".join(map(str, HIDE)))).write_text(json.dumps({"take": TEST, "hidden": "rows " + " ".join(map(str, HIDE)), "n": len(P), "scale_ratio_median": float(np.median(ratio)), "methods": res}, indent=1))
np.savez_compressed(OUT / ("examples_rows_%s.npz" % "_".join(map(str, HIDE))), frames=np.argsort(-(Yt > 0.05).sum(1))[:400:40], true=Xb, masked=Xm, P=P, mask=mask, valid=vL, **{"fill_" + k: v for k, v in fills.items()})
print("TRANSFER_DONE")
