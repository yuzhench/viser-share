"""Taxels for the web page (ours vs the CV baseline, take 2026-11-05/19-11-14): per part, 30 frames a second on the 3D view's
timeline (seconds from the part's first 3D frame; the recording puts 3D frame i at i x 0.2 s), both gloves, two kinds per glove:
  filtered : the contact value from setup/tactile_filter_1025.py (baseline-free residual where in contact, else 0)
  raw      : the calibrated pressure as recorded (gloves/<side>/tactile.npz)
each as uint8 (value 0 - 1 -> 0 - 255, nearest 120 Hz sample) for the 526 slots that carry a sensor.
Writes <out>/meta.json and <out>/part<k>.bin (left filtered, left raw, right filtered, right raw; each n x 526, row-major).
    python setup/export_tactile_web_1025.py [--out dir]     (default results/site_preview/pipeline/baseline/tactile)"""
import json
import sys
from pathlib import Path

import numpy as np

TB = Path("/root/autodl-tmp/tactile_baseline")
sys.path.insert(0, str(TB / "setup"))
from glove_ref import Take  # noqa: E402

OUT = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else TB / "results/site_preview/pipeline/baseline_1105/tactile"
OUT.mkdir(parents=True, exist_ok=True)
T = Take("2026-11-05/19-11-14"); src0 = int(T.meta["excerpt"]["source_t_start_us"])
TAKE = TB / "data/hilotah/full/2026-11-05/19-11-14"
FPS, H, W = 30, 24, 31
raw, flt, ts, vmask = {}, {}, {}, {}
for s in ("left", "right"):
    z = np.load(TAKE / "gloves" / s / "tactile.npz"); f = np.load(TB / "results/tactile_1105" / ("filtered_%s.npz" % s))
    raw[s], flt[s], ts[s] = z["data"], f["value"], (z["t_us"] - src0) / 1e6
    assert len(raw[s]) == len(flt[s]) and np.all(np.diff(ts[s]) >= 0)
    vmask[s] = f["valid"].astype(bool)
valid = vmask["left"] | vmask["right"]          # the two gloves of 11-05 differ: right has 534 sensor slots, left 526
slots = np.flatnonzero(valid)
absent = {s: np.flatnonzero(valid & ~vmask[s]).tolist() for s in ("left", "right")}   # slots of the union this glove does not have


def q(a):
    return np.round(np.clip(a.astype(np.float32), 0, 1) * 255).astype(np.uint8)


dead = {s: np.flatnonzero(vmask[s] & (np.where(raw[s] == -1, 0, raw[s]).max(0) < 0.01)).tolist() for s in ("left", "right")}   # dead row / column wires
meta = {"fps": FPS, "H": H, "W": W, "slots": slots.tolist(), "order": ["left/filtered", "left/raw", "right/filtered", "right/raw"],
        "value": "byte / 255", "raw_active": 0.1,
        "dead": dead, "absent": absent, "dead_rule": "a sensor slot that never reads above 0.01 in the whole take", "parts": {}}
for k in range(1, 6):
    tt = np.load(TB / "results/viser_baseline_1105/prep" / ("part%d.npz" % k))["t_take_s"].astype(np.float64)
    n3 = len(tt); n = int(np.ceil(n3 * 0.2 * FPS)) + 1
    tau = np.arange(n) / FPS
    take_t = tt[0] + tau * (tt[-1] - tt[0]) / (0.2 * (n3 - 1))
    chunks = []
    for s in ("left", "right"):
        i = np.clip(np.searchsorted(ts[s], take_t), 1, len(ts[s]) - 1)
        i = np.where(np.abs(ts[s][i - 1] - take_t) <= np.abs(ts[s][i] - take_t), i - 1, i)
        chunks += [q(flt[s][i][:, slots]), q(raw[s][i][:, slots])]
    (OUT / ("part%d.bin" % k)).write_bytes(np.concatenate([c.reshape(-1) for c in chunks]).tobytes())
    meta["parts"]["part%d" % k] = {"t0": round(float(tt[0]), 4), "n": n, "file": "part%d.bin" % k}
    cl, cr = (chunks[0] > 0).sum(1), (chunks[2] > 0).sum(1)
    print("part%d n=%d take %.2f - %.2f s | in contact per frame: left median %d max %d, right median %d max %d | raw >= 0.1: left median %d, right median %d" % (
        k, n, take_t[0], take_t[-1], np.median(cl), cl.max(), np.median(cr), cr.max(),
        np.median((chunks[1] >= 26).sum(1)), np.median((chunks[3] >= 26).sum(1))), flush=True)
(OUT / "meta.json").write_text(json.dumps(meta))
print("TACT_WEB_DONE", OUT)
