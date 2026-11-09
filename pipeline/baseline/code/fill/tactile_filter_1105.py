"""Tactile filtering for the Wuji gloves (526 taxels in a 24 x 31 slot grid, -1 = no sensor; 120 Hz), take 2026-11-05/19-11-14.
What is wrong with the raw residual: most of the 'noise' is not jitter (sample-to-sample change is tiny) but per-taxel offsets:
many taxels rest well above 0 (the glove's own no-contact baseline is off for them), so ~15 % of the taxels read > 0.1 in
every frame whether the hand touches anything or not. Filter:
  1 baseline : per taxel, its resting level = a low percentile over a long window (10th percentile of each 1 s block, then
               the 30th percentile of those over +-30 s), subtracted; negatives -> 0
  2 noise    : per taxel noise sigma = 1.4826 x MAD of what is left
  3 contact  : on when above max(4 sigma, 0.02), off when below max(2 sigma, 0.01) (hysteresis), at least 3 samples (25 ms)
  4 spatial  : a contact taxel needs at least one contact neighbour in the grid (single speckles dropped)
Writes results/tactile_1105/filtered_<side>.npz (t_us, contact (n, 744) bool, value (n, 744) float16 = baseline-free
residual where in contact, else 0) and prints how much is active before / after.
    python setup/tactile_filter_1025.py"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import convolve, percentile_filter

TB = Path("/root/autodl-tmp/tactile_baseline")
TAKE = TB / "data/hilotah/full/2026-11-05/19-11-14"
OUT = TB / "results/tactile_1105"; OUT.mkdir(parents=True, exist_ok=True)
H, W = 24, 31
for side in ("left", "right"):
    z = np.load(TAKE / "gloves" / side / "tactile.npz"); d = z["data"].astype(np.float32); t = z["t_us"]
    valid = (d != -1).all(0); v = d[:, valid]; n = len(v)
    B = 120; nb = n // B
    blk = np.percentile(v[:nb * B].reshape(nb, B, -1), 10, axis=1)                  # (nb, 526) resting level per second
    base_b = percentile_filter(blk, 30, size=(61, 1), mode="nearest")
    tb = (np.arange(nb) + 0.5) * B
    base = np.stack([np.interp(np.arange(n), tb, base_b[:, j]) for j in range(v.shape[1])], 1).astype(np.float32)
    r = np.clip(v - base, 0, None)
    sig = 1.4826 * np.median(np.abs(r - np.median(r, 0)), 0) + 1e-4
    on, off = np.maximum(4 * sig, 0.02), np.maximum(2 * sig, 0.01)
    state = np.zeros(v.shape[1], bool); C = np.zeros(r.shape, bool)
    for i in range(n):
        x = r[i]; state = np.where(state, x > off, x > on); C[i] = state
    # minimum duration 3 samples: drop runs shorter than that
    run = C.copy()
    keep = C & (np.roll(C, 1, 0) | np.roll(C, -1, 0))
    keep &= (np.roll(C, 1, 0) & np.roll(C, 2, 0)) | (np.roll(C, -1, 0) & np.roll(C, -2, 0)) | (np.roll(C, 1, 0) & np.roll(C, -1, 0))
    C = keep
    full = np.zeros((n, H * W), bool); full[:, valid] = C
    nbr = convolve(full.reshape(n, H, W).astype(np.uint8), np.ones((1, 3, 3), np.uint8), mode="constant") - full.reshape(n, H, W)
    full &= (nbr.reshape(n, H * W) > 0)
    val = np.zeros((n, H * W), np.float16); rv = np.zeros((n, H * W), np.float32); rv[:, valid] = r; val[full] = rv[full].astype(np.float16)
    np.savez_compressed(OUT / ("filtered_%s.npz" % side), t_us=t, contact=full, value=val, valid=valid, base=base.astype(np.float16), sigma=sig)
    raw_act = (v > 0.1).sum(1); flt_act = full.sum(1)
    print("%-5s raw > 0.1: median %3.0f taxels per frame (p90 %3.0f) | filtered contact: median %3.0f (p90 %3.0f, max %3.0f) | frames with any contact %.0f%% | noise sigma median %.4f" % (
        side, np.median(raw_act), np.percentile(raw_act, 90), np.median(flt_act), np.percentile(flt_act, 90), flt_act.max(), 100 * np.mean(flt_act > 0), np.median(sig)), flush=True)
print("TACTILE_FILTER_DONE")
