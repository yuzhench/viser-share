"""The no-contact test take 2026-11-07/23-12-44 (55 s: hands touch nothing; loose fist, single-finger curls): what both gloves
read when nothing is pressed. Per taxel: the resting level (5th percentile of raw), the noise (MAD of raw around it) and how far
it rises above rest (99th percentile of raw minus the resting level); per finger and hand: the summed rise over time.
    envs/vitra/bin/python setup/nocontact_check.py      -> results/tactile_fill/nocontact_2026-11-07_23-12-44.png/.json"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TB = Path("/root/autodl-tmp/tactile_baseline"); T = TB / "data/hilotah/full/2026-11-07/23-12-44"; OUT = TB / "results/tactile_fill"
H, W = 24, 31
REG = {"little": (slice(0, 24), slice(0, 5)), "ring": (slice(0, 10), slice(5, 11)), "middle": (slice(0, 10), slice(13, 19)),
       "index": (slice(0, 10), slice(21, 26)), "thumb": (slice(0, 24), slice(26, 31)), "palm": (slice(10, 24), slice(5, 26))}
BLUE, ORANGE, INK, INK2, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
S = {}
for side in ("left", "right"):
    z = np.load(T / "gloves" / side / "tactile.npz"); R = z["data"].astype(np.float64); t = (z["t_us"] - z["t_us"][0]) / 1e6
    valid = (R != -1).all(0)
    dead = valid & ((R > 0).mean(0) < 0.01)
    rest = np.full(744, np.nan); noise = np.full(744, np.nan); rise = np.full(744, np.nan)
    rest[valid] = np.percentile(R[:, valid], 5, axis=0)
    noise[valid] = 1.4826 * np.median(np.abs(R[:, valid] - np.median(R[:, valid], 0)), 0)
    rise[valid] = np.percentile(R[:, valid], 99, axis=0) - rest[valid]
    X = np.where(valid, np.clip(R - rest, 0, None), 0)
    ts = {}
    for k, (rs, cs) in REG.items():
        m = np.zeros((H, W), bool); m[rs, cs] = True; m = m.reshape(-1) & valid & ~dead
        ts[k] = X[:, m].sum(1)
    S[side] = dict(R=R, t=t, valid=valid, dead=dead, rest=rest, noise=noise, rise=rise, ts=ts, rate=len(t) / t[-1])
    print("%-5s %d samples, %.0f Hz, %d sensor slots, %d dead | rest level median %.3f (p90 %.3f) | noise median %.4f | rise above rest: median %.3f, p90 %.3f, max %.3f | taxels rising > 0.1: %d"
          % (side, len(t), S[side]["rate"], valid.sum(), dead.sum(), np.nanmedian(rest[valid & ~dead]), np.nanpercentile(rest[valid & ~dead], 90),
             np.nanmedian(noise[valid & ~dead]), np.nanmedian(rise[valid & ~dead]), np.nanpercentile(rise[valid & ~dead], 90), np.nanmax(rise[valid & ~dead]),
             int((rise[valid & ~dead] > 0.1).sum())), flush=True)
both = S["left"]["valid"] & S["right"]["valid"] & ~S["left"]["dead"] & ~S["right"]["dead"]
cr = np.corrcoef(S["left"]["rest"][both], S["right"]["rest"][both])[0, 1]
print("resting level at the same slot, left vs right: correlation %.2f, median |difference| %.3f" % (cr, np.median(np.abs(S["left"]["rest"][both] - S["right"]["rest"][both]))))
n = min(len(S["left"]["t"]), len(S["right"]["t"]))
for k in REG:
    a, b = S["left"]["ts"][k][:n], S["right"]["ts"][k][:n]
    c = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    print("  %-6s summed rise: left p99 %6.2f  right p99 %6.2f | left-right correlation over time %.2f" % (k, np.percentile(a, 99), np.percentile(b, 99), c))
fig = plt.figure(figsize=(14, 10.5), dpi=110, facecolor=BG)
gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.25], hspace=0.45, wspace=0.25, left=0.05, right=0.98, top=0.92, bottom=0.06)
fig.text(0.02, 0.975, "No-contact test take 2026-11-07 23-12-44: what the gloves read when nothing is pressed", fontsize=13, fontweight="bold", color=INK, va="top")
fig.text(0.02, 0.948, "55 s, loose fist and single-finger curls, no object touched. Raw = the SDK's calibrated tactile output. Grey: no sensor; blue x: dead (never above 0). Same row and column = the same spot on both gloves.",
         fontsize=9, color=INK2, va="top")
for j, side in enumerate(("left", "right")):
    s = S[side]
    for i, (key, title, vmax) in enumerate((("rest", "resting level (5th pct of raw)", 0.3), ("rise", "rise above rest (99th pct - rest)", 0.3))):
        ax = fig.add_subplot(gs[i, 2 * j:2 * j + 2])
        im = np.full((H, W), np.nan); im.reshape(-1)[s["valid"]] = s[key][s["valid"]]
        ax.imshow(np.where(np.isfinite(im), np.nan, 1), cmap="Greys", vmin=0, vmax=4, interpolation="nearest")
        h = ax.imshow(im, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        dr, dc = np.divmod(np.flatnonzero(s["dead"]), W); ax.scatter(dc, dr, marker="x", s=10, c="#2b8cff", linewidths=0.8)
        ax.set_title("%s glove · %s" % (side, title), fontsize=9.5, color=INK, loc="left"); ax.set_xticks(range(0, W, 5)); ax.set_yticks(range(0, H, 5))
        ax.tick_params(labelsize=7, colors=INK2)
        fig.colorbar(h, ax=ax, fraction=0.025, pad=0.01).ax.tick_params(labelsize=7)
ax = fig.add_subplot(gs[2, :]); off = 0
for k in REG:
    for side, col in (("left", BLUE), ("right", ORANGE)):
        ax.plot(S[side]["t"], S[side]["ts"][k] + off, color=col, lw=0.8)
    ax.text(-0.5, off, k, ha="right", va="bottom", fontsize=8.5, color=INK)
    off += max(1.0, 1.15 * max(np.percentile(S["left"]["ts"][k], 99.5), np.percentile(S["right"]["ts"][k], 99.5)))
ax.set_yticks([]); ax.set_xlabel("time (s)", fontsize=9, color=INK2); ax.set_xlim(0, max(S["left"]["t"][-1], S["right"]["t"][-1]))
ax.set_title("summed rise above rest per finger (blue: left glove, orange: right glove), stacked", fontsize=9.5, color=INK, loc="left")
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "nocontact_2026-11-07_23-12-44.png", dpi=110, facecolor=BG)
print("saved")
