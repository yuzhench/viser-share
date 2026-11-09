"""MANO (right) arrays and the glove's taxel binding for the MuJoCo tactile hand (run in an env that can unpickle MANO, e.g. vitra):
v_template, J_regressor, skinning weights, parents, faces (outward), the palm-view axes of setup/hand_mano_base.py (x toward the
thumb side, y toward the fingertips, z out of the palm), fingertip vertices, and every grid slot's place on the MANO surface
(triangle + barycentric weights, so it holds in any pose).
  binding: computed from setup/hand_taxel_map.py's output (results/tactile_hand/hand.json + hand_debug.npz) and the sensor masks
           (results/tactile_<take>/filtered_<side>.npz) when they are there, and saved as results/mujoco_tactile/taxel_binding.npz;
           otherwise read from taxel_binding.npz (next to this script, or in results/mujoco_tactile/), so only MANO_RIGHT.pkl is needed.
    MANO_RIGHT_PKL=... python mj_export_mano.py   -> results/mujoco_tactile/mano_assets.npz (+ taxel_binding.npz)"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

TB = Path(os.environ.get("TACTILE_BASELINE", "/root/autodl-tmp/tactile_baseline"))   # working dir (results/ under it); override by env
OUT = TB / "results/mujoco_tactile"; OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
TIPS = ("thumb", "index", "middle", "ring", "little")

m = pickle.load(open(os.environ.get("MANO_RIGHT_PKL", TB / "VITRA/weights/mano/MANO_RIGHT.pkl"), "rb"), encoding="latin1")   # MANO licence: not redistributed
vt = np.asarray(m["v_template"], float); f = np.asarray(m["f"], np.int64)
Jr = m["J_regressor"]; Jr = Jr.toarray() if hasattr(Jr, "toarray") else np.asarray(Jr)
Wt = np.asarray(m["weights"], float); par = np.asarray(m["kintree_table"])[0].astype(int); par[0] = -1
fn = np.cross(vt[f[:, 1]] - vt[f[:, 0]], vt[f[:, 2]] - vt[f[:, 0]]); vn = np.zeros_like(vt)
for k in range(3):
    np.add.at(vn, f[:, k], fn)
if np.mean(np.sum((vt - vt.mean(0)) * vn, 1)) < 0:
    f = f[:, ::-1].copy()
J = Jr @ vt
u = J[4] - J[0]; u /= np.linalg.norm(u); r = J[1] - J[7]; r -= u * (r @ u); r /= np.linalg.norm(r); z = np.cross(r, u)


def compute_binding():
    """closest point on the display-pose mesh (setup/hand_mano_base.py build_hand, setup/hand_taxel_map.py pose) for each placed
    slot; keeps the slots that are a working sensor in some take."""
    sys.path.insert(0, str(TB / "setup")); sys.path.insert(0, str(HERE.parent / "hand_view"))
    from hand_mano_base import TIPV, build_hand
    hj = json.loads((TB / "results/tactile_hand/hand.json").read_text()); dbg = np.load(TB / "results/tactile_hand/hand_debug.npz")
    Hd = build_hand(**hj["pose"]); P0, F0 = Hd["P0"], Hd["F0"]
    sid, sxyz = dbg["slot_ids"], dbg["slot_xyz"]; fin = np.isfinite(sxyz).all(1); sid, sxyz = sid[fin], sxyz[fin]
    A0, B0, C0 = P0[F0[:, 0]], P0[F0[:, 1]], P0[F0[:, 2]]
    bf, bw = [], []
    for p in sxyz:
        ab, ac, ap = B0 - A0, C0 - A0, p - A0
        d00, d01, d11 = (ab * ab).sum(1), (ab * ac).sum(1), (ac * ac).sum(1); d20, d21 = (ap * ab).sum(1), (ap * ac).sum(1)
        den = d00 * d11 - d01 * d01 + 1e-18; v = np.clip((d11 * d20 - d01 * d21) / den, 0, 1); w = np.clip((d00 * d21 - d01 * d20) / den, 0, 1)
        s = v + w; o = s > 1; v[o] /= s[o]; w[o] /= s[o]
        q = A0 + v[:, None] * ab + w[:, None] * ac; k = int(np.argmin(np.linalg.norm(q - p, axis=1)))
        bf.append(F0[k]); bw.append([1 - v[k] - w[k], v[k], w[k]])
    bf = np.array(bf); bw = np.array(bw)
    # F0 is the face list of build_hand (possibly reversed per face); we keep the vertex ids, which are MANO's own
    valid = np.zeros(744, bool)
    for tag in ("1025", "1105", "1105b"):
        for sd in ("left", "right"):
            valid |= np.load(TB / ("results/tactile_%s/filtered_%s.npz" % (tag, sd)))["valid"]
    keep = valid[sid]
    return sid[keep], bf[keep], bw[keep], np.array([TIPV[k] for k in TIPS]), len(sid)


if (TB / "results/tactile_hand/hand_debug.npz").exists():
    slot_ids, tax_vid, tax_bary, tips, n_placed = compute_binding(); how = "computed (%d placed)" % n_placed
    np.savez(OUT / "taxel_binding.npz", slot_ids=slot_ids, tax_vid=tax_vid, tax_bary=tax_bary, tips=tips)
else:
    bp = next(p for p in (HERE / "taxel_binding.npz", OUT / "taxel_binding.npz") if p.exists())
    b = np.load(bp); slot_ids, tax_vid, tax_bary, tips = b["slot_ids"], b["tax_vid"], b["tax_bary"], b["tips"]; how = "read from %s" % bp
np.savez(OUT / "mano_assets.npz", v_template=vt, faces=f, J_regressor=Jr, weights=Wt, parents=par, J=J, ax_x=r, ax_y=u, ax_z=z,
         tips=tips, slot_ids=slot_ids, tax_vid=tax_vid, tax_bary=tax_bary)
print("exported: %d vertices, %d faces, %d taxel slots, binding %s; axes x %s y %s z %s" % (len(vt), len(f), len(slot_ids), how, np.round(r, 3), np.round(u, 3), np.round(z, 3)))
