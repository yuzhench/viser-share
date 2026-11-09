"""MuJoCo 4.0 demos for the tactile hand: (1) power grasp + lift of a 500 ml bottle, (2) three-finger twisting of its cap (three regrasps
of 90 deg, the bottle held by the other hand = a soft weld). The hand is the dynamic build of ManoHand: 16 segments welded to mocap
targets that follow a scripted MANO pose. Every frame the 534 glove taxels are read from MuJoCo's tactile sensor (penetration depth)
and turned into a glove-like reading by a placeholder model (soft-layer spread + saturation; to be calibrated on the table-press
recording).
outputs: results/mujoco_tactile/demo_<name>.npz + summary.json; results/site_preview/tmp_check/mj_<name>.mp4 and mj_<name>_*.png
    envs/mujoco4/bin/python setup/mj_demo_bottle.py [power] [twist]"""
import ctypes
import os
from pathlib import Path
TB = Path(os.environ.get("TACTILE_BASELINE", "/root/autodl-tmp/tactile_baseline"))   # working dir (results/ under it); override by env
_EGL = os.environ.get("EGL_LIB", str(TB / "envs/egl/usr/lib/x86_64-linux-gnu/libEGL.so.1"))   # glvnd libEGL if not installed system-wide
if os.path.exists(_EGL):
    ctypes.CDLL(_EGL, mode=ctypes.RTLD_GLOBAL)
os.environ.setdefault("MUJOCO_GL", "egl")
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import imageio.v2 as iio  # noqa: E402
import matplotlib  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(TB / "setup")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import mj_bottle_scene as S  # noqa: E402
from mj_tactile_hand import ManoHand, rod  # noqa: E402

OUT = TB / "results/mujoco_tactile"; PREV = TB / "results/site_preview/tmp_check"
OUT.mkdir(parents=True, exist_ok=True); PREV.mkdir(parents=True, exist_ok=True)
FPS, W3, H3, CELL, TOP = 25, 640, 528, 16, 24
LUT = (matplotlib.colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
REGIONS = {"palm": [0], "thumb": [13, 14, 15], "index": [1, 2, 3], "middle": [4, 5, 6], "ring": [10, 11, 12], "little": [7, 8, 9]}
FONT = ImageFont.load_default(size=17); FONT_S = ImageFont.load_default(size=14)
ACTIVE = 0.05


def sm(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


class Glove:
    """placeholder glove response (to be calibrated on the table-press recording): soft-layer spread of the penetration depth over
    neighbouring taxels (Gaussian in rest-surface distance, sigma 2.5 mm), then a saturating reading 1 - exp(-d / d0), d0 = 1.5 mm."""

    def __init__(self, H, sigma=0.0025, d0=0.0015):
        P = H.tax_rest; D2 = ((P[:, None] - P[None]) ** 2).sum(-1)
        K = np.exp(-D2 / (2 * sigma ** 2)); K[D2 > (3 * sigma) ** 2] = 0
        self.K, self.d0 = K / K.sum(1, keepdims=True), d0

    def __call__(self, depth):
        return 1 - np.exp(-(self.K @ np.clip(depth, 0, None)) / self.d0)


def phase_at(seq, t):
    k, t0 = 0, 0.0
    while k < len(seq) - 1 and t >= t0 + seq[k][0]:
        t0 += seq[k][0]; k += 1
    return seq[k][1], sm((t - t0) / seq[k][0])


class Closer:
    """grip controller on the hand targets (what a hand does without vision), joint by joint like a closing hand: the MCP flexes until
    the finger presses with f_seg, then the PIP until the middle + distal phalanges do, then the DIP until the distal does (or a joint
    reaches its limit), so each finger wraps the object; the thumb moves from its pre-shape towards an over-closed target until it
    presses with f_thumb. While held it re-tightens if a finger loses its grip; closing=False opens the hand."""

    def __init__(self, G, pre, thumb_target, fingers=("index", "middle", "ring", "little"), f_seg=2.0, f_thumb=4.0,
                 omega=np.radians(130), maxdeg=(95, 100, 80), thumb_time=0.9):
        self.G, self.pre, self.tt, self.fingers = G, pre.copy(), thumb_target.copy(), fingers
        self.f_seg, self.f_thumb, self.omega, self.maxr, self.trate = f_seg, f_thumb, omega, np.radians(maxdeg), 1.0 / thumb_time
        self.ang = {f: np.zeros(3) for f in fingers}; self.stage = {f: 0 for f in fingers}; self.ut = 0.0

    def step(self, dt, F, closing):
        if closing:
            for f in self.fingers:
                ch = S.CHAINS[f]; k = self.stage[f]
                if k < 3:
                    if F[ch[k:]].sum() >= self.f_seg or self.ang[f][k] >= self.maxr[k]:
                        self.stage[f] += 1
                    else:
                        self.ang[f][k] += self.omega * dt
                elif F[ch].sum() < 0.5 * self.f_seg:
                    for kk in (1, 2):
                        if self.ang[f][kk] < self.maxr[kk]:
                            self.ang[f][kk] += 0.3 * self.omega * dt
            if F[[13, 14, 15]].sum() < self.f_thumb:
                self.ut = min(1.0, self.ut + self.trate * dt)
        else:
            for f in self.fingers:
                self.ang[f] = np.maximum(0.0, self.ang[f] - 1.5 * self.omega * dt); self.stage[f] = 0
            self.ut = max(0.0, self.ut - 1.5 * self.trate * dt)

    def pose(self):
        p = self.pre.copy()
        for f in self.fingers:
            ch, a = S.CHAINS[f], self.G.flex[f]
            for k in range(3):
                p[ch[k]] = self.pre[ch[k]] + a * self.ang[f][k]
        p[13:16] = self.pre[13:16] + self.ut * (self.tt[13:16] - self.pre[13:16])
        return p


class Wrapper:
    """power-grasp controller: each finger moves along its planned wrap (all joints together in the ratio that lays proximal, middle
    and distal phalanges on the object at u = 1), by position up to u = 1, then on (up to u = 1.3) while the finger presses with less
    than f_fing; the thumb likewise from its pre-shape to its planned contact and on until it presses with f_thumb. Re-tightens while
    held; closing=False opens."""

    def __init__(self, pre, wrap, fingers=("index", "middle", "ring", "little"), f_fing=4.0, f_thumb=4.0, t_close=1.0, t_thumb=0.9, umax=1.3):
        self.pre, self.wrap, self.fingers, self.f_fing, self.f_thumb, self.umax = pre.copy(), wrap.copy(), fingers, f_fing, f_thumb, umax
        self.rate, self.trate = 1.0 / t_close, 1.0 / t_thumb; self.u = {f: 0.0 for f in fingers}; self.ut = 0.0

    def step(self, dt, F, closing):
        for f in self.fingers:
            if closing:
                if self.u[f] < 1.0 or F[S.CHAINS[f]].sum() < self.f_fing:      # the planned wrap by position, then by force
                    self.u[f] = min(self.umax, self.u[f] + self.rate * dt)
            else:
                self.u[f] = max(0.0, self.u[f] - 1.5 * self.rate * dt)
        if closing:
            if self.ut < 1.0 or F[[13, 14, 15]].sum() < self.f_thumb:
                self.ut = min(self.umax, self.ut + self.trate * dt)
        else:
            self.ut = max(0.0, self.ut - 1.5 * self.trate * dt)

    def pose(self):
        p = self.pre.copy()
        for f in self.fingers:
            ch = S.CHAINS[f]; p[ch] = self.pre[ch] + self.u[f] * (self.wrap[ch] - self.pre[ch])
        p[13:16] = self.pre[13:16] + self.ut * (self.wrap[13:16] - self.pre[13:16])
        return p


class Pincher:
    """the same idea for a fingertip pinch: each listed finger moves from its open pose towards an over-closed one until it presses
    with f_des (re-tightening while held)."""

    def __init__(self, pop, psq, fingers=("thumb", "index", "middle"), f_des=3.0, t_close=0.45):
        self.pop, self.psq, self.fingers, self.f_des, self.rate = pop.copy(), psq.copy(), fingers, f_des, 1.0 / t_close
        self.u = {f: 0.0 for f in fingers}

    def step(self, dt, F, closing):
        for f in self.fingers:
            if closing:
                if F[S.CHAINS[f]].sum() < self.f_des:
                    self.u[f] = min(1.0, self.u[f] + self.rate * dt)
            else:
                self.u[f] = max(0.0, self.u[f] - 1.5 * self.rate * dt)

    def pose(self):
        p = self.pop.copy()
        for f in self.fingers:
            ch = S.CHAINS[f]; p[ch] = self.pop[ch] + self.u[f] * (self.psq[ch] - self.pop[ch])
        return p


def plan_power(H, G):
    Tg, pg, pre, info = S.power_grasp(H, G, z_h=0.07, sq_palm=0.0005, sq=0.0015, sq_thumb=0.0015); LIFT, PUSH = 0.12, 0.004
    C = Wrapper(pre, pg)
    seq = [(0.4, "settle"), (1.0, "approach"), (1.6, "close"), (1.5, "lift"), (1.2, "hold in the air"), (1.4, "put down"),
           (0.8, "release"), (0.8, "retract")]

    def at(dx=0.0, dz=0.0):
        T = Tg.copy(); T[0, 3] += dx; T[2, 3] += dz
        return T

    def ctrl(t, dt, F):
        ph, u = phase_at(seq, t)
        if ph in ("close", "lift", "hold in the air", "put down"):
            C.step(dt, F, True)
        elif ph == "release":
            C.step(dt, F, False)
        P = -PUSH   # the arm pulls the bottle into the palm while the fingers wrap (they re-tighten by force as it moves)
        T = {"settle": lambda: at(0.07), "approach": lambda: at(0.07 * (1 - u)), "close": lambda: at(P * u), "lift": lambda: at(P, LIFT * u),
             "hold in the air": lambda: at(P, LIFT), "put down": lambda: at(P, LIFT * (1 - u) + 0.0015 * u),
             "release": lambda: at(P, 0.0015), "retract": lambda: at(P + (0.07 - P) * u, 0.0015)}[ph]()
        return C.pose(), T, ph
    ctrl.closer = C
    return ctrl, sum(s[0] for s in seq), info, dict(lift=LIFT)


def plan_twist(H, G):
    Tg, p0, psq, pop, info = S.pinch_grasp(H, G, sq=0.006); Q = np.pi / 2
    P = Pincher(pop, psq, f_des=4.0)
    seq = [(0.4, "settle"), (0.9, "reach down")]
    for c in range(3):
        seq += [(0.6, "pinch %d" % (c + 1)), (1.0, "twist %d" % (c + 1)), (0.4, "let go %d" % (c + 1)), (0.6, "turn back %d" % (c + 1))]
    seq += [(0.8, "lift away")]

    def up(dz):
        T = Tg.copy(); T[2, 3] += dz
        return T

    def rise(k):          # the hand follows the thread: the cap rises PITCH per radian, the fingertips hold it
        return S.PITCH * Q * k

    def ctrl(t, dt, F):
        ph, u = phase_at(seq, t); c = int(ph.split()[-1]) if ph[-1].isdigit() else 0
        if ph.startswith("pinch") or ph.startswith("twist"):
            P.step(dt, F, True)
        elif ph.startswith("let go"):
            P.step(dt, F, False)
        if ph == "settle":
            T = up(0.06)
        elif ph == "reach down":
            T = up(0.06 * (1 - u))
        elif ph.startswith("pinch"):
            T = up(rise(c - 1))
        elif ph.startswith("twist"):
            T = S.rotz(Q * u) @ up(rise(c - 1 + u))
        elif ph.startswith("let go"):
            T = S.rotz(Q) @ up(rise(c))
        elif ph.startswith("turn back"):
            T = S.rotz(Q * (1 - u)) @ up(rise(c))
        else:
            T = up(rise(3) + 0.06 * u)
        return P.pose(), T, ph
    return ctrl, sum(s[0] for s in seq), info, {}


def plan_press(H, G):
    """the calibration motions on the table (what the table-press recording will contain): flat palm press, then fingertip press.
    Flat hand: fingers straight, thumb rotated up about the palm's long axis until it no longer hangs below the palm (MANO's rest
    thumb sits ~6 mm lower); fingertips: the same hand pitched 25 deg fingers-down, thumb tucked."""
    wx, wy, wz = np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, -1.0])     # palm down, fingers +y, thumb -x
    T0 = S.hand_T(H, wx, wy, wz, S.palm_ref(H), (0.19, -0.03, 0.0))
    rest = np.zeros((16, 3)); zp = float(G.pts(rest, T0, [0])[:, 2].min())

    def lift_thumb(T, level):          # smallest CMC rotation that brings the thumb's lowest point up to 'level' (soft-min of z)
        def res(r):
            p = rest.copy(); p[13] = r; z = G.pts(p, T, [13, 14, 15])[:, 2]; zs = -0.0008 * np.log(np.exp(-(z - z.min()) / 0.0008).sum()) + z.min()
            return np.r_[30.0 * max(0.0, level - zs), 0.01 * r]
        r = least_squares(res, np.zeros(3), bounds=(-1.2, 1.2)).x; p = rest.copy(); p[13] = r
        return p
    flat = lift_thumb(T0, zp); th_deg = float(np.degrees(np.linalg.norm(flat[13])))
    zf = float(G.pts(flat, T0, range(16))[:, 2].min())
    Rp = np.eye(4); Rp[:3, :3] = rod(np.array([np.radians(-25.0), 0, 0]))[0]; c = T0[:3, :3] @ S.palm_ref(H) + T0[:3, 3]
    Tp = Rp.copy(); Tp[:3, 3] = c - Rp[:3, :3] @ c; Tp = Tp @ T0                                  # pitched about the palm centre
    zfing = float(G.pts(flat, Tp, range(1, 13))[:, 2].min()); tips = lift_thumb(Tp, zfing + 0.02)
    zt = float(G.pts(tips, Tp, range(16))[:, 2].min()); PRESS, PP = 0.0025, 0.007   # fingertips / palm (the thenar is lowest)
    info = {"thumb_up_deg": th_deg, "flat_lowest_mm": round(1e3 * zf, 1), "palm_lowest_mm": round(1e3 * zp, 1), "tips_lowest_mm": round(1e3 * zt, 1),
            "press_mm": 1e3 * PRESS, "palm_press_mm": 1e3 * PP}
    seq = [(0.4, "settle"), (1.0, "palm down"), (1.0, "palm press"), (0.6, "lift"), (0.6, "tilt"), (1.0, "fingertips down"),
           (1.0, "fingertip press"), (0.6, "lift "), (0.3, "end")]

    def at(dz, T=T0):
        T = T.copy(); T[2, 3] += dz
        return T

    def ctrl(t, dt, F):
        ph, u = phase_at(seq, t); up = 0.05
        if ph == "settle":
            return flat, at(up - zf), ph
        if ph == "palm down":
            return flat, at(up - zf - (up + PP) * u), ph
        if ph == "palm press":
            return flat, at(-zf - PP), ph
        if ph == "lift":
            return flat, at(-zf - PP + (up + PP) * u), ph
        if ph == "tilt":
            Ru = np.eye(4); Ru[:3, :3] = rod(np.array([np.radians(-25.0) * u, 0, 0]))[0]; Ru[:3, 3] = c - Ru[:3, :3] @ c
            return flat + u * (tips - flat), at(up - zf + u * (zf - zt), Ru @ T0), ph
        if ph == "fingertips down":
            return tips, at(up - zt - (up + PRESS) * u, Tp), ph
        if ph == "fingertip press":
            return tips, at(-zt - PRESS, Tp), ph
        if ph == "lift ":
            return tips, at(-zt - PRESS + (up + PRESS) * u, Tp), ph
        return tips, at(up - zt, Tp), ph
    return ctrl, sum(s[0] for s in seq), info, {}


def cam_press(cam, d, bid):
    cam.lookat[:] = (0.19, 0.01, 0.01); cam.distance = 0.36; cam.azimuth = -145; cam.elevation = -22


def cam_power(cam, d, bid):
    p = d.xpos[bid]; cam.lookat[:] = (p[0] + 0.02, p[1], p[2] + 0.09); cam.distance = 0.37; cam.azimuth = 62; cam.elevation = -14


def cam_twist(cam, d, bid):
    cam.lookat[:] = (0.0, 0.0, S.CAPZ - 0.012); cam.distance = 0.31; cam.azimuth = -68; cam.elevation = -24


def add_taxels(scn, P, N, rd):
    for i in np.argsort(rd):
        if scn.ngeom >= scn.maxgeom:
            break
        on = rd[i] > ACTIVE
        rgba = np.r_[LUT[min(255, int(rd[i] * 255))] / 255.0, 1.0] if on else np.array([0.18, 0.2, 0.26, 0.55])
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([0.0018 if on else 0.0008, 0, 0]),
                            P[i] + 0.0005 * N[i], np.eye(3).ravel(), rgba.astype(np.float32))
        scn.ngeom += 1


def grid_img(rd, slots):
    img = np.full((24 * CELL, 31 * CELL, 3), 48, np.uint8); col = LUT[np.clip((rd * 255).astype(int), 0, 255)]
    for s, c in zip(slots, col):
        r, k = divmod(int(s), 31); img[r * CELL + 1:(r + 1) * CELL - 1, k * CELL + 1:(k + 1) * CELL - 1] = c
    return img


def compose(img, rd, slots, top, bottom):
    panel = np.full((H3, 31 * CELL, 3), 18, np.uint8); g = grid_img(rd, slots); panel[TOP:TOP + g.shape[0]] = g
    im = Image.fromarray(np.concatenate([img, panel], 1)); dr = ImageDraw.Draw(im)
    for i, s in enumerate(top):
        dr.text((12, 8 + 22 * i), s, fill=(12, 12, 12), font=FONT)
    for name, c in [("little", 2), ("ring", 7.5), ("middle", 15.5), ("index", 23), ("thumb", 28)]:
        dr.text((W3 + c * CELL + CELL / 2, 4), name, fill=(200, 200, 200), font=FONT_S, anchor="ma")
    for i, s in enumerate(bottom):
        dr.text((W3 + 10, TOP + g.shape[0] + 8 + 18 * i), s, fill=(225, 225, 225), font=FONT_S)
    return np.asarray(im)


def simulate(name, H, G, planner, cam_fn, hold, title, still_at, K=1500.0):
    plan, dur, info, extra = planner(H, G)
    m = mujoco.MjModel.from_xml_string(S.scene_xml(H, hold=hold, K=K)); d = mujoco.MjData(m); H.bind(m); glove = Glove(H)
    sub = int(round(1.0 / (FPS * m.opt.timestep)))
    F = np.zeros(16); pose, T, _ = plan(0.0, 0.0, F); H.teleport(m, d, pose, T); mujoco.mj_forward(m, d)
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bottle")
    qcap = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cap_rot")]
    qlift = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cap_lift")]
    gname = {i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(m.ngeom)}
    ren = mujoco.Renderer(m, H3, W3); opt = mujoco.MjvOption(); opt.geomgroup[1] = 0
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    nf = int(round(dur * FPS)); R = {k: [] for k in ["t", "phase", "depth", "read", "force", "bottle", "cap", "cap_lift", "track", "objF"]}
    wr = iio.get_writer(str(PREV / ("mj_%s.mp4" % name)), fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=16)
    b0 = d.xpos[bid].copy(); stills = dict(still_at); t0 = time.time()
    for fi in range(nf):
        for _ in range(sub):
            pose, T, ph = plan(d.time, m.opt.timestep, F); H.set_pose(m, d, pose, T); mujoco.mj_step(m, d); F = H.seg_forces(m, d)[0]
        depth, _ = H.read(m, d); rd = glove(depth); F, by = H.seg_forces(m, d)
        trk = float(np.linalg.norm(d.xpos[H.body] - d.mocap_pos[H.mocap], axis=1).max())
        objF = {}
        for g, v in by.items():
            objF[gname[g]] = objF.get(gname[g], 0.0) + v
        for k, v in [("t", d.time), ("phase", ph), ("depth", depth.astype(np.float32)), ("read", rd.astype(np.float32)), ("force", F),
                     ("bottle", d.xpos[bid].copy()), ("cap", float(d.qpos[qcap])), ("cap_lift", float(d.qpos[qlift])), ("track", trk), ("objF", objF)]:
            R[k].append(v)
        cam_fn(cam, d, bid); ren.update_scene(d, camera=cam, scene_option=opt)
        add_taxels(ren.scene, H.sensed_points(m, d), H.normals_world(d), rd)
        img = ren.render()
        fr = {r: F[s].sum() for r, s in REGIONS.items()}
        bottom = ["glove map (sim, 534 slots), palm view of the right glove",
                  "contact force [N]  palm %.1f  thumb %.1f  index %.1f" % (fr["palm"], fr["thumb"], fr["index"]),
                  "                   middle %.1f  ring %.1f  little %.1f" % (fr["middle"], fr["ring"], fr["little"]),
                  "active taxels (reading > %.2f): %d   max %.2f" % (ACTIVE, int((rd > ACTIVE).sum()), rd.max())]
        if name == "power":
            bottom.append("bottle height above the table: %+.1f mm" % (1e3 * (d.xpos[bid][2] - b0[2])))
        elif name == "press":
            bottom.append("table force on the hand: %.1f N" % objF.get("table", 0.0))
        else:
            bottom.append("cap turned %.0f deg, risen %.2f mm" % (np.degrees(d.qpos[qcap]), 1e3 * d.qpos[qlift]))
        bottom.append("reading = 1 - exp(-depth / 1.5 mm)  (uncalibrated placeholder)")
        frame = compose(img, rd, H.slots, [title, "t = %.2f s   %s" % (d.time, ph)], bottom)
        wr.append_data(frame)
        for lab, ts in list(stills.items()):
            if d.time >= ts:
                Image.fromarray(frame).save(PREV / ("mj_%s_%s.png" % (name, lab))); stills.pop(lab)
    wr.close()
    print("%s: %d frames in %.1f s" % (name, nf, time.time() - t0), flush=True)
    np.savez_compressed(OUT / ("demo_%s.npz" % name), t=np.array(R["t"]), phase=np.array(R["phase"]), depth=np.array(R["depth"]),
                        read=np.array(R["read"]), force=np.array(R["force"]), bottle=np.array(R["bottle"]), cap=np.array(R["cap"]),
                        cap_lift=np.array(R["cap_lift"]), track=np.array(R["track"]), slots=H.slots, tax_seg=H.tax_seg)
    return R, info, extra


def region_stats(H, rd, F):
    out = {}
    for r, segs in REGIONS.items():
        k = np.isin(H.tax_seg, segs); out[r] = dict(active=int((rd[k] > ACTIVE).sum()), of=int(k.sum()), mean_read=round(float(rd[k].mean()), 3),
                                                     force_N=round(float(F[segs].sum()), 2))
    return out


def summarize_power(H, R, info, extra):
    ph = np.array(R["phase"]); rd = np.array(R["read"]); F = np.array(R["force"]); b = np.array(R["bottle"]); trk = np.array(R["track"])
    hold = ph == "hold in the air"; sq = ph == "close"
    lift_done = float(b[hold][-1, 2] - b[0, 2])
    phal = {"proximal": [1, 4, 7, 10, 14], "middle": [2, 5, 8, 11], "distal": [3, 6, 9, 12, 15]}
    seg_on = (np.array(R["depth"])[hold] > 0).mean(0)
    return {"grasp": info, "hold_in_air": region_stats(H, rd[hold].mean(0), F[hold].mean(0)),
            "phalanx_active_taxels": {k: int(sum(((rd[hold].mean(0) > ACTIVE) & np.isin(H.tax_seg, v)).sum() for _ in [0])) for k, v in phal.items()},
            "segments_touching_in_air": [int(j) for j in range(16) if seg_on[np.isin(H.tax_seg, [j])].max(initial=0) > 0.5],
            "bottle_lifted_mm": round(1e3 * lift_done, 1), "commanded_mm": 1e3 * extra["lift"],
            "slip_in_hand_mm": round(1e3 * (extra["lift"] - lift_done), 2),
            "slip_during_hold_mm": round(1e3 * float(b[hold][0, 2] - b[hold][-1, 2]), 3),
            "total_force_in_air_N": round(float(F[hold].sum(1).mean()), 1), "total_force_squeeze_N": round(float(F[sq].sum(1).mean()), 1),
            "tracking_err_mm": {"max": round(1e3 * float(trk.max()), 2), "free_motion": round(1e3 * float(trk[ph == "approach"].max()), 3)}}


def summarize_press(H, R, info):
    ph = np.array(R["phase"]); rd = np.array(R["read"]); F = np.array(R["force"]); out = {"plan": info}
    for p in ("palm press", "fingertip press"):
        k = ph == p; out[p] = region_stats(H, rd[k][-5:].mean(0), F[k][-5:].mean(0))
        out[p + " table_N"] = round(float(np.mean([o.get("table", 0.0) for o, kk in zip(R["objF"], k) if kk][-5:])), 2)
    return out


def summarize_twist(H, R, info):
    ph = np.array(R["phase"]); rd = np.array(R["read"]); F = np.array(R["force"]); cap = np.array(R["cap"]); lift = np.array(R["cap_lift"])
    out = {"grasp": info, "cycles": []}
    for c in range(1, 4):
        tw = ph == "twist %d" % c; i0, i1 = np.flatnonzero(tw)[[0, -1]]
        out["cycles"].append({"cap_turn_deg": round(float(np.degrees(cap[i1] - cap[max(i0 - 1, 0)])), 1),
                              "during_twist": region_stats(H, rd[tw].mean(0), F[tw].mean(0))})
    out["cap_total_deg"] = round(float(np.degrees(cap[-1])), 1); out["cap_risen_mm"] = round(1e3 * float(lift[-1]), 2)
    out["commanded_deg"] = 270.0
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["power", "twist", "press"]
    H = ManoHand(); G = S.Grasp(H); summ = {}
    sp = OUT / "summary.json"
    if sp.exists():
        summ = json.loads(sp.read_text())
    if "power" in which:
        R, info, extra = simulate("power", H, G, plan_power, cam_power, False, "MuJoCo 4.0 - power grasp, lift, put down",
                                  {"grip": 2.95, "air": 5.0})
        summ["power"] = summarize_power(H, R, info, extra); print(json.dumps(summ["power"]), flush=True)
    if "twist" in which:
        R, info, _ = simulate("twist", H, G, plan_twist, cam_twist, True, "MuJoCo 4.0 - three-finger cap twist (x3)",
                              {"twist2": 5.0, "end": 9.4})
        summ["twist"] = summarize_twist(H, R, info); print(json.dumps(summ["twist"]), flush=True)
    if "press" in which:
        R, info, _ = simulate("press", H, G, plan_press, cam_press, True, "MuJoCo 4.0 - table press (the calibration motions)",
                              {"palm": 2.3, "tips": 5.3})
        summ["press"] = summarize_press(H, R, info); print(json.dumps(summ["press"]), flush=True)
    sp.write_text(json.dumps(summ, indent=1))
