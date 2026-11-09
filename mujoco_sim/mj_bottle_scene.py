"""MuJoCo 4.0 scene for the tactile hand: a table, a 500 ml bottle (body, shoulder, neck) and a screw cap, plus the grasp synthesis
(numpy kinematics on the MANO hand) used by the demos.
  cap  : hinge (turn) + slide (rise) tied to the thread pitch (2.7 mm per turn) by a fixed tendon held at zero length (a joint
         equality is ~1e4 too soft here: MuJoCo's diagonal approximation ignores its slope); dry friction on the hinge is the
         torque the hand must apply; it can only be unscrewed (0 .. 1.5 turns)
  hold : optional soft weld of the bottle to the world (the other hand holding it)
    from mj_bottle_scene import scene_xml, Grasp, power_grasp, pinch_grasp"""
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

TB = Path(os.environ.get("TACTILE_BASELINE", "/root/autodl-tmp/tactile_baseline"))   # working dir (results/ under it); override by env
sys.path.insert(0, str(TB / "setup")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from mj_tactile_hand import ManoHand, rod  # noqa: E402

RB, HB, RC, HC = 0.032, 0.17, 0.0155, 0.012              # bottle radius / body height, cap radius / height (m)
PITCH = 0.0027 / (2 * np.pi)                            # cap rise per radian
CAPZ = HB + 0.035 + HC / 2                              # cap centre height (bottle standing at the origin)
CYL = [(RB, 0.0, HB), (0.022, HB, HB + 0.02), (0.0125, HB + 0.02, HB + 0.035), (RC, CAPZ - HC / 2, CAPZ + HC / 2)]
CHAINS = {"index": [1, 2, 3], "middle": [4, 5, 6], "little": [7, 8, 9], "ring": [10, 11, 12], "thumb": [13, 14, 15]}
THUMB_ANGLES = [-90, -105, -120, -135]
PAD_SLOT = {"thumb": 9 * 31 + 28, "index": 2 * 31 + 23, "middle": 1 * 31 + 15, "ring": 2 * 31 + 7, "little": 6 * 31 + 2}


def scene_xml(H, hold=False, cap_friction=0.08, **hand_kw):
    X = H.mjcf(**hand_kw)
    eq = X["equality"] + '<tendon tendon1="thread" solref="0.004 1"/>'
    if hold:
        eq += '<weld name="bottle_hold" body1="bottle" relpose="0 0 0 1 0 0 0" solref="0.005 1"/>'
    return f"""<mujoco model="tactile_bottle">
<option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="20" noslip_iterations="3"/>
<visual><global offwidth="1600" offheight="1200"/><quality shadowsize="4096"/><headlight ambient="0.42 0.42 0.42" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/></visual>
<asset>{X['asset']}<texture type="skybox" builtin="gradient" rgb1="0.93 0.94 0.96" rgb2="0.70 0.73 0.78" width="512" height="512"/><texture name="tex" type="2d" builtin="checker" rgb1=".88 .88 .85" rgb2=".80 .80 .77" width="512" height="512"/>
<material name="table" texture="tex" texrepeat="12 12"/></asset>
<worldbody>
<light pos="0.3 -0.5 1.1" dir="-0.3 0.5 -1.1" diffuse="0.6 0.6 0.6" castshadow="true"/>
<geom name="table" type="box" size="0.4 0.4 0.02" pos="0 0 -0.02" material="table" contype="1" conaffinity="3" friction="0.8 0.01 0.001"/>
<body name="bottle" pos="0 0 0"><freejoint name="bottle_free"/>
  <geom name="bottle_body" type="cylinder" size="{RB} {HB / 2}" pos="0 0 {HB / 2}" mass="0.5" rgba="0.62 0.82 0.96 0.45" contype="1" conaffinity="3" friction="1 0.01 0.001"/>
  <geom name="bottle_shoulder" type="cylinder" size="0.022 0.01" pos="0 0 {HB + 0.01}" mass="0.01" rgba="0.62 0.82 0.96 0.45" contype="1" conaffinity="3"/>
  <geom name="bottle_neck" type="cylinder" size="0.0125 0.0075" pos="0 0 {HB + 0.0275}" mass="0.005" rgba="0.62 0.82 0.96 0.7" contype="1" conaffinity="3"/>
  <body name="cap" pos="0 0 {CAPZ}">
    <joint name="cap_rot" type="hinge" axis="0 0 1" range="0 540" limited="true" frictionloss="{cap_friction}" damping="0.0005" solreflimit="0.004 1"/>
    <joint name="cap_lift" type="slide" axis="0 0 1" range="0 0.004" limited="true" armature="0.05" solreflimit="0.004 1"/>
    <geom name="cap" type="cylinder" size="{RC} {HC / 2}" mass="0.01" rgba="0.12 0.42 0.9 1" contype="1" conaffinity="3" friction="1.2 0.01 0.001"/>
    <geom name="cap_mark" type="box" size="0.0018 0.0075 0.0004" pos="0 0.0065 {HC / 2 + 0.0004}" rgba="1 1 1 1" contype="0" conaffinity="0" mass="0"/>
  </body>
</body>
{X['body']}
</worldbody>
<tendon><fixed name="thread"><joint joint="cap_lift" coef="1"/><joint joint="cap_rot" coef="{-PITCH:.8f}"/></fixed></tendon>
<deformable>{X['skin']}</deformable>
<equality>{eq}</equality>
<sensor>{X['sensor']}</sensor></mujoco>"""


def obj_depth(P, center=(0.0, 0.0), cyl=CYL):
    """penetration depth (m, > 0 inside) of points P (n,3) into the bottle standing at 'center' (max over its cylinders)."""
    rho = np.linalg.norm(P[:, :2] - np.asarray(center)[None], axis=1); z = P[:, 2]
    return np.max([np.minimum(np.minimum(r - rho, z - z0), z1 - z) for r, z0, z1 in cyl], axis=0)


def hand_T(H, wx, wy, wz, p_ref, q):
    """4x4 placement taking MANO's axes (x: towards the thumb, y: towards the fingertips, z: palm normal) to world directions
    wx, wy, wz, with the rest point p_ref landing on q."""
    A = np.stack([H.ax["x"], H.ax["y"], H.ax["z"]], 1); U, _, Vt = np.linalg.svd(A); A = U @ Vt
    R = np.stack([wx, wy, wz], 1).astype(float) @ A.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.asarray(q, float) - R @ p_ref
    return T


def rotz(phi, c=(0.0, 0.0)):
    """4x4 rotation by phi about the vertical axis through (c, z)."""
    T = np.eye(4); R = rod(np.array([0, 0, phi]))[0]; c3 = np.array([c[0], c[1], 0.0])
    T[:3, :3] = R; T[:3, 3] = c3 - R @ c3
    return T


class Grasp:
    """grasp synthesis on the MANO hand (numpy, no physics): finger flexion axes, pad points, close-until-contact, fingertip IK."""
    BOUNDS = {"thumb": (np.array([-0.85, -0.85, -0.85, -0.3, -0.3]), np.array([0.85, 0.85, 0.85, 1.6, 1.6])),   # CMC <= ~70 deg (skin)
              "finger": (np.array([-0.2, -0.35, 0.0, 0.0]), np.array([1.7, 0.35, 1.9, 1.5]))}
    X0 = {"thumb": np.zeros(5), "finger": np.array([0.6, 0.0, 0.6, 0.4])}

    def __init__(self, H):
        self.H = H; J, z = H.J, H.ax["z"]
        self.flex = {}
        for f, ch in CHAINS.items():
            dvec = J[ch[2]] - J[ch[0]]; dvec /= np.linalg.norm(dvec)
            if f == "thumb":
                nt = H.tax_n[np.isin(H.tax_seg, [14, 15])].mean(0); nt /= np.linalg.norm(nt)
                a = np.cross(dvec, nt)
            else:
                a = np.cross(dvec, z)
            self.flex[f] = a / np.linalg.norm(a)
        self.pad = {f: int(np.flatnonzero(H.slots == s)[0]) for f, s in PAD_SLOT.items()}
        self.dth = (J[14] - J[13]) / np.linalg.norm(J[14] - J[13])   # thumb metacarpal axis: penalise CMC twist about it (LBS skin)

    def pts(self, pose, T, segs):
        R, t = self.H.fk(pose, T)
        return np.vstack([(self.H.v[self.H.vseg == j] - self.H.J[j]) @ R[j].T + t[j] for j in segs])

    def apply(self, pose, f, x):
        """finger x = (MCP flex, MCP abduction, PIP flex, DIP flex); thumb x = (CMC axis-angle (3), MCP flex, IP flex)."""
        p = pose.copy(); ch, a = CHAINS[f], self.flex[f]
        if f == "thumb":
            p[ch[0]] = x[:3]; p[ch[1]] = a * x[3]; p[ch[2]] = a * x[4]
        else:
            p[ch[0]] = a * x[0] + self.H.ax["z"] * x[1]; p[ch[1]] = a * x[2]; p[ch[2]] = a * x[3]
        return p

    def pad_world(self, pose, T, f):
        H = self.H; k = self.pad[f]; s = H.tax_seg[k]; R, t = H.fk(pose, T)
        return R[s] @ (H.tax_rest[k] - H.J[s]) + t[s], R[s] @ H.tax_n[k]

    def close(self, pose, T, finger, pen, squeeze=0.0015, maxdeg=(95, 100, 80), step=0.5):
        """flex the finger joint by joint (proximal first) until the segments beyond it are 'squeeze' inside the object (pen = depth)."""
        ch, a = CHAINS[finger], self.flex[finger]; got = []
        for k, j in enumerate(ch):
            base = pose[j].copy(); ang = 0.0
            for ang in np.arange(0, maxdeg[k] + 1e-9, step):
                pose[j] = base + a * np.radians(ang)
                if pen(self.pts(pose, T, ch[k:])).max() >= squeeze:
                    break
            got.append(float(ang))
        return pose, got

    def _x0(self, kind, x0):
        lb, ub = self.BOUNDS[kind]
        return np.clip(self.X0[kind] if x0 is None else np.asarray(x0, float), lb + 1e-6, ub - 1e-6)

    def tip_ik(self, pose, T, f, target, n_out=None, x0=None, iters=200):
        """place the finger's pad taxel at 'target' (pad facing against the object's outward normal n_out) -> pose, error (m), x."""
        kind = "thumb" if f == "thumb" else "finger"; lb, ub = self.BOUNDS[kind]

        def res(x):
            q, n = self.pad_world(self.apply(pose, f, x), T, f); r = list(q - target)
            if n_out is not None:
                r += list(0.01 * (n + n_out))
            return np.array(r + list(0.002 * x))
        sol = least_squares(res, self._x0(kind, x0), bounds=(lb, ub), max_nfev=iters * 10)
        return self.apply(pose, f, sol.x), float(np.linalg.norm(sol.fun[:3])), sol.x

    def multi_ik(self, R_T, p_ref, q0, targets, base_pose, depth_fn, allow=0.0003, dq=0.08, drot=0.5):
        """wrist (rotation delta on R_T, translation of the palm reference point around q0) + the joints of the fingers in 'targets'
        so that each pad taxel sits on targets[f] = (point, outward object normal) facing it, with no other hand point deeper than
        'allow' in the object (depth_fn)."""
        fingers = list(targets); kinds = ["thumb" if f == "thumb" else "finger" for f in fingers]
        lb = np.concatenate([[-drot] * 3, [-dq] * 3] + [self.BOUNDS[k][0] for k in kinds])
        ub = np.concatenate([[drot] * 3, [dq] * 3] + [self.BOUNDS[k][1] for k in kinds])

        def unpack(x):
            Rw = rod(x[:3])[0] @ R_T; q = q0 + x[3:6]
            T = np.eye(4); T[:3, :3] = Rw; T[:3, 3] = q - Rw @ p_ref
            pose = base_pose.copy(); i = 6
            for f, k in zip(fingers, kinds):
                n = 5 if k == "thumb" else 4; pose = self.apply(pose, f, x[i:i + n]); i += n
            return T, pose

        def res(x):
            T, pose = unpack(x); r = []
            for f in fingers:
                q, n = self.pad_world(pose, T, f); P, n_out = targets[f]
                r += list(q - P) + list(0.01 * (n + n_out))
            r += list(0.003 * x[:3]) + list(0.002 * x[6:])
            r += list(0.3 * np.clip(depth_fn(self.pts(pose, T, range(16))) - allow, 0, None))
            return np.array(r)
        x0 = np.concatenate([np.zeros(6)] + [self.X0[k] for k in kinds]); x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)
        sol = least_squares(res, x0, bounds=(lb, ub), max_nfev=6000)
        T, pose = unpack(sol.x)
        errs = {f: float(np.linalg.norm(self.pad_world(pose, T, f)[0] - targets[f][0])) for f in fingers}
        xs, i = {}, 6
        for f, k in zip(fingers, kinds):
            n = 5 if k == "thumb" else 4; xs[f] = sol.x[i:i + n]; i += n
        return T, pose, errs, xs


def palm_ref(H):
    return H.tax_rest[H.tax_seg == 0].mean(0)


def _power_once(H, G, z_h, y0, sq_palm, sq, sq_thumb=None):
    sq_thumb = sq if sq_thumb is None else sq_thumb
    wx, wy, wz = np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([-1.0, 0, 0])
    rest = np.zeros((16, 3)); p_ref = palm_ref(H); T = hand_T(H, wx, wy, wz, p_ref, (RB + 0.06, y0, z_h))
    for dx in np.arange(0, 0.1, 0.0002):
        Tt = T.copy(); Tt[0, 3] -= dx
        if obj_depth(G.pts(rest, Tt, [0])).max() >= sq_palm:
            break
    Tg = Tt; pose = rest.copy(); info = {"y0": y0, "palm_ref_world": [round(float(x), 4) for x in Tg[:3, :3] @ p_ref + Tg[:3, 3]]}
    for f in ["index", "middle", "ring", "little"]:
        pose, got = G.close(pose, Tg, f, obj_depth, squeeze=sq); info[f + "_deg"] = got
    best = None
    for th in np.radians(THUMB_ANGLES):          # thumb on the near side, opposing the wrapped fingers and pushing into the palm
        for dz in [0.0, 0.015, 0.03]:
            n_out = np.array([np.cos(th), np.sin(th), 0.0]); P = np.array([RB * np.cos(th), RB * np.sin(th), z_h + dz]) - sq_thumb * n_out
            pz, err, xs = G.tip_ik(pose, Tg, "thumb", P, n_out)
            other = float(obj_depth(G.pts(pz, Tg, [13, 14])).max())
            score = err + max(0.0, other - sq_thumb) + 0.002 * max(0.0, float(np.linalg.norm(xs[:3])) - 1.2)
            if best is None or score < best[0]:
                best = (score, pz, float(np.degrees(th)), dz, err, other, P, n_out, xs)
    pose = best[1]; info.update(thumb_angle=best[2], thumb_dz=best[3], thumb_err_mm=round(1e3 * best[4], 2), thumb_base_depth_mm=round(1e3 * best[5], 2))
    # pre-shape for the approach / release: fingers straight, thumb pad 25 mm off its contact point along the outward normal
    P, n_out, xb = best[6], best[7], best[8]
    pre, e_pre, _ = G.tip_ik(rest, Tg, "thumb", P + 0.025 * n_out, n_out, x0=xb)
    path = 0.0
    for s in np.linspace(0, 0.07, 15):
        Ts = Tg.copy(); Ts[0, 3] += s; path = max(path, float(obj_depth(G.pts(pre, Ts, range(1, 16))).max()))
    clos = max(float(obj_depth(G.pts(pre + u * (pose - pre), Tg, range(1, 16))).max()) for u in np.linspace(0, 1, 21))
    info.update(pre_thumb_err_mm=round(1e3 * e_pre, 2), approach_max_depth_mm=round(1e3 * path, 2), close_path_max_depth_mm=round(1e3 * clos, 2))
    info["pre"] = pre
    sd = [round(1e3 * float(obj_depth(G.pts(pose, Tg, [j])).max()), 2) for j in range(16)]; info["seg_depth_mm"] = sd
    info["n_touch"] = int(sum(x > 0.5 for x in sd)); info["max_depth_mm"] = max(sd)
    return Tg, pose, info


def power_grasp(H, G, z_h=0.085, y0s=(-0.035, -0.03, -0.025, -0.02, -0.015, -0.01), sq_palm=0.0015, sq=0.0015, sq_thumb=None):
    """right hand beside the bottle (palm normal -x, thumb up, fingers +y): slide in until the palm is sq_palm inside, curl each finger
    joint by joint until it is sq inside, thumb pad onto the bottle's near side by IK (best of a few contact points). The bottle's
    position along the palm (y0) is chosen to maximise the number of touching segments without anything deeper than 3 mm."""
    runs = [_power_once(H, G, z_h, y0, sq_palm, sq, sq_thumb) for y0 in y0s]
    ok = [r for r in runs if r[2]["max_depth_mm"] <= 1e3 * sq + 2.0] or runs
    Tg, pose, info = max(ok, key=lambda r: r[2]["n_touch"])
    info["search"] = {r[2]["y0"]: r[2]["n_touch"] for r in runs}; pre = info.pop("pre")
    return Tg, pose, pre, info


def pinch_grasp(H, G, sq=0.004, open_d=0.012, ang=None):
    """right hand above the cap, palm down, fingers +x, thumb +y: thumb pad on the cap's -x side, index and middle pads on the +x side
    (wrist pose and finger joints by one IK, no other hand point inside the bottle/cap); ring and little tucked. Returns the
    contact pose, the squeeze pose (pads sq inside the cap) and the open pose (pads open_d outside), same wrist."""
    wx, wy, wz = np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, -1.0])
    p_ref = palm_ref(H); R_T = hand_T(H, wx, wy, wz, p_ref, np.zeros(3))[:3, :3]
    base = np.zeros((16, 3))
    for f in ("ring", "little"):
        ch, a = CHAINS[f], G.flex[f]; base[ch[0]], base[ch[1]], base[ch[2]] = a * np.radians(65), a * np.radians(90), a * np.radians(50)
    ang = ang or {"thumb": 180.0, "index": 22.0, "middle": -18.0}
    tg = {}
    for f, a in ang.items():
        n = np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0]); tg[f] = (np.array([0, 0, CAPZ]) + RC * n, n)
    T, pose, errs, xs = G.multi_ik(R_T, p_ref, np.array([-0.035, 0.0, CAPZ + 0.045]), tg, base, obj_depth)
    sqz, opn, e_sq, e_op = pose.copy(), pose.copy(), {}, {}
    for f, (P, n) in tg.items():
        sqz, e_sq[f], _ = G.tip_ik(sqz, T, f, P - sq * n, n, x0=xs[f])
        opn, e_op[f], _ = G.tip_ik(opn, T, f, P + open_d * n, n, x0=xs[f])
    info = {"ik_err_mm": {f: round(1e3 * e, 2) for f, e in errs.items()}, "sq_err_mm": {f: round(1e3 * e, 2) for f, e in e_sq.items()},
            "open_err_mm": {f: round(1e3 * e, 2) for f, e in e_op.items()},
            "max_depth_contact_mm": round(1e3 * float(obj_depth(G.pts(pose, T, range(16))).max()), 2),
            "seg_depth_squeeze_mm": [round(1e3 * float(obj_depth(G.pts(sqz, T, [j])).max()), 2) for j in range(16)],
            "max_depth_open_mm": round(1e3 * float(obj_depth(G.pts(opn, T, range(16))).max()), 2),
            "palm_ref_world": [round(float(x), 4) for x in T[:3, :3] @ p_ref + T[:3, 3]]}
    return T, pose, sqz, opn, info
