"""MuJoCo 4.0 tactile hand: a right MANO hand made of 16 rigid segments (palm + 3 per finger), posed from MANO skinning (the segment of a
vertex = its largest skinning weight), with the glove's taxels on it.
  collision : per segment, the convex hull of its MANO vertices (hand geoms collide with objects and the table, not with each other)
  taxels    : every grid slot of the glove (setup/hand_taxel_map.py layout, 534 slots) bound to the MANO surface; per segment one
              mesh whose vertices are its taxels, on an invisible non-colliding mesh geom, read by MuJoCo's tactile sensor
              (per vertex: penetration depth into the geoms touching that segment)
  pose      : pose (16 x 3 axis-angle, MANO's frame, as MANO's own pose parameters) + a 4x4 wrist placement in the world
  two builds: mjcf_parts() - the segments are mocap bodies (kinematic, used by selftest());
              mjcf()       - the segments are free, gravity-compensated bodies, each held to a mocap target by a soft weld (the hand's
                             position controller): forces stay bounded, contacts see real hand velocities (friction can turn a cap),
                             plus a MANO skin (MuJoCo skin, bones = segments, MANO weights) for rendering.
Mesh geoms are re-centred by MuJoCo's compiler; the geom frame carries the inverse offset, so mesh vertices taken in the geom frame
land where we put them (checked in selftest()).
    from mj_tactile_hand import ManoHand; H = ManoHand(); X = H.mjcf(); ...; H.bind(m); H.teleport(m, d, pose, T); H.set_pose(...)"""
import os
from pathlib import Path

import mujoco
import numpy as np

TB = Path(os.environ.get("TACTILE_BASELINE", "/root/autodl-tmp/tactile_baseline"))   # working dir (results/ under it); override by env
ASSETS = Path(os.environ.get("MJ_MANO_ASSETS", TB / "results/mujoco_tactile/mano_assets.npz"))   # from mj_export_mano.py
SEG_NAMES = ["palm", "index1", "index2", "index3", "middle1", "middle2", "middle3", "little1", "little2", "little3",
             "ring1", "ring2", "ring3", "thumb1", "thumb2", "thumb3"]
DUMMY = np.array([[0, 0, 0], [0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002]])   # filler vertices on the bone (segments with < 4 taxels)


def fmt(a):
    return " ".join("%.6f" % x for x in np.asarray(a, float).ravel())


def rod(r):
    r = np.atleast_2d(r); th = np.linalg.norm(r, axis=1, keepdims=True) + 1e-12; k = r / th
    K = np.zeros((len(r), 3, 3)); K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]; K[:, 1, 0] = k[:, 2]; K[:, 1, 2] = -k[:, 0]; K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    s = np.sin(th)[:, :, None]; c = np.cos(th)[:, :, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def mat2quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1)); return q


class ManoHand:
    def __init__(self, prefix="h", friction=1.2, solref="0.004 1"):
        z = np.load(ASSETS)
        self.v, self.f, self.W, self.par, self.J = z["v_template"], z["faces"], z["weights"], z["parents"], z["J"]
        self.ax = {"x": z["ax_x"], "y": z["ax_y"], "z": z["ax_z"]}; self.tips = z["tips"]
        self.slots, tv, tb = z["slot_ids"], z["tax_vid"], z["tax_bary"]
        self.prefix, self.friction, self.solref = prefix, friction, solref
        self.vseg = np.argmax(self.W, 1)
        P = np.einsum("sk,skd->sd", tb, self.v[tv]); self.tax_rest = P
        self.tax_seg = np.argmax(np.einsum("sk,skj->sj", tb, self.W[tv]), 1)
        vn = np.zeros_like(self.v); fn = np.cross(self.v[self.f[:, 1]] - self.v[self.f[:, 0]], self.v[self.f[:, 2]] - self.v[self.f[:, 0]])
        for k in range(3):
            np.add.at(vn, self.f[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True); n = np.einsum("sk,skd->sd", tb, vn[tv]); self.tax_n = n / np.linalg.norm(n, axis=1, keepdims=True)
        self.seg_tax = {j: np.flatnonzero(self.tax_seg == j) for j in range(16)}   # the sensor reports in this order per segment
        self.dynamic = False

    # ---- kinematics
    def fk(self, pose, T=np.eye(4)):
        """pose (16,3) axis-angle (MANO convention, parent frame; row 0 = global rotation of the wrist) -> R (16,3,3), t (16,3)
        of each segment in the world (a rest point p goes to R_j (p - J_j) + t_j), after the 4x4 placement T."""
        R = rod(np.asarray(pose, float).reshape(16, 3)); G = np.zeros((16, 4, 4))
        for j in range(16):
            L = np.eye(4); L[:3, :3] = R[j]; L[:3, 3] = self.J[j] - (self.J[self.par[j]] if j else 0)
            G[j] = L if j == 0 else G[self.par[j]] @ L
        G = np.einsum("ab,jbc->jac", T, G)
        return G[:, :3, :3], G[:, :3, 3]

    def taxels_world(self, pose, T=np.eye(4)):
        R, t = self.fk(pose, T); s = self.tax_seg
        return np.einsum("sab,sb->sa", R[s], self.tax_rest - self.J[s]) + t[s]

    # ---- MJCF
    def _taxel_parts(self, j):
        p = self.prefix; ti = self.seg_tax[j]
        if not len(ti):
            return None
        tl = self.tax_rest[ti] - self.J[j]
        if len(tl) < 4:
            tl = np.vstack([tl, DUMMY[:4 - len(tl)]])
        return ('<mesh name="%s_tax%d" inertia="shell" vertex="%s"/>' % (p, j, fmt(tl)),
                '<geom name="%s_tax%d" type="mesh" mesh="%s_tax%d" contype="0" conaffinity="0" group="4" rgba="1 0 0 0" mass="0"/>' % (p, j, p, j),
                '<tactile name="%s_tac%d" geom="%s_tax%d" mesh="%s_tax%d"/>' % (p, j, p, j, p, j))

    def mjcf_parts(self):
        """kinematic build: the segments are the mocap bodies -> (assets, bodies, sensors) XML strings."""
        p = self.prefix; assets, bodies, sensors = [], [], []
        for j in range(16):
            loc = self.v[self.vseg == j] - self.J[j]
            assets.append('<mesh name="%s_col%d" vertex="%s"/>' % (p, j, fmt(loc)))
            g = ['<geom name="%s_col%d" type="mesh" mesh="%s_col%d" contype="2" conaffinity="1" friction="%g 0.01 0.001" solref="%s" rgba="0.55 0.6 0.68 1" group="1"/>' % (
                p, j, p, j, self.friction, self.solref)]
            tp = self._taxel_parts(j)
            if tp:
                assets.append(tp[0]); g.append(tp[1]); sensors.append(tp[2])
            bodies.append('<body name="%s_%s" mocap="true">%s</body>' % (p, SEG_NAMES[j], "".join(g)))
        self.dynamic = False
        return "".join(assets), "".join(bodies), "".join(sensors)

    def mjcf(self, K=3000.0, tc_weld=0.005, dmax=0.95, min_mass=0.03, obj_mass=0.5, skin_rgba="0.93 0.79 0.69 1"):
        """dynamic build -> dict(asset, body, sensor, equality, skin) XML strings.
        Segment j: free body 'h_<seg>' (gravcomp) + mocap target 'h_<seg>_t' + weld (time constant tc_weld).
        Segment mass = max(hull mass at water density, min_mass): MuJoCo's weld stiffness also scales with mass, and the light
        fingertips (2-3 g) would otherwise be too compliant to squeeze anything (gravity is compensated, so the mass only adds inertia).
        Contact stiffness: MuJoCo's soft contacts are mass-scaled (force/depth = m_eff d/(1-d) / (dmax tc)^2), so each segment gets
        its own solref time constant for about K N/m (impedance fixed at d = dmax); the hand's geoms have priority, so their
        friction/solref/solimp are used in every hand contact."""
        from scipy.spatial import ConvexHull
        p = self.prefix; A, B, S, E = [], [], [], []
        self.seg_mass, self.seg_tc = {}, {}
        for j in range(16):
            nm = "%s_%s" % (p, SEG_NAMES[j]); loc = self.v[self.vseg == j] - self.J[j]
            mj = max(1000.0 * ConvexHull(loc).volume, min_mass); meff = 1.0 / (1.0 / (0.7 * mj) + 1.0 / obj_mass)
            tc = max(0.0042, float(np.sqrt(meff * dmax / (1 - dmax) / (dmax ** 2 * K))))
            self.seg_mass[j], self.seg_tc[j] = mj, tc
            A.append('<mesh name="%s_col%d" vertex="%s"/>' % (p, j, fmt(loc)))
            g = ['<geom name="%s_col%d" type="mesh" mesh="%s_col%d" contype="2" conaffinity="1" priority="1" condim="3" friction="%g 0.01 0.001" mass="%.5f" '
                 'solref="%.4f 1" solimp="%g %g 0.001" rgba="0.55 0.6 0.68 1" group="1"/>' % (p, j, p, j, self.friction, mj, tc, dmax, dmax)]
            tp = self._taxel_parts(j)
            if tp:
                A.append(tp[0]); g.append(tp[1]); S.append(tp[2])
            B.append('<body name="%s_t" mocap="true"/>' % nm)
            B.append('<body name="%s" gravcomp="1"><freejoint name="%s_j"/>%s</body>' % (nm, nm, "".join(g)))
            E.append('<weld name="%s_w" body1="%s" body2="%s_t" relpose="0 0 0 1 0 0 0" solref="%g 1" solimp="%g %g 0.001"/>' % (nm, nm, nm, tc_weld, dmax, dmax))
        bones = []
        for j in range(16):
            w = self.W[:, j]; ids = np.flatnonzero(w > 1e-4)
            bones.append('<bone body="%s_%s" bindpos="%s" bindquat="1 0 0 0" vertid="%s" vertweight="%s"/>' % (
                p, SEG_NAMES[j], fmt(self.J[j]), " ".join(map(str, ids)), " ".join("%.5f" % x for x in w[ids])))
        skin = '<skin name="%s_skin" rgba="%s" vertex="%s" face="%s">%s</skin>' % (p, skin_rgba, fmt(self.v), " ".join(map(str, self.f.ravel())), "".join(bones))
        self.dynamic = True
        return dict(asset="".join(A), body="".join(B), sensor="".join(S), equality="".join(E), skin=skin)

    def bind(self, m):
        """after compiling: body / mocap / joint / geom / sensor ids."""
        p = self.prefix; nm = lambda j: "%s_%s" % (p, SEG_NAMES[j])
        bid = lambda s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s)
        self.body = np.array([bid(nm(j)) for j in range(16)])
        self.mocap = np.array([m.body_mocapid[bid(nm(j) + "_t" if self.dynamic else nm(j))] for j in range(16)])
        if self.dynamic:
            jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm(j) + "_j") for j in range(16)]
            self.qadr = np.array([m.jnt_qposadr[i] for i in jid]); self.vadr = np.array([m.jnt_dofadr[i] for i in jid])
        self.gcol = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "%s_col%d" % (p, j)) for j in range(16)])
        self.g2s = {int(g): j for j, g in enumerate(self.gcol)}
        self.sens = {}
        for j, ti in self.seg_tax.items():
            if len(ti):
                sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "%s_tac%d" % (p, j))
                gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "%s_tax%d" % (p, j))
                mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "%s_tax%d" % (p, j))
                self.sens[j] = (m.sensor_adr[sid], m.sensor_dim[sid], gid, mid)

    def set_pose(self, m, d, pose, T=np.eye(4)):
        """move the mocap bodies (kinematic build) or the mocap targets (dynamic build)."""
        R, t = self.fk(pose, T)
        for j in range(16):
            d.mocap_pos[self.mocap[j]] = t[j]; d.mocap_quat[self.mocap[j]] = mat2quat(R[j])

    def teleport(self, m, d, pose, T=np.eye(4)):
        """set targets and (dynamic build) the segment bodies themselves, at rest."""
        R, t = self.fk(pose, T)
        for j in range(16):
            q = mat2quat(R[j]); d.mocap_pos[self.mocap[j]] = t[j]; d.mocap_quat[self.mocap[j]] = q
            if self.dynamic:
                a, v = self.qadr[j], self.vadr[j]; d.qpos[a:a + 3] = t[j]; d.qpos[a + 3:a + 7] = q; d.qvel[v:v + 6] = 0

    def read(self, m, d):
        """-> depth (534,), slip (534, 2) in the order of self.slots."""
        depth = np.zeros(len(self.slots)); slip = np.zeros((len(self.slots), 2))
        for j, (adr, dim, gid, mid) in self.sens.items():
            nv = m.mesh_vertnum[mid]; x = d.sensordata[adr:adr + dim].reshape(dim // nv, nv)   # channel-major
            ti = self.seg_tax[j]; depth[ti] = x[0, :len(ti)]
            if x.shape[0] >= 3:
                slip[ti, 0] = x[1, :len(ti)]; slip[ti, 1] = x[2, :len(ti)]
        return depth, slip

    def sensed_points(self, m, d):
        """world positions of the sensor's mesh vertices (what MuJoCo samples), in self.slots order."""
        out = np.zeros((len(self.slots), 3))
        for j, (adr, dim, gid, mid) in self.sens.items():
            a, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]; V = m.mesh_vert[a:a + nv]
            Xw = V @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]; ti = self.seg_tax[j]; out[ti] = Xw[:len(ti)]
        return out

    def normals_world(self, d):
        R = d.xmat[self.body[self.tax_seg]].reshape(-1, 3, 3)
        return np.einsum("sab,sb->sa", R, self.tax_n)

    def seg_forces(self, m, d):
        """total normal contact force (N) on each hand segment, and on the hand from each other geom {geom id: N}."""
        F = np.zeros(16); by = {}; f6 = np.zeros(6)
        for i in range(d.ncon):
            g1, g2 = int(d.contact.geom[i][0]), int(d.contact.geom[i][1])
            j = self.g2s.get(g1, self.g2s.get(g2)); o = g2 if g1 in self.g2s else g1
            if j is None:
                continue
            mujoco.mj_contactForce(m, d, i, f6); F[j] += f6[0]; by[o] = by.get(o, 0.0) + f6[0]
        return F, by


def selftest():
    H = ManoHand(); A, B, S = H.mjcf_parts()
    xml = ('<mujoco><option timestep="0.002"/><asset>%s</asset><worldbody><light pos="0 0 1"/>%s'
           '<body name="ball" pos="0 0 0"><freejoint/><geom name="ball" type="sphere" size="0.02" contype="1" conaffinity="3" mass="0.1"/></body>'
           '</worldbody><sensor>%s</sensor></mujoco>') % (A, B, S)
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m); H.bind(m)
    pose = np.zeros((16, 3)); T = np.eye(4); T[:3, 3] = (0, 0, 0.3)
    H.set_pose(m, d, pose, T); mujoco.mj_forward(m, d)
    err = np.linalg.norm(H.sensed_points(m, d) - H.taxels_world(pose, T), axis=1)
    print("taxels %d on %d segments; sensed vs intended position error: max %.3f mm" % (len(H.slots), len(H.sens), 1e3 * err.max()))
    tw = H.taxels_world(pose, T); R, t = H.fk(pose, T); k = np.flatnonzero(H.tax_seg == 0)[len(np.flatnonzero(H.tax_seg == 0)) // 2]
    nw = R[0] @ H.tax_n[k]
    d.qpos[:3] = tw[k] + 0.015 * nw; d.qpos[3:7] = (1, 0, 0, 0); mujoco.mj_forward(m, d)
    depth, slip = H.read(m, d)
    near = np.linalg.norm(tw - d.qpos[:3], axis=1) < 0.02
    print("ball pressed 5 mm into the palm: ncon %d, taxels with depth > 0: %d (inside the ball: %d), max depth %.2f mm (expected ~5)" % (
        d.ncon, int((depth > 0).sum()), int(near.sum()), 1e3 * depth.max()))
    exp = np.clip(0.02 - np.linalg.norm(tw - d.qpos[:3], axis=1), 0, None)
    print("depth vs geometric overlap: max abs diff %.3f mm" % (1e3 * np.abs(depth - exp).max()))
    return H, m, d


if __name__ == "__main__":
    selftest()
