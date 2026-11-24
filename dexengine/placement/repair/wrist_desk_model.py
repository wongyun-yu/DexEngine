"""The arm-side and hand-side geometry of the wrist housing <-> desk repair (wrist_desk.Side solves with them).

Arm: the arm_6 pose from the EE target, arm_5 over joint-6 angles, and the desk penetration of a link's colliders by
the judge's own ruler. Hand: one side's hand in the tree model (root and finger joints, tips, hand bodies, and the hand
<-> object shapes). Their two constants are part of wrist_desk.CONSTS (wrist_desk imports them).
"""

from __future__ import annotations

import numpy as np

from dexengine.placement.repair import wrist_shift as WS

Q6_SAMPLES = 72  # joint-6 angles sampled for the arm_5 best case
OBJ_COLLISION_GROUP = 3  # the scene writers' object collision geoms (scene builder · base_adapter: convex parts,
# or the visual mesh standing in when an object has none) — the hand-object shapes


def _rot(axis, th):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Arm:
    """The arm side's wrist geometry in the scene frame: arm_6 pose from the EE target, arm_5 over joint-6 angles,
    desk penetration of a link's colliders by the judge's own function (collision._desk_each — ``pen``, the only
    exact depth here) on the deciding ruler (arm_proxy.mesh_ruler — the original mesh, never the search proxy).

    arm_5 and arm_6 must be one hull each (no capsule) and the scene must have a desk; anything else dies — the bound
    below is written and tested for exactly that. Bound (pruning only, never a result): ztop − the lowest hull vertex
    inside the footprint (``_lb`` on ``_verts``) is clip_min_z's first term, every later term only lowers z, and float
    subtraction is monotone — so it is ≤ ``pen`` bit for bit as long as ``_verts`` forms the ruler's own vertex bytes
    (einsum's row results do not depend on the batch). ``_verts`` restates the two-line vertex formation of
    arm_surface._hull_desk_pen; moving it into arm_surface would change collision.ruler_id (the IK pool clearance
    sidecar identity)."""

    BLOCK = 8  # poses per arm_5 sweep block (speed only — no result depends on it)

    def __init__(self, arm: str, hand: str, side: str, desk_xy):
        import xml.etree.ElementTree as ET  # noqa: PLC0415

        # measured on the deciding (mesh) ruler
        from dexengine.placement.collision import arm_proxy as AP  # noqa: PLC0415
        from dexengine.placement.collision import core as SCL  # noqa: PLC0415 — the judge's ruler and its tolerance

        if desk_xy is None:
            raise ValueError(
                f"{arm}/{side}: no desk box — 2e counts penetration only inside the box the caller gives "
                "for this tree (desk_of; never replaced by the canonical box)"
            )
        self.SCL, self.desk_xy, self.AP = SCL, desk_xy, AP
        with AP.mesh_ruler():
            c = self.ctx = SCL.build_arm_ctx(arm, hand, side)
        self.k6 = len(c["joints"]) - 1
        self.k5 = self.k6 - 1
        self.ee_inv = np.linalg.inv(c["ee_local"])
        self.A6inv = np.linalg.inv(c["A"][self.k6])
        self.ax6 = c["axes"][self.k6]
        lim = {j.get("name"): j.find("limit") for j in ET.parse(c["urdf"]).getroot().findall("joint")}
        L = lim[c["joints"][self.k6]]
        self.q6 = np.linspace(float(L.get("lower")), float(L.get("upper")), Q6_SAMPLES)
        self.ztop = float(c["ztop"])
        self.thr = SCL.DESK_SURF_PEN_MM / 1000.0
        self.sub = {
            k: {
                "surf": {
                    "caps": [x for x in c["surf"]["caps"] if x["k"] == k],
                    "hulls": [x for x in c["surf"]["hulls"] if x["k"] == k],
                },
                "ztop": c["ztop"],
                "base": np.zeros(3),
            }
            for k in (self.k5, self.k6)
        }
        for k in (self.k5, self.k6):
            s = self.sub[k]["surf"]
            if len(s["hulls"]) != 1 or s["caps"]:
                raise ValueError(
                    f"{arm}/{side}: arm link {k} colliders = {len(s['hulls'])} hulls · {len(s['caps'])} "
                    "capsules — the 2e ruler bound handles exactly one hull per link"
                )
        self.reach5 = np.linalg.norm(
            self.sub[self.k5]["surf"]["hulls"][0]["v"] @ self.A6inv[:3, :3].T + self.A6inv[:3, 3], axis=1
        ).max()
        R6 = np.stack([_rot(self.ax6, -q) for q in self.q6])
        M = np.zeros((len(self.q6), 4, 4))
        M[:, :3, :3] = R6
        M[:, 3, 3] = 1
        self.M6 = M @ self.A6inv  # arm_6 pose → arm_5 pose at each joint-6 sample

    def _ts(self, k: int, Tk: np.ndarray) -> np.ndarray:
        """Link k's poses in the ruler's link-pose array (the links before k at identity) — the one layout."""
        Ts = np.broadcast_to(np.eye(4), (len(Tk), k + 1, 4, 4)).copy()
        Ts[:, k] = Tk
        return Ts

    def pen(self, k: int, Tk: np.ndarray) -> np.ndarray:
        with self.AP.mesh_ruler():
            return self.SCL._desk_each(self.sub[k], self._ts(k, Tk), 0.0, self.desk_xy)

    def T6(self, ee_pos, ee_R) -> np.ndarray:
        E = np.zeros(ee_pos.shape[:-1] + (4, 4))
        E[..., :3, :3] = ee_R
        E[..., :3, 3] = ee_pos
        E[..., 3, 3] = 1
        return E @ self.ee_inv

    def _verts(self, k: int, Tk: np.ndarray) -> np.ndarray:
        """Link k's hull vertices in the scene (P, nv, 3) as arm_surface._hull_desk_pen forms them from what
        collision._desk_each hands it (the link's rotation · origin + base) — the bound's input, nothing else."""
        T = self._ts(k, Tk)[:, k]
        t = T[:, :3, 3] + (self.sub[k]["base"] + np.array([0.0, 0.0, 0.0]))
        return np.einsum("tij,vj->tvi", T[:, :3, :3], self.sub[k]["surf"]["hulls"][0]["v"]) + t[:, None, :]

    def _lb(self, k: int, V) -> np.ndarray:
        """ztop − the lowest vertex inside the footprint (−1e9 where none) ≤ ``pen`` (see the class note)."""
        (x0, x1), (y0, y1) = self.desk_xy
        inb = (V[..., 0] >= x0) & (V[..., 0] <= x1) & (V[..., 1] >= y0) & (V[..., 1] <= y1)
        z = np.amin(np.where(inb, V[..., 2], 1e9), -1)
        return np.where(z < 1e8, self.sub[k]["ztop"] - z, -1e9)

    def pen5_best(self, T6: np.ndarray, lb6=None, skip=None) -> np.ndarray:
        """Least arm_5 penetration over joint 6 per pose (−1e9 where arm_5 cannot reach the desk). ``lb6``/``skip``:
        see need_depth (a pose whose bound max(lb6, least arm_5 bound) ``skip`` holds for is +inf)."""
        out = np.full(len(T6), -1e9)
        near = np.nonzero(T6[:, 2, 3] - self.reach5 < self.ztop + self.thr)[0]
        for i in range(0, len(near), self.BLOCK):
            idx = near[i : i + self.BLOCK]
            T5 = np.einsum("pij,qjk->pqik", T6[idx], self.M6).reshape(-1, 4, 4)
            out[idx] = self._sweep(T5, len(idx), None if skip is None else lb6[idx], skip)
        return out

    def _sweep(self, T5, P: int, lb6, skip) -> np.ndarray:
        """Least ruler depth of arm_5 over the joint-6 samples of P poses (T5 = their P·Q arm_5 poses, pose-major):
        measured at the angle whose lowest vertex is highest, then only at the angles whose bound is not ≥ that value —
        any other angle's depth (≥ its bound) cannot be the least. With ``skip``: +inf where skip(bound) holds."""
        Q, k = len(self.q6), self.k5
        V = self._verts(k, T5)
        lb = self._lb(k, V).reshape(P, Q)
        best = np.full(P, np.inf)
        live = np.arange(P) if skip is None else np.nonzero(~skip(np.maximum(lb6, lb.min(1))))[0]
        if len(live):
            q0 = np.argmax(V[..., 2].min(-1).reshape(P, Q)[live], 1)
            best[live] = self.pen(k, T5[live * Q + q0])
            ev = ~(lb[live] >= best[live][:, None])
            ev[np.arange(len(live)), q0] = False
            pi, qi = np.nonzero(ev)
            if len(pi):
                np.minimum.at(best, live[pi], self.pen(k, T5[live[pi] * Q + qi]))
        return best

    def need_depth(self, T6: np.ndarray, skip=None) -> np.ndarray:
        """max(arm_6, arm_5 at its best joint-6 angle) per pose (m). ``skip`` (Side.choose): a predicate, monotone in
        the depth, on the bound of it — a pose it holds for is +inf, not measured (its depth would hold it too).
        An aerial scene dies: the ruler is −1e9 there (no desk) while the bound is finite, so skip would be unsound —
        the 2e step writes NA for those trees and never builds a Side."""
        from dexengine.placement.coords import core as VC  # noqa: PLC0415

        if VC.scene_mode() == "aerial":
            raise ValueError(
                "need_depth on an aerial scene — the ruler is −1e9 there (no desk) while the bound is "
                "finite, so skip would be unsound · 2e leaves aerial trees NA"
            )
        V = self._verts(self.k6, T6)
        lb6 = self._lb(self.k6, V)
        live = np.arange(len(T6)) if skip is None else np.nonzero(~skip(lb6))[0]
        out = np.full(len(T6), np.inf)
        if len(live):
            out[live] = np.maximum(self.pen(self.k6, T6[live]), self.pen5_best(T6[live], lb6[live], skip))
        return out


class Hand:
    """One side's hand in the tree model: root joints, finger joints (clamped to the reference finger range), tips,
    hand bodies, and the hand ↔ object shapes for the penetration check."""

    def __init__(self, m, lay, ti, side: str, hand: str):
        import mujoco as mj  # noqa: PLC0415

        from dexengine.placement.fk import body_positions as BP  # noqa: PLC0415 — the FK owner's site names

        self.m, self.side = m, side
        off, H = int(lay.hands[side]), int(lay.hand_qpos_dim)
        j_of = {int(m.jnt_qposadr[j]): j for j in range(m.njnt)}
        root = [j_of[c] for c in range(off, off + 6)]
        fing = [j_of[c] for c in range(off + 6, off + H)]
        kinds, bodies = sorted(int(m.jnt_type[j]) for j in root), {int(m.jnt_bodyid[j]) for j in root}
        if kinds != [2, 2, 2, 3, 3, 3] or len(bodies) != 1:
            raise ValueError(f"{side}: the 6 hand-root columns are not 3 slides + 3 hinges on one body")
        if any(int(m.jnt_dofadr[j]) != int(m.jnt_qposadr[j]) for j in root + fing):
            raise ValueError(f"{side}: qpos and qvel addresses of the hand columns differ — a quaternion precedes them")
        self.cols = (off, H)
        self.body = int(m.jnt_bodyid[root[0]])
        self.rq = np.array([m.jnt_qposadr[j] for j in root])
        self.rd = np.array([m.jnt_dofadr[j] for j in root])
        self.fq = np.array([m.jnt_qposadr[j] for j in fing])
        self.fd = np.array([m.jnt_dofadr[j] for j in fing])
        names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        rng = WS.scene_finger_ranges(hand, m, ti["embodiment_type"])
        self.lo = np.array([rng[names[j]][0] for j in fing])
        self.hi = np.array([rng[names[j]][1] for j in fing])
        tips = [n.replace("right_", f"{side}_", 1) for n in BP.RIGHT_SITE_NAMES if n.endswith("_tip")]
        self.tip_names = [n.split("_", 1)[1][:-4] for n in tips]
        self.sids = np.array([mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, n) for n in tips])
        if (self.sids < 0).any():
            raise ValueError(f"{side}: fingertip sites {tips} are not in the scene")
        under = [b for b in range(m.nbody) if self._root_of(b) == self.body]
        self.bodies = np.array(under)
        objs = {str(o["name"]) for o in ti.get("objects") or []}
        self.hand_g = sorted(
            {
                (int(m.geom_bodyid[g]), int(m.geom_dataid[g])): g
                for g in range(m.ngeom)
                if int(m.geom_bodyid[g]) in set(under) and int(m.geom_type[g]) == 7
            }.values()
        )
        roots = {n: mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, n) for n in objs}
        coll = [g for g in range(m.ngeom) if int(m.geom_type[g]) == 7 and int(m.geom_group[g]) == OBJ_COLLISION_GROUP]
        own = {
            n: [g for g in coll if r >= 0 and self._has_ancestor(int(m.geom_bodyid[g]), r)] for n, r in roots.items()
        }
        bare = sorted(n for n, gs in own.items() if not gs)
        if bare:
            raise ValueError(
                f"objects {bare} have no collision parts (mesh, group {OBJ_COLLISION_GROUP}) in the scene "
                "— hand-object penetration cannot be measured"
            )
        self.obj_g = sorted({g for gs in own.values() for g in gs})

    def _root_of(self, b: int) -> int:
        while int(self.m.body_parentid[b]) > 0 and b != self.body:
            b = int(self.m.body_parentid[b])
        return b

    def _has_ancestor(self, b: int, a: int) -> bool:
        while b > 0:
            if b == a:
                return True
            b = int(self.m.body_parentid[b])
        return False

    def obj_depth(self, d) -> float:
        """Deepest hand ↔ object (collision parts) penetration of the pose in ``d`` (m, ≥ 0) — MuJoCo signed distance
        on the pairs whose bounding spheres overlap."""
        import mujoco as mj  # noqa: PLC0415

        if not self.obj_g:
            return 0.0
        hg, og = np.asarray(self.hand_g), np.asarray(self.obj_g)
        cd = (
            np.linalg.norm(d.geom_xpos[hg][:, None] - d.geom_xpos[og][None], axis=2)
            - self.m.geom_rbound[hg][:, None]
            - self.m.geom_rbound[og][None]
        )
        deep = 0.0
        for i, j in zip(*np.nonzero(cd < 0.0), strict=True):
            deep = max(deep, -float(mj.mj_geomDistance(self.m, d, int(hg[i]), int(og[j]), 0.01, None)))
        return deep
