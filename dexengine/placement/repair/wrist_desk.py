#!/usr/bin/env python3
"""Wrist housing <-> desk reference repair — the single owner of the rule (stage 1 step 2e: after the contact sidecar
2d, before the case step 3).

The robot hand is mounted on the arm's last link (arm_6), which sits where the human wrist was; with a robot hand
bigger than the human hand that housing can lie inside the desk while the palm is above it. arm_6 is rigid with the
hand and the desk is fixed in the scene, so no placement can fix it and the strict judge drops the clip — the reference
itself must change. arm_5 (behind joint 6) follows the arm IK, so here it is held to a necessary condition: its best
joint-6 angle must clear the desk (the judge measures the angle it actually gets).

Rule (constants below; thresholds that belong to the judge are read from it):
  need      a frame where the arm_6 colliders (sim_collider.bodies — the judge's surface ruler, arm_surface through
            collision._desk_each) lie deeper than collision.DESK_SURF_PEN_MM below the desk top, or arm_5 does at its
            best joint-6 angle.
  defect    in a need frame the palm body (the IK target's own body) is below the desk top -> UNRESOLVED (a data
            defect, not a body gap — it is never "repaired").
  candidate rotation about two horizontal axes through the pivot (centre of the fingertips in MANO contact — the 2d
            contact_all sidecar, any object; no contact -> all five tips), total angle <= WRIST_TILT_MAX_DEG on a
            TILT_STEP_DEG grid, then the least lift <= LIFT_MAX_MM that puts arm_6 (and arm_5 at its best angle)
            WRIST_DESK_CLEAR_MM above the top. A candidate that puts a hand body or tip newly below the top is dropped.
  selection least total angle first, ties by lift (LIFT_WEIGHT_DEG_PER_MM = 0 is the only value the code implements;
            any other dies at import); the first whose finger re-IK keeps every contact tip within TIP_RESID_MM and
            whose hand-object penetration grows <= OBJ_PEN_INC_MM wins.
  time      the chosen transforms of the need frames -> wrist_shift.smooth_ramp (3-point + C¹ ramp of RAMP_FRAMES);
            the lift the smoothed transforms still lack is added as one profile (wrist_shift.ramp_dilate — each
            frame's shortfall with the same ramp around it, never a one-frame step); fingers re-solved frame by frame
            warm-started from the previous frame; every touched frame (contact or not) is verified again on the final
            state, and the added arm_6 motion between consecutive frames (boundaries included) stays under
            WRIST_JUMP_MAX_MPS. Any failure -> UNRESOLVED and the tree stays as it was.
  speed     the same bytes, less work (Arm · Side.choose): the grid is measured lazily in angle order, the arm_5 sweep
            over joint 6 measures only the angles a rigorous lower bound cannot rule out, and a candidate whose bound
            already needs more lift than LIFT_MAX_MM is never measured. One hull per arm_5/arm_6 link on a desk scene
            — anything else dies.
The object never moves. Writes (only when something changed): qpos (+ qpos_rollout, qvel by base_adapter's forward
rule, the ikrollout qpos) with the one backup/undo of wrist_shift (tag pre_ws), body_positions by the same FK function
base_adapter uses, task_info rules.WRIST_SLACK — the stage driver is wrist_desk_stage.py. Marker:
``[WRIST_SLACK] on clips_changed=k/N``. The UNRESOLVED reason strings are ledger and task_info data: keep their bytes.
"""

from __future__ import annotations

import numpy as np

from dexengine.placement.repair import wrist_shift as WS
from dexengine.placement.repair.wrist_desk_model import Arm, Hand, _rot  # noqa: F401 — re-exported (WDS.Arm · WDS.Hand)

WRIST_TILT_MAX_DEG = 20.0  # total rotation angle of the hand (not per axis)
TILT_STEP_DEG = 2.5  # rotation grid (two horizontal axes)
LIFT_MAX_MM = 30.0  # largest lift after the rotation
WRIST_DESK_CLEAR_MM = 2.0  # arm_6 housing and arm_5 (best joint-6 angle — stage 1 has no arm IK) above the desk top
# after the repair (the arm-surface tolerance)
TIP_RESID_MM = 5.0  # contact fingertip residual after the finger re-IK
OBJ_PEN_INC_MM = 5.0  # hand-object penetration may grow at most this much (every touched frame)
LIFT_WEIGHT_DEG_PER_MM = 0.0  # least rotation first, lift only breaks ties — the one value Side.choose implements
# (any other dies below)
WRIST_JUMP_MAX_MPS = 0.5  # added arm_6 speed between frames (x dt per frame): the C¹ ramp of a full lift
# (LIFT_MAX_MM over RAMP_FRAMES — step <= 0.25 · 30 mm = 7.5 mm at 50 Hz) passes,
# per-frame steps (measured p90 11.7 · max 30.7 mm/frame) do not
RAMP_FRAMES = 5  # C¹ ramp width outside the repaired episodes
# the model's two constants, imported here so CONSTS keeps its key order (OBJ_COLLISION_GROUP is in it, Q6_SAMPLES not)
from dexengine.placement.repair.wrist_desk_model import OBJ_COLLISION_GROUP, Q6_SAMPLES  # noqa: E402,F401

CODE_VERSION = "WS-3"  # bump when the rule changes (the 2e column then reads stale and the step re-runs)
RULE_KEY = "WRIST_SLACK"  # task_info rules key = rule-table card id
TAG = "pre_ws"  # wrist_shift backup tag of this repair
ROOT_KEYS = ("real_step_new", "real_step_max_deg")  # root_audit's record — the 2e ledger's last columns, never a tree
STATUSES = ("CLEAN", "REPAIRED", "UNRESOLVED", "NA")
#: The rule's numeric constants (part of the 2e input fingerprint, wrist_desk_fresh.input_fp).
CONSTS = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, (int, float)) and k != "Q6_SAMPLES"}
if LIFT_WEIGHT_DEG_PER_MM != 0:  # Side.choose implements least angle with lift breaking ties — nothing else
    raise ValueError("LIFT_WEIGHT_DEG_PER_MM != 0 — Side.choose implements only 0 (least rotation, lift breaks ties)")


def desk_of(tree) -> tuple:
    """The desk as this repair measures it → (footprint ((x0, x1), (y0, y1)) in the scene, tree → scene z offset):
    the one place 2e reads desk geometry (the stage driver assembles every Arm and Side from it). The footprint is
    the tree's own box (coords.desk_bounds_scene_for, no edge margin) and a depth counts as penetration only inside
    it: the ruler (collision._desk_each), its bound (Arm._lb) and the hand tests (Side.inxy) all take this box, so a
    hand going below the top beside the desk is not a penetration. The z offset is coords.scene_z_offset (the desk
    height statement qpos_layout.desk_z_of behind it); the top's height in the scene is the judge ruler's own
    (collision.build_arm_ctx ``ztop``). The box is where the tree sits before placement: the judge moves the clip
    over the desk (dx · dy · yaw), so 'inside the footprint' here and in an adopted placement are two statements."""
    from dexengine.placement.coords import core as VC  # noqa: PLC0415

    return VC.desk_bounds_scene_for(tree, margin=0.0), VC.scene_z_offset(tree)


def tilt_grid(u, cap_deg: float):
    """Rotations about z×u and u (two horizontal axes) on the TILT_STEP_DEG grid with total angle ≤ cap → (R (G,3,3),
    angle deg (G,)). Identity first."""
    ang = np.radians(np.arange(-cap_deg, cap_deg + 1e-9, TILT_STEP_DEG))
    a1 = np.cross([0.0, 0.0, 1.0], u)
    R = np.stack([_rot(a1, x) @ _rot(u, y) for x in ang for y in ang])
    geo = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    keep = geo <= cap_deg + 1e-6
    o = np.argsort(geo[keep], kind="stable")
    return R[keep][o], geo[keep][o]


def _expm(rv):
    th = float(np.linalg.norm(rv))
    return np.eye(3) if th < 1e-12 else _rot(rv / th, th)


class Side:
    """One side's solve on one tree: frame series in the scene frame, need/defect, candidate choice, the smoothed
    final state and its verification. ``cap_deg``/``lift_max_mm`` default to the decision constants (tests vary
    them)."""

    def __init__(
        self,
        m,
        d,
        qpos,
        hand: Hand,
        arm: Arm,
        z_off: float,
        palm_body: int,
        palm_off,
        contact,
        dt: float,
        cap_deg: float = WRIST_TILT_MAX_DEG,
        lift_max_mm: float = LIFT_MAX_MM,
    ):
        import mujoco as mj  # noqa: PLC0415

        self.m, self.d, self.q, self.H, self.A, self.z = m, d, np.asarray(qpos, float), hand, arm, float(z_off)
        self.palm, self.off, self.cap, self.lmax = palm_body, np.asarray(palm_off, float), cap_deg, lift_max_mm / 1000
        self.cm = np.asarray(contact, bool)
        self.jmax = WRIST_JUMP_MAX_MPS * float(dt)
        T = len(self.q)
        self.tips = np.zeros((T, len(hand.sids), 3))
        self.B = np.zeros((T, len(hand.bodies), 3))
        P = np.zeros((T, 3))
        R = np.zeros((T, 3, 3))
        for t in range(T):
            d.qpos[:] = self.q[t]
            mj.mj_kinematics(m, d)
            P[t], R[t] = d.xpos[palm_body], d.xmat[palm_body].reshape(3, 3)
            self.tips[t], self.B[t] = d.site_xpos[hand.sids], d.xpos[hand.bodies]
        self.T6 = arm.T6(*WS.ee_target(P, R, self.off, self.z))
        self.Ps, self.tips_s, self.B_s = self.sc(P), self.sc(self.tips), self.sc(self.B)
        self.depth = arm.need_depth(self.T6)
        self.need = self.depth > arm.thr
        self.piv = np.stack(
            [self.tips_s[t][self.cm[t]].mean(0) if self.cm[t].any() else self.tips_s[t].mean(0) for t in range(T)]
        )
        self._d0: dict = {}
        self.root: dict = {}  # root_audit of the rows this side wrote (verify_rows) — ROOT_KEYS
        self._prev = None  # finish: (frame, row) just written — pose continues its root change

    def sc(self, X):
        return X @ WS.RZ_NEG90.T + np.array([0.0, 0.0, self.z])

    def inxy(self, X):
        (x0, x1), (y0, y1) = self.A.desk_xy
        return (X[..., 0] >= x0) & (X[..., 0] <= x1) & (X[..., 1] >= y0) & (X[..., 1] <= y1)

    def below(self, B, tips, B0, tips0):
        """Hand bodies / tips newly below the desk top (bodies: top · tips: top − tolerance), per candidate."""
        zt, thr = self.A.ztop, self.A.thr
        nb = (self.inxy(B) & (B[..., 2] < zt) & ~(B0[..., 2] < zt)).any(-1)
        return nb | (self.inxy(tips) & (tips[..., 2] < zt - thr) & ~(tips0[..., 2] < zt - thr)).any(-1)

    def defect(self) -> str:
        """The palm itself below the desk top in a need frame — a tracking/desk defect, not a body gap. Finger links
        below the top are not a defect here (the judge does not filter hand ↔ desk; the f0 repair takes deep starts)."""
        if (self.need & self.inxy(self.Ps) & (self.Ps[:, 2] < self.A.ztop)).any():
            return "suspected defect: palm below the desk top"  # ledger / task_info reason (data — keep the bytes)
        return ""

    def pose(self, t: int, R, c, warm=None):
        """Frame t moved by the scene transform x → R·x + c, fingers re-solved onto its contact tips →
        (qpos row, contact residual m, tips scene). While finish writes the frames in order, the root coordinates
        continue the frame before (its root change carried to this frame's original — WS.move_root ``ref``)."""
        Rm = WS.RZ_NEG90.T @ R @ WS.RZ_NEG90
        cm_ = WS.RZ_NEG90.T @ ((R - np.eye(3)) @ np.array([0.0, 0.0, self.z]) + c)
        H, ref = self.H, None
        if self._prev is not None and self._prev[0] == t - 1:
            ref = self.q[t][H.rq] + (self._prev[1][H.rq] - self.q[t - 1][H.rq])
        row = WS.move_root(self.m, self.d, self.q[t], H.body, H.rq, H.rd, Rm, cm_, ref=ref)
        if warm is not None:
            row[H.fq] = np.clip(row[H.fq] + warm, H.lo, H.hi)
        which = np.nonzero(self.cm[t])[0]
        row, res = WS.finger_ik(self.m, self.d, row, H.fq, H.fd, H.lo, H.hi, H.sids[which], self.tips[t][which])
        return row, res, self.sc(self.d.site_xpos[H.sids].copy())

    def obj_inc(self, t: int, row) -> float:
        import mujoco as mj  # noqa: PLC0415

        if t not in self._d0:
            self.d.qpos[:] = self.q[t]
            mj.mj_kinematics(self.m, self.d)
            self._d0[t] = self.H.obj_depth(self.d)
        self.d.qpos[:] = row
        mj.mj_kinematics(self.m, self.d)
        return self.H.obj_depth(self.d) - self._d0[t]

    BATCH = 16  # grid candidates measured per step after the identity (speed only — no result depends on it)

    def _over(self, depth) -> np.ndarray:
        """The lift a depth needs exceeds the cap — the ok test's lift half, monotone in the depth (choose's skip)."""
        return np.maximum(0.0, depth + WRIST_DESK_CLEAR_MM / 1000) > self.lmax

    def choose(self, t: int):
        """Least-angle feasible candidate of need frame t → (R, c) or None: the ok candidates (lift ≤ cap, nothing newly
        below the top) are tried by angle, ties by lift, then grid order, and the first whose fingers re-solve wins.
        The grid is measured lazily: whole equal-angle groups in ascending angle (tilt_grid's order — the identity alone
        first, then about BATCH at a time), each measured batch tried before the next is measured — every later
        candidate has a larger angle, so the tries are exactly those of measuring the whole grid first. A candidate
        whose depth bound already needs more lift than the cap is not measured (Arm.need_depth skip) — it could never
        be ok."""
        p, T6 = self.piv[t], self.T6[t]
        u = T6[:3, 3] - p
        u[2] = 0.0
        u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0.0, 0.0])
        Rg, geo = tilt_grid(u, self.cap)
        Tc = np.repeat(T6[None], len(Rg), 0)
        Tc[:, :3, :3] = Rg @ T6[:3, :3]
        Tc[:, :3, 3] = (Rg @ (T6[:3, 3] - p)) + p
        n, a = len(geo), 0
        ends = [int(e) for e in np.nonzero(np.diff(geo))[0] + 1] + [n]
        while a < n:
            b = next(e for e in ends if e > a)
            while a and b < n and b - a < self.BATCH:
                b = next(e for e in ends if e > b)
            dz = np.maximum(0.0, self.A.need_depth(Tc[a:b], skip=self._over) + WRIST_DESK_CLEAR_MM / 1000)
            m = np.nonzero(dz <= self.lmax)[0]  # a skipped candidate is +inf here
            R, up = Rg[a:b][m], dz[m][:, None, None] * np.array([0.0, 0.0, 1.0])
            Bm, Tm = (np.einsum("gij,kj->gki", R, X - p) + p + up for X in (self.B_s[t], self.tips_s[t]))
            m = m[~self.below(Bm, Tm, self.B_s[t], self.tips_s[t])]
            for g in m[np.lexsort((dz[m], geo[a:b][m]))]:
                c = p - Rg[a + g] @ p + np.array([0.0, 0.0, dz[g]])
                row, res, _ = self.pose(t, Rg[a + g], c)
                if res <= TIP_RESID_MM / 1000 and self.obj_inc(t, row) <= OBJ_PEN_INC_MM / 1000:
                    return Rg[a + g], c
            a = b
        return None

    def solve(self):
        """→ (status, reason, rows {t: qpos row} | None, stats)."""
        if not self.need.any():
            return "CLEAN", "", None, {}
        if not self.H.obj_g and self.cm.any():
            return "UNRESOLVED", "no object collision piece to measure hand–object penetration", None, {}
        why = self.defect()
        if why:
            return "UNRESOLVED", why, None, {}
        T = len(self.q)
        rv = np.zeros((T, 3))
        cc = np.zeros((T, 3))
        own = {}
        for t in np.nonzero(self.need)[0]:
            pick = self.choose(int(t))
            if pick is None:
                return (
                    "UNRESOLVED",
                    f"frame {t}: no solution within {self.cap:g}° · lift {self.lmax * 1000:g} mm",
                    None,
                    {},
                )
            own[int(t)] = pick
            rv[t], cc[t] = WS._rotvec(pick[0]), pick[1]
        w = WS.smooth_ramp([rv, cc], self.need, RAMP_FRAMES)
        extra = WS.ramp_dilate(self.lift_short(np.nonzero(w > 0)[0], rv, cc), RAMP_FRAMES)
        cc[:, 2] += extra
        return self.finish(np.nonzero((w > 0) | (extra > 0))[0], rv, cc, own)

    def lift_short(self, idx, rv, cc) -> np.ndarray:
        """(T,) the lift each touched frame's smoothed transform still lacks (m, 0 when enough): a need frame must
        clear WRIST_DESK_CLEAR_MM, a ramp frame must stay within the judge's tolerance."""
        e = np.zeros(len(self.q))
        if len(idx):
            Rs = np.stack([_expm(rv[t]) for t in idx])
            T6 = self.T6[idx].copy()
            T6[:, :3, :3] = Rs @ T6[:, :3, :3]
            T6[:, :3, 3] = np.einsum("tij,tj->ti", Rs, T6[:, :3, 3]) + cc[idx]
            dep, clr = self.A.need_depth(T6), WRIST_DESK_CLEAR_MM / 1000
            e[idx] = np.where(np.where(self.need[idx], dep > -clr, dep > self.A.thr), dep + clr, 0.0)
        return e

    def finish(self, idx, rv, cc, own):
        """Final state of the touched frames, each verified (contact or not). A frame tries the smoothed transform,
        then — a need frame — its own choice, — a ramp frame — the ramp scaled down, and last the frame's own
        original row (valid there: the frame did not need a repair; no finger warm start reaches it). A need frame
        passing neither → UNRESOLVED; so is an added arm_6 step between consecutive frames (the untouched ones add 0 —
        boundaries count) above ``jmax``."""
        st = {"angle": 0.0, "lift": 0.0, "resid": 0.0, "objinc": 0.0, "nontip": 0.0, "jump": 0.0}
        rows, warm, prev, add = {}, None, -2, np.zeros((len(self.q), 3))
        self._prev = None
        for t in (int(x) for x in idx):
            tries = [(_expm(rv[t]), cc[t].copy())]
            tries += [own[t]] if t in own else [(_expm(a * rv[t]), a * cc[t]) for a in (0.75, 0.5, 0.25)] + [None]
            first = None
            for tr in tries:
                if tr is None:  # ramp frame left exactly as it was
                    why, row, sv = "", self.q[t].copy(), {"h6": self.T6[t][:3, 3].copy()}
                    break
                why, row, sv = self._final(t, tr[0], tr[1].copy(), warm if prev == t - 1 else None)
                first = first or why
                if not why:
                    break
            else:
                return "UNRESOLVED", f"frame {t} final check failed: {first}", None, {}
            add[t] = sv.pop("h6") - self.T6[t][:3, 3]
            st = {k: max(st[k], sv.get(k, 0.0)) for k in st}
            rows[t], warm, prev = row, row[self.H.fq] - self.q[t][self.H.fq], t
            self._prev = (t, row)  # the next frame continues this one's root change
        st["jump"] = float(np.linalg.norm(np.diff(add, axis=0), axis=1).max(initial=0.0))
        if st["jump"] > self.jmax + 1e-9:
            return (
                "UNRESOLVED",
                f"extra wrist move between frames {st['jump'] * 1000:.1f} mm > cap {self.jmax * 1000:g} mm",
                None,
                {},
            )
        self._prev = None
        return "REPAIRED", "", rows, {k: round(v * (1 if k == "angle" else 1000), 2) for k, v in st.items()}

    def _final(self, t: int, R, c, warm):
        """One frame's final pose under x → R·x + c (the lift is the caller's — one profile, never raised here) →
        (failure | "", row, stats incl. the new arm_6 origin ``h6``)."""
        A, clr = self.A, WRIST_DESK_CLEAR_MM / 1000
        T6 = self.T6[t].copy()
        T6[:3, :3] = R @ T6[:3, :3]
        T6[:3, 3] = R @ T6[:3, 3] + c
        dep = float(A.need_depth(T6[None])[0])
        row, res, tips = self.pose(t, R, c, warm)
        lift = float((R @ self.piv[t] + c - self.piv[t])[2])
        B = self.sc(self.d.xpos[self.H.bodies].copy())
        inc, ang = self.obj_inc(t, row), float(np.degrees(np.linalg.norm(WS._rotvec(R))))
        bad = (
            "angle"
            if ang > self.cap + 1e-6
            else "housing clearance"
            if (self.need[t] and dep > -clr + 1e-6) or dep > A.thr
            else "lift"
            if lift > self.lmax + 1e-9
            else "residual"
            if res > TIP_RESID_MM / 1000
            else "object penetration"
            if inc > OBJ_PEN_INC_MM / 1000
            else "newly below the desk top"
            if self.below(B[None], tips[None], self.B_s[t], self.tips_s[t])[0]
            else ""
        )
        nc = ~self.cm[t]
        return (
            bad,
            row,
            {
                "angle": ang,
                "lift": lift,
                "resid": res,
                "objinc": inc,
                "h6": T6[:3, 3].copy(),
                "nontip": float(np.linalg.norm(tips - self.tips_s[t], axis=1)[nc].max(initial=0)),
            },
        )

    def root_audit(self, rows) -> dict:
        """The hand's real turn between neighbour frames in the written rows, measured on them (FK of the root body)
        → {ROOT_KEYS}: the steps turning more than WS.ROOT_TURN_REPORT_DEG where the original did not, and the largest
        of them (°). Counted, never judged: every row's pose is the solver's round trip (WS.ROOT_TOL_M), so such a step
        is what the per-frame tilts ask for (neighbours tilted in different directions), not a solve defect."""
        import mujoco as mj  # noqa: PLC0415

        new = self.q.copy()
        for t, row in rows.items():
            new[t] = row

        def turn(q, t) -> float:
            R = []
            for u in (t, t + 1):
                self.d.qpos[:] = q[u]
                mj.mj_kinematics(self.m, self.d)
                R.append(self.d.xmat[self.H.body].reshape(3, 3).copy())
            return float(np.degrees(np.linalg.norm(WS._rotvec(R[0].T @ R[1]))))

        lim, big = WS.ROOT_TURN_REPORT_DEG, []
        for s in sorted({s for t in rows for s in (t - 1, t) if 0 <= s < len(new) - 1}):
            a = turn(new, s)
            if a > lim and turn(self.q, s) <= lim:
                big.append(a)
        return dict(zip(ROOT_KEYS, (len(big), round(max(big, default=0.0), 2)), strict=True))

    def verify_rows(self, rows) -> float:
        """Independent re-measure on the written state (FK of the new rows → EE → arm_6/arm_5): worst depth (m) over
        the need frames — the caller requires ≤ −WRIST_DESK_CLEAR_MM (plus solver tolerance). The hand's real turn
        between neighbour frames is measured on the same rows and left in ``self.root`` (root_audit — a record)."""
        import mujoco as mj  # noqa: PLC0415

        self.root = self.root_audit(rows)
        ts = [t for t in rows if self.need[t]]
        P = np.zeros((len(ts), 3))
        R = np.zeros((len(ts), 3, 3))
        for i, t in enumerate(ts):
            self.d.qpos[:] = rows[t]
            mj.mj_kinematics(self.m, self.d)
            P[i], R[i] = self.d.xpos[self.palm], self.d.xmat[self.palm].reshape(3, 3)
        return float(self.A.need_depth(self.A.T6(*WS.ee_target(P, R, self.off, self.z))).max(initial=-1e9))


def root_mark(rows) -> str:
    """The run's root audit line from the ledger rows (a side that wrote rows carries ROOT_KEYS)."""
    a = [r for r in rows if ROOT_KEYS[0] in r]
    hit = [r for r in a if r[ROOT_KEYS[0]]]
    return (
        f"[WRIST_ROOT] repaired sides {len(a)} · new neighbour-frame real turns > {WS.ROOT_TURN_REPORT_DEG:g}° "
        f"after the repair {sum(r[ROOT_KEYS[0]] for r in hit)} steps ({len(hit)} sides · max "
        f"{max((r[ROOT_KEYS[1]] for r in hit), default=0.0):g}°) — counted only · what dies is a pose round-trip "
        "residual or a non-converged solve"
    )
