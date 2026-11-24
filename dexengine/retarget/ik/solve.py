"""The per-frame tracking solve.

A gravity-on copy of the scene, with every collision and actuator off, is pulled onto the per-frame targets by the
soft equality constraints of the IK model. Each frame: move the targets, let the constraint solver settle, project
onto the joint limits, apply the prior's slew limit, then record the raw configuration and diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from dexengine.retarget.ik.inputs import FINGERS, Keypoints
from dexengine.retarget.ik.model import LimitSets, Tracked
from dexengine.retarget.ik.posture import PosturePrior

SETTLE_STEPS = 30
INIT_SETTLE_STEPS = 300
RESETTLE_STEPS = max(5, SETTLE_STEPS // 2)
VIOLATION_RESETTLE = 1e-9
MULTISTART_ROTATIONS = ((math.pi / 2, 0.0, 0.0), (0.0, math.pi / 2, 0.0), (0.0, 0.0, math.pi / 2))
MULTISTART_MARGIN = 0.02


@dataclass
class RawSolve:
    """Per raw frame records of the solve."""

    qpos: np.ndarray  # (T, nq)
    contact_pos: np.ndarray  # (T, 10, 3) world positions of the rig's track_hand sites (scene model)
    palm_err: dict[str, np.ndarray]
    tip_err: np.ndarray  # (T, 10)
    viol_pre: np.ndarray  # (T,)
    viol_final: np.ndarray  # (T,)
    multistart_choice: int


class Tracker:
    """Runs the solve of one clip on a compiled IK model."""

    def __init__(
        self,
        ik: mujoco.MjModel,
        scene: mujoco.MjModel,
        tracked: list[Tracked],
        mocap: np.ndarray,
        kp: Keypoints,
        limits: LimitSets,
        prior: PosturePrior | None,
        hinges: list[tuple[int, np.ndarray]],
        sides: tuple[str, ...],
    ):
        self.m, self.scene = ik, scene
        self.tracked, self.mocap = tracked, mocap
        self.kp, self.limits, self.prior, self.hinges, self.sides = kp, limits, prior, hinges, sides
        self.site_ids = [mujoco.mj_name2id(ik, mujoco.mjtObj.mjOBJ_SITE, tr.site) for tr in tracked]
        self.track_hand = [
            mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_SITE, f"track_hand_{s}_{f}_tip")
            for s in ("right", "left")
            for f in FINGERS
        ]

    # --- one physics step and its pieces -------------------------------------------------------------------------
    def set_targets(self, d: mujoco.MjData, t: int) -> None:
        """Move every target mocap body to its entity's frame-t reference pose (quaternion as stored)."""
        ref = self.kp.ref[t]
        for tr, mid in zip(self.tracked, self.mocap, strict=True):
            d.mocap_pos[mid] = ref[tr.col, :3]
            d.mocap_quat[mid] = ref[tr.col, 3:7]

    def impose_hinges(self, d: mujoco.MjData, t: int) -> None:
        """Overwrite each articulated-object hinge angle with its frame-t reference."""
        for adr, angles in self.hinges:
            d.qpos[adr] = angles[t]

    def physics_step(self, d: mujoco.MjData, t: int, q_prev: np.ndarray | None) -> None:
        """Prior forces of frame `t` → one `mj_step` on the IK model → hinge angles back to the frame-t reference."""
        if self.prior is not None:
            self.prior.write(d, t, q_prev)
        mujoco.mj_step(self.m, d)
        self.impose_hinges(d, t)

    def settle(self, d: mujoco.MjData, t: int, steps: int, q_prev: np.ndarray | None) -> None:
        """`steps` physics steps toward the frame-t targets."""
        for _ in range(steps):
            self.physics_step(d, t, q_prev)

    # --- frame-0 start ---------------------------------------------------------------------------------------------
    def start_state(self) -> np.ndarray:
        """Zero configuration with every object at its frame-0 pose and hinges at their frame-0 angles.

        Extra objects sit at their own free-joint address. The primary objects' poses are placed in the trailing
        entries of `qpos` (right then left when both exist, the only one otherwise) — where existing outputs were
        produced from; with a hinge after a free joint this is one entry off.
        """
        q = np.zeros(self.m.nq)
        ref0 = self.kp.ref[0]
        for tr in self.tracked:
            if tr.site.startswith("extra_object_"):
                j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{tr.site}_joint")
                if j >= 0:
                    a = int(self.m.jnt_qposadr[j])
                    q[a : a + 7] = ref0[tr.col]
        objects = [tr for tr in self.tracked if tr.site in ("right_object", "left_object")]
        if len(objects) == 2:
            q[-14:-7] = ref0[objects[0].col]
            q[-7:] = ref0[objects[1].col]
        elif len(objects) == 1:
            q[-7:] = ref0[objects[0].col]
        for adr, angles in self.hinges:
            q[adr] = angles[0]
        return q

    def _rot_addresses(self) -> list[tuple[int, int, int]]:
        out = []
        for s in self.sides:
            ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_rot_{a}") for a in "xyz"]
            if min(ids) >= 0:
                out.append(tuple(int(self.m.jnt_qposadr[j]) for j in ids))
        return out

    def _probe_score(self, q0: np.ndarray) -> float:
        d = mujoco.MjData(self.m)
        d.qpos[:] = q0
        d.qvel[:] = 0.0
        d.qacc_warmstart[:] = 0.0
        self.set_targets(d, 0)
        self.settle(d, 0, INIT_SETTLE_STEPS, None)
        mujoco.mj_kinematics(self.m, d)
        dist = [
            np.linalg.norm(d.site_xpos[sid] - self.kp.ref[0, tr.col, :3])
            for tr, sid in zip(self.tracked, self.site_ids, strict=True)
            if tr.kind in ("palm", "tip")
        ]
        return float(np.mean(dist))

    def multistart(self, q0: np.ndarray) -> tuple[np.ndarray, int, list[float]]:
        """Pick the frame-0 start among the plain start and three base-rotation alternatives."""
        rots = self._rot_addresses()
        cands = [q0]
        if rots:
            for angles in MULTISTART_ROTATIONS:
                c = q0.copy()
                for adr in rots:
                    c[list(adr)] = angles
                cands.append(c)
        scores = [self._probe_score(c) for c in cands]
        best = 0
        for i in range(1, len(cands)):
            if scores[i] < scores[best] - MULTISTART_MARGIN:
                best = i
        return cands[best], best, scores

    # --- the solve -------------------------------------------------------------------------------------------------
    def run(self) -> RawSolve:
        T, m = self.kp.T, self.m
        q_start, choice, scores = self.multistart(self.start_state())
        if choice:
            print(f"[IK] multistart: alternative start {choice} wins (scores {np.round(scores, 4).tolist()})")
        d = mujoco.MjData(m)
        d.qpos[:] = q_start
        mujoco.mj_kinematics(m, d)
        ds = mujoco.MjData(self.scene)
        qpos = np.zeros((T, m.nq))
        cpos = np.zeros((T, 10, 3))
        palm = {s: np.full(T, np.nan) for s in ("right", "left")}
        tips = np.full((T, 10), np.nan)
        v_pre, v_fin = np.zeros(T), np.zeros(T)
        q_prev = None
        for t in range(T):
            self.set_targets(d, t)
            self.settle(d, t, INIT_SETTLE_STEPS if t == 0 else SETTLE_STEPS, q_prev)
            v_pre[t] = self.limits.violation(d.qpos)
            self.limits.project(d.qpos)
            if v_pre[t] > VIOLATION_RESETTLE:
                self.settle(d, t, RESETTLE_STEPS, q_prev)
                self.limits.project(d.qpos)
            if self.prior is not None:
                if q_prev is not None:
                    self.prior.slew(d.qpos, q_prev)
                self.limits.project(d.qpos)
            mujoco.mj_kinematics(m, d)
            if self.prior is not None:
                q_prev = d.qpos.copy()
            for tr, sid in zip(self.tracked, self.site_ids, strict=True):
                err = float(np.linalg.norm(d.site_xpos[sid] - self.kp.ref[t, tr.col, :3]))
                if tr.kind == "palm":
                    palm[tr.side][t] = err
                elif tr.kind == "tip":
                    tips[t, (0 if tr.side == "right" else 5) + tr.finger] = err
            v_fin[t] = self.limits.violation(d.qpos)
            qpos[t] = d.qpos
            ds.qpos[:] = d.qpos
            mujoco.mj_kinematics(self.scene, ds)
            for k, sid in enumerate(self.track_hand):
                if sid >= 0:
                    cpos[t, k] = ds.site_xpos[sid]
        return RawSolve(qpos, cpos, palm, tips, v_pre, v_fin, choice)
