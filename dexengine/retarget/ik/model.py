"""IK model assembly: tracked entities, mocap targets, soft equality constraints, joint limit sets.

The IK model is the scene file edited before compiling (collisions and actuator force limits off, one mocap target
body per tracked entity, one weld/connect equality from the entity site to its target) and then edited after
compiling (options, hyperextension clip, prior damping). The scene model itself is left as compiled.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from dexengine.retarget.ik import prior as P
from dexengine.retarget.ik.inputs import FINGERS, object_col, tip_col, wrist_col

#: Equality parameters per entity class: (type, solref, solimp, torque scale).
_EQ = {
    "palm": (mujoco.mjtEq.mjEQ_WELD, (0.02, 1.0), (0.0, 0.95, 0.1, 0.5, 2.0), 10.0),
    "object": (mujoco.mjtEq.mjEQ_WELD, (0.002, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0), 10.0),
    "tip_firm": (mujoco.mjtEq.mjEQ_CONNECT, (0.01, 1.0), (0.0, 0.95, 0.01, 0.5, 2.0), 1.0),
    "tip_soft": (mujoco.mjtEq.mjEQ_CONNECT, (0.01, 1.0), (0.0, 0.95, 0.03, 0.5, 2.0), 1.0),
}
_FIRM_TIPS = ("thumb", "index", "middle")
TARGET_RGBA = [0.0, 1.0, 0.0, 0.5]
WRAP_WIDTH = 2.0 * math.pi - 1e-6


@dataclass
class Tracked:
    """One tracked site and the reference column it follows."""

    site: str
    kind: str  # "palm" | "tip" | "object"
    col: int
    side: str | None = None
    finger: int | None = None

    @property
    def eq_class(self) -> str:
        if self.kind == "tip":
            return "tip_firm" if FINGERS[self.finger] in _FIRM_TIPS else "tip_soft"
        return self.kind


def tracked_entities(scene: mujoco.MjModel, sides: tuple[str, ...], n_extra: int) -> list[Tracked]:
    """Candidate list filtered to the embodiment and to object bodies present in the scene."""
    cands: list[Tracked] = []
    for side in ("right", "left"):
        cands.append(Tracked(f"{side}_palm", "palm", wrist_col(side), side))
        cands.extend(
            Tracked(f"{side}_{finger}_tip", "tip", tip_col(side, k), side, k) for k, finger in enumerate(FINGERS)
        )
    cands.extend(Tracked(f"extra_object_{i}", "object", 12 + i) for i in range(n_extra))
    cands.extend(
        Tracked(f"{side}_object", "object", object_col(f"{side}_object", n_extra), side) for side in ("right", "left")
    )
    if len(sides) == 1:
        cands = [c for c in cands if sides[0] in c.site or c.site.startswith("extra_object_")]
    out = []
    for c in cands:
        if c.kind == "object" and mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_BODY, c.site) < 0:
            continue
        if mujoco.mj_name2id(scene, mujoco.mjtObj.mjOBJ_SITE, c.site) < 0:
            raise AssertionError(f"[IK] tracked site {c.site!r} missing from the scene (rig / scene contract)")
        out.append(c)
    return out


def build_ik_model(scene_xml: str, tracked: list[Tracked], h: float) -> mujoco.MjModel:
    """Compile the IK model from the scene file; targets are mocap bodies after every scene body."""
    spec = mujoco.MjSpec.from_file(scene_xml)
    for joint in spec.joints:
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    for i, tr in enumerate(tracked):
        body = spec.worldbody.add_body(name=f"ik_target_{i}_{tr.site}", mocap=True)
        site = body.add_site(name=f"ik_target_{i}_{tr.site}", rgba=TARGET_RGBA, group=1)
        if tr.kind == "tip":
            site.type = mujoco.mjtGeom.mjGEOM_SPHERE
            site.size = [0.01, 0.01, 0.01]
        else:
            site.type = mujoco.mjtGeom.mjGEOM_BOX
            site.size = [0.01, 0.02, 0.03]
    for i, tr in enumerate(tracked):
        eq_type, solref, solimp, torque = _EQ[tr.eq_class]
        data = np.zeros(11)
        data[10] = torque
        spec.add_equality(
            name=f"ik_track_{i}_{tr.site}",
            type=eq_type,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=tr.site,
            name2=f"ik_target_{i}_{tr.site}",
            data=data,
            solref=list(solref),
            solimp=list(solimp),
        )
    model = spec.compile()
    model.opt.timestep = h
    model.opt.iterations = 20
    model.opt.ls_iterations = 50
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    return model


def target_mocap_ids(model: mujoco.MjModel, tracked: list[Tracked]) -> np.ndarray:
    """Mocap index of each tracked entity's target body, in tracked order."""
    ids = []
    for i, tr in enumerate(tracked):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ik_target_{i}_{tr.site}")
        ids.append(int(model.body_mocapid[b]))
    return np.array(ids, dtype=int)


def apply_hyperext_clip(model: mujoco.MjModel, conf: dict | None) -> tuple[list[str], list[str]]:
    """Narrow the IK model's ranges with the prior's hyperextension clip; returns (considered, changed) names."""
    considered: list[str] = []
    changed: list[str] = []
    if conf is None or not conf["limit_hyperextension"]:
        return considered, changed
    for name, sign, neutral in P.hyperext_clip_targets(conf):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            raise P.PriorSchemaError(f"[PRIOR] hyperextension clip joint {name!r} is not in the scene")
        lo, hi = (float(x) for x in model.jnt_range[j])
        new = P.hyperext_clip_range(lo, hi, sign, neutral)
        considered.append(name)
        if (float(new[0]), float(new[1])) != (lo, hi):
            model.jnt_range[j] = new
            changed.append(name)
    return considered, changed


@dataclass
class LimitSets:
    """Projection targets: wrapping hinges (reduced modulo their range width) and clipped joints."""

    wrap_adr: np.ndarray
    wrap_lo: np.ndarray
    wrap_w: np.ndarray
    clip_adr: np.ndarray
    clip_lo: np.ndarray
    clip_hi: np.ndarray

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> LimitSets:
        wrap, clip = [], []
        for j in range(model.njnt):
            if not model.jnt_limited[j] or model.jnt_type[j] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            lo, hi = float(model.jnt_range[j][0]), float(model.jnt_range[j][1])
            adr = int(model.jnt_qposadr[j])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and hi - lo >= WRAP_WIDTH:
                wrap.append((adr, lo, hi - lo))
            else:
                clip.append((adr, lo, hi))

        def cols(rows, k):
            return np.array([r[k] for r in rows], dtype=int if k == 0 else float)

        return cls(cols(wrap, 0), cols(wrap, 1), cols(wrap, 2), cols(clip, 0), cols(clip, 1), cols(clip, 2))

    def violation(self, q: np.ndarray) -> float:
        """Largest excursion beyond [lo, hi] among the clipped (non-wrapping) joints; 0 if none."""
        if self.clip_adr.size == 0:
            return 0.0
        x = q[self.clip_adr]
        return float(max(0.0, float(np.max(self.clip_lo - x)), float(np.max(x - self.clip_hi))))

    def project(self, q: np.ndarray) -> None:
        """In place: wrapping hinges to `lo + mod(q - lo, w)`, all other limited joints clipped to [lo, hi]."""
        if self.wrap_adr.size:
            q[self.wrap_adr] = self.wrap_lo + np.mod(q[self.wrap_adr] - self.wrap_lo, self.wrap_w)
        if self.clip_adr.size:
            q[self.clip_adr] = np.clip(q[self.clip_adr], self.clip_lo, self.clip_hi)


def hinge_table(model: mujoco.MjModel, arti: dict[str, np.ndarray]) -> list[tuple[int, np.ndarray]]:
    """(qpos address, per-frame angle) of every articulated-object hinge in the scene."""
    out = []
    for side in ("right", "left"):
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_object_hinge")
        if j < 0:
            continue
        if side not in arti:
            raise ValueError(f"[IK] scene has {side}_object_hinge but the archive has no qpos_obj_arti_{side}")
        out.append((int(model.jnt_qposadr[j]), np.asarray(arti[side][:, 0], dtype=np.float64)))
    return out
