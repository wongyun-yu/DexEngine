"""Finger branch re-solve: after the physics solve, put a finger back on its natural branch without moving its tip.

Why: a finger with more joints than its tip has coordinates has a null space, and the physics solve has no posture
term on it (template v2). On a joint whose range is wide enough to hold a mirrored solution (DG-5F-S thumb
`joint_1_2`, 153°), the solve drifts along that null space into the mirror branch: the proximal phalanx swings across
the palm and the distal joints fold back to the same tip ("Z" thumb), and the mirror branch's centre lies past the
hardware stop, so the joint parks on its limit (thumb −153°).

What: for every joint the hand's prior lists in `branch_window_deg` (`PRIOR_TEMPLATE`; empty = off, every output
byte-identical), the finger that owns the joint is re-solved frame by frame — kinematics only, the rest of the hand
untouched — so its tip stays where the physics solve put it, with the listed joint kept inside its window:

    min_q |tip(q) - tip_raw(t)|^2 + MU |q - q(t-1)|^2 + RHO |q - q_mid|^2,   q in the IK ranges ∩ window

(frame 0 starts from q_mid without the continuity term)

The continuity term stops the null-space drift; the small pull to the centre of the joint ranges picks one posture
when the tip leaves freedom. Only that finger's columns of the raw solve change, and the smoothing that follows is
per coordinate, so every other column of the output stays byte-identical.

The same windows bound the reference finger range downstream (`wrist_shift.finger_ranges`, through the window
`hyperext_limits`, key `branch_windows`): the repairs that re-solve fingers after the IK (wrist ↔ desk, f0, the cross
repair) and the finger range gate keep the joint inside them. The sim body's joint limits (truncate.py) never read them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco

#: Continuity weight (m² per rad²): a 0.01 rad step costs as much as a 0.03 mm tip offset. Not used on frame 0.
MU = 1e-5
#: Pull to the centre of the joint ranges (m² per rad²) — small against the tip and continuity terms.
RHO = 1e-7
ITERS = 25
INIT_ITERS = 200


@dataclass
class BranchReport:
    finger: str
    window_joint: str
    window_deg: tuple[float, float]
    frames: int
    outside_raw: int  # raw frames whose window joint was outside the window
    tip_dev_p95_m: float
    tip_dev_max_m: float
    max_step_rad: float

    def marker(self) -> str:
        return (
            f"[BRANCH] {self.finger} {self.window_joint} window=[{self.window_deg[0]:.1f},{self.window_deg[1]:.1f}]deg "
            f"frames={self.frames} outside_raw={self.outside_raw} tip_dev_p95={self.tip_dev_p95_m * 1e3:.3f}mm "
            f"tip_dev_max={self.tip_dev_max_m * 1e3:.3f}mm max_step={math.degrees(self.max_step_rad):.2f}deg"
        )

    def as_json(self) -> dict:
        return {
            "finger": self.finger,
            "window_joint": self.window_joint,
            "window_deg": list(self.window_deg),
            "frames": self.frames,
            "outside_raw": self.outside_raw,
            "tip_dev_p95_m": self.tip_dev_p95_m,
            "tip_dev_max_m": self.tip_dev_max_m,
            "max_step_rad": self.max_step_rad,
        }


def parse_windows(val, fingers, want, die) -> list[dict]:
    """The prior's `branch_window_deg` → [{joint, finger, lo, hi}] (radians) for the requested sides (called by
    `prior.build_prior_conf`, which owns the key). Each joint must belong to one declared finger chain (mcp/pip/dip
    or abd) of its side; a joint of a side the scene does not request is skipped (a one-hand scene of a two-hand
    yaml). The window must be a non-empty [lo, hi]; one window per finger. `die(what)` raises the prior's error."""
    if val is None:
        return []
    if not isinstance(val, dict):
        die(f"branch_window_deg: must be a mapping {{joint: [lo_deg, hi_deg]}}, got {val!r}")
    owner = {}
    for f in fingers:
        for role in ("mcp", "pip", "dip"):
            if role in f["chain"]:
                owner[f["chain"][role]["joint"]] = f["name"]
        for a in f["abd"]:
            owner[a["joint"]] = f["name"]
    out = []
    for joint, win in val.items():
        side = str(joint).split("_", 1)[0]
        if side not in want:
            continue
        if joint not in owner:
            die(f"branch_window_deg: {joint!r} is not a joint of a declared {side} finger chain")
        if not isinstance(win, list | tuple) or len(win) != 2 or not float(win[0]) < float(win[1]):
            die(f"branch_window_deg.{joint}: must be [lo_deg, hi_deg] with lo < hi, got {win!r}")
        if any(w["finger"] == owner[joint] for w in out):
            die(f"branch_window_deg: two windows on finger {owner[joint]!r} — one per finger")
        out.append(
            {
                "joint": str(joint),
                "finger": owner[joint],
                "lo": math.radians(float(win[0])),
                "hi": math.radians(float(win[1])),
            }
        )
    return out


def _finger_joints(finger: dict) -> list[str]:
    chain = finger["chain"]
    return [chain[r]["joint"] for r in ("mcp", "pip", "dip") if r in chain] + [a["joint"] for a in finger["abd"]]


def _solve(m, d, base, qa, da, site, x, q0, qp, lo, hi, qm, iters, mu):
    """Bounded Levenberg–Marquardt on one finger with every other coordinate fixed at `base`."""
    import mujoco  # noqa: PLC0415

    jacp = np.zeros((3, m.nv))
    n = len(qa)

    def tip(q):
        d.qpos[:] = base
        d.qpos[qa] = q
        mujoco.mj_kinematics(m, d)
        return d.site_xpos[site].copy()

    def cost(q, r):
        return float(r @ r + mu * (q - qp) @ (q - qp) + RHO * (q - qm) @ (q - qm))

    q = np.clip(q0, lo, hi)
    r = tip(q) - x
    c, lam = cost(q, r), 1e-6
    for _ in range(iters):
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacSite(m, d, jacp, None, site)
        J = jacp[:, da]
        H = J.T @ J + (mu + RHO + lam) * np.eye(n)
        g = J.T @ r + mu * (q - qp) + RHO * (q - qm)
        qn = np.clip(q - np.linalg.solve(H, g), lo, hi)
        rn = tip(qn) - x
        cn = cost(qn, rn)
        if cn < c:
            q, r, c, lam = qn, rn, cn, max(lam * 0.3, 1e-12)
        else:
            lam *= 10.0
            tip(q)  # restore the data to the accepted point for the next Jacobian
            if lam > 1e3:
                break
    return q


def resolve_branches(m: mujoco.MjModel, qpos: np.ndarray, conf: dict | None) -> list[BranchReport]:
    """In place on the raw solve `qpos` (T, nq): re-solve every finger that owns a windowed joint. Returns one report
    per re-solved finger (empty when the prior lists no window)."""
    import mujoco  # noqa: PLC0415 (prior.py imports this module on interpreters without mujoco)

    windows = (conf or {}).get("branch_windows") or []
    reports: list[BranchReport] = []
    if not windows:
        return reports
    d = mujoco.MjData(m)
    by_name = {f["name"]: f for f in conf["fingers"]}
    for w in windows:
        finger = by_name[w["finger"]]
        names = _finger_joints(finger)
        jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        if min(jids) < 0:
            raise ValueError(f"[BRANCH] {w['finger']}: joints {names} not all in the IK model")
        qa = np.array([int(m.jnt_qposadr[j]) for j in jids])
        da = np.array([int(m.jnt_dofadr[j]) for j in jids])
        lo = np.array([float(m.jnt_range[j][0]) for j in jids])
        hi = np.array([float(m.jnt_range[j][1]) for j in jids])
        k = names.index(w["joint"])
        lo[k], hi[k] = max(lo[k], w["lo"]), min(hi[k], w["hi"])
        if lo[k] > hi[k]:
            raise ValueError(f"[BRANCH] {w['joint']}: window {w['lo']:.4f}..{w['hi']:.4f} misses the IK range")
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{finger['side']}_{finger['role']}_tip")
        if site < 0:
            raise ValueError(f"[BRANCH] {w['finger']}: tip site missing in the IK model")
        T = qpos.shape[0]
        x = np.zeros((T, 3))
        for t in range(T):
            d.qpos[:] = qpos[t]
            mujoco.mj_kinematics(m, d)
            x[t] = d.site_xpos[site]
        raw = qpos[:, qa].copy()
        outside = int(((raw[:, k] < lo[k] - 1e-9) | (raw[:, k] > hi[k] + 1e-9)).sum())
        qm = 0.5 * (lo + hi)
        q = qm.copy()
        new = np.zeros_like(raw)
        for t in range(T):
            base = qpos[t].copy()
            first = t == 0
            iters, mu = (INIT_ITERS, 0.0) if first else (ITERS, MU)
            q = _solve(m, d, base, qa, da, site, x[t], q, q, lo, hi, qm, iters, mu)
            new[t] = q
        qpos[:, qa] = new
        dev = np.zeros(T)
        for t in range(T):
            d.qpos[:] = qpos[t]
            mujoco.mj_kinematics(m, d)
            dev[t] = np.linalg.norm(d.site_xpos[site] - x[t])
        step = float(np.abs(np.diff(new, axis=0)).max()) if T > 1 else 0.0
        reports.append(
            BranchReport(
                w["finger"],
                w["joint"],
                (math.degrees(w["lo"]), math.degrees(w["hi"])),
                T,
                outside,
                float(np.percentile(dev, 95)),
                float(dev.max()),
                step,
            )
        )
    return reports
