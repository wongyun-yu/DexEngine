"""Fingertip-force and finger-speed oracle — the caps a hand applies (hands.finger_effort_applied ·
hands.finger_velocity_vector), measured on its right retargeting rig and held against the vendor figures the hand spec
states. A measurement checked against a statement, not the statement written twice.

Force: F_max = min_j cap_j / |J_j · d| over the finger chain (J = fingertip-site position Jacobian column, d = the
distal phalanx normal, the last joint's column direction), in three postures (straight · 30 % · 50 % from the rest-side
limit). The hand's figure = the index finger's best posture (a vendor "fingertip force" is a favourable-pose maximum).
  · stated nominal → figure ≤ nominal × (1 + TOL) (above it the cap is beyond the hardware)
  · scale ≠ 1      → figure ≥ nominal × (1 − TOL) (a scale that does not reproduce the nominal has no ground)
Speed: closing and opening a joint over its full rig range at the vendor repetition frequency f needs 2·R·f; an
applied speed above it is beyond the hardware (checked for every joint of a hand whose speed scale is not 1).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from dexengine.core import paths  # noqa: E402
from dexengine.core.hands import HAND_REGISTRY, finger_effort_applied, finger_velocity_vector  # noqa: E402
from dexengine.core.hands.helpers import per_joint  # noqa: E402

TOL = 0.03
POSTURES = (0.0, 0.3, 0.5)
SIDE = "right"


def _rig(hand):
    m = mujoco.MjModel.from_xml_path(str(paths.hand_rig_dir(paths.hand_spec_dir(hand)) / f"{SIDE}.xml"))
    return m, mujoco.MjData(m)


def _index_force(m, d, effort) -> float:
    """The best static index distal-pad force (N) at the per-joint caps ``effort`` over the three postures."""
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{SIDE}_index_tip")
    assert sid >= 0, f"no {SIDE}_index_tip site on the rig"
    chain, b = [], int(m.site_bodyid[sid])
    while b > 0:
        chain += [
            j for j in range(m.njnt) if int(m.jnt_bodyid[j]) == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        b = int(m.body_parentid[b])
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in chain]
    keep = [i for i, n in enumerate(names) if not re.fullmatch(rf"{SIDE}_(pos|rot)_[xyz]", n)]
    chain = [chain[i] for i in keep][::-1]  # proximal → distal
    caps = list(per_joint(effort, [names[i] for i in keep]))[::-1]
    best = 0.0
    for frac in POSTURES:
        d.qpos[:] = 0.0
        for j in chain:
            lo, hi = (float(x) for x in m.jnt_range[j])
            rest, far = (lo, hi) if abs(lo) < abs(hi) else (hi, lo)
            d.qpos[m.jnt_qposadr[j]] = rest + frac * (far - rest) if frac > 0 else min(max(0.0, lo), hi)
        mujoco.mj_forward(m, d)
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, None, sid)
        cols = [jacp[:, m.jnt_dofadr[j]] for j in chain]
        dn = cols[-1] / np.linalg.norm(cols[-1])
        best = max(
            best,
            min(c / abs(float(col @ dn)) for c, col in zip(caps, cols, strict=True) if abs(float(col @ dn)) > 1e-4),
        )
    return best


@pytest.mark.parametrize("hand", sorted(HAND_REGISTRY))
def test_applied_effort_cap_meets_the_vendor_fingertip_force(hand):
    spec = HAND_REGISTRY[hand]
    nominal, scale = spec.vendor_fingertip_force_n, float(spec.finger_effort_peak_scale)
    if nominal is None:
        assert scale == 1.0
        return
    assert not spec.coupled_joint_names_right, "the Jacobian model does not know coupled joints"
    m, d = _rig(hand)
    force = _index_force(m, d, finger_effort_applied(spec))
    assert force <= nominal * (1 + TOL), f"{hand}: index pad {force:.2f} N > nominal {nominal} N"
    if scale != 1.0:
        assert force >= nominal * (1 - TOL), f"{hand}: scale ×{scale} gives {force:.2f} N, nominal {nominal} N"


@pytest.mark.parametrize("hand", sorted(HAND_REGISTRY))
def test_applied_finger_speed_within_a_full_range_open_close(hand):
    spec = HAND_REGISTRY[hand]
    if float(spec.finger_velocity_peak_scale) == 1.0:
        return  # the spec's speed is the vendor's maximum as stated
    f = float(spec.vendor_close_repetition_hz)
    m, _ = _rig(hand)
    over = {}
    for name, v in zip(spec.finger_joint_names_right, finger_velocity_vector(spec), strict=True):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert j >= 0, f"{hand}: no joint {name} on the rig"
        limit = 2.0 * float(m.jnt_range[j][1] - m.jnt_range[j][0]) * f
        if v > limit * (1 + 1e-9):
            over[name] = (round(v, 3), round(limit, 3))
    assert not over, f"{hand}: applied finger speed above a full-range open-close at {f} Hz: {over}"
