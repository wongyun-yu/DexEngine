"""Gate speed rule of placement augment variants (called by augment.place_gen after the variant bake).

The same function as the post-bake gate J-gate-speed: gate.speed_ratio(stretch.frame_ratio, arm + fingers) p99 <=
gate.GATE_SPEED_P99_MAX · arm limits = the same arm_specs.per_joint_limits("velocity") x stretch.FRAME_DT as retime ·
fingers = hand_surface.finger_budget · finger values = the variant tree (its own time axis). Retiming guarantees it
by construction, so a failure is a signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dexengine.core import paths as P
from dexengine.record.augment.judged import die  # a shape mismatch refuses the clip like a judge input


def speed(src, tree: Path, qs: dict) -> tuple[bool, str, dict]:
    """Gate speed rule (module doc): fingers of the variant tree tree and arm cache qs {active side: q} -> (pass,
    reason, per side [p99, max])."""
    from dexengine.placement.collision import hand_surface as HSF
    from dexengine.placement.retime import stretch as VST

    bud = np.asarray(P.load_registry("arm_specs").ARM_REGISTRY[src.arm].per_joint_limits("velocity"), float)
    qpos = np.load(Path(tree) / "0/trajectory_kinematic.npz")["qpos"]
    fv = HSF.guard_finger_q(tree, src.hand, idle=(), qpos=qpos)
    return speed_check(qs, bud * VST.FRAME_DT, fv, HSF.finger_budget(src.hand))


def speed_check(qs: dict, bud, fv: dict, fbud) -> tuple[bool, str, dict]:
    """Per side gate.speed_ratio(arm + that side's fingers) p99 <= gate.GATE_SPEED_P99_MAX; a side with a reason
    (length mismatch etc.) also fails."""
    from dexengine.placement.gate import core as GT

    vals, why = {}, ""
    for s, q in qs.items():
        if q.shape[1] != len(bud):
            die(f"{s}: joints {q.shape[1]} != arm limits {len(bud)}")
        rp, rm, _lp, _lm, sp_why = GT.speed_ratio(q, bud, fv[s], fbud)
        vals[s] = [round(rp, 4), round(rm, 4)]
        if sp_why or not rp <= GT.GATE_SPEED_P99_MAX:
            why = why or f"{s} r_p99={rp:.2f} > {GT.GATE_SPEED_P99_MAX}" + (f" ({sp_why})" if sp_why else "")
    return not why, why, vals
