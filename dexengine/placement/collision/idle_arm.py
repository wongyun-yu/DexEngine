"""One-hand rule: the idle arm's rest pose (single owner) and the active / idle sides of a clip or tree. Judging,
the bake, cross-repair and the gate read these; nothing else writes the values.
"""

from __future__ import annotations

import numpy as np

from dexengine.placement.collision.geom import arm_link_transforms

# ── One-hand rule: the idle arm's rest pose ──
# In a clip where only one hand is active (case.active is one side), the idle arm does not solve IK and simply stands
# in its rest pose, its fingers at their rest pose (hand_surface.rest_finger_q). Judging (judge --active-sides auto),
# the bake (idle_rest), cross-repair (frozen) and the gate all read the pose through idle_qpos — its value is the arm
# spec's (ArmSpec.idle_qpos, configs/robots/arms/<arm>.yaml, with the reasons for the numbers).


def idle_qpos(arm: str, side: str) -> np.ndarray:
    """The idle arm's rest pose of ``arm`` on ``side`` (joint order, rad). An arm without a stated pose dies."""
    from dexengine.core.arms import ARM_REGISTRY  # noqa: PLC0415

    q = ARM_REGISTRY[arm].idle_qpos.get(side)
    if q is None:
        raise ValueError(f"[idle] arm {arm!r} states no idle_qpos for {side} (configs/robots/arms/{arm}.yaml)")
    return np.array(q)


#: IK npz `solver` value — the marker by which the gate and audits recognise an "idle arm".
IDLE_SOLVER_TAG = "idle_rest"
#: The key copied from adopted.json into the tree task_info (the bake copies it, downstream reads it).
IDLE_SIDES_KEY = "idle_sides"


def active_idle_from_case(case) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """task_info.case → (active sides, idle sides) — the **single owner** of the one-hand rule (judging
    `judge --active-sides auto` and the bake_drive re-bake call the same function). Only when case.active is **exactly
    one side** is the other side idle; with 0 or 2 sides both arms are active and nothing is idle. Missing case/active
    raises ValueError — never a silent both."""
    if not isinstance(case, dict) or "active" not in case:
        raise ValueError(
            "task_info.case(.active) missing — this tree's task_info has no case (no silent both).\n"
            "  · the source layer of every dataset is base (paths.default_src_layer):\n"
            "    run dexengine/data/cases/taxonomy.py --dataset <ds> --write first to write the case."
        )
    act = tuple(s for s in ("right", "left") if s in case["active"])
    if len(act) == 1:
        return act, tuple(s for s in ("right", "left") if s not in act)
    return ("right", "left"), ()


def active_idle_for_tree(case, layout_hands, manip_sides=None) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """One tree's (active sides, idle sides, rule) — the **single entry point** of the one-hand rule.
    · If layout_hands (the hands in qpos_layout, i.e. the hands the tree actually has) is **one side**, that side is
      active and the absent hand is idle (rule "layout"; the one-hand embodiments — case is UNI_R/UNI_L and
      case.active is not consulted).
    · With two sides, `active_idle_from_case` (rule "case"); with manip_sides, an active hand with no main object is
      idle too (rule "case∩manip").
    · No hands, or values that are neither, raise ValueError."""
    hands = tuple(s for s in ("right", "left") if s in (layout_hands or ()))
    if len(hands) == 1:
        return hands, tuple(s for s in ("right", "left") if s not in hands), "layout"
    if len(hands) != 2:
        raise ValueError(f"unexpected tree hand list: {layout_hands}")
    act, idle = active_idle_from_case(case)
    if manip_sides is not None:
        # Even when case.active says active, a **hand with no main object** (contact 0) is not a manipulating hand —
        # it goes to the idle side on the same criterion as f0 (clip_sides=manip_sides) and the contact derivation.
        # Otherwise judging reads a wrist shadow that f0 never made.
        m = tuple(s for s in ("right", "left") if s in (manip_sides or ()))
        act2 = tuple(s for s in act if s in m)
        if not act2:
            raise ValueError(
                f"no manipulating hand (case.active={act}, hands with a main object={m}) — a no_main_object clip"
            )
        if act2 != act:
            idle = tuple(s for s in ("right", "left") if s not in act2)
            return act2, idle, "case∩manip"
    return act, idle, "case"


def idle_arm_q(arm: str, side: str, T: int) -> np.ndarray:
    """The idle arm's joint trajectory (T, n_joints) — the rest pose (idle_qpos) on every frame."""
    return np.repeat(idle_qpos(arm, side)[None, :], int(T), axis=0)


def idle_ee_pose(ctx: dict) -> np.ndarray:
    """The EE (hand attachment frame) 4×4 at the rest pose of the context's (arm, side) — **robot-local** (base not
    included; the same frame as the coarse-path target Tt)."""
    arm, _hand, side = ctx["combo"]
    Tlast = arm_link_transforms(ctx, idle_qpos(arm, side)[None, :])[0, -1]
    return Tlast @ ctx["ee_local"]


def idle_sides_of_tree(tree_dir) -> list[str]:
    """The idle-side list in the tree's task_info ([] if absent). Provenance of the value = judging
    plan → adopted.json → the bake copies it into task_info."""
    import json
    from pathlib import Path

    p = Path(tree_dir) / "task_info.json"
    if not p.is_file():
        return []
    v = json.loads(p.read_text()).get(IDLE_SIDES_KEY) or []
    bad = [s for s in v if s not in ("right", "left")]
    if bad:
        raise ValueError(f"[idle] {p}: {IDLE_SIDES_KEY}={v} — only right/left are allowed")
    return list(v)
