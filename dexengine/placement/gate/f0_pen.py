#!/usr/bin/env python3
"""Frame-0 (spawn) penetration — robot↔object (sets ⑤⑥, gate J-gate-f0pen) and object↔object (set ③ at frame 0,
gate J-gate-f0objobj) measured with the shapes the sim collides with.

The sets (⑤⑥ extend cross_sets' ①–④; ③ is cross_sets' own object pair set, read at the spawn pose):
  ③ 3_obj_obj      right main object ↔ left main object (cross_sets' ③ pairs — the same pairs, frame 0 only)
  ⑤ 5_arm_obj      every arm link (ArmSpec.body_side, incl. the EE link that cross_sets files under the hand) ↔ every
                    main object (own · opposite · shared)
  ⑥ 6_hand_ownobj  every hand link ↔ that hand's own main object(s) (its side's, or the shared one)
Shapes = the colliders the sim spawns (cross_sets geom "collision": URDF <collision> bodies, arm links from
sim_collider — the one generator of the arm colliders, incl. the revision hulls the USD generator writes), not the
visual hull of ①–④: the spawn collision is PhysX's. Distances = cross_sets.CrossSetModel (same mj_geomDistance,
hull graphs, 0.0 re-query). Pose = what the sim spawns: arm-IK npz frame 0 (both arms, robot root) + tree qpos frame 0.
Thresholds = coords.F0_OBJ_OBJ_PEN_M (③, a gate reason) · coords.F0_ARM_OBJ_PEN_M (⑤, a gate reason) ·
coords.F0_HAND_OWNOBJ_MODE (⑥ — "report" = measured and reported, never a violation). Callers: gate.core (both sides
fail, reason 'F0_pen:…' for ⑤ · 'F0_obj_obj:…' for ③ — gate-table data) and dry scans — the same function
frame0_penetration. The gate's frame-0 rules live here too (gate_reason · reason form F0_REASON_HEAD · the declared
data property _data_exc); the unmeasurable branches and the systematic stop are the gate's non-verdict rule
(ledger_states.measured · error_systematic); gate.core applies them per side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dexengine.placement import ledger_states as LS  # the gate's one non-verdict rule (measured · GateUnmeasured)
from dexengine.placement.collision import cross_sets as CC

F0_SETS = ("3_obj_obj", "5_arm_obj", "6_hand_ownobj")
#: The side letter of the frame-0 pair classes (f0_pairs: `<S>arm-ownobj` · `<S>arm-oppobj` · `<S>hand-ownobj`).
SIDE_LETTER = {"right": "R", "left": "L"}


def f0_pairs(model) -> list:
    """③⑤⑥ pairs of a built CrossSetModel, in its pair form (geom i, geom j, set, class). ③ = the model's own ③ pairs
    (cross_sets._pairs — one definition of which objects pair)."""
    by: dict[str, list[int]] = {}
    for i, (g, _) in enumerate(model.ginfo):
        by.setdefault(g, []).append(i)
    out = [p for p in model._pairs() if p[2] == "3_obj_obj"]
    side_of = model.arm_spec.body_side  # an arm link's side (ArmSpec) — the EE link is an arm link here too
    for side, A, H in (("right", "RA", "RH"), ("left", "LA", "LH")):
        S = SIDE_LETTER[side]
        arm = [i for i in by.get(A, []) + by.get(H, []) if side_of(model.ginfo[i][1]) == side]
        hand = [i for i in by.get(H, []) if side_of(model.ginfo[i][1]) is None]
        own = ("OR" if S == "R" else "OL", "OS")
        for og in ("OR", "OL", "OS"):
            out += [
                (gi, gj, "5_arm_obj", f"{S}arm-{'own' if og in own else 'opp'}obj")
                for gi in arm
                for gj in by.get(og, [])
            ]
        out += [(gi, gj, "6_hand_ownobj", own_class(side)) for og in own for gi in hand for gj in by.get(og, [])]
    return out


def own_class(side: str) -> str:
    """The ⑥ pair class of one side (hand ↔ its own main object)."""
    return f"{SIDE_LETTER[side]}hand-ownobj"


def hand_ownobj_mm(pairs) -> dict:
    """{side: the deepest ⑥ pair of that side (mm · 0.0 = none)} — the gate's per-side column (reported only)."""
    return {s: max([r[4] for r in pairs if r[1] == own_class(s)], default=0.0) for s in SIDE_LETTER}


def set_thresholds() -> dict:
    """Set → threshold (mm) or None (measured, never a violation). Owners: coords.F0_OBJ_OBJ_PEN_M ·
    coords.F0_ARM_OBJ_PEN_M · coords.F0_HAND_OWNOBJ_MODE ("report" → None; any other mode dies — it has no consumer)."""
    from dexengine.placement.coords import core as _coords

    if _coords.F0_HAND_OWNOBJ_MODE != "report":
        raise ValueError(f"coords.F0_HAND_OWNOBJ_MODE={_coords.F0_HAND_OWNOBJ_MODE!r} — the only known mode is report")
    return {
        "3_obj_obj": float(_coords.F0_OBJ_OBJ_PEN_M) * 1000.0,
        "5_arm_obj": float(_coords.F0_ARM_OBJ_PEN_M) * 1000.0,
        "6_hand_ownobj": None,
    }


def violations(pairs, thresholds: dict) -> list:
    """The pairs deeper than their set's threshold; a set whose threshold is None never violates."""
    return [r for r in pairs if thresholds.get(r[0]) is not None and r[4] > thresholds[r[0]]]


def frame0_penetration(tree_dir, ik_right, ik_left, assets_root, urdf=None) -> dict:
    """Frame-0 penetration of one baked clip (sets ③⑤⑥ on the sim shapes).

    Returns {"thresh_mm": {set: mm | None}, "worst_mm": {set: depth ≥ 0}, "pairs": [(set, class, robot link, object
    body, depth mm)] — every penetrating pair (deepest over the pair's shapes), deepest first, "violations": the pairs
    deeper than their set's threshold, "hand_ownobj_mm": {side: ⑥ depth ≥ 0 of that side's pairs}}."""
    tree_dir = Path(tree_dir)
    ik = {"right": CC.load_ik(Path(ik_right)), "left": CC.load_ik(Path(ik_left))}
    q0 = np.asarray(np.load(tree_dir / "0/trajectory_kinematic.npz")["qpos"][0], float)
    layout = CC.HandLayout(tree_dir, urdf=urdf, quiet=True)
    # An unbuildable collider or a main object without a shape dies inside CrossSetModel — no pass on nothing.
    model = CC.CrossSetModel(layout, tree_dir=tree_dir, assets_root=assets_root, geom="collision")
    model.pairs = f0_pairs(model)  # this model measures ③⑤⑥ only
    model._pf = None  # (the bounding-sphere prefilter is rebuilt for these pairs)
    fq, obj = layout.hand_blocks(q0)
    model.set_pose({s: (ik[s][0], ik[s][1][0]) for s in ("right", "left")}, fq, obj, ik["right"][2])
    _mins, _h, rows = model.distances(1e-3, sets=F0_SETS)
    deep: dict = {}
    for sn, cls, a, b, d, _ft in rows:
        if d < 0.0:
            k = (sn, cls, a, b)
            deep[k] = max(deep.get(k, 0.0), -d * 1000.0)
    pairs = sorted((k + (round(v, 2),) for k, v in deep.items()), key=lambda r: -r[4])
    thr = set_thresholds()
    return {
        "thresh_mm": thr,
        "worst_mm": {sn: max([r[4] for r in pairs if r[0] == sn], default=0.0) for sn in F0_SETS},
        "pairs": pairs,
        "violations": violations(pairs, thr),
        "hand_ownobj_mm": hand_ownobj_mm(pairs),
    }


# ── The gate's frame-0 rules (J-gate-f0pen ⑤ · J-gate-f0objobj ③) — reason form and the declared data property, next
#    to the measuring function above; gate.core applies them per side (_f0_reason · _term_gate · _verdict). The
#    unmeasurable branches and the systematic stop are the gate's non-verdict rule: ledger_states.measured (a declared
#    data property = a reason · any other exception = GateUnmeasured, both sides ERROR) · ledger_states.error_systematic
#    (gate.checks.chunk_rc's F0 term -> [GATE_ERROR_FAIL]).
#: The only measuring exception declared as a data property for J-gate-f0pen is the objects.hull contract's HullMissing
#: (the object USD has no rigid body / collision mesh) — a reason (FAIL -> GUARD_FAIL). Every other exception (invalid
#: cooked hull, part mismatch, missing file, import ...) is not a data verdict: both sides of the clip are ERROR
#: (ledger_states.GATE_VERDICT_STATE -> BAKE_ERROR, re-bake). HullMissing also covers "no USD" (environment); many of
#: those trip the systematic gate.


def _data_exc() -> tuple:
    from dexengine.data.objects import hull as OH  # noqa: PLC0415 — owner of the declared exception classes

    return (OH.HullMissing,)


def gate_reason(tree, out, assets_root, n_f0) -> tuple[str | None, dict]:
    """J-gate-f0pen: the spawn pose must not put an arm collider (the sim shape, incl. the wrist hull) deeper than
    coords.F0_ARM_OBJ_PEN_M inside any main object (set ⑤). PhysX resolves it as a pop the policy never caused, and no
    placement moves it (the wrist comes from the data). A scene property at spawn -> the caller fails both sides.

    A clip it cannot measure goes through ledger_states.measured — a declared data property (_data_exc) fails with
    that reason; any other exception raises ledger_states.GateUnmeasured (both sides ERROR) — never a silent pass.
    Set ⑥ (hand <-> own object) is reported only (coords.F0_HAND_OWNOBJ_MODE = "report") — counted in the [F0_PEN]
    marker, never a reason; its depth per side is the second value, {side: mm} (a gate table column; {} = not
    measured). The caller is gate.checks._f0_reason."""
    if not (tree.is_dir() and out["ik_right"].is_file() and out["ik_left"].is_file()):
        return None, {}  # no tree: the gate's reads die on it · no arm IK: that side is ERROR
    from dexengine.placement.coords import core as _coords

    if not n_f0.get("said"):
        n_f0["said"] = True
        print(
            f"[F0_PEN] thresh_mm={set_thresholds()} (None = reported only) "
            "(f0_pen.frame0_penetration · sim collider shapes)",
            flush=True,
        )
    f0, why = LS.measured(
        n_f0, "F0_pen", _data_exc(), lambda: frame0_penetration(tree, out["ik_right"], out["ik_left"], assets_root)
    )
    if why:
        return why, {}
    n_f0["hand_report"] += f0["worst_mm"]["6_hand_ownobj"] > _coords.F0_ARM_OBJ_PEN_M * 1000.0
    return f0_reason_text(f0["violations"], n_f0), f0["hand_ownobj_mm"]


#: The gate reason head of each frame-0 set that can violate — J-gate-f0objobj (③) · J-gate-f0pen (⑤). Gate-table data.
F0_REASON_HEAD = {"3_obj_obj": "F0_obj_obj", "5_arm_obj": "F0_pen"}


def f0_reason_text(v: list, n_f0: dict) -> str | None:
    """The reason of the frame-0 violations (rows of violations()) — one head per set (F0_REASON_HEAD; a set
    without a head dies) · up to three pairs each · counted in n_f0 per set."""
    if not v:
        return None
    out = []
    for sn in dict.fromkeys(r[0] for r in v):
        if sn not in F0_REASON_HEAD:
            raise ValueError(f"F0 set {sn!r} has no gate reason head — F0_REASON_HEAD")
        rs = [r for r in v if r[0] == sn]
        n_f0[f"violation_{sn}"] = n_f0.get(f"violation_{sn}", 0) + 1
        out.append(
            F0_REASON_HEAD[sn]
            + ":"
            + ",".join(f"{a}~{b}{d:.0f}mm" for _s, _c, a, b, d in rs[:3])
            + (f"(+{len(rs) - 3})" if len(rs) > 3 else "")
        )
    n_f0["violation"] += 1
    return ";".join(out)
