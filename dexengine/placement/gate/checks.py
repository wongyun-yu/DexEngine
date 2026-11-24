"""Per-clip verdict helpers of the final gate (gate.core): the speed ratio, the support report, the scene-term wrapper,
the per-side verdict, the height contract and the chunk's systematic-failure rc.

The reason strings these helpers put into a side's ``why`` / ``err`` end up in the gate table's reason column (data):
keep their bytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dexengine.placement import ledger_states as LS
from dexengine.placement.collision import support_surface as SSF
from dexengine.placement.coords import core as _VC
from dexengine.placement.gate import f0_pen as F0P
from dexengine.placement.judge import input_record as _JFP
from dexengine.placement.retime import stretch as VST

#: The gate's non-verdict failures follow one rule (ledger_states.measured, plus the artifact-defect term): when a scene
#: term (J-gate-f0pen · J-gate-fingerrange) cannot be measured, only an exception declared as a property of the data
#: is a reason (FAIL -> GUARD_FAIL) — for f0 the objects.hull contract's HullMissing (no rigid body / collision mesh in
#: the object USD), for the finger range none. Every other exception (invalid cooked hull, scene compile failure,
#: missing file, import ...) and every artifact defect of a bake-OK clip (no arm IK, no contact label, finger column
#: length, left/right or tree generation mismatch) is not a data verdict: that side is ERROR
#: (ledger_states.GATE_VERDICT_STATE -> BAKE_ERROR, re-bake). The classification lives in ledger_states because two
#: measuring functions use it (this package and f0_pen.gate_reason).
#: Systematic gate: when a term's unmeasurable count or the artifact defects are ledger_states.error_systematic, the
#: table is written and the gate returns GATE_RC_SYSTEMATIC with a final [GATE_ERROR_FAIL] line — the stage-2 driver
#: stops the chunk and writes no state. A single clip never stops the chunk.
GATE_RC_SYSTEMATIC = 3


def systematic_rc(terms: dict) -> int:
    """The chunk's non-verdict failures per term ({head: counts}) — any systematic (ledger_states.error_systematic) ->
    one [GATE_ERROR_FAIL] line per term + GATE_RC_SYSTEMATIC, else 0."""
    rc = 0
    for head, n in terms.items():
        bad, tried = n["unmeasurable"], n["unmeasurable"] + n["measured"]
        if LS.error_systematic(bad, tried):
            print(
                f"[GATE_ERROR_FAIL] {head} {bad}/{tried} (undeclared exceptions {n.get('error', 0)}) — systematic "
                f"(>= {LS.ERROR_SYSTEMATIC_MIN} clips · all or >= {LS.ERROR_SYSTEMATIC_FRAC:.0%}) · not a clip "
                f"verdict · chunk rc {GATE_RC_SYSTEMATIC}",
                flush=True,
            )
            rc = GATE_RC_SYSTEMATIC
    return rc


def speed_ratio(q, arm_budget, fingers, finger_budget):
    """J-gate-speed for one side's final motion: per frame the worst of the arm joints and that side's finger joints
    (stretch.frame_ratio — the definition the retiming stretched with) -> (p99, max, p99 label, max label, reason); a
    label names the finger group when it set the value. A finger column of another length (another time axis) has no
    ratio: NaN (fails every threshold) + its own reason — never a silent pass."""
    if len(fingers) != len(q):
        return float("nan"), float("nan"), "", "", f"finger rows length≠joints({len(fingers)}≠{len(q)})"
    if len(q) < 2:
        return 0.0, 0.0, "", "", ""
    r, r_f = VST.frame_ratio(q, arm_budget), VST.frame_ratio(fingers, finger_budget)
    r_all = np.maximum(r, r_f)
    lab = (
        "(fingers)" if np.percentile(r_f, 99) > np.percentile(r, 99) else "",
        "(fingers)" if r_f.max() > r.max() else "",
    )
    return float(np.percentile(r_all, 99)), float(r_all.max()), *lab, ""


def _support_report(clip, side, ctx, q, fq, zoff, sup, why, n_sup) -> float | None:
    """[SUPPORT_GUARD] counts of one side: a guard failure on a support object, and hand <-> support at the spawn frame
    (support_surface.hand_support_mm — reported, never a reason; [SUPPORT_HAND] per side with a depth). Returns that
    depth in mm for the table column (None = the clip has no support object)."""
    from dexengine.placement.collision import core as _SCL

    n_sup["arm_fail"] += _SCL.SUPPORT_PEN_WORD in why
    if not sup:
        return None
    mm, lk, ob = SSF.hand_support_mm(ctx, q[:1], fq[:1], zoff, sup)
    n_sup["hand_report"] += mm > _SCL.DESK_SURF_PEN_MM
    if mm > 0.0:
        print(f"[SUPPORT_HAND] {clip} {side} {lk}~{ob} {mm:.1f}mm (reported only)", flush=True)
    return mm


def _term_gate(res, reason) -> None:
    """One scene term (J-gate-f0pen · J-gate-fingerrange — ``reason`` = a thunk calling its reason function) on both
    sides: a reason fails them; an undeclared measuring exception (GateUnmeasured) marks them ERROR — not a data
    verdict (a real FAIL elsewhere still wins, _verdict)."""
    try:
        _fail_both(res, reason())
    except LS.GateUnmeasured as e:
        for side in ("right", "left"):
            _err(res[side], str(e))


def _err(d, msg) -> None:
    """A non-verdict failure on one side — the side goes ERROR unless it FAILs."""
    d["err"] = ((d["err"] + ";") if d.get("err") else "") + msg


def _f0_reason(res, tree, out, assets_root, n_f0) -> str | None:
    """J-gate-f0pen's reason (f0_pen.gate_reason, under ledger_states.measured); its hand <-> own-object depth per side
    goes to res[side]["f0own"] (a gate table column, reported only)."""
    why, own = F0P.gate_reason(tree, out, assets_root, n_f0)
    for side, mm in own.items():
        res[side]["f0own"] = mm
    return why


def _verdict(d) -> tuple[str, str]:
    """A side's (verdict, reason) — words = ledger_states.GATE_VERDICT_STATE: FAIL beats ERROR (an undeclared measuring
    exception only) beats PASS; the ERROR reason follows the side's own reasons."""
    return (
        "FAIL" if not d["ok"] else "ERROR" if d.get("err") else "PASS",
        ";".join(x for x in (d["why"], d.get("err")) if x),
    )


def _fail_both(res, msg) -> None:
    """A scene property failed — both sides FAIL and carry the reason (no-op for None)."""
    if not msg:
        return
    for side in ("right", "left"):
        res[side]["ok"] = False
        res[side]["why"] = ((res[side]["why"] + ";") if res[side]["why"] else "") + msg


def chunk_rc(n_f0, n_fr, n_height, n_art) -> int:
    """The chunk's rc after the table — systematic_rc over the gate's non-verdict terms (one head each)."""
    return systematic_rc(
        {
            "F0 penetration unmeasurable": n_f0,
            "finger range unmeasurable": n_fr,
            "height unmeasurable": n_height,
            "artifact defect": n_art,
        }
    )


def height_reason(tree, fp, n_h: dict) -> str | None:
    """Height contract of an aerial baked tree: a re-stamp that recomputed the floor proxy from a trajectory a later
    step moved would put the scene at a height the robot was not solved for — a reason fails both sides.

    No fingerprint (a segment that inherited its placement without a judge — input_record.no_fingerprint) is counted,
    not a reason. Every read of an existing fingerprint runs inside ledger_states.measured, so one it cannot read
    (another schema, an empty setup field) or a missing aerial stamp is GateUnmeasured (both sides ERROR)."""
    if not (Path(tree).is_dir() and _VC.scene_mode(tree) == "aerial"):
        return None
    if _JFP.no_fingerprint(fp):
        n_h["no_fingerprint"] += 1
        return None
    got, why = LS.measured(n_h, "height", (), lambda: _JFP.height_violation(tree, fp))
    if why:
        return why
    n_h["bad"] += bool(got)
    return got


def _joint_model(xml: Path):
    """The scene's kinematic tree compiled by MuJoCo with every mesh geom a small sphere and no mesh or texture asset:
    a geom carries no joint, so joint names, types, ranges and qpos addresses are the full scene's (names, jnt_type,
    jnt_range and jnt_qposadr checked equal) at ~0.05 s instead of compiling ~180 meshes."""
    import mujoco as mj

    s = mj.MjSpec.from_file(str(xml))
    for g in s.geoms:
        if g.type == mj.mjtGeom.mjGEOM_MESH:
            g.type, g.size, g.meshname = mj.mjtGeom.mjGEOM_SPHERE, [0.01, 0.0, 0.0], ""
    for x in list(s.meshes) + list(s.textures):
        s.delete(x)
    for ma in s.materials:
        ma.textures = [""] * len(ma.textures)
    return s.compile()
