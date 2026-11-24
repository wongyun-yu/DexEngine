"""Rule S-desk-lite: the scene desk is the dataset's fixed box, so a clip whose recorded hand goes under the desk top
inside that box cannot be played on it; objects standing outside the box are flagged.

Why: the desk top is one box per dataset (`dexengine.core.desk.desk_top_for`), not the captured table. Where the
captured table was smaller or ended earlier, the person's hand passed below the table height at a place the scene
desk still covers, and the robot hand following that reference would pass through the desk. The judge's arm guard
measures the arm links (wrist housing included) against the desk; the hand itself (palm, fingers) is not measured
there, so this gate reads the recorded hand.

Measurement, on a baked desk tree (aerial trees have no desk: out of scope):
  hand     every manipulating hand's base (the wrist, `qpos_layout.hands`) over every tree frame, in the scene frame
           (capture → scene rotation, z + `scene_z_offset`)
  below    inside the desk box (x · y, margin 0) and lower than the desk top by more than the D14 depth
           (`candidate_rules.WRIST_BELOW_M`, the same depth the first pass reads on the MANO wrist)
  gate     any frame below → the clip is gated (reason `desk_lite:<side> <depth> mm @<frame>`)
  flag     a spawned object (main objects, support objects, slotted extras) whose frame-0 footprint lies more than
           `place_rule`'s off-desk fraction outside the box is listed (`off_desk`); a flag, not a gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RULE_ID = "S-desk-lite"


def _coords():
    from dexengine.placement.coords import core as C  # noqa: PLC0415

    return C


def on(ds: str) -> bool:
    """Is S-desk-lite on for dataset `ds` (rules registry scope `default:scene_rules`)?"""
    from dexengine.data import rules as RR  # noqa: PLC0415

    return ds in RR.ALL and RR.rule_on(RULE_ID, ds)


def hand_below(tree_dir, ti: dict, q: np.ndarray) -> dict:
    """{side: {depth_m: lowest wrist z − desk top inside the box (None when never inside), frame}} of the
    manipulating hands."""
    from dexengine.core import paths as P  # noqa: PLC0415

    C = _coords()
    lay = P.qpos_layout.read(str(tree_dir), nq=q.shape[1])
    (x0, x1), (y0, y1) = C.desk_bounds_scene_for(tree_dir, margin=0.0, ti=ti)
    top, dz = float(C.desk_height()), float(C.scene_z_offset(tree_dir))
    out = {}
    for side in lay.manip_sides():
        h = int(lay.hands[side])
        xy = C.capture_to_scene(q[:, h : h + 2].astype(np.float64))
        z = q[:, h + 2].astype(np.float64) + dz - top
        inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
        if not inside.any():
            out[side] = {"depth_m": None, "frame": None}
            continue
        f = int(np.flatnonzero(inside)[np.argmin(z[inside])])
        out[side] = {"depth_m": round(float(z[f]), 4), "frame": f}
    return out


def off_desk(tree_dir, ti: dict, q: np.ndarray) -> list[str]:
    """Spawned objects whose frame-0 footprint is off the desk box (place_rule's measure and threshold)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.placement.scene import place_rule as PR  # noqa: PLC0415
    from dexengine.placement.scene import static_desk as SD  # noqa: PLC0415

    C = _coords()
    placed, _ = C._placers(tree_dir, ti, q)
    slotted = {r["name"] for r in XS.clip_rows(str(tree_dir), ti, q, strict=False) if r["cls"] in XS.SLOTTED}
    box = C.desk_bounds_scene_for(tree_dir, margin=0.0, ti=ti)
    out = []
    for o in ti.get("objects") or []:
        if not o.get("mesh_dir") or o.get("qpos_start") is None:
            continue
        if o.get("role") in XS.SPAWNED_ROLES or str(o["name"]) in slotted:
            xy = C.capture_to_scene(PR.footprint(placed(o, 0)[:, :2]))
            if SD.overhang_frac(xy, box=box) > SD.OVERHANG_MAX_FRAC:
                out.append(str(o["name"]))
    return out


def measure(tree_dir, ti: dict | None = None, q: np.ndarray | None = None) -> dict:
    """{scope, hands, gate, why, off_desk} of one baked tree (module doc). scope False (aerial, or the rule off for the
    dataset) → no measurement, gate False."""
    from dexengine.data import candidate_rules as CR  # noqa: PLC0415 — owner of the D14 depth

    tree_dir = Path(tree_dir)
    if ti is None:
        ti = json.loads((tree_dir / "task_info.json").read_text(encoding="utf-8"))
    ds = str(ti.get("dataset_name") or "")
    from dexengine.placement.scene import place_rule as PR  # noqa: PLC0415

    if not on(ds) or not PR.in_scope(tree_dir, ti):
        return {"scope": False, "hands": {}, "gate": False, "why": "", "off_desk": []}
    if q is None:
        q = np.load(tree_dir / "0" / "trajectory_kinematic.npz")["qpos"]
    hands = hand_below(tree_dir, ti, q)
    low = [(s, h) for s, h in hands.items() if h["depth_m"] is not None and h["depth_m"] < CR.WRIST_BELOW_M]
    why = " ".join(f"{RULE_ID}:{s} {h['depth_m'] * 1000:.0f}mm@{h['frame']}" for s, h in low)
    return {"scope": True, "hands": hands, "gate": bool(low), "why": why, "off_desk": off_desk(tree_dir, ti, q)}
