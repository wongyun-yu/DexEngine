#!/usr/bin/env python3
"""Still objects added at render time (assignment gate verdict `valid_static`) — the rule and the poses.

The training scene lacks an object the requirement asks for while that object is still (reference displacement < 1 cm):
training stays as is and the object is added only at render time. Read by the judge's and the staleness check's desk
check (static_desk). What the render actually spawns is decided by the render's extras rule, not here.

Rule      add_names(static_required, tree_ti) = the gate's static_required − the tree's supports (role=support — already
          spawned fixed in support slots). The gate already removed the scene's main objects.
Pose      rows(tree_dir, ti, names) — per object the **tree** qpos frame-0 7-pose (capture x·y·z · wxyz), not the base
          frame 0: the tree (<clip>_pl_rt) is the placed and retimed world every training spawn uses (the base and
          tree poses of one object can be ~0.9 m apart). Scene pose = coords.support_scene_pose(pose,
          coords.scene_z_offset(tree)) — the support objects' function. USD = object hull usd_file (the training USD).
Still     the whole-tree displacement span_m is reported — definition and threshold are the gate's
          (hand_object_assign._span_m · TH_STATIC_SUPPORT_M · max per-axis range). The gate measures its window, this
          the whole tree. span ≥ threshold → moving=True (the reader decides).
Float     rows(float_rule=True) — a floating object the hands do not approach is lowered, one they approach is dropped
          (rule · thresholds · marker [STATIC_FLOAT]: static_desk.float_rule).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

TAG = "[STATIC_ADD]"


class StaticAddError(RuntimeError):
    """The rule cannot be applied — never passed over silently."""


def _static():
    from dexengine.data.cases import hand_object_assign as H  # noqa: PLC0415 — still definition (_span_m) · threshold

    return H._span_m, float(H.TH_STATIC_SUPPORT_M)


def add_names(static_required, tree_ti: dict | None) -> list[str]:
    """The module rule — static_required − the tree's supports. No tree (None) → raises (supports cannot be removed)."""
    if tree_ti is None:
        raise StaticAddError("no tree task_info — cannot remove the supports (role=support)")
    supports = {str(o["name"]) for o in tree_ti.get("objects") or [] if o.get("role") == "support"}
    return [str(n) for n in static_required if str(n) not in supports]


def rows(tree_dir, ti: dict, names, *, scene: bool = True, usd: bool = True, float_rule: bool = False) -> list[dict]:
    """Module "Pose" · "Still" — [{name, obj_id, usd, pose (7 · tree f0 capture), span_m, moving, scene_xyz,
    scene_wxyz}]. scene=False skips the scene transform (coords · z offset; no such fields). usd=False does not resolve
    the USD place (usd '') — for readers that never open the USD (judge · staleness · static_desk, which read surface
    samples, so the USD need not exist on the judging machine).
    float_rule=True (default off): applies the float rule (static_desk.float_rule) and adds per row float (decision
    dict) · float_action · pose_render (z lowered by the float when lowered, else pose); the scene pose follows
    pose_render. Dropped rows are **kept** (marked by float_action — the render skips them). `pose` is always the
    original pose (the value the desk measures use)."""
    names = [str(n) for n in names]
    if not names:
        return []
    tree_dir = Path(tree_dir)
    by = {str(o["name"]): o for o in ti.get("objects") or []}
    if lost := [n for n in names if n not in by]:
        raise StaticAddError(f"{tree_dir.name}: objects to add {lost} are not in the tree's objects")
    if art := [n for n in names if by[n].get("articulation")]:
        raise StaticAddError(
            f"{tree_dir.name}: objects to add {art} are articulated — only still rigid objects are added"
        )
    q = np.load(tree_dir / "0" / "trajectory_kinematic.npz")["qpos"]
    span_of, th = _static()
    out = []
    z_off = None
    for n in names:
        st = int(by[n]["qpos_start"])
        if q.shape[1] < st + 7:
            raise StaticAddError(f"{tree_dir.name}: {n} qpos_start {st} + 7 > width {q.shape[1]}")
        pose = [float(v) for v in q[0, st : st + 7]]
        span = span_of(q[:, st : st + 3])
        out.append(
            {
                "name": n,
                "obj_id": str(by[n]["obj_id"]),
                "usd": _usd(ti, by[n]) if usd else "",
                "pose": pose,
                "span_m": span,
                "moving": span >= th,
            }
        )
    if float_rule:
        from dexengine.placement.scene import static_desk as _sd  # noqa: PLC0415 — the float rule

        for r, dec in zip(out, _sd.float_rule(tree_dir, ti, out), strict=True):
            r.update(
                float=dec,
                float_action=dec["action"],
                pose_render=[*r["pose"][:2], r["pose"][2] - dec["dz_m"], *r["pose"][3:]],
            )
    if scene:
        from dexengine.placement.coords import core as _coords  # noqa: PLC0415 — support pose · z offset

        for r in out:
            if z_off is None:
                z_off = float(_coords.scene_z_offset(tree_dir))
            xyz, wxyz = _coords.support_scene_pose(r.get("pose_render", r["pose"]), z_off)
            r.update(scene_xyz=[float(v) for v in xyz], scene_wxyz=[float(v) for v in wxyz], z_off=z_off)
    return out


def _usd(ti: dict, o: dict) -> str:
    from dexengine.data.objects import hull as _oh  # noqa: PLC0415 — the training USD path

    usd = str(_oh.usd_file(str(ti["dataset_name"]), str(o["obj_id"]), False))
    if not os.path.isfile(usd):
        raise StaticAddError(f"{o['name']} has no USD: {usd}")
    return usd
