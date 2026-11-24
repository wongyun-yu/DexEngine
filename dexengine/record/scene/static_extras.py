"""Display-only static extras at the render entry: called by the campaign physics render and the check videos (the clip
boundary of rl.render.clip_video).

Of the static extras not standing in the training and judging scene, those the record rule (scene.extras.decide) puts
are placed in the render in display-only form (no physics API: hands and objects pass without pushing them). Wider
than extra_slots VISUAL_ONLY: static objects dropped from the training scene by the robot keep-out rule (STATIC
robot_keepout) stand too when decide puts them. Which object, where, and how it spawns is decided by the one owner it
shares with the record render, scene.extras (decide, spawn); this file only reads that and swaps the prims per clip
(no second classification). One difference from the record render: there is no rollout record, so the rollout overlap
gate (scene.extras_apply.rollout_overlap) does not run.
After the spawn this file counts the physics APIs under the prims again in its own way (it does not trust the spawn's
claim of removal); any left dies.
Prim paths get a new name per clip (owner scene.extras.prim_path: hanging another USD straight on the same path leaves
the previous clip's object meshes and materials on screen).
Marker ``[RENDER_STATIC] <clip> n=<put> no collision (physics API <removed> removed · 0 left) · not drawn n=<count
standing nowhere> (per reason) · in the training scene <count> · extras k``, printed even without extras (a render
without the marker did not go through this entry). The overhang marker ``[EXTRAS_OVERHANG] on …``
follows on one line (extras_rules; the rule is always on).
Tests: tests/record/test_static_extras.py (pxr only, no Isaac).
"""

from __future__ import annotations

import json
from pathlib import Path

MARK = "[RENDER_STATIC]"


class RenderStaticError(RuntimeError):
    """The entry cannot apply, or physics is left on a prim that must be display-only: never passes silently."""


def tree_of(seq_dir, clip: str) -> Path:  # noqa: V103 — used by dexengine/rl/render/clip_video.py
    """That clip's tree under the session tree folder (--sequence_dir); missing dies (which objects to put is
    unknown)."""
    from dexengine.core import paths as P

    tree = Path(seq_dir or "") / (P.clip_name(clip) + P.TREE_SFX)
    if not seq_dir or not (tree / "task_info.json").is_file():
        raise RenderStaticError(
            f"{MARK} {clip}: tree {tree} has no task_info.json (--sequence_dir {seq_dir!r}): the "
            "display-only static extras cannot be decided"
        )
    return tree


def physics_left(prim) -> list[str]:
    """Physics APIs (collision, rigid body) left under the prim, as a 'path:API' list. An independent measurement (does
    not call the spawn's removal function)."""
    from pxr import Usd, UsdPhysics

    apis = (
        UsdPhysics.CollisionAPI,
        UsdPhysics.MeshCollisionAPI,
        UsdPhysics.RigidBodyAPI,
        UsdPhysics.ArticulationRootAPI,
    )
    return [f"{sub.GetPath()}:{api.__name__}" for sub in Usd.PrimRange(prim) for api in apis if sub.HasAPI(api)]


def assert_visual_only(prim) -> None:
    left = physics_left(prim)
    if left:
        raise RenderStaticError(f"{MARK} {len(left)} physics APIs left on a prim that must be display-only: {left[:3]}")


def nowhere(rows: list[dict]) -> tuple[int, str]:
    """Extras standing nowhere (neither in the training scene nor drawn by the render) -> (count, 'reason n …'). Owner
    of
    the reason names and of the 'drawn elsewhere' reasons = scene.extras (REASONS, DRAWN_ELSEWHERE)."""
    from dexengine.record.scene import extras as EXV

    _put, drop, _by = EXV.summary(rows)
    by: dict = {}
    for r in drop:
        if r["reason"] not in EXV.DRAWN_ELSEWHERE:
            by[r["reason"]] = by.get(r["reason"], 0) + 1
    return sum(by.values()), " ".join(f"{k} {v}" for k, v in sorted(by.items())) or "-"


def put_points(ds: str, dec: dict, put: list) -> dict:
    """{row name: scene points} of the rows to put, each read through its object in `dec["objs"]` (scene.extras.decide:
    row name → object, render-only rows included — they are not in the tree's objects[])."""
    from dexengine.record.scene import extras as EXV

    return {r["name"]: EXV.scene_points(ds, dec["objs"][r["name"]], r["pose_render"], dec["z_off"]) for r in put}


def apply(raw_env, tree_dir, clip: str) -> dict:
    """Clip boundary (before that clip's first frame): remove the previous clip's prims and put this clip's -> one
    marker line. Returns the put and dropped rows."""
    import omni.usd

    from dexengine.record.scene import extras as EXV
    from dexengine.record.scene import extras_rules as OCC

    tree_dir = Path(tree_dir)
    ti = json.loads((tree_dir / "task_info.json").read_text(encoding="utf-8"))
    dec = EXV.decide(tree_dir, ti)
    rows = dec["rows"]
    put, drop, _by = EXV.summary(rows)
    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath(EXV.ROOT).IsValid():
        stage.RemovePrim(EXV.ROOT)
    placed = []
    if put:
        origin = raw_env.scene.env_origins[0].detach().cpu().numpy().astype(float)
        placed = EXV.spawn(put, put_points(str(ti["dataset_name"]), dec, put), origin, 0.0)
        assert_visual_only(stage.GetPrimAtPath(EXV.ROOT))
    n_no, why = nowhere(rows)
    print(
        f"{MARK} {clip} n={len(placed)} no collision (physics API {sum(p['physics_api_removed'] for p in placed)} "
        f"removed · 0 left) · not drawn n={n_no} ({why}) · in the training scene {len(drop) - n_no} · extras "
        f"{len(rows)} · objects {[r['obj_id'] for r in put]} · rule scene.extras.decide (no rollout overlap gate)",
        flush=True,
    )
    OCC.overhang_say(clip, rows)
    return {
        "placed": [dict(name=r["name"], obj_id=r["obj_id"], **p) for r, p in zip(put, placed, strict=True)],
        "dropped": [{"name": r["name"], "reason": r["reason"]} for r in drop],
    }
