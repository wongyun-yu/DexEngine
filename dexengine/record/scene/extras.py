"""Extra objects of the record render (visual only): single owner of the rule, poses, reasons, the rollout overlap gate
and the spawn.

Extra objects = tree objects[] with role=extra (objects not spawned in the training and judging scene) + the
render-only list (render_only.tree_rows, row field source: original objects outside the tree; missing dies). Only the
stage-2 render session (render.episode) puts those passing the rule as **visual-only prims** (no physics API, constant
pose); the training and judging env and its USD stay as they are. The rule is per object (dataset names are not
read). An object is dropped for the first reason that hits:
  in_tree · no_asset  (list) an object the tree holds · no mesh asset of the same mesh
  no_pose        no trajectory place in the tree (qpos_start)
  in_sim         the training scene already spawns it (extra slots on; rule owner extra_slots.slot_rows): a visual
                 prim too would draw it twice
  aerial         not a desk scene (coords.scene_mode)
  part           the object is an articulation (objects[].articulation; a visual prim cannot carry joint state). A part
                 group sibling that is a main object or hand-touched is not a reason by itself: parts merged into
                 another body are not in the merged tree's objects[]
  moving         over the tree's frames, a move >= STATIC_MOVE_M or a rotation >= STATIC_ROT_DEG relative to f0
  off_desk       the xy centre of the f0 surface samples (static_desk.surface) lies outside the canonical desk box
                 (static_desk.desk_box)
  penetrate      f0 lowest point - desk top < -DESK_TOL_M
  desk_overhang  area outside the desk > threshold (canonical on; 0 = off for comparison; owner extras_rules)
  hand_near      the target robot hand (ref_hand_geom.RefHands: FK of the tree reference hand meshes) comes within
                 static_desk.HAND_NEAR_M during the clip
  float_dropped  dropped by the float rule (static_desk.float_rule)
  rollout_overlap (episode gate, render session) some render frame has the nearest distance between the real rollout
                 robot link surface samples and the object surface samples < ROLL_CLEAR_M: dropped in that episode
                 only
  occlude        (episode gate) it occludes a moving object (extras_rules)
Objects the float rule lowered are put at the lowered pose (z - float). Pose = tree f0 -> coords.support_scene_pose
(the same scene transform as supports) · USD = the render cover (dexengine.core.artifact_paths.render_spawn_usds;
without one the training USD, noted in the marker) of the training USD (dexengine.data.objects.hull.usd_file).
Oracle (measured vs stated): the main objects' tree f0 scene position vs the scene object slot position at step 0
(ORACLE_TOL_M) · the spawned prim's world bounds bottom and xy centre vs the surface samples (SPAWN_TOL_M); beyond them
it dies.
Marker ``[EXTRAS_VISUAL] <clip> put n · dropped m (per reason …) · extras k · source <source> put · dropped (…) …``,
printed even with 0 extras (session.launch requires it). Meta = render_meta.extras_visual -> the episode row
extras_visual (store.meta.render_fix).
The CPU decision (decide) runs without Isaac; self-test ``python3 dexengine/record/scene/extras.py --tree <tree> …``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from dexengine.record.scene import extras_rules as OCC

if TYPE_CHECKING:  # the render session imports this module before AppLauncher: numpy inside functions only ([PRE_APP])
    import numpy as np

MARK = "[EXTRAS_VISUAL]"
ROOT = "/World/CollectLook/extras"
STATIC_MOVE_M = 0.02  # static: max move relative to f0
STATIC_ROT_DEG = 5.0  # static: max rotation relative to f0
DESK_TOL_M = 0.02  # penetration threshold (f0 lowest point below the desk top)
ROLL_CLEAR_M = 0.01  # nearest rollout robot surface to object surface threshold
ROLL_MEASURE_M = 0.10  # robot points within the object box + this margin are all measured · with none, a sparse
# sample is measured (1/this; true value >= margin: being farther than the threshold shows in the box already)
ROLL_FAR_STRIDE = 16
ROBOT_SAMPLE_M = 0.005  # robot link mesh surface sample spacing
ORACLE_TOL_M = 0.02  # main object f0 scene position vs scene slot at step 0
SPAWN_TOL_M = 0.02  # spawned prim world bounds vs surface samples (bottom z, xy centre)
REASONS = (
    "no_pose",
    "in_sim",
    "in_tree",
    "no_asset",
    "aerial",
    "part",
    "moving",
    "off_desk",
    "penetrate",
    OCC.OVERHANG_REASON,
    "hand_near",
    "float_dropped",
    "rollout_overlap",
    OCC.REASON,
)
DRAWN_ELSEWHERE = ("in_sim", "in_tree")  # reasons whose object is drawn elsewhere anyway; the others appear nowhere
SOURCES = ("tree", "render_only")  # row sources: tree objects[] role=extra · render-only list (render_only)
PUT_WORD, SOURCE_WORD = " put ", " · source "  # marker words; log_problems checks them


class ExtrasError(RuntimeError):
    """The rule or oracle cannot be applied: never passes silently."""


def rule() -> dict:
    """Thresholds stored in the meta (values owned by this file's constants; float and hand thresholds by
    static_desk)."""
    from dexengine.placement.scene import static_desk as SD

    return (
        {
            "static_move_m": STATIC_MOVE_M,
            "static_rot_deg": STATIC_ROT_DEG,
            "desk_tol_m": DESK_TOL_M,
            "hand_near_m": SD.HAND_NEAR_M,
            "float_max_m": SD.FLOAT_MAX_M,
            "roll_clear_m": ROLL_CLEAR_M,
            "roll_measure_m": ROLL_MEASURE_M,
            "robot_sample_m": ROBOT_SAMPLE_M,
        }
        | OCC.overhang_rule_fields()
        | OCC.rule_fields()
    )


def _kind(o: dict) -> str:
    """'' = an independent object · part (the object is an articulation). Being touched by the human hand is not a
    reason: cells whose training scene had no extras put them as visuals like cells with them on (moved objects are
    dropped by moving, robot overlap by hand_near and rollout_overlap)."""
    return "part" if o.get("articulation") else ""


def _motion(q: np.ndarray, st: int) -> tuple[float, float]:
    import numpy as np

    pos = q[:, st : st + 3].astype(float)
    qw = q[:, st + 3 : st + 7].astype(float)
    qw = qw / np.linalg.norm(qw, axis=1, keepdims=True)
    rot = np.degrees(2 * np.arccos(np.clip(np.abs((qw * qw[0]).sum(1)), 0, 1)))
    return float(np.linalg.norm(pos - pos[0], axis=1).max()), float(rot.max())


def scene_points(ds: str, o: dict, pose, z_off: float) -> np.ndarray:
    """Object surface samples (static_desk.surface) -> 7-field pose (capture, wxyz) -> scene (env-frame) coordinates
    (N, 3)."""
    from dexengine.placement.coords import core as C
    from dexengine.placement.scene import static_desk as SD

    P = C.capture_to_scene(SD._cap_points(pose, SD.surface(ds, o)))
    P[:, 2] += z_off
    return P


def scene_frame(pose, z_off: float):
    """The body → scene (env-frame) rigid transform of `scene_points` as (R (3,3), c (3,)): scene = R p + c. Rule
    S-prox reads it (penetration is measured in the static's body frame, `overlap.penetration`)."""
    import numpy as np

    from dexengine.placement.coords import core as C
    from dexengine.placement.scene import static_desk as SD

    A = C.capture_to_scene(SD._cap_points(pose, np.vstack([np.zeros(3), np.eye(3)])))
    c = A[0].copy()
    c[2] += z_off
    return (A[1:] - A[0]).T, c


def _collect_rows(extras, rows, ob, in_sim, q, tree_dir, ro_rows):
    """One row per tree extra (pose, motion) and per render-only row; ob = name -> object."""
    for o in extras:
        r = {"name": str(o["name"]), "obj_id": str(o.get("obj_id")), "source": "tree", "reason": ""}
        rows.append(r)
        ob[r["name"]] = o
        if o.get("qpos_start") is None:
            r["reason"] = "no_pose"
            continue
        if r["name"] in in_sim:
            r["reason"] = "in_sim"
            continue
        st = int(o["qpos_start"])
        if q.shape[1] < st + 7:
            raise ExtrasError(f"{tree_dir.name}: {r['name']} qpos_start {st} + 7 > width {q.shape[1]}")
        r["move_m"], r["rot_deg"] = (round(v, 4) for v in _motion(q, st))
        r["pose"] = [float(v) for v in q[0, st : st + 7]]
    for r in ro_rows:
        if r["name"] in ob:
            raise ExtrasError(f"{tree_dir.name}: render-only row name {r['name']} overlaps a tree object")
        rows.append(r)
        ob[r["name"]] = {"name": r["name"], "obj_id": r["obj_id"]}


def _desk_reasons(rows, mode, ob, ds, z_off, desk, bx0, bx1, by0, by1):
    """Reasons from the scene, object kind, motion and desk (off desk, penetrating, overhang)."""
    for r in rows:
        if r["reason"]:
            continue
        if mode != "table":
            r["reason"] = "aerial"
            continue
        kind = _kind(ob[r["name"]])
        if kind:
            r["reason"] = kind
            continue
        if r["move_m"] >= STATIC_MOVE_M or r["rot_deg"] >= STATIC_ROT_DEG:
            r["reason"] = "moving"
            continue
        P = scene_points(ds, ob[r["name"]], r["pose"], z_off)
        cx, cy = (float(v) for v in P[:, :2].mean(0))
        r.update(f0_cx=round(cx, 4), f0_cy=round(cy, 4), f0_gap_m=round(float(P[:, 2].min()) - desk, 4))
        if not (bx0 <= cx <= bx1 and by0 <= cy <= by1):
            r["reason"] = "off_desk"
        elif r["f0_gap_m"] < -DESK_TOL_M:
            r["reason"] = "penetrate"
        else:
            OCC.overhang_judge(r, P[:, :2], ((bx0, bx1), (by0, by1)))


def _hand_reasons(rows, tree_dir, ti, ds, ob):
    """Rows a reference hand comes near (hand_near)."""
    from dexengine.placement.scene import static_desk as SD

    live = [r for r in rows if not r["reason"]]
    if live:
        from dexengine.placement.scene import ref_hand_geom as RH

        ref = RH.RefHands(tree_dir, ti)
        for r in live:
            pts = (
                ref.object_points(r["name"], r["pose"])
                if r["source"] == "tree"
                else SD._cap_points(r["pose"], SD.surface(ds, ob[r["name"]]))
            )
            dmin, k, side = ref.min_dist(pts)
            r.update(hand_min_m=round(float(dmin), 4), hand_frame=int(k), hand_side=side)
            if dmin <= SD.HAND_NEAR_M:
                r["reason"] = "hand_near"


def _float_rows(rows, ti, objs, ob, tree_dir, z_off, ds):
    """Float rule of the live rows: dropped, or lowered to the render pose."""
    from dexengine.placement.coords import core as C
    from dexengine.placement.scene import static_desk as SD

    live = [r for r in rows if not r["reason"]]
    if live:
        fl = [{"name": r["name"], "pose": r["pose"], "moving": False, "surf": r["source"] != "tree"} for r in live]
        # list object surfaces
        ti2 = dict(ti, objects=[*objs, *(ob[r["name"]] for r in rows if r["source"] != "tree")])
        for r, dec in zip(live, SD.float_rule(tree_dir, ti2, fl), strict=True):
            r.update(float_action=dec["action"], lowered_dz_m=round(float(dec["dz_m"]), 4))
            if dec["action"] == SD.FLOAT_DROPPED:
                r["reason"] = "float_dropped"
                continue
            pr = [*r["pose"][:2], r["pose"][2] - float(dec["dz_m"]), *r["pose"][3:]]
            xyz, wxyz = C.support_scene_pose(pr, z_off)
            r.update(
                pose_render=pr,
                scene_xyz=[float(v) for v in xyz],
                scene_wxyz=[float(v) for v in wxyz],
                usd=_usd(ds, r["obj_id"]),
            )


def decide(tree_dir, ti: dict | None = None) -> dict:
    """CPU decision (header rules, rollout gate excluded) -> {scene, z_off, rows, objs, render_only}. Row = one extra
    object (tree, then list order); reason '' means a candidate to put (fields scene_xyz, scene_wxyz, pose_render,
    usd). objs = {row name: object}."""
    import numpy as np

    from dexengine.data.objects import extra_slots as XS
    from dexengine.placement.coords import core as C
    from dexengine.placement.scene import static_desk as SD
    from dexengine.record.scene import render_only as RO

    tree_dir = Path(tree_dir)
    if ti is None:
        ti = json.loads((tree_dir / "task_info.json").read_text(encoding="utf-8"))
    objs = ti.get("objects") or []
    extras = XS.extra_objects(ti)
    mode = C.scene_mode(tree_dir)
    out = {"scene": mode, "z_off": None, "rows": [], "objs": {}, "render_only": None}
    q = np.load(tree_dir / "0" / "trajectory_kinematic.npz")["qpos"]
    ro_rows, out["render_only"] = RO.tree_rows(tree_dir, ti, q, ORACLE_TOL_M)
    if not extras and not ro_rows:
        return out
    ds = str(ti["dataset_name"])
    z_off = out["z_off"] = float(C.scene_z_offset(tree_dir)) if mode == "table" else None
    desk = float(C.desk_height())
    (bx0, bx1), (by0, by1) = SD.desk_box(tree_dir)
    in_sim = (
        {r["name"] for r in XS.slot_rows(str(tree_dir), ti, strict=False)[0]} if XS.requested() and extras else set()
    )
    rows, ob = [], out["objs"]
    _collect_rows(extras, rows, ob, in_sim, q, tree_dir, ro_rows)
    _desk_reasons(rows, mode, ob, ds, z_off, desk, bx0, bx1, by0, by1)
    _hand_reasons(rows, tree_dir, ti, ds, ob)
    _float_rows(rows, ti, objs, ob, tree_dir, z_off, ds)
    out["rows"] = rows
    return out


def _usd(ds: str, obj_id: str) -> str:
    import os

    from dexengine.data.objects import hull as OH

    usd = str(OH.usd_file(ds, obj_id, False))
    if not os.path.isfile(usd):
        raise ExtrasError(f"{obj_id}: no training USD: {usd}")
    return usd


def summary(rows: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """(put, dropped, count per reason). Unknown reasons die."""
    if bad := sorted({r["reason"] for r in rows if r["reason"] and r["reason"] not in REASONS}):
        raise ExtrasError(f"unknown reasons {bad}")
    put = [r for r in rows if not r["reason"]]
    drop = [r for r in rows if r["reason"]]
    return put, drop, dict(Counter(r["reason"] for r in drop))


def by_source(rows: list[dict]) -> dict:
    """{source: {put, drop, by_reason}}; unknown sources die."""
    if bad := sorted({r.get("source") for r in rows} - set(SOURCES), key=str):
        raise ExtrasError(f"unknown sources {bad}")
    out = {}
    for src in SOURCES:
        put, drop, by = summary([r for r in rows if r["source"] == src])
        out[src] = {"put": len(put), "drop": len(drop), "by_reason": by}
    return out


def marker(clip: str, rows: list[dict]) -> str:
    put, drop, by = summary(rows)
    why = " ".join(f"{k} {v}" for k, v in sorted(by.items())) or "-"
    src = " · ".join(
        f"{k}{PUT_WORD}{v['put']} dropped {v['drop']} ("
        + (" ".join(f"{a} {b}" for a, b in sorted(v["by_reason"].items())) or "-")
        + ")"
        for k, v in by_source(rows).items()
    )
    return f"{MARK} {clip}{PUT_WORD}{len(put)} · dropped {len(drop)} ({why}) · extras {len(rows)}{SOURCE_WORD}{src}"


def log_problems(text: str) -> list[str]:
    # one line must carry the marker, put and source (proof that the list was consumed)
    ok = any(ln.startswith(f"{MARK} ") and PUT_WORD in ln and SOURCE_WORD in ln for ln in text.splitlines())
    miss = [] if ok else [f"extra object marker '{MARK} …{PUT_WORD}…{SOURCE_WORD}' missing"]
    return miss + OCC.overhang_log_problems(text)


_GEN = [0]  # spawn call count: generation of the prim names


def prim_path(gen: int, k: int) -> str:
    """First generation e<k> (record meta name) · later generations e<k>_g<gen>: hanging another USD straight on the
    same
    path leaves the old prim behind."""
    return f"{ROOT}/e{k}" + (f"_g{gen}" if gen else "")


def mesh_bound(prim) -> tuple[np.ndarray, np.ndarray]:
    """World bounds of the drawn mesh vertices (the BBoxCache aligned range uses rotated box corners, which lowers the
    bottom)."""
    import numpy as np

    from pxr import Usd, UsdGeom

    from dexengine.record.render import body_geom as BG

    P = BG.mesh_points(prim, purposes=(UsdGeom.Tokens.default_, UsdGeom.Tokens.render))
    if not len(P):
        raise ExtrasError(f"{MARK} no drawn mesh vertices under {prim.GetPath()}")
    W = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    V = P @ W[:3, :3].T + W[:3, 3]
    return V.min(0), V.max(0)


def spawn(rows: list[dict], pts: dict, origin: np.ndarray, shift_z: float) -> list[dict]:
    """Visual-only prim per row to put (the same UsdFileCfg function as the training spawn, physics API removed); world
    bounds checked against the surface samples."""
    import numpy as np

    import omni.usd
    from pxr import UsdGeom

    import isaaclab.sim as su

    from dexengine.core import artifact_paths as TP
    from dexengine.record.looks import look_d as LOOK

    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, ROOT)
    usds, fallback = TP.render_spawn_usds([r["usd"] for r in rows])
    placed, gen = [], _GEN[0]
    _GEN[0] += 1
    for k, (r, u) in enumerate(zip(rows, usds, strict=True)):
        path = prim_path(gen, k)
        xyz = np.asarray(r["scene_xyz"], float) + origin + np.array([0.0, 0.0, shift_z])
        cfg = su.UsdFileCfg(usd_path=u)
        cfg.func(path, cfg, translation=tuple(float(v) for v in xyz), orientation=tuple(r["scene_wxyz"]))
        removed, left = LOOK._strip_physics(stage.GetPrimAtPath(path))
        if left:
            raise ExtrasError(f"{MARK} {r['name']}: {left} physics APIs left")
        lo, hi = mesh_bound(stage.GetPrimAtPath(path))
        P = pts[r["name"]] + origin + np.array([0.0, 0.0, shift_z])
        err = {
            "bottom_m": float(lo[2] - P[:, 2].min()),
            "center_xy_m": float(np.linalg.norm((lo[:2] + hi[:2]) / 2 - (P[:, :2].min(0) + P[:, :2].max(0)) / 2)),
        }
        if max(abs(v) for v in err.values()) > SPAWN_TOL_M:
            raise ExtrasError(
                f"{MARK} {r['name']} spawn bounds vs surface samples {err} > {SPAWN_TOL_M} m: USD units, "
                "axes or pose are wrong"
            )
        placed.append(
            {
                "prim": path,
                "usd": u,
                "train_usd_fallback": r["usd"] in fallback,
                "physics_api_removed": removed,
                "bbox_world": [lo.round(4).tolist(), hi.round(4).tolist()],
                "spawn_err_m": {k2: round(v, 4) for k2, v in err.items()},
            }
        )
    return placed


# ── self-test (CPU) ──────────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="CPU decision of the extra objects (rollout gate excluded): one line + rows per tree"
    )
    ap.add_argument("--tree", action="append", required=True, help="tree folder (<clip>_pl_rt); repeatable")
    ap.add_argument(
        "--render-only-root", default="", help="read the render-only list from this root (generator --out dry run)"
    )
    a = ap.parse_args()
    if a.render_only_root:
        from dexengine.record.scene import render_only as RO

        RO.read_from(a.render_only_root)
    for t in a.tree:
        dec = decide(t)
        print(
            marker(Path(t).name, dec["rows"]) + f" · scene {dec['scene']} · render only {dec['render_only']}",
            flush=True,
        )
        for r in dec["rows"]:
            print(
                f"  {r['name']:<28} {r['obj_id']:<28} {r['source']:<11} {r['reason'] or 'PUT':<14} "
                + " ".join(
                    f"{k}={r[k]}"
                    for k in (
                        "move_m",
                        "rot_deg",
                        "f0_gap_m",
                        "f0_cx",
                        "f0_cy",
                        "hand_min_m",
                        "float_action",
                        "lowered_dz_m",
                        OCC.OVERHANG_FIELD,
                    )
                    if k in r
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
