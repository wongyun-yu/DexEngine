#!/usr/bin/env python3
"""Render-only scene object list of the record: single owner of its shape and reading (writer render_only_build;
reader scene.extras.decide).

List = objects that are in the original capture scene but not in the tree's objects[] (the record render shows every
scene object as a visual mesh; the tree stays as it is and the training and judging env do not read this list). The
rule (what goes into the list) is owned by the generator's module doc.
Shape: layer data_layout 'render_only' (data/<ds>/render_only): index.json {version, dataset, axis_hand,
  clips: {clip: object count}}
  + for clips with objects > 0 a <clip>.json {version, dataset, clip, axis_hand, base_tree, window, how,
  oracle, objects[{name, rid, obj_id, reason, mesh, pose (base row 0 x·y·z · wxyz), move_m, rot_deg, asset_*}],
  anchors[]}.
Reading (read_clip): no index · another version · clip missing from the index · count > 0 without a file · count
  mismatch = dies. The read root = the canonical layer; the only way to change it is read_from (explicit argument of
  the CPU decision CLI, marked).
To the tree (tree_rows): into the world of the tree consuming the record (axis hand base row 0):
  coords.place_pose(tree_placement) -> for a placement-augment variant tree (task_info aug_place) the variant transform
  (variant_tf = augment.scene.record_tf) is chained · anchor oracle (anchors = anchor_starts: object root pose blocks,
  articulation base links included; the base tree's base row 0 -> the same chain vs that row's qpos of this tree; max
  > tol or 0 anchors dies). Check qpos = anchor_ref: a tree that went through cross repair (xr, cross_repair: moves
  objects held by carry_object with the hand by <= 30 mm) uses the pre-retiming row of its input copy
  (wrist_shift._bk(kin, "pre_xr")) (coords._base_to_tree(src_idx=None), f0 padding P), otherwise the
  coords._base_to_tree(0) row of the final qpos (f0 padding, retiming map). The record holds the pre-xr pose, so
  checking against the final one reads the xr move as a mismatch (the same rule as dexengine.data.contact.derive G2a;
  raising the threshold blinds the oracle) · rids this tree already holds = in_tree
  (readers/raw/scene_classify.tree_scene, obj_key of the dataset's measuring module; not drawn twice). Row field source
  = render_only · pose and static measurements as recorded.
Measuring module: dexengine.data.readers.raw.<dataset> (oracle_module) — the dataset is its one owner. A key_module
  field in older records names the module that wrote them and is not read.
"""

from __future__ import annotations

import json
from pathlib import Path

MARK = "[RENDER_ONLY]"
VERSION = 2
INDEX = "index.json"
LAYER = "render_only"
NAME_PREFIX = "render_only_"  # row and prim name (never clashes with tree objects[] names)
REASONS = ("no_pose", "no_asset")  # reasons the generator sets (the same words in the consumer scene.extras.REASONS)


class RenderOnlyError(RuntimeError):
    """The list cannot be made or read."""


# read root: None = the canonical layer. Only read_from changes it (CPU decision CLI argument only)
_READ_ROOT: list = [None]


def read_from(root: str) -> None:
    """Point reading at a dry-run root (generator --out); only when the caller says so, with a marker."""
    _READ_ROOT[0] = str(root)
    print(f"{MARK} read root = {root} (explicit, not the canonical layer)", flush=True)


# ── places ────────────────────────────────────────────────────────────────────
def list_dir(ds: str, root: str | None = None) -> Path:
    from dexengine.core import paths as P

    if root is None:
        return P.layer_path(LAYER, ds)
    return Path(root) / P.data_layout.render(LAYER, ds)


def clip_file(ds: str, clip: str, root: str | None = None) -> Path:
    return list_dir(ds, root) / f"{clip}.json"


def read_clip(ds: str, clip: str, root: str | None = None) -> dict:
    """That clip's record (objects [] when 0 objects). Dies on the conditions in the module doc's 'Reading'."""
    ip = list_dir(ds, root) / INDEX
    if not ip.is_file():
        raise RenderOnlyError(
            f"{ds}: render-only object index missing {ip}: run the generator (scene/render_only_build.py) first"
        )
    idx = json.loads(ip.read_text(encoding="utf-8"))
    if idx.get("version") != VERSION:
        raise RenderOnlyError(f"{ip}: version {idx.get('version')} != {VERSION}")
    if clip not in idx["clips"]:
        raise RenderOnlyError(f"{ds}/{clip}: clip not in the index: run the generator for that clip again")
    n = int(idx["clips"][clip])
    if n == 0:
        return {"version": VERSION, "dataset": ds, "clip": clip, "objects": [], "anchors": []}
    f = clip_file(ds, clip, root)
    if not f.is_file():
        raise RenderOnlyError(f"{ds}/{clip}: the index says {n} objects but the file is missing {f}")
    rec = json.loads(f.read_text(encoding="utf-8"))
    if len(rec["objects"]) != n or rec.get("version") != VERSION:
        raise RenderOnlyError(
            f"{f}: objects {len(rec['objects'])} · version {rec.get('version')} != index {n} · {VERSION}"
        )
    return rec


def render_inputs(ti: dict) -> tuple[list[Path], list[str]]:
    """What a render reads from this layer for the tree's clip (read_clip · tree_rows · scene.extras): ([the index,
    the clip record when it lists objects, the anchor frame files, each listed object's surface samples — scene.extras
    places it on the desk through static_desk.surface], [the training USD of each listed object — the render spawns it
    as its overlay]). Read through read_clip, so a missing or stale index dies here too."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data.contact import fingerprint as SF  # noqa: PLC0415 — the surface sample place static_desk reads
    from dexengine.data.objects import hull as OH  # noqa: PLC0415

    ds, clip = str(ti["dataset_name"]), str(ti["clip_id"])
    rec = read_clip(ds, clip)
    if not rec["objects"]:
        return [list_dir(ds) / INDEX], []
    files = [list_dir(ds) / INDEX, clip_file(ds, clip), *(Path(f) for f in frame_files(ti, rec))]
    files += [SF.surface_path(P.main_root(), ds, o) for o in rec["objects"]]
    return files, [str(OH.usd_file(ds, str(o["obj_id"]), False)) for o in rec["objects"]]


def anchor_starts(tree_dir, ti: dict, nq: int) -> dict[str, int]:
    """Anchor rule (shared by generator and consumer): objects[] whose qpos_start is the start column of a 7-wide object
    block (root rigid pose) of that tree's qpos layout, {name: start column}. Articulations have their root pose (base
    link) in such a block too and ride the same coordinate chain; joint angles are separate 1-wide blocks, not anchors.
    Truth of the blocks = qpos_layout.layout_of(...); blocks() that do not cover qpos without gaps die."""
    from dexengine.core import paths as P

    lay = P.qpos_layout.layout_of(str(tree_dir), ti, nq=nq)
    lay.blocks()
    starts = set(lay.object_starts())
    return {
        str(o["name"]): int(o["qpos_start"])
        for o in ti.get("objects") or []
        if o.get("qpos_start") is not None and int(o["qpos_start"]) in starts
    }


def anchor_ref(tree_dir, ti: dict, q) -> tuple:
    """Anchor check reference (module doc 'To the tree') -> (qpos, row, source 'pre_xr' | 'kinematic'). With an xr copy,
    its pre-retiming row; a column count different from the final dies. Without one (tree without xr) the final qpos's
    base row 0. The source is kept as a meta marker."""
    import numpy as np

    from dexengine.placement.coords import core as C
    from dexengine.placement.repair import wrist_shift as WS

    tree_dir = Path(tree_dir)
    xr, _meta = WS._bk(tree_dir / "0" / "trajectory_kinematic.npz", "pre_xr")
    if not xr.is_file():
        return q, int(C._base_to_tree(tree_dir, ti)(0)), "kinematic"
    qx = np.load(xr)["qpos"]
    if np.shape(qx)[1] != np.shape(q)[1]:
        raise RenderOnlyError(f"{tree_dir.name}: xr copy {xr.name} columns {np.shape(qx)[1]} != final {np.shape(q)[1]}")
    return qx, int(C._base_to_tree(tree_dir, ti, T=len(qx), src_idx=None)(0)), "pre_xr"


def variant_tf(ti: dict):
    """For a placement-augment variant tree (task_info aug_place) that variant's tree coordinate transform
    (augment.scene.record_tf: the same transform and owner as the rigid_any that built the variant tree); None for an
    original tree. A variant is a rigid move of the whole scene, so the render-only objects move with it (the variant
    task_info's placement record tree_placement is the original's)."""
    if ti.get("aug_place") is None:
        return None
    from dexengine.record.augment import scene as aug_scene

    return aug_scene.record_tf(ti)


def frame_pairs(ti: dict, rec: dict) -> list[tuple[str, str, str]]:
    """Anchors whose identity must be decided by the key module's frame_key (rigid frame) = (name, record id, tree id):
    only anchors with the same name in the tree and an id that differs as text (same id = same object). The tree_rows
    check and the lane statement (frame_files) read this one list."""
    tid = {str(o["name"]): str(o.get("obj_id")) for o in ti.get("objects") or []}
    return [
        (a["name"], a["obj_id"], tid[a["name"]])
        for a in rec["anchors"]
        if a["name"] in tid and tid[a["name"]] != a["obj_id"]
    ]


def oracle_module(ds: str):
    """The raw-capture scene reader of dataset ds (dexengine.data.readers.raw.<ds>); a dataset without one dies."""
    import importlib
    import importlib.util

    name = f"dexengine.data.readers.raw.{ds}"
    if importlib.util.find_spec(name) is None:
        raise RenderOnlyError(f"{ds}: no raw scene reader {name}")
    return importlib.import_module(name)


def frame_module(rec: dict):
    """The record's measuring module; missing anchor identity (frame_key) or its inputs (frame_inputs) dies."""
    km = oracle_module(rec["dataset"])
    if not (hasattr(km, "frame_key") and hasattr(km, "frame_inputs")):
        raise RenderOnlyError(
            f"{rec['clip']}: key module {km.__name__} has no frame_key or frame_inputs: the "
            "identity of anchors with differing ids cannot be decided"
        )
    return km


def frame_files(ti: dict, rec: dict) -> list[str]:  # noqa: V103 — public API for plugins (statement lane)
    """Files the tree_rows anchor identity reads (the record lane statement carries them as nodes): frame_inputs for
    both ids of each frame_pairs entry."""
    pairs = frame_pairs(ti, rec)
    if not pairs:
        return []
    km = frame_module(rec)
    return sorted({f for _n, *ids in pairs for i in ids for f in km.frame_inputs(i)})


def _anchor_errors(rec, tobj, roots, tree_dir, pairs, km, errs, to_tree, qa, f0):
    """Each anchor of the record against the tree root pose (none dropped silently); position errors in m."""
    import numpy as np

    for a in rec["anchors"]:  # no anchor is dropped silently: missing, not a root or another frame = dies
        o = tobj.get(a["name"])
        if o is None or a["name"] not in roots:
            raise RenderOnlyError(
                f"{tree_dir.name}: anchor {a['name']} {'is not in the tree' if o is None else 'is not a root block'}"
            )
        if a["name"] not in pairs:  # ids equal as text = the same object (no frame file read)
            fa = ft = a["obj_id"]
        else:
            try:
                fa, ft = (km.frame_key(i) for i in pairs[a["name"]])
            except ValueError as e:
                raise RenderOnlyError(f"{tree_dir.name}: frame of anchor {a['name']} cannot be decided: {e}") from e
        if fa != ft:
            raise RenderOnlyError(
                f"{tree_dir.name}: rigid frame {fa} ({a['obj_id']}) of anchor {a['name']} != tree "
                f"{ft} ({o.get('obj_id')}): same name, different object"
            )
        s0 = roots[a["name"]]
        errs.append(float(np.linalg.norm(np.asarray(to_tree(a["pose"]))[:3] - np.asarray(qa[f0, s0 : s0 + 3], float))))


def tree_rows(tree_dir, ti: dict, q, tol_m: float) -> tuple[list[dict], dict]:
    """Render-only rows of the consumer (scene.extras.decide): moves the record (read_clip) into the world of **this
    tree**. Space: coords.place_pose(base pose, *coords.tree_placement(ti)) -> for a variant tree variant_tf(ti) is
    chained. Oracle: the anchors (base row 0 of the base tree's rigid bodies) are moved by the same chain and checked
    against the anchor_ref qpos (pre-retiming row of the xr copy | final base row 0); max > tol_m or 0 anchors to check
    dies. rids this tree already holds = reason in_tree (readers/raw/scene_classify.tree_scene; not drawn twice).
    -> (rows, {n, anchors, anchor_err_m})."""
    import numpy as np

    from dexengine.data.readers.raw import scene_classify as SC
    from dexengine.placement.coords import core as C

    tree_dir = Path(tree_dir)
    ds, clip = str(ti["dataset_name"]), str(ti["clip_id"])
    rec = read_clip(ds, clip, _READ_ROOT[0])
    if not rec["objects"]:
        return [], {"n": 0, "anchors": 0, "anchor_err_m": None}
    qa, f0, src = anchor_ref(tree_dir, ti, q)
    place = C.tree_placement(ti)
    vt = variant_tf(ti)

    def to_tree(p7) -> list:
        p = [float(v) for v in (C.place_pose(p7, *place) if place is not None else p7)]
        return p if vt is None else vt.pose7(p)

    tobj = {str(o["name"]): o for o in ti.get("objects") or []}
    roots = anchor_starts(tree_dir, ti, int(np.shape(q)[1]))
    km = oracle_module(rec["dataset"])  # owner of keys and anchor identity
    pairs = {n: ids for n, *ids in frame_pairs(ti, rec)}
    if pairs:
        frame_module(rec)
    errs = []
    _anchor_errors(rec, tobj, roots, tree_dir, pairs, km, errs, to_tree, qa, f0)
    if not errs:
        raise RenderOnlyError(f"{tree_dir.name}: 0 anchors to check: no object proves the tree coordinate chain")
    if max(errs) > tol_m:
        raise RenderOnlyError(
            f"{tree_dir.name}: anchor check ({src}) max {max(errs) * 1000:.1f} mm > "
            f"{tol_m * 1000:.0f} mm: the record to tree coordinate chain is off (when the tree was "
            "rebuilt after the record, run the generator again)"
        )
    scene = SC.tree_scene(ti, km.obj_key)
    rows = []
    for ob in rec["objects"]:
        r = {
            "name": str(ob["name"]),
            "obj_id": str(ob["obj_id"] or "-"),
            "rid": str(ob["rid"]),
            "source": "render_only",
            "reason": "in_tree" if ob["rid"] in scene else str(ob["reason"]),
        }
        if ob["pose"] is not None:
            r.update(pose=to_tree(ob["pose"]), move_m=ob["move_m"], rot_deg=ob["rot_deg"])
        rows.append(r)
    return rows, {
        "n": len(rows),
        "anchors": len(errs),
        "anchor_err_m": round(max(errs), 6),
        "anchor_src": src,
        "variant": "aug_place" if vt is not None else None,
    }
