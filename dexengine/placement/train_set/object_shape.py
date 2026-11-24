"""Object identity and shape of a training clip — the object key the bundles group by (`target_object`, from the base
layer's main object) and the shape family that says which keys are similar (`key_shape`, from the object's surface
points). Called by the bundle generator (select.partition); bundle_pack owns the rule that turns extents into a
family.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from dexengine.core.errors import InputMissingError
from dexengine.core.paths import FIELD_SEP, TAKE_SEP, layer_path
from dexengine.data import rules as RR
from dexengine.placement.train_set import bundle_pack as BP

_OBJ_CACHE: dict = {}


def obj_key_marker(dataset: str, emb: str, hand: str) -> str:
    """The object-key marker line `[OBJ_KEY] dataset=… obj_key=…` — printed by every caller that keys objects."""
    return (
        f"[OBJ_KEY] dataset={dataset} emb={emb} hand={hand} obj_key={RR.DATASET_PROFILE[dataset].obj_key} "
        "(base = mesh_dir of the base-layer main object (role main) · clip = last task field of the clip id) — "
        "owner datasets.DATASET_PROFILE"
    )


def target_object_file(clip_id: str, dataset: str, emb: str, hand: str = "") -> Path | None:
    """What target_object reads: the base task_info (obj_key "base"), None for "clip" — also stated as a campaign
    input by bundle/run_inputs.py. An unknown obj_key dies."""
    key = RR.DATASET_PROFILE[dataset].obj_key  # an unknown dataset dies here with KeyError
    if key not in ("base", "clip"):
        raise SystemExit(f"[select] unknown obj_key {key!r} ({dataset}) — set base|clip in DATASET_PROFILE")
    return (
        None
        if key == "clip"
        else (layer_path("base", dataset, hand=hand or RR.hand_dir(), emb=emb) / clip_id / "task_info.json")
    )


def target_object(clip_id: str, dataset: str, emb: str, hand: str = "") -> str:
    """The object key that groups a training set. Its source is declared by
    `datasets.DATASET_PROFILE[dataset].obj_key`:

      "clip"  the last task field of the clip id (`brush_brush_bowl` → `bowl`) — the TACO convention, where that
              field is the object.
      "base"  the mesh directory name (`objects[].mesh_dir`, else the top-level `<name>_mesh_dir`) of the main object
              (`objects[].role == "main"`) of the BASE-layer task_info. Two main objects that differ are joined as
              `A+B` (sorted, distinct). No main object dies.

    The base layer is where roles are declared; a tree's roles are a copy from bake time. Outside TACO the last task
    field is a subject, a session time, a verb or a hash — not an object."""
    if target_object_file(clip_id, dataset, emb, hand) is None:  # obj_key "clip" — the clip id says it
        return clip_id.split(TAKE_SEP, 1)[0].split(FIELD_SEP)[-1]
    return "+".join(main_meshes(clip_id, dataset, emb, hand))


def main_meshes(clip_id: str, dataset: str, emb: str, hand: str = "") -> list[str]:
    """The clip's main object mesh ids (sorted, distinct) from the **base** layer task_info — the objects whose applied
    role is main (objects[].mesh_dir or `<name>_mesh_dir`). The one read behind the object key (`target_object`, base
    mode) and the shape family. Unreadable or no main object → dies."""
    if (ck := (dataset, hand or RR.hand_dir(), emb, clip_id)) in _OBJ_CACHE:
        return _OBJ_CACHE[ck]
    ti = target_object_file(clip_id, dataset, emb, hand)
    if ti is None:
        raise InputMissingError(f"[select] obj_key of {dataset} is not base — no main object mesh ({clip_id})")
    if not ti.is_file():
        raise InputMissingError(f"[select] base-layer task_info missing (cannot read the object key): {ti}")
    t = json.loads(ti.read_text())
    mains = [o for o in (t.get("objects") or []) if o.get("role") == "main"]
    names = set()
    for o in mains:
        md = o.get("mesh_dir") or t.get(f"{o.get('name')}_mesh_dir")
        if md:
            names.add(Path(str(md)).name)
    if not names:
        raise SystemExit(
            f"[select] cannot resolve the main object in the base layer "
            f"(role=main {[o.get('name') for o in mains]}): {ti}"
        )
    _OBJ_CACHE[ck] = out = sorted(names)
    return out


_EXT_CACHE: dict = {}


def shape_files(dataset: str, key: str) -> list[Path]:
    """What the shape family of an object key reads (`object_extents` of each member): <meshes layer>/<mesh>/
    surface_points.npz for a base key, nothing for a clip key — also stated as a campaign input by
    bundle/run_inputs.py (reader verify:shape)."""
    if RR.DATASET_PROFILE[dataset].obj_key != "base":
        return []
    return [layer_path("meshes", dataset) / m / "surface_points.npz" for m in key.split("+")]


def object_extents(dataset: str, mesh: str) -> tuple[float, float, float]:
    """PCA extents (bundle_pack.pca_extents) of one object's surface points (path: `shape_files`). Missing →
    dies."""
    if (dataset, mesh) not in _EXT_CACHE:
        import numpy as np

        f = shape_files(dataset, mesh)[0]
        if not f.is_file():
            raise InputMissingError(f"[select] no surface points to measure the shape family: {f}")
        _EXT_CACHE[(dataset, mesh)] = BP.pca_extents(np.load(f)["points"])
    return _EXT_CACHE[(dataset, mesh)]


def key_shape(dataset: str):
    """(family(key), near(k1, k2)) for one dataset's object keys. Base keys name meshes: family = bundle_pack.family_of
    over the members' extents, near = distance of log extents (the largest member of each key). Clip keys (TACO — the
    key is the object category, no mesh named): the key is its own family, near = 0."""
    base = RR.DATASET_PROFILE[dataset].obj_key == "base"

    def ext(m):
        return object_extents(dataset, m) if base else None

    def kext(k):
        return max(object_extents(dataset, m) for m in k.split("+")) if base else None

    def family(k):
        return BP.family_of(k, ext)

    def near(a, b):
        if not base:
            return 0.0
        ea, eb = kext(a), kext(b)
        if min(ea + eb) <= 0:
            raise ValueError(
                f"shape distance: object of zero thickness ({a} {ea} · {b} {eb}) — the surface points "
                "are flat or a line (no log)"
            )
        return sum((math.log(x) - math.log(y)) ** 2 for x, y in zip(ea, eb, strict=True)) ** 0.5

    return family, near
