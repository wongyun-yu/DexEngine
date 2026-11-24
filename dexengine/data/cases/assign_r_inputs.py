"""Inputs of rule R (assign_r): reads the source statements (OakInk2 program_info · TACO source folder names) and builds
the R input F.

Same version as assign_r (dexengine.data.cases.assign_version.ASSIGN2_VERSION covers assign_r, assign_r_frames and this
module). R's decision depends on the source statements and these inputs, so a change in how they are read must move the
ASSIGN-2 version; that is why the reading lives here and not in the case stage (cases.taxonomy, under another version).
assign_r and assign_r_frames hold pure functions only. Unreadable input is an error; there is no silent path that skips
the source layer and uses the measurement alone.
  OakInk2  clip name → sequence hash · interval number → the stage's per-hand object lists in program_info (owner of
           the interval order) (assign_r.oak_native).
  TACO     the file names of the source folder `(<verb>, <tool>, <target>)/<sequence>/{tool_<id>,target_<id>}.npy` give
           the roles (raw_roles).
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path

import numpy as np

from dexengine.core import paths as P
from dexengine.data.cases import assign_r as AR2  # rule R (source-layer readers oak_native · taco_native)
from dexengine.data.cases import assign_r_frames as ARF  # frame inputs of R (hand points, objects, hand–object pairs)
from dexengine.data.cases import hand_object_assign as H  # owner of object units (units_of)


# TACO source statement.
def triplet_name(dirname: str) -> str:
    """Raw triplet folder `(<verb>, <tool>, <target>)` → clip head `<verb>_<tool>_<target>` with spaces as `-`.

    The rule runs forward over the folder list (one direction only): the clip head cannot be split back, because a verb
    may hold a hyphen already (`stir-fry`) or spaces (`pour in some` → `pour-in-some`)."""
    return dirname[1:-1].replace(", ", "_").replace(" ", "-")


@functools.lru_cache(maxsize=1)
def triplets() -> dict:
    """{clip head: source folder}; the root belongs to side_objects (_raw_object_poses). A missing root is an
    error."""
    from dexengine.data import side_objects as SO  # noqa: PLC0415 — owner of the TACO source object-pose root

    root = SO._raw_object_poses()
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is missing — cannot read the TACO source tool/target (file names)")
    out: dict = {}
    for d in root.iterdir():
        if d.is_dir():
            k = triplet_name(d.name)
            if k in out:
                raise ValueError(f"two TACO source folders read as the same clip head {k!r} ({out[k].name} · {d.name})")
            out[k] = d
    return out


def raw_roles(base_seq: str) -> dict[str, str] | None:
    """Role → mesh id as the TACO **raw data** gives it: {'tool': id, 'target': id}.

    The raw data puts the role **in the file name**:
    `Object_Poses/(<verb>, <tool>, <target>)/<seqid>/{tool_<id>,target_<id>}.npy`.
    So which object is the tool and which is the target never has to be inferred — it is
    in the data (confirmed). None if the sequence folder is absent; the raw root itself missing raises.
    """
    head, _, sid = base_seq.partition("__")
    trip = triplets().get(head)
    d = trip / sid if trip is not None else None
    if d is None or not d.is_dir():
        return None
    out: dict[str, str] = {}
    for f in d.glob("*.npy"):
        if f.name.endswith(".grid50.npy"):
            continue
        role, _, mid = f.stem.partition("_")
        out[role] = mid
    return out if {"tool", "target"} <= set(out) else None


# OakInk2 source statement.
@functools.lru_cache(maxsize=1)
def oak_program_index() -> dict:
    """OakInk2 source program_info (owner of the interval order): {12-char sequence hash: file}. A missing folder is an
    error."""
    d = P.layer_path("raw", "oakink2") / "hf" / "program" / "program_info"
    if not d.is_dir():
        raise FileNotFoundError(
            f"{d} is missing — cannot read rule R's OakInk2 source statement (per-hand object lists)"
        )
    out = {}
    for f in d.glob("*.json"):
        m = re.search(r"seq__([0-9a-f]+)__", f.name)
        if m:
            out[m.group(1)[:12]] = f
    return out


def oak_entries(task: str) -> list:
    """program_info (key, info) list of the clip's sequence (key order = interval order). Fails on an unreadable name or
    a missing file."""
    m = re.match(r"oakink2_.+?__scene_\d+_([0-9a-f]{12})_s\d+$", task)
    if not m:
        raise ValueError(
            f"OakInk2 clip name {task!r}: cannot read the sequence hash (the measurement is not used alone "
            "without the source layer)"
        )
    f = oak_program_index().get(m.group(1))
    if f is None:
        raise FileNotFoundError(
            f"OakInk2 program_info has no sequence {m.group(1)} ({task}) — source statement missing"
        )
    return list(json.loads(Path(f).read_text()).items())


# R input F.
def parts(sz) -> dict:
    """Articulated surface samples → {"base"|"moving"|"all": (points, KD tree)}; part coordinates of rule R
    (assign_r_frames._q_part)."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    keys = ("points",) if "points" in sz.files else ("base_points", "moving_points")
    return {("all" if k == "points" else k.split("_")[0]): (sz[k], cKDTree(sz[k])) for k in keys}


def native(ds: str, ti: dict, F: dict) -> list:
    """Read rule R's source statements into F["nat"] and return the task object names that must always be in the scene
    (TACO tool and target)."""
    units, members = F["units"], F["members"]
    obj_id_of = {str(o["name"]): str(o["obj_id"]) for o in ti.get("objects") or []}
    if ds == "oakink2":
        from dexengine.data.oakink2 import (
            parts as OP,  # noqa: PLC0415 — owner of part identity (official object_part_tree)
        )

        task = ti.get("task") or ti["clip_id"]
        sg = ti.get("segment") or {}
        F["nat"] = {
            "oak": AR2.oak_native(
                task,
                int(sg.get("start", 0) or 0),
                int(((sg.get("frame_map") or {}).get("offset", 0)) or 0),
                F["T"],
                F["fps"],
                units,
                members,
                obj_id_of,
                oak_entries(task),
                OP.rid_of,
                OP.root_of,
            )
        }
        return []
    if ds == "taco":
        parent = ti.get("parent_clip") or ti["clip_id"]
        roles = raw_roles(parent)
        if roles is None:
            raise FileNotFoundError(
                f"TACO source tool/target statement missing ({parent}) — "
                "rule R always keeps the source task objects in the scene"
            )
        t = AR2.taco_native(roles, units, members, obj_id_of)
        F["nat"] = {"taco": t}
        return [*members[t["tool_unit"]], *members[t["target_unit"]]]
    return []


def build(
    ds: str, ti: dict, objs: list, hands: list, T: int, fps: float, kpm: dict, mano: dict, thr: float, objsrc: dict
) -> tuple[dict, list]:
    """R input F and the task object names, from what the case stage (cases.taxonomy) has read already (hand points
    kpm · MANO sidecar mano · object pose and surface objsrc = {object: (o7, joint, joint column, small object, surface
    samples)})."""
    F = {"T": T, "fps": fps, "hands": hands, "pair": {}, "obj": {}, "nat": {}}
    for n, (o7, art, hinge, small, sz) in objsrc.items():
        F["obj"][n] = ARF.object_features(o7, fps, small, art, hinge, parts(sz) if art else None)
    for s in hands:
        P6 = ARF.hand_points(kpm["tips"][s], kpm["wrist"][s])
        z, oidx, _ = mano[s]
        for n in objsrc:
            oi = oidx[n]
            F["pair"][(s, n)] = ARF.pair_features(
                P6,
                np.asarray(z["dist"][:, oi, :], float),
                np.asarray(z["palm_dist"][:, oi], float),
                F["obj"][n],
                fps,
                thr,
            )
    F["units"], F["members"] = H.units_of(objs)
    return F, native(ds, ti, F)
