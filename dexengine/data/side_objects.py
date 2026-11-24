#!/usr/bin/env python3
"""The **single source of truth** for hand↔object assignment — (mesh id, qpos slot) is
resolved in exactly one place.

WHY (measured)
    Two conventions for the assignment:

      A) **fixed slot** — right = qpos slot 0 = the first object in scene.xml.
         Used by: the retargeting staging task_info · the env trajectory loader ·
         the tree bake (which copies the staging task_info).

      B) **measured grasp** — an earlier generator (not shipped)
             assigned by the fingertip-to-surface
         sustained contact rate. Used by the V1 base trees' task_info.json (dexengine.core.layout.roster_root).

    They agree by accident on most clips (31 of 32) and diverge where the human holds the tool
    **in the left hand**: reading the mesh from B but the trajectory and the contact from A makes
    the two disagree (e.g. `pour in some_kettle_bowl__20231019_009`: the right hand grasps the bowl
    (191) at a sustain rate of 1.00, the left hand holds the kettle 199 at 0.73, but the scene order
    is [199,191] — the env would spawn and score the kettle as the 'right object', and the right hand
    stirs thin air while nobody touches the kettle).

    **B is the measurement, so B is canonical.** This module resolves B into one
    (mesh, slot) pair and every consumer looks only here — the convention is never
    duplicated into a consumer (pipeline doc §3-6).

⚠ `side_assign` was **always** recorded by that V1 generator (default grasp), and
    `assignment_provenance()` below makes "measured / unmeasured" queryable — a tree
    without the field was never measured (it carries the A-family displacement heuristic, which
    swaps left and right wholesale on the clips where "the tool is in the right hand" breaks).
    A gate that looks at the **geometry** rather than the format sits in the canonical loader
    (record.viz.mano_gate.taco_alignment_gate) — a format-agreement check is a tautology (it compares three
    copies derived from one file), so it cannot catch this failure.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from dexengine.core.errors import InputMissingError

#: `--help` is an operator-facing string, so it stays Korean while
#: the docstring above is English. They have different readers: the docstring is for
#: whoever edits this file, `--help` is for whoever runs it. Wiring argparse to `__doc__`
#: would merge the two.
HELP = "The **single canonical** hand↔object assignment: resolves (mesh ID, qpos slot) in one place."

SIDES = ("right", "left")


#: The raw TACO layout the pipeline reads, relative to the data root: stage 1's lift writes it from the TACO source
#: (taco_lift — shifted Hand_Poses, the source's Object_Poses and object_models_released). It is the **one place**
#: TACO object poses are read from (stage-3 cases, the scene-object oracle readers/raw/taco); the public TACO
#: Object_Poses are not read (they differ from the released tracks — taco_source).
TACO_STAGE_RAW_REL = "data/taco_retarget/raw/taco"


def taco_stage_raw() -> Path:
    from dexengine.core import (
        paths as _P,  # noqa: PLC0415 — accessor of the data root (the caller's _path_* already set sys.path)
    )

    return _P.main_root() / TACO_STAGE_RAW_REL


def _raw_object_poses() -> Path:
    return taco_stage_raw() / "Object_Poses"


def scene_object_ids(tree_dir: Path) -> list[str]:
    """The object mesh ids scene.xml references — **order of appearance = the order of
    the qpos object slots**.

    Measured (R32 32/32): this order is exactly the raw data's
    **[tool, target]**. So slot 0 = tool, slot 1 = target is a dataset invariant (not an
    inference).
    ⚠ But "the right hand holds the tool" is **not an invariant** — the corpus holds 45
    clips where the human held the tool in the left hand. For hand↔object the measured
    grasp (task_info) is canonical.
    """
    xml = (Path(tree_dir) / "scene.xml").read_text()
    ids: list[str] = []
    for m in re.finditer(r"objects/(\d+)/visual\.obj", xml):
        if m.group(1) not in ids:
            ids.append(m.group(1))
    return ids


def scene_visual_ids(tree_dir: Path) -> tuple[str, str]:
    """(right, left) = the mesh ids of scene.xml's `right_visual` and `left_visual` meshes: the objects qpos object
    slots 0 and 1 carry (convention A). Measured over the 2,317 V1 TACO trees: always the scene order
    (scene_object_ids). A scene without both dies."""
    xml = (Path(tree_dir) / "scene.xml").read_text()
    got = dict(re.findall(r'<mesh name="(right|left)_visual" file="objects/(\d+)/visual\.obj"', xml))
    if set(got) != set(SIDES):
        raise InputMissingError(f"[side_objects] {tree_dir}: scene.xml has no right_visual and left_visual mesh {got}")
    return got["right"], got["left"]


def resolve(tree_dir: Path) -> dict[str, tuple[str, int]]:
    """{'right': (mesh id, slot), 'left': (mesh id, slot)}.

    Matches the task_info mesh assignment (B) onto the scene order (= the slot). For the
    standard layout that is right→0 / left→1, byte-identical to the previous behaviour;
    only the left-hand-tool clips get their slots flipped. If it cannot be decided (not
    exactly 2 objects, or task_info missing), the standard layout is used.
    """
    tree_dir = Path(tree_dir)
    ids = scene_object_ids(tree_dir)
    ti_p = tree_dir / "task_info.json"
    ti = json.loads(ti_p.read_text()) if len(ids) == 2 and ti_p.is_file() else None
    return resolve_ids(ids, ti, tree_dir.name)


def resolve_ids(ids: list[str], ti: dict | None, name: str) -> dict[str, tuple[str, int]]:
    """`resolve` from its two inputs: the scene object ids in slot order and the task_info (None = none) of clip
    `name` — the one rule, for a tree and for the TACO source pack (taco_source) alike."""
    if len(ids) != 2 or ti is None:
        return {"right": (ids[0] if ids else "", 0), "left": (ids[1] if len(ids) > 1 else "", 1)}
    out: dict[str, tuple[str, int]] = {}
    for side in SIDES:
        v = ti.get(f"{side}_object_mesh_dir")
        if not v:
            return {"right": (ids[0], 0), "left": (ids[1], 1)}
        mid = str(v).rstrip("/").split("/")[-1]
        if mid not in ids:
            raise InputMissingError(
                f"[side_objects] {name}: task_info {side}={mid} is not among scene.xml "
                f"objects {ids} — the assignment and the scene are different data (check by hand)."
            )
        out[side] = (mid, ids.index(mid))
    if out["right"][1] == out["left"][1]:
        raise SystemExit(f"[side_objects] {name}: both sides have the same slot {out}")
    return out


def assignment_provenance(tree_dir: Path) -> str:
    """**How** this tree's hand↔object assignment was decided (a required lookup).

    Returns
      "grasp[...]" / "grasp-left[...]"     **measured** grasp sustain rate — canonical
      "motion-fallback(...)"               not measurable → the displacement
                                           heuristic (treat it as suspect)
      "motion"                             the displacement heuristic, explicitly requested
      "unmeasured"                         no `side_assign` field = **never measured**.
                                           Trees baked before the field existed are this.
    """
    p = Path(tree_dir) / "task_info.json"
    return provenance_of(json.loads(p.read_text()) if p.is_file() else None)


def provenance_of(ti: dict | None) -> str:
    """`assignment_provenance` from a task_info (None = no task_info)."""
    if ti is None:
        return "unmeasured"
    return str(ti.get("side_assign") or "unmeasured")


def contact_pairing_problems(tree_dir: Path) -> list[str]:
    """Whether the contact sidecar's `tool_mesh_id` agrees with the assignment — the list
    of reasons if it does not.

    A contact annotation is the distance to 'the object that hand touches'. For a clip
    whose assignment was flipped, the existing file was computed against a **different
    object**, so the whole reward is wrong. Judged by content, not by count.
    """
    import numpy as np

    tree_dir = Path(tree_dir)
    want = resolve(tree_dir)
    bad: list[str] = []
    for side in SIDES:
        p = tree_dir / "0" / f"contact_mano_{side}.npz"
        if not p.is_file():
            bad.append(f"{side}: contact_mano_{side}.npz missing")
            continue
        z = np.load(p)
        if "tool_mesh_id" not in z.files:
            # The old schema means 'cannot be judged', not 'fine' — passing it lets old files with
            # left and right swapped slip through the net.
            bad.append(f"{side}: old schema (no tool_mesh_id) — annotate again")
            continue
        got = str(z["tool_mesh_id"])
        if got != want[side][0]:
            bad.append(
                f"{side}: contact tool={got} but the assignment is {want[side][0]} "
                f"(slot {want[side][1]}) — annotate the contact again"
            )
    return bad


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trees", nargs="+", help="tree directories")
    a = ap.parse_args()
    n_bad = 0
    for t in a.trees:
        r = resolve(Path(t))
        probs = contact_pairing_problems(Path(t))
        prov = assignment_provenance(Path(t))
        flag = "standard" if r["right"][1] == 0 else "**slots flipped**"
        if prov == "unmeasured":
            probs.insert(0, "no assignment source (side_assign not recorded) — the assignment was never measured")
        print(
            f"{Path(t).name[:56]:56s} R={r['right'][0]}@{r['right'][1]} "
            f"L={r['left'][0]}@{r['left'][1]} {flag} [{prov[:28]}]" + ("  ⚠ " + "; ".join(probs) if probs else "")
        )
        n_bad += bool(probs)
    raise SystemExit(1 if n_bad else 0)
