"""Scene manifest of one trajectory: every body of the scene with its kind, asset, collision geometry and pose source,
and the collision matrix with the reason of each cell — so a train or eval env built from the dataset gets the same
physics. Each cell carries two values: `delivered` (the delivered scene: full physics) and `training` (the training
scene: the robot's static collisions off for exploration).

It restates nothing: the bodies are the slot rows (data/objects/extra_slots.clip_rows, scene rules applied), the cells
are the collision owner's tables (rl/tasks/scene/collision_presets: preset rows, extra_collide with and without full
physics, FULL_PHYSICS_OBJECTS_OFF), and the collision shapes are the cooked dumps the simulation collides
(data/objects/hull), named relative to the data root.

Statements beside the bodies and cells, each read from its owner:
  constants         the thresholds the scene rules read, keyed `<owner module>.<name>` (`constants`)
  ruler             the collider-shape ruler (data/objects/sdf_grid.ruler_id: the git blobs of
                    collider_shape.RULER_MODULES; field collider_shape_blob) and the collider statement of the slots
                    (collider_shape.statement_fp, the value the env prints in `[TRI_COLL] statement`)
  scene_rules       every registered scene rule (rules registry scope default:scene_rules): on for the dataset, or
                    its opt-out reason (Profile.scene_rules_off)
  self_penetration  the delivered trajectory's robot self-penetration, measured on its executed joints by the owner
                    (placement.collision.self_pen: max, class, pair, frame, verdict, threshold, counted classes, ruler)
  pair_off          the both-dynamic pairs S-dyn-pairs turns off, training and delivered (`pairs_off`: cells of
                    data/objects/dyn_pairs.scene_pair through collision_presets.dyn_pair_cells)
Robot self-collision (preset rows R1 R2 R3, one group): the preset value in training and in the delivered scene.

`problems(scene, traj)` is the oracle side of one statement: unknown or missing keys, constants that differ from their
owners, another ruler, a slot collider other than the dispatcher's (collider_shape.row_shape), a collider statement
other than collider_shape.statement_fp of the stated slots, another preset or rule version, preset rows or slot cells
other than the collision owner's tables, a self-penetration of another rule, a verdict its max does not give, a
delivered trajectory over the line (> the drop line fails), a pair-off cell other than the collision owner's value or
not between two main objects, and, given the episode, a self-penetration other than the one measured again on its
joints. Both sides refuse a process whose all-object collision switch is on (the training
column states the tables with it off).
Trajectory = (an episode file of the final, the final's meta/info.json): `traj(parquet, info)`.

  <retarget venv> -m dexengine.record.scene.manifest <tree folder> --episode <parquet> --info <meta/info.json>
      [--preset P] [--out file.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dexengine.core import stdio

FORMAT = "dexengine.scene_manifest/2"
PRESET = "canon_r2r3off"  # the recipe's preset (configs/rl/recipe.yaml COLL_PRESET)
SELF_ROWS = ("R1", "R2", "R3")  # robot self-collision rows of the preset table, stated as one group
#: A stated self-penetration max this close (mm) to the drop line is rounded; its verdict is not read from it.
SELF_PEN_BAND_MM = 0.005
SCENE_KEYS = (
    "bodies",
    "collisions",
    "constants",
    "dataset",
    "format",
    "pair_off",
    "preset",
    "rule",
    "ruler",
    "scene_rules",
    "self_penetration",
    "tree",
)
BODY_KEYS = ("cls", "collision", "kind", "name", "obj_id", "pose", "pose_f0", "side", "static", "why")
COLLISION_KEYS = ("collider", "cooked", "links", "parts", "usd", "why")
CELL_KEYS = ("a", "b", "delivered", "row", "training", "why")
RULER_KEYS = ("collider_shape_blob", "statement_fp")

POSE = {
    "main": "simulated dynamic body; reference = the tree trajectory columns",
    "moving": "kinematic, follows the tree trajectory columns",
    "carried_attached": "kinematic at the simulated carrier pose × the recorded relative pose",
    "still": "kinematic, pinned at its frame-0 tree pose",
    "visual": "render only (no body)",
    "absent": "not in the scene",
}


def constants() -> dict:
    """The thresholds the scene rules read, keyed `<owner module>.<name>`, each value read from its owner."""
    from dexengine.data.objects import hand_clear as HC  # noqa: PLC0415
    from dexengine.data.objects import scene_requirement as SR  # noqa: PLC0415
    from dexengine.placement.collision import core as CO  # noqa: PLC0415
    from dexengine.placement.collision import self_pen as SP  # noqa: PLC0415
    from dexengine.placement.scene import object_desk as OD  # noqa: PLC0415

    named = (
        (SR, "TRUE_SHAPE_OVERLAP_M"),
        (HC, "CLEAR_TOL_M"),
        (OD, "OBJ_BELOW_M"),
        (SP, "ROBOT_SELF_PEN_MAX_M"),
        (CO, "DESK_SURF_PEN_MM"),
        (CO, "POLICY_MOTION_ALLOWANCE_MM"),
    )
    return {f"{m.__name__}.{n}": getattr(m, n) for m, n in named}


def scene_rules(ds: str) -> list:
    """Every registered scene rule with its state for dataset `ds`: on, or the opt-out reason of the dataset profile.
    An unknown dataset raises."""
    from dexengine.data import rules as RR  # noqa: PLC0415

    off = dict(RR.DATASET_PROFILE[ds].scene_rules_off)
    return [{"id": r.id, "on": RR.covers(r, ds), "opt_out": off.get(r.id)} for r in RR.RULES if RR.is_scene_rule(r)]


def pairs_off(tree_dir, ti: dict, q) -> list:
    """The pair-off cells of one tree: the pair its scene turns off (data/objects/dyn_pairs.scene_pair, the statement
    the env reads) in the collision owner's cell form (collision_presets.dyn_pair_cells)."""
    from dexengine.data.objects import dyn_pairs as DP  # noqa: PLC0415
    from dexengine.rl.tasks.scene import collision_presets as CP  # noqa: PLC0415

    with stdio.capture():
        p = DP.scene_pair(tree_dir, ti, q)
    return CP.dyn_pair_cells([p] if p else [])


def _pair_off_problems(scene: dict, cp) -> list:
    """Each pair-off cell: the cell keys, the collision owner's row and values, between two distinct main bodies."""
    mains = {b["name"] for b in scene["bodies"] if b.get("kind") == "main"}
    bad = []
    for c in scene["pair_off"]:
        if not isinstance(c, dict) or set(c) != set(CELL_KEYS):
            bad.append(f"pair_off cell {c!r}: keys != {sorted(CELL_KEYS)}")
            continue
        if (c["row"], c["delivered"], c["training"]) != (cp.DYN_PAIR_ROW, *cp.DYN_PAIR_COLLIDE.values()):
            bad.append(f"pair_off cell {c['a']}×{c['b']}: {c['row']} {c['delivered']}/{c['training']} != the owner's")
        if c["a"] == c["b"] or not {c["a"], c["b"]} <= mains:
            bad.append(f"pair_off cell {c['a']}×{c['b']}: not two main bodies of the scene ({sorted(mains)})")
    return bad


def _rel(p) -> str:
    """A data path relative to the data root (paths.main_root); a path outside it raises."""
    from dexengine.core import paths as P  # noqa: PLC0415

    return Path(p).relative_to(P.main_root()).as_posix()


def _collision(ds: str, o: dict) -> dict:
    """The collision geometry the simulation uses for object `o`: its USD, cooked dump and part count."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data.objects import hull as _oh  # noqa: PLC0415

    art = bool(o.get("articulation"))
    usd = _oh.usd_file(ds, str(o["obj_id"]), art)
    try:
        links = _oh.collision_parts(ds, str(o["obj_id"]), art).links
    except _oh.HullMissing as e:
        return {"usd": _rel(usd), "cooked": None, "parts": 0, "why": str(e)[:200]}
    return {
        "usd": _rel(usd),
        "cooked": _rel(P.data_layout.cooked_hull_file(str(usd))),
        "parts": sum(len(v) for v in links.values()),
        "links": len(links),
    }


def _pose_source(r: dict, xs) -> str:
    """The pose source of a slot row: a slot's by its kind, an attached row (a carried slot, or a still riding a main
    object: S-dyn-pairs) at its carrier, a still pinned at frame 0, a visual-only body rendered only."""
    if r["cls"] == xs.SLOT or r.get("kind") == xs.KIND_ATTACHED:
        return POSE.get(r.get("kind"), POSE["moving"])
    return {xs.STATIC_SPAWN: POSE["still"], xs.VISUAL_ONLY: POSE["visual"]}.get(r["cls"], POSE["absent"])


def _slot_cells(r: dict, cp, ds: str) -> list:
    """Collision cells of one slot row: robot and main objects, delivered and training values, with reasons."""
    d, t = cp.extra_collide(r["kind"], full=True, ds=ds), cp.extra_collide(r["kind"], full=False, ds=ds)
    why_o = cp.FULL_PHYSICS_OBJECTS_OFF.get(r["kind"])
    tg = r.get("touch_gate")
    if tg and d["objects"] == "OFF":
        why_o = f"S-touch-gate: {tg['why']} " + json.dumps({k: v["depth_m"] for k, v in tg["pairs"].items()})
    why_t = f"training table of kind {r['kind']}"
    if r["kind"] in cp.ROBOT_STATIC_KINDS and cp.robot_static(ds):
        why_t += " · S-robot-static: robot on"
    if cy := r.get("carry"):
        why_t += f" · S-carried: attached on {cy['carrier']} for the whole clip"
    cell = {"row": cp.EXTRA_ROW, "b": r["name"]}
    return [
        cell | {"a": "robot (hand · arm)", "delivered": d["hand"], "training": t["hand"], "why": why_t},
        cell | {"a": "main objects", "delivered": d["objects"], "training": t["objects"], "why": why_o or why_t},
    ]


def _preset_cells(cp, preset: str) -> list:
    """The preset rows (ROW_ORDER): the same value in training and in the delivered scene; R1 R2 R3 as one group."""
    rows_p = cp.EXPECTED[preset]
    group = " · robot self-collision rows " + " ".join(SELF_ROWS) + " as one group · delivered: self_penetration"
    return [
        {
            "row": k,
            "a": cp.ROW_MEANING[k],
            "b": "",
            "delivered": rows_p[k],
            "training": rows_p[k],
            "why": f"preset {preset} row {k}" + (group if k in SELF_ROWS else ""),
        }
        for k in cp.ROW_ORDER
        if k in rows_p
    ]


def _bodies(ti: dict, q, ds: str, rows: list) -> tuple[list, list, list]:
    """(bodies, slot cells, slot rows) of one tree."""
    from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415 — owner of the collider shape
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.rl.tasks.scene import collision_presets as CP  # noqa: PLC0415

    objs = {str(o["name"]): o for o in ti.get("objects") or []}
    bodies = [
        {"name": n, "kind": "main", "side": o.get("side"), "obj_id": o.get("obj_id"), "pose": POSE["main"]}
        | {"collision": _collision(ds, o)}
        for n, o in objs.items()
        if o.get("role") == "main"
    ]
    cells, slots = [], []
    for r in rows:
        o = objs[r["name"]]
        b = {"name": r["name"], "cls": r["cls"], "why": r.get("why"), "kind": r.get("kind"), "obj_id": o.get("obj_id")}
        b["pose"] = _pose_source(r, XS)
        if r["cls"] in XS.SLOTTED:
            slots.append(r)
            b |= {"obj_id": r["obj_id"], "static": bool(r.get("static"))}
            b["collision"] = _collision(ds, o) | {"collider": CS.row_shape(ds, r)}
            if r["cls"] == XS.STATIC_SPAWN:
                b["pose_f0"] = [round(float(v), 6) for v in q[0, int(o["qpos_start"]) : int(o["qpos_start"]) + 7]]
            cells += _slot_cells(r, CP, ds)
        bodies.append(b)
    return bodies, cells, slots


def traj(parquet, info: dict) -> dict:
    """One delivered trajectory: an episode file and the info.json of its dataset."""
    return {"parquet": Path(parquet), "info": info}


def self_penetration(t: dict) -> dict:
    """The self-penetration statement of trajectory `t`, measured by its owner."""
    from dexengine.placement.collision import self_pen as SP  # noqa: PLC0415

    return SP.statement(SP.measure_parquet(t["parquet"], t["info"]))


def build(tree_dir, t: dict, preset: str = PRESET) -> dict:
    """The manifest of one trajectory `t` recorded from one tree (module doc)."""
    from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.data.objects import sdf_grid as SG  # noqa: PLC0415 — owner of the ruler identity
    from dexengine.rl.tasks.scene import collision_presets as CP  # noqa: PLC0415

    _training_tables(CP)
    tree_dir = Path(tree_dir)
    ti = json.loads((tree_dir / "task_info.json").read_text(encoding="utf-8"))
    q = np.load(tree_dir / "0" / "trajectory_kinematic.npz")["qpos"]
    ds = str(ti["dataset_name"])
    with stdio.capture():
        rows = XS.clip_rows(str(tree_dir), ti, q)
    bodies, cells, slots = _bodies(ti, q, ds, rows)
    return {
        "format": FORMAT,
        "tree": tree_dir.name,
        "dataset": ds,
        "preset": preset,
        "rule": XS.rule_version([ds]),
        "constants": constants(),
        "ruler": {"collider_shape_blob": SG.ruler_id(), "statement_fp": CS.statement_fp(ds, slots)},
        "scene_rules": scene_rules(ds),
        "self_penetration": self_penetration(t),
        "pair_off": pairs_off(tree_dir, ti, q),
        "bodies": bodies,
        "collisions": cells + _preset_cells(CP, preset),
    }


def _key_problems(where: str, d, allowed, exact: bool) -> list:
    """Keys of one statement part: unknown keys always fail; with `exact`, missing keys fail too."""
    if not isinstance(d, dict):
        return [f"{where}: not an object"]
    bad = [f"{where}: unknown keys {sorted(set(d) - set(allowed))}"] * bool(set(d) - set(allowed))
    return bad + [f"{where}: missing keys {sorted(set(allowed) - set(d))}"] * (exact and bool(set(allowed) - set(d)))


def _shape_problems(scene: dict) -> list:
    """Unknown keys anywhere in the statement (scene, ruler, bodies, collision parts, cells)."""
    bad = _key_problems("scene", scene, SCENE_KEYS, True)
    if bad:
        return bad
    bad += _key_problems("ruler", scene["ruler"], RULER_KEYS, True)
    for b in scene["bodies"]:
        bad += _key_problems(f"body {b.get('name')}", b, BODY_KEYS, False)
        if "collision" in b:
            bad += _key_problems(f"body {b.get('name')} collision", b["collision"], COLLISION_KEYS, False)
    for c in scene["collisions"]:
        bad += _key_problems(f"cell {c.get('row')} {c.get('b')}", c, CELL_KEYS, True)
    return bad


def _collider_problems(scene: dict) -> list:
    """Each slot's stated collider against the dispatcher, and the stated collider statement against its owner."""
    from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    ds = scene["dataset"]
    slots = [b for b in scene["bodies"] if b.get("cls") in XS.SLOTTED]
    stated = {b["name"]: (b.get("collision") or {}).get("collider") for b in slots}
    bad = [
        f"slot {b['name']}: collider {stated[b['name']]!r} != dispatcher {CS.row_shape(ds, b)!r}"
        for b in slots
        if stated[b["name"]] != CS.row_shape(ds, b)
    ]
    fp = CS.statement_fp(ds, slots)
    return bad + [f"collider statement {scene['ruler']['statement_fp']} != {fp}"] * (
        fp != scene["ruler"]["statement_fp"]
    )


def _training_tables(cp) -> None:
    """The training column states the tables with the all-object collision switch at its default (off); a process
    with the switch on cannot state or check it."""
    if cp.all_obj_collision():
        raise ValueError(f"{cp.ALL_OBJ_TAG} {cp.ALL_OBJ_ENV}=1: the manifest states the training tables with it off")


def _slot_cell_problems(scene: dict, cp) -> list:
    """The stated slot cells (row RE) against the collision owner's tables for each slot's stated kind."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    def key(c: dict) -> tuple:
        return c["row"], c["a"], c["b"], c["delivered"], c["training"]

    slots = [b for b in scene["bodies"] if b.get("cls") in XS.SLOTTED]
    want = sorted(key(c) for b in slots for c in _slot_cells(b, cp, scene["dataset"]))
    got = sorted(key(c) for c in scene["collisions"] if c["row"] == cp.EXTRA_ROW)
    return (
        [] if got == want else [f"slot cells {sorted(set(got) - set(want))} != tables {sorted(set(want) - set(got))}"]
    )


def _self_pen_problems(stated, t) -> list:
    """The stated self-penetration against its owner's rule, its own verdict (outside the rounding band of the line), a
    drop, and, given the trajectory, the joints measured again."""
    from dexengine.placement.collision import self_pen as SP  # noqa: PLC0415

    bad = _key_problems("self_penetration", stated, SP.STATEMENT_KEYS, True)
    if bad:
        return bad
    if stated["state"] != "measured" or stated["rule"] != SP.rule():
        return [f"self_penetration {stated['state']} rule {stated['rule']} != owner measured {SP.rule()}"]
    line, mm = SP.rule()["threshold_mm"], float(stated["max_mm"])
    if abs(mm - line) > SELF_PEN_BAND_MM and stated["verdict"] != ("drop" if mm > line else "keep"):
        bad.append(f"self_penetration verdict {stated['verdict']} != the verdict of {mm} mm at the {line} mm line")
    if stated["verdict"] != "keep":
        bad.append(f"self_penetration {mm} mm ({stated['pair']} frame {stated['frame']}) over the line: delivered")
    if t is not None and (got := self_penetration(t)) != stated:
        bad.append(f"self_penetration {stated} != measured again {got}")
    return bad


def problems(scene: dict, t: dict | None = None) -> list:
    """The oracle of one statement (module doc): [] when it holds, else one line per problem. `t` = the trajectory the
    statement was built from (its self-penetration is measured again)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.data.objects import sdf_grid as SG  # noqa: PLC0415
    from dexengine.rl.tasks.scene import collision_presets as CP  # noqa: PLC0415

    bad = _shape_problems(scene)
    if bad:
        return bad
    if scene["format"] != FORMAT:
        return [f"format {scene['format']!r} != {FORMAT!r}"]
    _training_tables(CP)
    ds = scene["dataset"]
    want = {
        "preset": PRESET,
        "rule": XS.rule_version([ds]),
        "constants": constants(),
        "scene_rules": scene_rules(ds),
    }
    bad += [f"{k} {scene[k]} != owner {v}" for k, v in want.items() if scene[k] != v]
    if scene["ruler"]["collider_shape_blob"] != SG.ruler_id():
        bad.append(f"ruler {scene['ruler']['collider_shape_blob']} != ruler id {SG.ruler_id()}")
    cells = {c["row"]: c for c in scene["collisions"] if c["row"] in CP.ROW_ORDER}
    bad += [
        f"preset row {k}: {cells.get(k)} != {v}"
        for k, v in CP.EXPECTED[PRESET].items()
        if k not in cells or (cells[k]["delivered"], cells[k]["training"]) != (v, v)
    ]
    bad += _self_pen_problems(scene["self_penetration"], t)
    return bad + _slot_cell_problems(scene, CP) + _collider_problems(scene) + _pair_off_problems(scene, CP)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tree")
    ap.add_argument("--episode", required=True, help="an episode file of the trajectory (its data parquet)")
    ap.add_argument("--info", required=True, help="the meta/info.json of the episode's dataset")
    ap.add_argument("--preset", default=PRESET)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t = traj(a.episode, json.loads(Path(a.info).read_text(encoding="utf-8")))
    text = json.dumps(build(a.tree, t, a.preset), indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
