#!/usr/bin/env python3
"""Rule and slot count for extra-object slots: extra objects a person moved (role extra) stand in the RL scene as
kinematic bodies that follow the reference (env side: `dexengine.rl.env_setup.build.build_extra_slots`; on by default,
off only with an explicit `DEXENGINE_EXTRA_SLOTS=0`).

Why: the scene spawns one main object per hand plus supports. Other objects a person held or moved (held extras,
OakInk2 free parts) would be missing from both training and rendering, so the policy would learn motions on thin air and
the videos would be false. Reassignment cannot fix it, since each hand has only one main-object slot.

Rule, applied the same way to any tree; for every role=extra object in objects[]:
  moved          = max position change from frame 0 ≥ `scene_requirement.MOVE_M` (scene requirement statement, read by
                   production and the oracle alike) or max rotation ≥ `ROT_DEG`
  jump           = max frame-to-frame position change ≥ `JUMP_M` (suspected tracking defect; not counted as motion)
  duplicate      = same identity as a spawned object (main · support) or an earlier object in objects[]
                   (OakInk2: official part id `dexengine.data.oakink2.parts.rid_of`)
  human contact  = tree stamp `case.contact_frac[name][side]` (stage 1 contact measurement, the value the judgement and
                   assignment use)
  merged part    = an official part that is not free in the per-object joint spec (`dexengine.data.oakink2.part_kinds`:
  hinge · slide ·
                   fixed · lever · deformable)
Gate before classification (independent of role, motion and contact): a merged part whose instance still has another
merged part in objects[] is PART_UNMERGED. Such parts belong to the part merge (2b, D1-g) alone; a merged tree has no
such object, so one left over means the merge failed. Standing it up as a slot, static slot, second main object or
support would give the part two owners (a door opening by itself) or leave a piece floating (a lamp head). strict
(default: training and collection entry points) fails with the reason (the tree's rules.D1g status); the scene audit
(strict=False) takes it as a row and records a PART_UNMERGED defect. Free parts (bottle caps, cut fruit) follow the
ordinary rules below.
Classification, in this order: duplicate → DUPLICATE · moved ∧ ¬jump ∧ contact > 0 → SLOT (any hand; contact by the
idle hand alone gives `why=idle_hand`: there is no robot hand on it, so it follows the reference) · moved (the rest) →
SUSPECT (with a mesh, `why=suspect` and on to ③ · ④ below) · anything else → STATIC.
Only three kinds of extras stand in the training and judging scene: moved ones (SLOT), still objects a hand touched
(①②) and still objects that receive (③). The rest (④) do not.
Static spawn (STATIC_SPAWN: takes a slot, pose fixed at frame 0):
  ① an extra a human hand touched that did not move (STATIC ∧ contact > 0, `why=touched`; a touched jump also lands here
     with `why=suspect`)
  ② a SLOT that moved only slightly without a held interval (`case.grasp_s`): max motion < `MOVE_M` ∧ max rotation <
     `MICRO_ROT_DEG`, `why=micro` (tracking jitter, a light push; thresholds from the motion and rotation distributions
     of held vs not-held SLOTs).
  ③ an untouched still extra (with a mesh) that **receives** (`why=receiver`, row `recv_by`): a spawned object stands on
     it, or a moved object comes up to it at the end and touches it (measured by `extra_slots_geom.receivers`). More
     than the support band below the desk surface → STATIC `why=below_work_surface`; more than the support band above
     the desk → it stands on a pedestal (`coords.static_pedestals`).
  Among ①②, those that **receive the main object** (bulb–lamp, putting into a bowl) only change kind to receiver (row
     `recv_by` = that main object; `why` stays touched · micro · suspect). The measurement is the same
     `extra_slots_geom.receivers` as ③, but only the main object counts as partner, because kind receiver enables
     collision with the main object alone (moved extras and support objects are kinematic or fixed, so there is no other
     pair to enable). Spawning, the placement rule and must-place (`static_fixed` reads `why=receiver`) are unchanged.
     Objects without geometry (mesh_dir) cannot be measured and are out of scope (no collision shape either).
  Placement rule (`place_rule`; baked desk trees, every dataset): a touched static-spawn candidate of ①② that was only
     **brushed** (`place_rule.drop_why`, the formula the judgement reads per batch) and that overlaps the robot's
     footprint or lies off the desk does not stand: STATIC `why=robot_keepout` · `why=off_desk`. Everything else stands
     (batches that violate it are dropped by the judgement). Untouched candidates skip this rule: they are ③ (must
     place) when they receive, otherwise ④ (visual only).
Visual only (VISUAL_ONLY, no slot): ④ untouched still extras that do not receive (`why=still` · untouched
`why=suspect`). Collection rendering (`dexengine.record.scene.extras.decide`) takes them as 'extras the env did not
spawn' and adds visual-only prims (no collision).
Row `kind` (slot rows only) = moving · touched · receiver, the key of the collision partners
(`collision_presets.EXTRA_KIND_COLLIDE`). The kind of a static spawn comes from `recv_by` alone: rows with a receiving
partner (③ and ①② that receive the main object) = receiver, the rest = touched.
Carried (SLOT, `why=carried`): an extra moved by at least MOVE_M without hand contact (not a jump; the 'carried'
category of `scene_requirement`). Kind moving (follows the world-recorded reference) until the scene rules run.
Scene rules (`scene_rules.apply`, on for every dataset unless its profile opts out — rules registry
`default:scene_rules`; off = these rows as above): S-carried (carried rows are attached to their carrier: kind
carried_attached, row `carry`; no carrier found = kind moving), S-touch-gate (touched static spawns collide with the
main objects unless the recording passes through them: kind touched_solid, row `touch_gate`) and S-envelope (visual-only
still extras next to the recorded motion stand: STATIC_SPAWN `why=envelope`, kind envelope, row `envelope`; the slot
cap drops envelope rows first). cls of carried and touched rows is unchanged, so placement readers see the same
slots; envelope rows are static spawns like touched ones.
SLOT and STATIC_SPAWN take slots (row `static` true = still); PART_UNMERGED stays outside the slots and the marker
counts it.
Slot count: SLOT + STATIC_SPAWN ≤ `K_MAX` per clip; more is an error once no envelope row is left to drop.
Rule version `RULE`: the name the run ledger and marker carry; bump it when the classification changes.
"""

from __future__ import annotations

import json
import os

import numpy as np

from dexengine.data.objects import (
    extra_slots_geom as XG,
)  # measurements this rule reads (motion · receiving · below the desk surface · distribution line)
from dexengine.data.objects.scene_requirement import (
    MOVE_M,
    ROT_DEG,
)  # owner of the scene requirement statement (motion and rotation thresholds, 'carried' category)

RULE = "ES-2026-11-24-recv-hull"  # rule version (ledger and marker name; bump it when the classification changes)
K_MAX = 9  # slot cap per clip (more is an error); measured maximum 6 with receivers only (HOT3D XHand base sample)

JUMP_M = 0.05  # frame-to-frame position jump (m, 50 Hz); above it, a suspected tracking defect

SLOT, DUPLICATE, SUSPECT, STATIC = "SLOT", "DUPLICATE", "SUSPECT", "STATIC"
STATIC_SPAWN = "STATIC_SPAWN"
BELOW_WORK_SURFACE = "below_work_surface"  # STATIC reason: untouched still extra below the desk surface (not spawned)
ROBOT_KEEPOUT = "robot_keepout"  # STATIC reason: brushed still extra overlaps the robot's footprint (placement rule)
OFF_DESK = (
    "off_desk"  # STATIC reason: brushed still extra extends past the desk top beyond the threshold (placement rule)
)
PLACE_DROPPED = (
    ROBOT_KEEPOUT,
    OFF_DESK,
)  # reasons the placement rule drops from spawning (same words as place_rule.WHYS, checked below)
KEEPOUT_TAG = "[STATIC_SPAWN]"
SLOTTED = (SLOT, STATIC_SPAWN)  # classes that take a slot (STATIC_SPAWN = pose fixed at frame 0)
VISUAL_ONLY = "VISUAL_ONLY"  # untouched still extra that does not receive; not placed in the scene (visual only)
RECEIVER = "receiver"  # STATIC_SPAWN reason: receives a spawned object
UNTOUCHED_STILL = ("still", "suspect")  # reasons that go through the receiving check (when the row's touched is false)
KIND_MOVING, KIND_TOUCHED, KIND_RECEIVER = "moving", "touched", "receiver"  # kind of a slot row
KINDS = (KIND_MOVING, KIND_TOUCHED, KIND_RECEIVER)
# Scene-rule kinds (scene_rules). They are **not** in KINDS — the statements that list kinds (ledger collide tables,
# markers, authoring) add them only when a row has one (`kinds_present`), so runs without them stay byte-identical.
# spawn_key counts the carried kind in the moving bucket (it moves with its carrier) and the static ones in the touched
# bucket (pinned at frame 0, not receiving).
KIND_ATTACHED = "carried_attached"
KIND_TOUCHED_SOLID, KIND_ENVELOPE = "touched_solid", "envelope"
SCOPED_KINDS = (KIND_ATTACHED, KIND_TOUCHED_SOLID, KIND_ENVELOPE)
KIND_BUCKET = {
    KIND_ATTACHED: KIND_MOVING,
    KIND_TOUCHED_SOLID: KIND_TOUCHED,
    KIND_ENVELOPE: KIND_TOUCHED,
}
MICRO_ROT_DEG = 15.0  # 'micro' rotation cap (°) of motion without a held interval
PART_UNMERGED = "PART_UNMERGED"  # articulated part left unmerged (2b); not a slot (strict fails)
ROUTED = {PART_UNMERGED: "part merge failed (2b) — defect"}  # outside-slot share → where it goes
SPAWNED_ROLES = ("main", "support")
EXTRA_ROLE = "extra"  # remaining role in objects[]: objects the training scene does not spawn (the target of this rule)
ENV = "DEXENGINE_EXTRA_SLOTS"  # on/off switch: empty or 1 = on (default) · 0 = off (explicit) · anything else fails
DEFAULT = "1"  # default: clips with extra objects are kept


def extra_objects(ti: dict) -> list[dict]:
    """Extra objects of the tree (objects[] order); shared by the render decision (dexengine.record.scene.extras.decide)
    and the remote statement."""
    return [o for o in (ti.get("objects") or []) if o.get("role") == EXTRA_ROLE]


def requested() -> bool:
    """Whether extra-object slots are on (read by the env and the scene-audit oracle); an unknown value fails."""
    v = os.environ.get(ENV, "").strip() or DEFAULT
    if v not in ("0", "1"):
        raise RuntimeError(f"[EXTRA_SLOTS] {ENV}={v!r} — only 0|1")
    return v == "1"


def _ident(ds: str, obj_id: str) -> str:
    if ds == "oakink2":
        from dexengine.data.oakink2 import (
            parts as _op,  # noqa: PLC0415 — owner of part identity (official object_part_tree)
        )

        return _op.rid_of(str(obj_id))
    return str(obj_id)


def _part_root(ds: str, obj_id: str) -> str | None:
    """Root id of the official part tree (None for non-parts or datasets without a part tree)."""
    if ds != "oakink2":
        return None
    from dexengine.data.oakink2 import parts as _op  # noqa: PLC0415

    rid = _op.rid_of(str(obj_id))
    return _op.root_of(rid) if _op.is_part(rid) else None


def _kind(ds: str, obj_id: str) -> str | None:
    """Joint-spec kind of an official part (dexengine.data.oakink2.part_kinds; fails when missing from the table) · None
    for non-parts."""
    if _part_root(ds, obj_id) is None:
        return None
    from dexengine.data.oakink2 import part_kinds as _pk  # noqa: PLC0415 — owner of the per-object joint kinds
    from dexengine.data.oakink2 import parts as _op  # noqa: PLC0415

    return _pk.kind_of(_op.rid_of(str(obj_id)))[0]


def _unmerged_why(tree_dir: str, ti: dict, o: dict, sibs: list[dict]) -> str:
    """One-line reason for an unmerged part: spec kind · remaining parts of the same instance · the tree's part-merge
    record (rules.D1g)."""
    kind = _kind(str(ti.get("dataset_name") or ""), o["obj_id"])
    d1g = (ti.get("rules") or {}).get("D1g")
    st = (
        f"{d1g.get('version')}:{d1g.get('status')}"
        if isinstance(d1g, dict)
        else "no rules.D1g (tree not merged by 2b; see the oakink2_artic_convert audit row)"
    )
    return (
        f"[EXTRA_SLOTS] {PART_UNMERGED} {os.path.basename(str(tree_dir))}: {o.get('role')} {o['name']}"
        f"({o['obj_id']}) is a {kind} part in the spec but stays apart from the merged parts "
        f"{[x['name'] for x in sibs]} "
        f"of the same instance — the part merge (2b D1-g) must make one object; not placed as a slot, main object "
        f"or support (one part, two owners) · 2b record {st}"
    )


def _unmerged_sibs(ds: str, o: dict, objs: list[dict]) -> list[dict]:
    """Other merged parts (spec kind ≠ free) of the same instance as merged part `o` still in objects[] ([] if none)."""
    root = _part_root(ds, o["obj_id"])
    if root is None or _kind(ds, o["obj_id"]) == "free":
        return []
    return [x for x in objs if x is not o and _part_root(ds, x["obj_id"]) == root and _kind(ds, x["obj_id"]) != "free"]


def _unmerged_rows(tree_dir: str, ti: dict, ds: str, objs: list[dict], strict: bool) -> tuple:
    """(unmerged (object, merged siblings) pairs, PART_UNMERGED rows of the spawned ones). strict fails on the first
    pair, whatever its role (second main object, support or extra alike)."""
    unmerged = [(o, sb) for o in objs for sb in [_unmerged_sibs(ds, o, objs)] if sb]
    if unmerged and strict:  # regardless of role: second main object, support or extra alike
        raise ValueError(_unmerged_why(tree_dir, ti, *unmerged[0]))
    um_rows = [
        {
            "name": str(o["name"]),
            "obj_id": str(o["obj_id"]),
            "qpos_start": o.get("qpos_start"),
            "cls": PART_UNMERGED,
            "role": o.get("role"),
            "why": _unmerged_why(tree_dir, ti, o, sb),
            "static": False,
        }
        for o, sb in unmerged
        if o.get("role") in SPAWNED_ROLES
    ]
    return unmerged, um_rows


def _motion_class(
    tree_dir: str, ti: dict, q: np.ndarray, o: dict, move, rot, step, c_act: float, c_idle: float, grasp: dict
) -> tuple:
    """(class, why) of an extra that is neither an unmerged part nor a duplicate: motion, human contact, mesh, then the
    placement rule for brushed static spawns."""
    moved = move >= MOVE_M or rot >= ROT_DEG
    jump = step >= JUMP_M
    grasped = any(iv for iv in (grasp.get(o["name"]) or {}).values())
    why = ""
    if moved and not jump and (c_act > 0 or c_idle > 0):
        cls, why = SLOT, ("" if c_act > 0 else "idle_hand")  # any hand (scene_requirement: moved · touch)
        if cls == SLOT and not grasped and move < MOVE_M and rot < MICRO_ROT_DEG:
            cls, why = STATIC_SPAWN, "micro"
    elif moved and not jump and move >= MOVE_M:
        cls, why = SLOT, "carried"  # moved without hand contact (carried on another object); follows the reference
    elif moved and o.get("mesh_dir"):
        cls, why = STATIC_SPAWN, "suspect"  # rotation only or jumps (tracking not trusted); kept, fixed at frame 0
    elif moved:
        cls = SUSPECT
    elif c_act > 0 or c_idle > 0:
        cls, why = STATIC_SPAWN, "touched"
    elif o.get("mesh_dir") and XG.below_work_surface(tree_dir, ti, q, o):
        # untouched and below the work surface (desk); not put inside the desk volume
        cls, why = (STATIC, BELOW_WORK_SURFACE)
    elif o.get("mesh_dir"):
        # extra that did not move: support, container, the rest of the human scene (fixed at frame 0)
        cls, why = (STATIC_SPAWN, "still")
    else:
        cls = STATIC
    if cls == STATIC_SPAWN and (c_act > 0 or c_idle > 0) and (drop := _place_drop(tree_dir, ti, q, o)):
        # placement rule: a brushed object on the robot's footprint or off the desk does not stand
        cls, why = (STATIC, drop)
    return cls, why


def _extra_row(tree_dir: str, ti: dict, q: np.ndarray, ds: str, objs: list[dict], o: dict, ctx: dict) -> tuple:
    """(identity, row) of one non-spawned object: its class and the values behind it. ctx = the tree-wide inputs
    (um_ids · spawned · seen · cf · idle · grasp) of the first pass."""
    if o.get("role") != "extra":
        raise ValueError(f"{tree_dir}: unknown role {o.get('role')!r} ({o.get('name')}) — only main|support|extra")
    st = o.get("qpos_start")
    if st is None:
        raise ValueError(f"{tree_dir}: extra {o.get('name')!r} has no qpos_start")
    ident = _ident(ds, o["obj_id"])
    move, rot, step = XG.motion(q, int(st))
    c = ctx["cf"].get(o["name"]) or {}
    c_act = max((float(v) for s, v in c.items() if s not in ctx["idle"]), default=0.0)
    c_idle = max((float(v) for s, v in c.items() if s in ctx["idle"]), default=0.0)
    why = ""
    if id(o) in ctx["um_ids"]:
        cls, why = PART_UNMERGED, _unmerged_why(tree_dir, ti, o, _unmerged_sibs(ds, o, objs))
    elif ident in ctx["spawned"] or ident in ctx["seen"]:
        cls = DUPLICATE
    else:
        cls, why = _motion_class(tree_dir, ti, q, o, move, rot, step, c_act, c_idle, ctx["grasp"])
    return ident, {
        "name": str(o["name"]),
        "obj_id": str(o["obj_id"]),
        "qpos_start": int(st),
        "cls": cls,
        "move_m": round(move, 4),
        "rot_deg": round(rot, 1),
        "step_m": round(step, 4),
        "contact_active": round(c_act, 3),
        "contact_idle": round(c_idle, 3),
        "touched": bool(c_act > 0 or c_idle > 0),
        "why": why,
        "static": cls == STATIC_SPAWN,
    }


def _first_pass(tree_dir: str, ti: dict, q: np.ndarray, ds: str, objs: list[dict], um_ids: set, um_rows: list) -> list:
    """Rows of the unmerged spawned objects, then one classified row per extra (objects[] order)."""
    idle = set(ti.get("idle_sides") or [])
    cf = (ti.get("case") or {}).get("contact_frac")
    if not isinstance(cf, dict):
        raise ValueError(f"{tree_dir}: no case.contact_frac stamp — cannot classify without human contact")
    spawned = {_ident(ds, o["obj_id"]) for o in objs if o.get("role") in SPAWNED_ROLES}
    grasp = (ti.get("case") or {}).get("grasp_s") or {}
    rows, seen = list(um_rows), set()
    ctx = {"um_ids": um_ids, "spawned": spawned, "seen": seen, "cf": cf, "idle": idle, "grasp": grasp}
    for o in objs:
        if o.get("role") in SPAWNED_ROLES:
            continue
        ident, row = _extra_row(tree_dir, ti, q, ds, objs, o, ctx)
        seen.add(ident)
        rows.append(row)
    return rows


def _receiver_pass(tree_dir: str, ti: dict, q: np.ndarray, rows: list[dict]) -> None:
    """Second pass: untouched still extras stand only when they receive (③ · ④); the spawned objects of the first pass
    are the partners. Rows are updated in place."""
    cand = [r for r in rows if r["cls"] == STATIC_SPAWN and r["why"] in UNTOUCHED_STILL and not r["touched"]]
    names = [r["name"] for r in cand]
    still = [r["name"] for r in rows if r["cls"] == STATIC_SPAWN and r["name"] not in names]
    recv = XG.receivers(tree_dir, ti, q, names, [r["name"] for r in rows if r["cls"] == SLOT], still) if cand else {}
    for r in cand:
        if r["name"] in recv:
            r.update(still=r["why"], why=RECEIVER, recv_by=recv[r["name"]])
        else:
            r.update(cls=VISUAL_ONLY, static=False)


def _main_receiver_pass(tree_dir: str, ti: dict, q: np.ndarray, objs: list[dict], rows: list[dict]) -> None:
    """Third pass: touched static spawns (①②) that receive the main object also record the partner. Same measurement
    as the second pass, but the only candidate partner is the main object (moved · still emptied); other partners are
    not recorded. Rows are updated in place."""
    mains = {str(o["name"]) for o in objs if o.get("role") == "main"}
    meshed = {str(o["name"]) for o in objs if o.get("mesh_dir")}
    hand = [r["name"] for r in rows if r["cls"] == STATIC_SPAWN and r["why"] != RECEIVER and r["name"] in meshed]
    got = XG.receivers(tree_dir, ti, q, hand, [], []) if hand and mains & meshed else {}
    for r in rows:
        if r["name"] in hand and got.get(r["name"]) in mains:
            r["recv_by"] = got[r["name"]]


def clip_rows(
    tree_dir: str, ti: dict | None = None, q: np.ndarray | None = None, strict: bool = True, kinds_only: bool = False
) -> list[dict]:
    """One row per role=extra object of a tree (objects[] order): class and the values behind it. strict (default) fails
    on PART_UNMERGED with the reason; False emits the row as PART_UNMERGED (the scene audit counts it as a defect).
    kinds_only=True is for a reader of the classes and kinds only (f0): the measures no class or kind reads are left out
    or cut short — the envelope rows' recorded overlap (envelope.overlap: `envelope.depth_m` and its clip drop) is not
    measured, and S-touch-gate stops a pair at its first frame deeper than the true-shape overlap (its `depth_m` then a
    lower bound, the pair marked `depth_at_least`; scene_rules.apply). A placement variant (`<clip>_pvNN`) takes its
    source clip's classification (owner extra_slots_variant: placement augment does not change object relations)."""
    if ti is None:
        with open(os.path.join(tree_dir, "task_info.json"), encoding="utf-8") as f:
            ti = json.load(f)
    from dexengine.data.objects import extra_slots_variant as XV  # noqa: PLC0415 — variant -> source tree

    src = XV.source_dir(tree_dir, ti)
    if src is not None:
        return XV.rows(tree_dir, ti, q, strict, src, _own_rows, kinds_only)
    return _own_rows(tree_dir, ti, q, strict, kinds_only)


def _own_rows(tree_dir: str, ti: dict, q: np.ndarray | None, strict: bool, kinds_only: bool = False) -> list[dict]:
    """clip_rows' body: the classification measured on this tree's own geometry and trajectory (a variant uses it for
    its own record and the placement verdict only)."""
    objs = ti.get("objects")
    if not isinstance(objs, list):
        raise ValueError(f"{tree_dir}: no objects[] — the extra rule reads base-layer trees only")
    ds = str(ti.get("dataset_name") or "")
    unmerged, um_rows = _unmerged_rows(tree_dir, ti, ds, objs, strict)
    um_ids = {id(o) for o, _sb in unmerged}
    if not any(o.get("role") not in SPAWNED_ROLES for o in objs):
        return um_rows
    if q is None:
        q = np.load(os.path.join(tree_dir, "0", "trajectory_kinematic.npz"))["qpos"]
    rows = _first_pass(tree_dir, ti, q, ds, objs, um_ids, um_rows)
    _receiver_pass(tree_dir, ti, q, rows)
    _main_receiver_pass(tree_dir, ti, q, objs, rows)
    for r in rows:
        if r["cls"] in SLOTTED:
            r["kind"] = KIND_MOVING if r["cls"] == SLOT else KIND_RECEIVER if r.get("recv_by") else KIND_TOUCHED
    from dexengine.data.objects import (
        scene_rules as SC,  # noqa: PLC0415 — the scene-rule passes (they read this module)
    )

    SC.apply(tree_dir, ti, q, rows, lambda o: _place_drop(tree_dir, ti, q, o), kinds_only=kinds_only)
    return rows


def kinds_present(rows) -> tuple:
    """KINDS + the scene-rule kinds some row of `rows` (rows, or lists of rows) has — the kind list a statement
    names."""
    flat = [x for r in rows for x in (r if isinstance(r, list) else [r])]
    return KINDS + tuple(k for k in SCOPED_KINDS if any(isinstance(x, dict) and x.get("kind") == k for x in flat))


def rule_version(datasets) -> str:
    """The rule name a run's ledger and marker carry: RULE, joined with the version of every scene rule that is on for
    a dataset of the run (`scene_rules.rule_version`; a run whose datasets opt out of all of them: RULE as is)."""
    from dexengine.data.objects import scene_rules as SC  # noqa: PLC0415

    return SC.rule_version(RULE, datasets)


def hand_receiver(r: dict) -> bool:
    """Touched receiver: a static spawn a hand touched (`why` ≠ receiver) that receives the main object (`recv_by`).
    Kind receiver (collision with the main object on). Read by the [EXTRA_SLOTS] marker and the hold measurement
    `recv_clearance`."""
    return r.get("cls") == STATIC_SPAWN and r.get("why") != RECEIVER and bool(r.get("recv_by"))


def _place_drop(tree_dir: str, ti: dict, q: np.ndarray, o: dict) -> str:
    """Reason the placement rule drops a static-spawn candidate (one of PLACE_DROPPED) or ''; the formula is
    `place_rule.drop_why`. Trees before placement have no scene position and give '' (the roster audit counts the set
    above)."""
    from dexengine.placement.scene import place_rule as PR  # noqa: PLC0415 — owner of the placement rule

    why = PR.drop_why(tree_dir, ti, q, o)
    if why and why not in PLACE_DROPPED:
        raise RuntimeError(f"[EXTRA_SLOTS] placement-rule reason {why!r} outside the vocabulary {PLACE_DROPPED}")
    return why


def keepout_marker(tree_dir: str, rows: list[dict]) -> str:
    """Marker line `[STATIC_SPAWN] skip place_rule n=<count> <tree> <name:reason …>` (slot_rows · audit)."""
    names = [f"{r['name']}:{r['why']}" for r in rows if r.get("why") in PLACE_DROPPED]
    tree = os.path.basename(str(tree_dir).rstrip("/"))
    return f"{KEEPOUT_TAG} skip place_rule n={len(names)} {tree} {','.join(names) or '-'}"


held_intervals = XG.held_intervals  # held intervals (tree frames); measured in extra_slots_geom


def env_inputs(ti: dict, tree_dir: str | None = None) -> dict:
    """Files the env reads because of this rule (meshes, obj_id of the slot USD, the tree's frame-map sidecars); read by
    the bundle input statement run_inputs."""
    return XG.env_inputs(ti, SPAWNED_ROLES, tree_dir)


def slot_rows(
    tree_dir: str, ti: dict | None = None, q: np.ndarray | None = None, strict: bool = True
) -> tuple[list[dict], dict]:
    """(rows that take a slot: SLOT · STATIC_SPAWN (row static true = fixed at frame 0), in qpos column order = slot
    order;
    outside-slot shares {class: [names]}). Fails when the two together exceed K_MAX. strict as in clip_rows."""
    return slot_rows_of(tree_dir, clip_rows(tree_dir, ti, q, strict))


def slot_rows_of(tree_dir: str, rows: list[dict]) -> tuple[list[dict], dict]:
    """`slot_rows` of the tree's clip_rows `rows` (a caller that already holds them reads its slots without measuring
    the rows again)."""
    if any(r.get("why") in PLACE_DROPPED for r in rows):  # report what was not spawned
        print(keepout_marker(tree_dir, rows), flush=True)
    routed = {c: [r["name"] for r in rows if r["cls"] == c] for c in ROUTED}
    out = sorted((r for r in rows if r["cls"] in SLOTTED), key=lambda r: r["qpos_start"])
    if len(out) > K_MAX:
        raise RuntimeError(
            f"[EXTRA_SLOTS] {os.path.basename(tree_dir)}: slot targets {len(out)} > K_MAX {K_MAX} "
            f"({[r['name'] for r in out]}) — nothing is dropped"
        )
    return out, routed


def spawn_key(rows: list[dict]) -> str:
    """Slot counts of one clip by kind: `<moving>/<touched static>/<receiving static>` (KINDS order)."""
    n = {
        k: sum(1 for r in rows if r["cls"] in SLOTTED and KIND_BUCKET.get(r.get("kind"), r.get("kind")) == k)
        for k in KINDS
    }
    return "/".join(str(n[k]) for k in KINDS)


def spawn_dist_text(dist: dict) -> str:
    return XG.dist_text(dist, KINDS)
