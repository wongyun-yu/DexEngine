"""The single owner of the hand x dataset feature schema ("form"): one schema, sides in meta, absent-feature marks.

All bimanual, right and left episodes of one hand x dataset share the same features. store.store writes this form
(piece form = final form); the merge only checks that forms are equal (store.merge_check.shape_diff). Features:
  observation.joint_pos.robot [J]  robot joint state; order = joint_order (bimanual order: right articulation (head,
                                   right arm, right hand) then left articulation (left arm, left hand)); names = joint
                                   names. A one-hand record (one articulation) split stably with split_artic.side_of
                                   gives the same order as the two articulations of a bimanual record.
  action.joint_target.robot [J]    joint targets, same order and names
  action [2W]                      policy action: one action term per side (SIDES order), width W each (both sides
                                   have the same hand, so the same width). In a one-hand episode the resting arm's slot
                                   is the hold command (zero residual, RESTING_RULE; line action_fill).
  observation.object_pose.<obj>    per side the main object main_object(side) = that hand's object pose (one object
                                   for both hands: the same pose in both, shared_object) + extra slots 0..K-1
                                   (name = rl.tasks.scene.extra_slots.PREFIX + k)
  observation.object_joint_pos.<obj> [J]  joint values of articulated objects, paired with object_pose. J = the
                                   hand x dataset statement's object_joints (store.slots; 1 if any articulated object,
                                   task_info articulation has one joint); J = 0 means no such features; rigid / empty
                                   slots = NaN (+ absent_features). Joint names go to the line object_joint_names.
  reference.object_pose.<side> · tracking.pos_err_m.<side> · tracking.rot_err_rad.<side>   one set per side
Absent features are NaN. Episode line fields: side (both|right|left), resting_arm, absent_features (wholly NaN
features), action_fill (per side recorded | hold_residual0), object_present ({obj: bool}), side_object ({side: main
object entity}), shared_object, and scene helpers outside the object features (first-frame pose): supports (support
discs) and support_objects (receiving objects). info.collect.form = spec() (form rules, K, fill rules). K = the max
number of extra slots (data.objects.extra_slots.slot_rows) over the clips of the hand x dataset input trees
(statement owner: store.slots).

Receiving objects: the training env places receiving objects (task_info objects with role=support: static objects
holding up an ungrasped main object) in kinematic slots support_obj_<k> fixed at their first-frame pose. Like support
discs they are scene helpers outside the object features; the line field SUPPORT_OBJECTS_KEY holds per slot
{slot, entity, name, obj_id, pose (first frame, 7)}. The object in slot k = the env's slot assignment
placement.coords.support_slot_objects(tree)[k]. Slots k >= n are stand-ins for other clips of the same bundle (the
env parks them at coords.SLOT_HIDDEN_Z): checked against that pose, not stored.

RESTING_RULE and FILL_RULE are description texts stored in info.collect.form (outside the fingerprint).
"""

from __future__ import annotations

import functools
import json

import numpy as np

from dexengine.core.errors import input_missing
from dexengine.record.store.lerobot import ABSENT

MARK = "[COLLECT_FORM]"
# Form version (integer). Bump it by hand when the structure or meaning of the form changes (feature names, order,
# dtype, shape, names, units, fill rules). The merge compares fingerprints only: a different structure with the same
# version dies with "bump the version" (form_diff). The description fields (FORM_DESC) are not compared.
FORM_VERSION = 4  # 4 = articulated-object joint features (observation.object_joint_pos.*)
FORM_DESC = ("resting", "fill", "k_source")  # description fields of info.collect.form, left out of the fingerprint
FORM_KEY = "form"  # field of info.collect
BOTH = "both"  # line side value of a bimanual episode (one-hand: the side name)
JOINT = "observation.joint_pos."  # recorded columns per entity; form column = JOINT + ROBOT
TARGET = "action.joint_target."
ROBOT = "robot"
ACTION = "action"
OBJ = "observation.object_pose."
OBJ_JOINT = "observation.object_joint_pos."  # articulated-object joint values (recorded per entity)
JOINTS_KEY = "object_joints"  # info.collect.form field and statement field: joint width J per form object
JOINT_NAMES_KEY = "object_joint_names"  # line field {obj: [joint names]}, only objects with joints
REF = "reference.object_pose."
TRACK = ("tracking.pos_err_m.", "tracking.rot_err_rad.")
SHARED_KEY = "shared_object"  # line field: both hands share one object (same pose in both main slots)
FILL_KEY = "action_fill"  # line field: source of each side's action slot {side: RECORDED | HOLD}
RECORDED = "recorded"  # recorded policy output (also the bimanual idle side: its residual remains)
HOLD = "hold_residual0"  # one-hand resting arm = hold command (wrist delta 0, finger residual 0)
HOLD_TERM = "ResidualArmIKAction"  # the action term class for which zero residual is a hold (class name in
#                                       rl/tasks/mdp; checked against the recorded term_modes, both bases on)
#: Name prefix of support discs (kinematic helpers the training env puts under floating static objects; not in the
#: capture). The name is built in dexengine/rl/env_setup/build.py (`f"support_disc_{k}"`).
SUPPORT_PREFIX = "support_disc_"
SUPPORT_OBJECTS_KEY = "support_objects"  # line field next to supports
STAND_IN_TOL = 1e-2  # m; kinematic with gravity off, so a parked stand-in does not move
RESTING_RULE = (
    "in a one-hand episode the resting arm's action columns = hold command = residual 0 (wrist Δ 0 · finger residual "
    "0) — the recorded action is the ResidualArmIKAction residual (reference-based · trajectory-based), so finger "
    "target = lo + ½(clamp(base(reference) + residual·rs) + 1)·range · wrist target = reference wrist ⊕ residual Δ. "
    "When the resting arm's base is the rest pose (hand_surface.rest_finger_q · arm default pose — the base the "
    "bimanual resting hand idle_side uses), residual 0 yields that pose as it is. The record statement (action "
    "column term_modes) must name that term with both bases on, else it dies · the line action_fill names the "
    "source per side (recorded | hold_residual0)"
)
FILL_RULE = (
    "one hand × dataset = one form (collect_form) · missing columns = NaN · line absent_features (whole) · stats "
    "(stats_skip): a wholly absent column is left out of that episode's stats (LeRobot aggregate_stats gathers only "
    "the episodes that have the column) · actions are always filled (resting arm = hold command) → action stats = "
    "every episode"
)


def die(msg: str):
    raise SystemExit(f"{MARK} {msg}")


@functools.lru_cache(maxsize=1)
def _shared():
    """(split_artic: side rule side_of and mirror · extra_slots: slot name PREFIX · qpos_layout: side order SIDES)."""
    from dexengine.core import qpos_layout
    from dexengine.rl.tasks.scene import extra_slots, split_artic

    return split_artic, extra_slots, qpos_layout


@functools.lru_cache(maxsize=1)
def support_obj_prefix() -> str:
    """Receiving-object slot name prefix (owner: rl.tasks.scene.object_pin.SUPPORT_OBJ_PREFIX)."""
    from dexengine.rl.tasks.scene import object_pin

    return object_pin.SUPPORT_OBJ_PREFIX


def sides() -> tuple[str, ...]:
    return tuple(_shared()[2].SIDES)


def other(side: str) -> str:
    return next(s for s in sides() if s != side)


def joint_order(names: list[str]) -> list[str]:
    """Bimanual joint order: the given order split stably into not-left (right, neutral) | left
    (split_artic.side_of)."""
    sa = _shared()[0]
    return [n for n in names if sa.side_of(n) != "left"] + [n for n in names if sa.side_of(n) == "left"]


def mirror(name: str, to: str) -> str:
    """Move a name to side `to`, per dot-separated part (action dim names "<term>.<joint>" carry the side in both).
    Uses the hand-spec mirror rule (first side word of a part) and its inverse; parts without a side word
    (wrist_pos_x) are kept. Dies if nothing changed or the inverse does not round-trip."""
    sa = _shared()[0]
    fwd = sa._to_left if to == "left" else (lambda n: n.replace("left_", "right_", 1))
    back = (lambda n: n.replace("left_", "right_", 1)) if to == "left" else sa._to_left
    parts = name.split(".")
    out = [fwd(x) if sa.side_of(x) == other(to) else x for x in parts]
    if out == parts or [back(y) if y != x else x for x, y in zip(parts, out, strict=True)] != parts:
        die(input_missing(f"cannot mirror name {name} to side {to} (no side word, or the inverse differs)"))
    return ".".join(out)


def main_object(side: str) -> str:
    """Main object entity of a side: object_<side> (checked against the recorded meta.sides in check_sides)."""
    return f"object_{side}"


def shared(side_object: dict) -> bool:
    """Do both hands share one object (both sides map to the same entity)?"""
    return len(side_object) == len(sides()) and len(set(side_object.values())) == 1


def main_source(col: str, side_object: dict, present: set, live: set) -> str | None:
    """Recorded entity for form object column col: main column object_<side> = that side's object (side_object; a
    shared object fills both); a side without an object takes a same-named recorded entity; extra columns only occupied
    slots (live). None = NaN."""
    for s in sides():
        if col == main_object(s):
            ent = side_object.get(s) or (col if col in present else None)
            return ent if ent in present else None
    return col if col in present and col in live else None


def extra_name(k: int) -> str:
    return f"{_shared()[1].PREFIX}{k}"


def objects(K: int) -> list[str]:
    return [main_object(s) for s in sides()] + [extra_name(k) for k in range(K)]


def check_sides(meta_sides: list[dict], present: set[str]) -> dict:
    """Recorded meta.sides -> {side: main object entity}; the entity must be object_<side> or the other side's
    (shared object) and must be recorded."""
    out = {}
    for s in meta_sides:
        side, ent = s["side"], s["entity"]
        if side not in sides() or ent not in {main_object(side), main_object(other(side))} or ent not in present:
            die(input_missing(f"recorded side {side} object {ent} is outside the object_<side> rule or not recorded"))
        out[side] = ent
    return out


#: info.collect fields that may differ per episode and are stored as a union (values themselves are per line).
UNION_FIELDS = ("channels", "force_keys", "force_clean_keys", "spawn_hold_steps_set")


def union_field(key: str, values: list) -> list:
    """Union of one UNION_FIELDS field (store: across episodes; merge: final | piece). channels = distinct recorded
    column specs without the per-record lo/hi, ordered by (name, spec); force keys = sorted union."""
    if key != "channels":
        return sorted({x for v in values for x in v})
    rows = {
        json.dumps({k: x for k, x in c.items() if k not in ("lo", "hi")}, sort_keys=True) for v in values for c in v
    }
    return [json.loads(r) for r in sorted(rows, key=lambda r: (json.loads(r)["name"], r))]


def spec(K: int, k_source: dict, joints: list[str], width: int, J: int = 0) -> dict:
    """info.collect.form: form rules, K, articulated-object width J and fill rules."""
    W = width // 2
    return dict(
        form_version=FORM_VERSION,
        K=K,
        k_source=k_source,
        sides=list(sides()),
        joints=len(joints),
        **{JOINTS_KEY: int(J)},
        action={"width": width, "slots": {s: [i * W, (i + 1) * W] for i, s in enumerate(sides())}},
        objects=objects(K),
        resting=RESTING_RULE,
        fill=FILL_RULE,
    )


def fingerprint(info: dict) -> dict:
    """Structural fingerprint of the form: form_version, form fields (without FORM_DESC), features (order, name,
    dtype, shape, names) and units (fps, video fps). Description texts do not change it."""
    c = info.get("collect") or {}
    f = c.get(FORM_KEY) or {}
    return {
        "form_version": f.get("form_version"),
        "form": {k: v for k, v in f.items() if k not in (*FORM_DESC, "form_version")},
        "features": [
            [n, v.get("dtype"), v.get("shape"), v.get("names")] for n, v in (info.get("features") or {}).items()
        ],
        "units": {"fps": info.get("fps"), "video_fps": c.get("video_fps")},
    }


def form_diff(a: dict, b: dict) -> list[str]:
    """Compare the forms of two infos (merge): [] when fingerprints match, else the problems."""
    fa, fb = fingerprint(a), fingerprint(b)
    if fa == fb:
        return []
    keys = [k for k in fa if fa[k] != fb[k]]
    if fa["form_version"] is None or fb["form_version"] is None:
        return [f"collect.{FORM_KEY}.form_version missing (piece stored before form versions; store it again)"]
    if fa["form_version"] == fb["form_version"]:
        return [
            f"form structure {keys} differs with the same form_version ({fa['form_version']}) — "
            "bump store.form.FORM_VERSION"
        ]
    return [f"form version differs {fa['form_version']} != {fb['form_version']} ({keys})"]


def stats_skip(line: dict) -> list[str]:
    """Features to leave out of the episode stats: the line's absent_features (wholly NaN)."""
    return sorted(line.get(ABSENT, []))


# ---- one episode -> form -------------------------------------------------------------------------------------
def _robot(cols, names, prefix, ents) -> tuple[np.ndarray | None, list[str] | None]:
    keys = [prefix + e for e in ents if prefix + e in cols]
    if not keys:
        return None, None
    if len(keys) != len(ents) or any(not names.get(k) for k in keys):
        die(
            input_missing(
                f"{prefix}* joint columns {keys}: some of articulations {ents} are missing or have no joint names"
            )
        )
    return np.concatenate([np.asarray(cols[k]) for k in keys], axis=1), [n for k in keys for n in names[k]]


def _reorder(a, joints_in: list[str], joints_form: list[str], what: str) -> np.ndarray:
    if sorted(joints_in) != sorted(joints_form) or len(set(joints_in)) != len(joints_in):
        die(
            f"{what} joint names != form joints (extra {sorted(set(joints_in) - set(joints_form))[:3]} · "
            f"missing {sorted(set(joints_form) - set(joints_in))[:3]})"
        )
    ix = {n: i for i, n in enumerate(joints_in)}
    return a[:, [ix[n] for n in joints_form]]


def check_hold_term(c: dict, clip: str) -> None:
    """Zero residual is a hold only for HOLD_TERM with reference_base and traj_ref on: check the action column's
    term_modes statement (missing or different is an error)."""
    modes = c.get("term_modes")
    bad = (
        modes is None
        or len(modes) != len(c["names"])
        or any(m.get("cls") != HOLD_TERM or not m.get("reference_base") or not m.get("traj_ref") for m in modes)
    )
    if bad:
        die(
            input_missing(
                f"{clip}: a zero-residual hold for the resting arm needs {HOLD_TERM} (reference and "
                f"trajectory base) — action term_modes {modes} (none = record again)"
            )
        )


def _action(cols, channels, side, L, clip):
    """Recorded action column -> (form [L, 2W], names | None, action_fill). Side of a term = side_of(term name);
    the resting arm gets the hold command."""
    sa = _shared()[0]
    ch = [c for c in channels if c["kind"] == ACTION]
    if len(ch) != 1 or ACTION not in cols:
        die(f"{len(ch)} recorded action columns — need one")
    c, a = ch[0], np.asarray(cols[ACTION])
    terms, dims = list(c["names"]), list(c.get("term_dims") or ([a.shape[1]] if len(c["names"]) == 1 else []))
    if len(dims) != len(terms) or sum(dims) != a.shape[1]:
        die(input_missing(f"action terms {terms} widths {dims} != column width {a.shape[1]} (no term_dims)"))
    tside = [sa.side_of(t) for t in terms]
    want = sides() if side == BOTH else (side,)
    if sorted(tside, key=str) != sorted(want) or len(set(dims)) != 1:
        die(f"action term sides {tside} != episode sides {list(want)}, or term widths differ {dims}")
    W, dn = dims[0], c.get("dim_names")
    out, names, lo = np.zeros((L, len(sides()) * W), dtype=np.float32), [None] * (len(sides()) * W), 0
    for _t, s, d in zip(terms, tside, dims, strict=True):
        k = sides().index(s) * W
        out[:, k : k + W] = a[:, lo : lo + d]
        if dn:
            names[k : k + W] = dn[lo : lo + d]
        lo += d
    fill = dict.fromkeys(tside, RECORDED)
    if side != BOTH:  # resting arm slot = hold command (the zeros above)
        check_hold_term(c, clip)
        r, k = other(side), sides().index(other(side)) * W
        fill[r] = HOLD
        if dn:
            src = sides().index(side) * W
            names[k : k + W] = [mirror(n, r) for n in names[src : src + W]]
    return out, (names if dn else None), {s: fill[s] for s in sides()}


def _object_joints(cols, names, o, src, J, L, clip) -> tuple[np.ndarray, list | None]:
    """Joint values [L, J] of form object o (OBJ_JOINT column of recorded entity src, NaN-padded to J), names | None."""
    raw = OBJ_JOINT + src if src else None
    a = np.asarray(cols[raw]) if raw in cols else None
    if a is None or a.shape[1] == 0:
        return np.full((L, J), np.nan, dtype=np.float32), None
    if a.shape[1] > J:
        die(f"{clip}: {src} has {a.shape[1]} joints > statement J={J} (stale statement)")
    out = np.full((L, J), np.nan, dtype=a.dtype)
    out[:, : a.shape[1]] = a
    return out, list(names.get(raw) or [])


def support_objects(cols: dict, present: set, obj: str, prefix: str, meta: dict, clip: str, die) -> tuple[list, set]:
    """Receiving-object slots (prefix<k>) among the recorded objects -> (line meta rows, recorded names to drop from
    the object features). cols[obj + name] = recorded pose column (L, 7); meta = record meta (env.sequence_dir = the
    tree root the env read); die = the form's error function."""
    rec = {o for o in present if o.startswith(prefix)}
    if not rec:
        return [], set()
    ks = {}
    for o in sorted(rec):
        if not o[len(prefix) :].isdigit():
            die(f"{clip}: recorded object {o} is not a receiving-object slot name {prefix}<k>")
        ks[int(o[len(prefix) :])] = o
    from dexengine.record import tree_ref as TR

    seq = (meta.get("env") or {}).get("sequence_dir")
    tree, why = TR.meta_tree(seq, clip)
    if tree is None:
        die(
            f"{clip}: no tree to resolve the objects of receiving slots {sorted(rec)} (record meta "
            f"env.sequence_dir={seq!r} · {why})"
        )
    from dexengine.placement.coords import core as CO

    ti = json.loads((tree / "task_info.json").read_text(encoding="utf-8"))
    names = CO.support_slot_objects(tree, ti, clip)
    by = {str(o.get("name")): o for o in ti.get("objects") or []}
    roles = sorted(n for n, o in by.items() if o.get("role") == "support")
    if bad := [n for n in names if (by.get(n) or {}).get("role") != "support"]:
        die(
            f"{clip}: env receiving-slot assignment {names}: {bad} have tree role != support "
            f"({[(by.get(n) or {}).get('role') for n in bad]})"
        )
    if len(names) != len(roles):
        die(f"{clip}: {len(names)} env receiving slots ({names}) != {len(roles)} tree role=support objects ({roles})")
    if miss := [f"{prefix}{k}" for k in range(len(names)) if k not in ks]:
        die(
            f"{clip}: receiving slots {miss} ({[names[int(m[len(prefix) :])] for m in miss]}) are not recorded — "
            f"recorded {sorted(rec)}"
        )
    out = []
    for k, o in sorted(ks.items()):
        pose = [float(x) for x in np.asarray(cols[obj + o])[0]]
        if k >= len(names):
            if abs(pose[2] - CO.SLOT_HIDDEN_Z) > STAND_IN_TOL:
                die(
                    f"{clip}: {o} is a stand-in slot (assignment has {len(names)}) but its first z {pose[2]:.4f} != "
                    f"parked {CO.SLOT_HIDDEN_Z}"
                )
            continue
        out.append({"slot": k, "entity": o, "name": names[k], "obj_id": str(by[names[k]]["obj_id"]), "pose": pose})
    return out, rec


def form_episode(ep: dict, entities: list[dict], meta: dict, clip: str, K: int, J: int = 0) -> dict:
    """store.store episode columns (after render_fix) -> form columns. J = articulated-object width (statement;
    0 = no such features, then recorded articulated objects are an error).

    Returns dict(cols, names, absent, line, joints, width)."""
    cols, names, L = ep["cols"], ep["names"], ep["L"]
    emb = meta["env"].get("embodiment")
    if emb not in (*sides(), "bimanual"):
        die(f"{clip}: recorded embodiment {emb!r} — must be one of {list(sides())} or bimanual")
    side = BOTH if emb == "bimanual" else emb
    robots = [e["name"] for e in entities if JOINT + e["name"] in cols]
    st, jn = _robot(cols, names, JOINT, robots)
    if st is None:
        die(input_missing(f"{clip}: no joint state columns"))
    order = joint_order(jn)
    out = {
        "timestamp": cols["timestamp"],
        "frame_index": cols["frame_index"],
        JOINT + ROBOT: _reorder(st, jn, order, "joint state"),
    }
    nm, absent = {JOINT + ROBOT: order}, []
    tg, tn = _robot(cols, names, TARGET, robots)
    out[TARGET + ROBOT] = (
        _reorder(tg, tn, order, "joint target")
        if tg is not None
        else np.full((L, len(order)), np.nan, dtype=np.float32)
    )
    nm[TARGET + ROBOT] = order
    absent += [TARGET + ROBOT] * (tg is None)
    present = {k[len(OBJ) :] for k in cols if k.startswith(OBJ)}
    supports = {
        o: [float(x) for x in np.asarray(cols[OBJ + o])[0]] for o in sorted(present) if o.startswith(SUPPORT_PREFIX)
    }
    present -= set(supports)  # support discs are scene helpers, outside the object features
    sobj, gone = support_objects(cols, present, OBJ, support_obj_prefix(), meta, clip, die)
    present -= gone  # receiving-object slots are scene helpers too
    side_object = check_sides(meta["sides"], present)
    objs = objects(K)
    if extra := sorted(present - set(objs)):
        die(f"{clip}: recorded objects {extra} are outside the form (main + extras 0..{K - 1})")
    slots = (meta.get("extra_slots") or {}).get(clip) or []
    if any(o not in map(main_object, sides()) for o in present) and "extra_slots" not in meta:
        die(input_missing(f"{clip}: extra entities recorded but the record meta has no extra_slots statement"))
    live = {extra_name(int(d["slot"])) for d in slots}
    if not J and (art := sorted(k for k in cols if k.startswith(OBJ_JOINT) and np.asarray(cols[k]).shape[1] > 0)):
        die(f"{clip}: articulated objects {art} recorded but the statement says J=0 (stale statement)")
    obj_present, jnames = {}, {}
    for o in objs:
        src = main_source(o, side_object, present, live)
        obj_present[o] = src is not None
        out[OBJ + o] = np.asarray(cols[OBJ + src]) if src else np.full((L, 7), np.nan, dtype=np.float32)
        absent += [OBJ + o] * (src is None)
        if J:
            out[OBJ_JOINT + o], jn = _object_joints(cols, ep["names"], o, src, J, L, clip)
            absent += [OBJ_JOINT + o] * (jn is None)
            if jn is not None:
                jnames[o] = jn
    for s in sides():
        for k in (REF, *TRACK):
            have = k + s in cols
            out[k + s] = (
                cols[k + s]
                if have
                else np.full((L, 7) if k == REF else (L,), np.nan, dtype=np.float32 if k == REF else np.float64)
            )
            absent += [k + s] * (not have)
    out[ACTION], nm[ACTION], fill = _action(cols, ep["channels"], side, L, clip)
    line = dict(
        side=side,
        resting_arm=None if side == BOTH else other(side),
        **{FILL_KEY: fill},
        object_present=obj_present,
        side_object=side_object,
        **{SHARED_KEY: shared(side_object)},
        supports=supports,
        **{SUPPORT_OBJECTS_KEY: sobj},
        **({JOINT_NAMES_KEY: jnames} if J else {}),
    )
    return {"cols": out, "names": nm, "absent": absent, "line": line, "joints": order, "width": out[ACTION].shape[1]}
