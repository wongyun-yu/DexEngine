"""Binding entry of the record replay render: fill empty extra slots that exist only in the scene with the hidden pose,
then call the canonical binding (traj_record.bind).

Why: the render scene is built from every clip of the session (slot count K = the per-clip maximum,
build_extra_slots). An augment-round session loads the original and the placement variants into one scene, but the
original trajectory was recorded with that clip alone: slots only the variants have (objects that become 'touched
static' depending on the placement) are missing from the original trajectory file and exist only in the scene. bind
has no state for scene-only entities and dies. The scene statement fixes those bodies' state: an empty place of that
trajectory gets the hidden pose (`dexengine.rl.tasks.scene.extra_slots.hidden_root` = the value the reset `_pose` writes
to empty places). So those columns are appended to the file entity list and the end of the chosen trajectory, and bind
alone checks (bind itself is unchanged). Scene-only entities that the statement does not call empty are not filled: bind
kills them as before.
Support disc slots (`support_disc_<j>`, _build_disc_slots; slot count = the fixed session K) work the same way: the
original stage 1 is recorded in a one-clip scene without discs, and when a variant uses discs they exist only in the
render scene ('scene-only support_disc_*'). Empty disc place = slots whose row of that clip in the reset event table
(`reset_support_discs` params table; the reader object_pin.reset_discs uses the rule 'z > -5 = a real disc') has
z <= -5; pose = the value reset_discs writes to empty places ([x, y, hidden_z], unit quaternion). A real disc by the
statement that is missing from the file is not filled: bind kills it.
Marker `[REPLAY_MAP] <clip>: empty slots by the scene statement [...] … filled with hidden pose …` (only when filled).
Slot renumbering (`renumber_slots`): in a one-env render with axis alignment (`cfg.extra_slot_axis`,
_extra_slot_axis; other contexts have no field and pass through) the scene slot number = the object identity axis
across clips, while the file's (stage-1 record) slot number = that clip's own number in the record session (rank of
the remaining rows); the file names are moved to axis numbers before bind matches by name (check = slot -> obj of the
record meta extra_slots[clip] vs the scene statement extra_slot_objs). Marker `[REPLAY_MAP] <clip>: slot numbers …`.
The opposite direction (`compact_recorded_axis`): when the scene has no axis (one clip) but the record session (one
env in stage 1) had one, file slot numbers = record axis fields; they move to the clip's own numbers by the slot rank
of the record meta extra_slots[clip], the rest become extra_obj_rec<s> (hidden check = bind).
Part of the render fingerprint (record.cache.RENDER_CODE_FILES).
"""

from __future__ import annotations

from dexengine.core.errors import InputMissingError


def pad_empty_slots(cfg, k: int, clip: str, ents: list[dict], st):
    """(entity list, trajectory) with the `extra_obj_*` that the scene statement (`cfg.extra_slot_valid`,
    `extra_slot_clips`) calls empty for loader clip k but that are missing from the file appended as hidden-pose
    columns at the very end of the trajectory (after the channel columns). Nothing to fill returns the input. The
    statement's clip k != this trajectory, or a file entity column end > trajectory width, dies."""
    import numpy as np

    from dexengine.core import paths as P

    xe = P.load_registry("extra_slots_env")  # owner of empty slots and the hidden pose
    names = list(getattr(cfg, "extra_slot_clips", None) or [])
    if names and P.tree_name(names[k]) != P.tree_name(clip):
        raise SystemExit(f"[REPLAY_MAP] {clip}: clip {k} of the slot statement is {names[k]}, another clip")
    have = {e["name"] for e in ents}
    rows = [(n, xe.hidden_root(cfg)) for n in xe.empty_slot_names(cfg, k) if n not in have]
    rows += [(n, r) for n, r in empty_disc_roots(cfg, k) if n not in have]
    if not rows:
        return ents, st
    pad = [n for n, _r in rows]
    D = int(st.shape[1])  # trajectory width = entity columns + channel columns; fill at the end
    if max(int(e["hi"]) for e in ents) > D:
        raise SystemExit(
            f"[REPLAY_MAP] {clip}: file entity column end {max(int(e['hi']) for e in ents)} > trajectory width {D}"
        )
    add, lo = [], D
    for n, r in rows:
        add.append({"name": n, "kind": "rigid", "joint_names": [], "aliases": [], "lo": lo, "hi": lo + len(r)})
        lo += len(r)
    col = np.concatenate([np.tile(np.asarray(r, dtype=st.dtype), (st.shape[0], 1)) for _n, r in rows], axis=1)
    print(
        f"[REPLAY_MAP] {clip}: empty slots by the scene statement {pad}, missing from the file, filled with hidden "
        f"pose {[r for _n, r in rows]} (render scene = every session clip)",
        flush=True,
    )
    return [*ents, *add], np.concatenate([st, col], axis=1)


def empty_disc_roots(cfg, k: int) -> list[tuple[str, list[float]]]:
    """[(disc slot name, env-local root pose [x y z | qw qx qy qz])]: from the reset event table
    (`cfg.events.reset_support_discs` params; writer _build_disc_slots) the **empty** disc slots of loader clip k (row
    z <= -5; the reader object_pin.reset_discs uses 'z > -5 = a real disc') and the pose reset_discs writes there
    ([x, y, hidden_z], unit quaternion). A scene without discs -> []. A clip number outside the table dies."""
    ev = getattr(getattr(cfg, "events", None), "reset_support_discs", None)
    if ev is None:
        return []
    prm = ev.params
    table, names, hz = prm["table"], list(prm["disc_names"]), float(prm["hidden_z"])
    if not 0 <= int(k) < len(table):
        raise SystemExit(f"[REPLAY_MAP] clip number {k} outside the disc table's {len(table)} clips")
    return [
        (names[j], [float(r[0]), float(r[1]), hz, 1.0, 0.0, 0.0, 0.0])
        for j, r in enumerate(table[int(k)])
        if float(r[2]) <= -5.0
    ]


def renumber_slots(cfg, k: int, clip: str, ents: list[dict], meta: dict) -> list[dict]:
    """Entity list with the file slot entities (`extra_obj_<j>`) renamed to scene axis numbers; without
    `cfg.extra_slot_axis` the input as it is (training, judging, single-clip original render: byte-identical). File j =
    this clip's j-th slot in the record session (its own number) -> scene field `extra_slot_axis[k][j]`. Slots the
    record session did not stand up for this clip (j >= its row count: hidden stand-ins) become `extra_obj_rec<j>` and
    stay in the file only; bind checks they are at hidden z in every frame, else kills.
    Check: the object of slot j in the record meta extra_slots[clip] != the scene statement extra_slot_objs[k][field],
    the record has no or more own slots, or the file lacks its own slot entities: dies."""
    axis = getattr(cfg, "extra_slot_axis", None)
    if axis is None:
        return compact_recorded_axis(cfg, k, clip, ents, meta)
    from dexengine.core import paths as P

    xe = P.load_registry("extra_slots_env")
    names = list(cfg.extra_slot_clips)
    if P.tree_name(names[k]) != P.tree_name(clip):
        raise SystemExit(f"[REPLAY_MAP] {clip}: clip {k} of the slot statement is {names[k]}, another clip")
    pre, where, objs = xe.PREFIX, [int(a) for a in axis[k]], cfg.extra_slot_objs[k]
    rec = ((meta or {}).get("extra_slots") or {}).get(clip)
    if rec is None and where:
        raise InputMissingError(
            f"[REPLAY_MAP] {clip}: the record meta extra_slots lacks this clip: the objects of the "
            "file slot numbers are unknown"
        )
    said = {int(r["slot"]): r.get("obj") for r in rec or []}
    bad = [
        f"slot {j}: record {said.get(j, 'none')!r} != scene field {a} {objs[a]!r}"
        for j, a in enumerate(where)
        if j not in said or (said[j] is not None and str(said[j]) != str(objs[a]))
    ]
    bad += [
        f"record slot {j} {o!r}: the scene statement gives this clip {len(where)} slots"
        for j, o in said.items()
        if j >= len(where) and o is not None
    ]
    if bad:
        raise SystemExit(f"[REPLAY_MAP] {clip}: record session slots != this scene's slots: {' · '.join(bad[:4])}")
    out, got = [], set()
    for e in ents:
        n = str(e["name"])
        if n.startswith(pre) and n[len(pre) :].isdigit():
            j = int(n[len(pre) :])
            got.add(j)
            e = dict(e, name=f"{pre}{where[j]}" if j < len(where) else f"{pre}rec{j}")
        out.append(e)
    if miss := [j for j in range(len(where)) if j not in got]:
        raise InputMissingError(
            f"[REPLAY_MAP] {clip}: the file lacks its own slot entities {[f'{pre}{j}' for j in miss]}"
        )
    if where != list(range(len(where))) or len(got) > len(where):
        print(
            f"[REPLAY_MAP] {clip}: slot numbers file -> scene axis "
            + " ".join(f"{j}→{a}({objs[a]})" for j, a in enumerate(where))
            + (
                f" · record hidden stand-ins {sorted(j for j in got if j >= len(where))} -> {pre}rec<j> (file only)"
                if len(got) > len(where)
                else ""
            )
            + " (one-env render axis alignment · check = record meta extra_slots)",
            flush=True,
        )
    return out


def compact_recorded_axis(cfg, k: int, clip: str, ents: list[dict], meta: dict) -> list[dict]:
    """The other direction of `renumber_slots`: the scene has no slot axis (one clip, or not a one-env render) but the
    recording session did — a one-env stage 1 over clips whose slot identities differ (e.g. the same static extra is
    `receiver` in one tree and `touched` in another — SPIDER variants vs their kin orig) put this clip's
    slots at its axis cells, so the file's `extra_obj_<s>` numbers are axis cells, not the clip's own numbers.
    Rule: the clip's own slot j = the j-th smallest recorded slot in meta extra_slots[clip]; file `extra_obj_<s>` of
    that slot → `extra_obj_<j>`; every other file slot entity → `extra_obj_rec<s>` (file only — `traj_record.bind`
    keeps it only if it is hidden in every frame, else dies). Recorded slots 0..n-1 (own numbering, every earlier
    file) = input unchanged. Checks: recorded count = the scene's valid slots for this clip · recorded obj =
    scene obj per slot — else dies."""
    rec = ((meta or {}).get("extra_slots") or {}).get(clip)
    if not rec:
        return ents
    slots = sorted(int(r["slot"]) for r in rec)
    if slots == list(range(len(slots))):
        return ents
    from dexengine.core import paths as P

    xe = P.load_registry("extra_slots_env")
    pre = xe.PREFIX
    names = list(getattr(cfg, "extra_slot_clips", None) or [])
    if not names or P.tree_name(names[k]) != P.tree_name(clip):
        raise SystemExit(
            f"[REPLAY_MAP] {clip}: clip {k} of the slot statement is {names[k] if names else None}, another clip"
        )
    valid = [bool(v) for v in (getattr(cfg, "extra_slot_valid", None) or [[]])[k]]
    objs = list((getattr(cfg, "extra_slot_objs", None) or [[]])[k])
    if valid != [True] * len(slots) + [False] * (len(valid) - len(slots)):
        raise SystemExit(
            f"[REPLAY_MAP] {clip}: record slots {slots} (axis fields) {len(slots)} != this scene's clip slots {valid}"
        )
    said = {int(r["slot"]): r.get("obj") for r in rec}
    bad = [
        f"slot {j}: record field {s} {said[s]!r} != scene {objs[j]!r}"
        for j, s in enumerate(slots)
        if said[s] is not None and str(said[s]) != str(objs[j])
    ]
    if bad:
        raise SystemExit(f"[REPLAY_MAP] {clip}: record session slots != this scene's slots: {' · '.join(bad[:4])}")
    own = {s: j for j, s in enumerate(slots)}
    out, got = [], set()
    for e in ents:
        n = str(e["name"])
        if n.startswith(pre) and n[len(pre) :].isdigit():
            s = int(n[len(pre) :])
            got.add(s)
            e = dict(e, name=f"{pre}{own[s]}" if s in own else f"{pre}rec{s}")
        out.append(e)
    if miss := [s for s in slots if s not in got]:
        raise InputMissingError(
            f"[REPLAY_MAP] {clip}: the file lacks record slot entities {[f'{pre}{s}' for s in miss]}"
        )
    print(
        f"[REPLAY_MAP] {clip}: record session axis fields -> the clip's own numbers "
        + " ".join(f"{s}→{j}({objs[j]})" for s, j in own.items())
        + f" · other file slots {sorted(got - set(own))} -> {pre}rec<s> (file only · hidden check = bind)"
        + " (no scene axis · check = record meta extra_slots)",
        flush=True,
    )
    return out


def bind(scene, cfg, k: int, clip: str, tf, st, dev):
    """(binding, trajectory): `pad_empty_slots`, then the canonical binding `traj_record.bind` (the hidden z of the
    file-only empty slot rule = the scene builder's `AERIAL_HIDDEN_Z`). The trajectory is the filled one; writing
    (write_states) uses it too."""
    from dexengine.rl.env_setup.objects import AERIAL_HIDDEN_Z
    from dexengine.rl.eval import traj_record as TR

    ents = renumber_slots(cfg, k, clip, tf.entities, tf.meta)
    ents, st = pad_empty_slots(cfg, k, clip, ents, st)
    return TR.bind(scene, ents, dev, trajs=[st], hidden_z=AERIAL_HIDDEN_Z), st
