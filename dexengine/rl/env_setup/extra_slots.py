"""Extra object slots of the scene: which static objects of a clip are spawned, their poses per env and their collision
partners."""

from __future__ import annotations

import os
from collections import Counter

from dexengine.data.objects import collider_shape as _cs  # owner of each slot body's collider shape
from dexengine.data.objects import extra_slots as _xs  # owner of the rule and the slot count
from dexengine.rl.env_setup import extra_slots_plan as XP  # Kit-free plan, shared with the trainer preflight
from dexengine.rl.env_setup.extra_slots_ledger import _extra_slots_marker, extra_slots_requested
from dexengine.rl.env_setup.objects import AERIAL_HIDDEN_Z, _render_object_usds, _slot_entity


def _extra_slot_assets(_su, _cp, rows: list, k: int, fill: tuple | None) -> list:
    """Per-clip assets of slot k (env i -> clip i % C): kinematic bodies; an empty place absent from a clip gets another
    clip's USD of the same slot, hidden. Colliders follow the statement `collision_presets.extra_has_collider`: only
    kinds with an ON partner in their table get one; all-OFF kinds and empty places stand with collision off (bodies
    with
    nothing to collide with; extra objects stand per env without physics replication, so their collision meshes cost
    memory per env). The authoring `author_extra_collision` measures the stage against this statement. The stand-in USD
    of empty places is chosen by the owner `scene.extra_slots.stand_in` (switch DEXENGINE_EXTRA_STANDIN) and logged in
    one marker line (stand-ins are hidden and have no collider, so the physics does not depend on the rule). Valid slots
    spawn through the same path as main objects (`_render_object_usds`: texture overlays in a render context with a
    `[RENDER_OBJ_USD] extra slot k` marker, the training USD as is in training). The asset identity
    (`cfg.extra_slot_usd`) is the training USD.
    `fill` = filling empty places in a one-env render (`_extra_slot_fills`; render context, one env, several clips only;
    otherwise None = the stand-in body above, byte-identical): an empty place stands that object instead of the
    stand-in (overlay path, visible, collider off; the render does not run physics), so the valid slots of variants
    placed on env 0 are visible. Marker `[EXTRA_SLOTS] empty place fill slot k`."""
    from dexengine.placement.coords import core as _coords  # noqa: F401 — import kept for its side effect
    from dexengine.rl.tasks.scene import extra_slots as _xe  # the registry window (one module, no Isaac app, tests too)

    n_empty = sum(_slot_cell(r, k) is None for r in rows)
    if fill is None:
        rule = _xe.standin_rule()
        stand_in = _xe.stand_in([[None if x is None else x["usd"] for x in r] for r in rows], k, rule)
        print(
            f"{_xe.MARK} empty place stand-in body slot {k}: rule {rule} · {stand_in} "
            f"{os.path.getsize(stand_in)} B · empty places {n_empty}/{len(rows)} clips",
            flush=True,
        )
    else:
        stand_in = _render_object_usds([fill[0]], f"extra slot {k} empty place fill")[0]
        print(
            f"{_xe.MARK} empty place fill slot {k}: one-env render — object of the clips with the slot {fill[0]} · "
            f"kind {fill[1]} · "
            f"spawn {stand_in} · empty places {n_empty}/{len(rows)} clips · visible · collider off (instead of the "
            f"stand-in body)",
            flush=True,
        )
    off = _su.CollisionPropertiesCfg(collision_enabled=False)
    kinds = [None if (x := _slot_cell(r, k)) is None else x["kind"] for r in rows]  # empty place = None
    have = [i for i, r in enumerate(rows) if _slot_cell(r, k) is not None]
    spawn = dict(zip(have, _render_object_usds([rows[i][k]["usd"] for i in have], f"extra slot {k}"), strict=True))
    return [
        _su.UsdFileCfg(
            usd_path=spawn.get(i, stand_in),
            visible=_slot_cell(r, k) is not None or fill is not None,
            activate_contact_sensors=True,
            rigid_props=_su.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=None if _cp.extra_has_collider(kind) else off,
        )
        for i, (r, kind) in enumerate(zip(rows, kinds, strict=True))
    ]


def _extra_slot_ids(_cp, rows: list) -> list:
    """Clip × slot identity (training USD, kind statement): the same form as `cfg.extra_slot_usd` · `extra_slot_kinds`
    (reader `scene.extra_slots.ExtraSlots.sig`)."""
    return [[None if x is None else (x["usd"], _cp.extra_collide(x["kind"]) and x["kind"]) for x in r] for r in rows]


def _extra_slot_fills(cfg, _cp, rows: list, K: int) -> list | None:
    """Empty place fill of a one-env render (identity | None per slot).
    On only in: a render context (`robots.selection.render_context` · `_render_ctx`) · one env (`cfg.scene.num_envs`) ·
    several clips. Everywhere else (training, judge, several envs, one clip) None: stand-in bodies and the statement as
    they are (byte-identical). Owner of the rule that picks the object = `scene.extra_slots.render_fill` (a split
    fails).
    Why: the collection render places the original and its variants on env 0 (asset = first sorted clip = the
    original) one after another. When a variant has a slot the original lacks, that place on env 0 is a hidden
    stand-in, the variant's object stays invisible and `_check_clip_binding` fails."""
    if not _one_env_render(cfg, rows, K):
        return None
    from dexengine.rl.tasks.scene import extra_slots as _xe

    ids = _extra_slot_ids(_cp, rows)
    fills = [_xe.render_fill(ids, k) for k in range(K)]
    return fills if any(f is not None for f in fills) else None


def _slot_cell(r: list, k: int):
    """Cell k of clip slot row r (empty place = None). Accepts both the clip's own numbering (slot_rows' compact form,
    only the tail empty) and the axis alignment of a one-env render (`_extra_slot_axis`, None cells in the middle). In
    the compact form it equals `len(r) > k`."""
    return r[k] if len(r) > k else None


def _one_env_render(cfg, rows: list, K: int) -> bool:
    """Is this a one-env render (collection stage 2: the original and its placement variants placed on env 0 one after
    another)? Single owner of the condition that turns on the empty place fill (`_extra_slot_fills`) and the slot axis
    alignment (`_extra_slot_axis`): slots present · several clips · one env · render context
    (`robots.selection.render_context`). Everywhere else (training, judge, several envs, one clip) false; the statement
    stays byte-identical."""
    return bool(K) and len(rows) >= 2 and int(cfg.scene.num_envs) == 1 and _render_ctx()


def _extra_slot_axis(_xs, _cp, rows: list) -> tuple[list, list]:
    """Slot axis of a one-env render: (clip × axis cell = that clip's row | None, per clip [the axis cell of its j-th
    row]).
    Axis = the union of the identities of every clip's slot rows (identity = (qpos column start, training USD, kind
    statement); placement variants of one source clip are rigid transforms, so columns and objects agree) · order =
    qpos column -> USD -> kind (with one clip, the same as slot_rows' compact order).
    Why: when the placement rule (place_rule, place-dependent) drops a static object at a variant's place, the clip's
    own numbering (the rank of the remaining rows) moves up, and slot k of the original on env 0 and slot k of the
    variant become different objects (`ExtraSlots._check_clip_binding` fails). An empty cell on the axis is that clip's
    empty place (hidden, filled by `_extra_slot_fills`); the placement rule's output (what stands) is unchanged.
    The same identity twice in one clip, or an axis above K_MAX, fails."""
    ids = _extra_slot_ids(_cp, rows)
    keys = [[(int(x["qpos_start"]), *i) for x, i in zip(r, ir, strict=True)] for r, ir in zip(rows, ids, strict=True)]
    for c, ks in enumerate(keys):
        if len(set(ks)) != len(ks):
            raise RuntimeError(
                f"[EXTRA_SLOTS] one-env render slot axis: clip {c} has the same identity twice {ks} — the axis "
                f"cell cannot be decided"
            )
    axis = sorted({k for ks in keys for k in ks}, key=lambda t: (t[0], str(t[1]), str(t[2])))
    if len(axis) > _xs.K_MAX:
        raise RuntimeError(
            f"[EXTRA_SLOTS] one-env render slot axis of {len(axis)} cells > K_MAX {_xs.K_MAX} — the union of "
            f"the clips' slot identities {axis}"
        )
    pos = {t: a for a, t in enumerate(axis)}
    where = [[pos[t] for t in ks] for ks in keys]
    cells = [[None] * len(axis) for _ in rows]
    for c, (r, w) in enumerate(zip(rows, where, strict=True)):
        for x, a in zip(r, w, strict=True):
            cells[c][a] = x
    return cells, where


def _full_physics_marker(_cp, rows: list) -> None:
    """Marker `[FULL_PHYSICS]` (switch on only): every slot kind collides with the robot and the main objects."""
    if not _cp.full_physics():
        return
    kinds = Counter(x["kind"] for r in rows for x in r if x is not None)
    print(
        f"{_cp.FULL_PHYSICS_TAG} on · every slot kind collides with the robot and the main objects (objects off: "
        f"{sorted(_cp.FULL_PHYSICS_OBJECTS_OFF)}) · slots by kind {dict(kinds)}",
        flush=True,
    )


def _render_ctx() -> bool:
    """Is this a render context? Owner of the decision: `robots.selection.render_context` (a lazy import, since it pulls
    in the package; tests swap it)."""
    from dexengine.robots.selection import render_context

    return render_context()


def _collider_cell(_cp, x: dict | None) -> str | None:
    """The collider statement of one slot cell: the row's shape (`collider_shape.row_shape`) when a collider stands on
    that kind (`collision_presets.extra_has_collider`), None for an empty place or a body without a collider."""
    return None if x is None or not _cp.extra_has_collider(x["kind"]) else x["shape"]


def _extra_slot_statement(
    cfg, _cp, rows: list, seqs: list, K: int, fills: list | None, axis: list | None = None
) -> None:
    """Cfg statement of the extra slots (clip × slot: valid · kind · name · asset identity · held spans · loader
    columns), read by `scene.extra_slots`, the loader and the collision authoring. `cfg.extra_slot_fill` is written only
    with `fills` (the empty place fill of a one-env render); otherwise the cell is not created (training and judge cfgs
    stay byte-identical). `cfg.extra_slot_axis` is written only with `axis` (the axis alignment of a one-env render: per
    clip [the axis cell of its j-th row]), and then rows may have None cells in the middle (`_slot_cell`)."""
    cfg.extra_object_slots = K
    cfg.extra_slot_clips = list(seqs)
    cfg.extra_slot_valid = [[_slot_cell(r, k) is not None for k in range(K)] for r in rows]
    # Slot kind (the rule row's kind: moving · touched · receiver) -> the key of the collision partners
    # (collision_presets.EXTRA_KIND_COLLIDE, read by the authoring author_extra_collision). Empty place = None. A kind
    # missing from the table fails here.
    cfg.extra_slot_kinds = [
        [None if x is None else _cp.extra_collide(x["kind"]) and x["kind"] for x in r] + [None] * (K - len(r))
        for r in rows
    ]
    cfg.extra_slot_hidden_z = AERIAL_HIDDEN_Z
    cfg.extra_slot_objs = [
        [None if x is None else x["obj_id"] for x in r] for r in rows
    ]  # names for the reference line
    cfg.extra_slot_usd = [
        [None if x is None else x["usd"] for x in r] for r in rows
    ]  # asset identity (checked against the placed clip)
    if fills is not None:
        cfg.extra_slot_fill = [None if f is None else list(f) for f in fills]  # the object standing in an empty place
    if (
        axis is not None
        # only the axis alignment of a one-env render (`_extra_slot_axis`); the collection replay binding moves file
        # numbers
    ):
        cfg.extra_slot_axis = [list(w) for w in axis]
    cfg.extra_slot_held = [[None if x is None else x["held"] for x in r] for r in rows]  # {side: [[start, end]]} | None
    _full_physics_marker(_cp, rows)
    # Collider shape per cell (S-true-shape, owner collider_shape.shape_of) where a collider stands; written only with a
    # triangles cell (else no cell: the convex parts as spawned, byte-identical).
    colliders = [[_collider_cell(_cp, x) for x in r] for r in rows]
    if any(c == _cs.TRIANGLES for r in colliders for c in r):
        cfg.extra_slot_colliders = colliders
    cfg.trajectory.n_extra_slots = K
    cfg.trajectory.extra_slot_cols = (
        {q: [None if x is None else x["qpos_start"] for x in r] for q, r in zip(seqs, rows, strict=True)} if K else None
    )  # None cell = hidden (loader)
    cfg.trajectory.extra_slot_static = (
        {q: [None if x is None else bool(x["static"]) for x in r] for q, r in zip(seqs, rows, strict=True)}
        if K
        else None
    )


def build_extra_slots(cfg, args_cli, ctx_label) -> None:
    """Extra object slots (default on · off = `DEXENGINE_EXTRA_SLOTS=0`): per clip, the objects chosen by the rule
    (dexengine/data/objects/extra_slots.slot_rows: role=extra objects a person handled and moved) spawn as K (= the
    per-clip maximum ≤ extra_slots.K_MAX) kinematic bodies; assets are fixed slots like the supports (`_slot_entity`,
    env i -> clip i % C), and empty slots get another clip's USD of the same slot at the hidden pose. One contact sensor
    per slot (body base_link, reference line). Motion, reset and the reference line are owned by `scene.extra_slots`
    (a teleport to the next frame after every policy step) · trajectory = the loader slot columns
    (`cfg.trajectory.extra_slot_cols`, the same dz and Rz(−90°) as the main objects) · collision =
    `collision_authoring.author_extra_collision` (per-kind table `collision_presets.EXTRA_KIND_COLLIDE`: moved extras:
    hand and arm on, main objects off / hand-touched static slots: main objects off / receiving static slots: main
    objects on; the robot cells of static slots per S-robot-static (`collision_presets.extra_collide`); slot kinds =
    `cfg.extra_slot_kinds`). Observations, rewards and terminations do not
    read the slots. Off does nothing (byte-identical). On with 0 slot targets still prints the marker.
    Shares outside the slots (PART_UNMERGED -> part merge defect) do not spawn and are counted in the marker.
    Static spawns (rule STATIC_SPAWN: extras the hand touched without moving them, extras that moved only slightly
    without a held span, and receiving static extras) are pinned in the same slots at their frame-0 pose
    (`cfg.trajectory.extra_slot_static`). Static extras the hand never touches and that receive nothing get no slot
    (rule VISUAL_ONLY: visual only in the collection render).
    Rule S-carried (`env_setup.extra_carry`): carried rows (kind carried_attached) are kinematic at the simulated
    carrier for the whole clip; statement `cfg.extra_slot_carry` only when such a row exists.
    Marker `[EXTRA_SLOTS] n=<extras> K=<slots> clips=<clips with extras>/<all> · outside the slots …`."""
    if not extra_slots_requested():
        XP.slots_off(args_cli, ctx_label)
        return
    from dexengine.placement.coords import core as _coords  # noqa: F401, I001 — import kept for its side effect
    import isaaclab.sim as _su  # lazy import, Isaac runtime only
    from isaaclab.managers import EventTermCfg as _EventTerm

    ContactSensorCfg, _cp, _mdp, _xe, plan, K, Ks, axis, cells = _plan_extra_slot_rows(cfg, args_cli)
    fills = _extra_slot_fills(cfg, _cp, cells, Ks)  # empty place fill of a one-env render (None elsewhere)
    for k in range(Ks):
        assets = _extra_slot_assets(_su, _cp, cells, k, None if fills is None else fills[k])
        _slot_entity(cfg, f"{_xe.PREFIX}{k}", f"{_xe.PRIM_LEAF}{k}", assets, contact=True)
        setattr(
            cfg.scene,
            f"{_xe.SENSOR_PREFIX}{k}",
            ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/" + f"{_xe.PRIM_LEAF}{k}/base_link", update_period=0.0, history_length=0
            ),
        )
    _extra_slot_statement(cfg, _cp, cells, plan.seqs, Ks, fills, axis)
    from dexengine.rl.env_setup import extra_carry as _xc  # S-carried: the statement when a carried row exists

    _xc.carry_statement(cfg, cells, Ks)
    if K:
        cfg.events.reset_extra_objects = _EventTerm(func=_mdp.reset_extra_objects, mode="reset", params={})
    rule = plan.rule()
    _extra_slots_marker(_xs, _cp, plan.rows, plan.seqs, Ks, plan.routed, ctx_label, rule)
    XP.record_spawn_list(plan, rule, ctx_label)


def _plan_extra_slot_rows(cfg, args_cli):
    """The run's slot plan (`extra_slots_plan.plan_rows`, Kit-free) and its Isaac side: the multi-clip contract on the
    cfg and, in a one-env render, the slot axis across clips. Returns (ContactSensorCfg, collision_presets, mdp terms,
    scene.extra_slots, plan, K, slot count, axis, slot cells)."""
    from isaaclab.sensors import ContactSensorCfg  # noqa: I001 — the import order of the original body is kept
    from dexengine.rl.tasks.scene import collision_presets as _cp
    from dexengine.rl.tasks.scene import extra_slots as _xe
    from dexengine.rl.tasks.mdp import terms as _mdp

    plan = XP.plan_rows(getattr(args_cli, "sequence_dir", None), args_cli.sequence)
    rows, seqs = plan.rows, plan.seqs
    K = max((len(r) for r in rows), default=0)  # the env -> clip contract of the slots: objects._slot_entity
    cells, Ks, axis = rows, K, None  # slot cells (clip × slot); training and judge keep the clip's own numbering
    if _one_env_render(cfg, rows, K):  # one-env render only: slot number = the object identity axis across clips
        al, where = _extra_slot_axis(_xs, _cp, rows)
        moved = [q for q, r, w in zip(seqs, rows, where, strict=True) if w != list(range(len(r)))]
        if moved:  # with 0 moves the axis = own numbering (tail empty places only); the statement as is
            cells, Ks, axis = al, len(al[0]), where
        print(
            f"[EXTRA_SLOTS] one-env render slot axis = union of object identities across clips K={len(al[0])} (clip's "
            f"own max {K}) · "
            f"clips with moved numbers {len(moved)}/{len(seqs)} {moved[:3]} · "
            + (
                f"empty cells {sum(x is None for r in al for x in r)} (hidden · filled) — the collection replay "
                f"binding moves file slot numbers "
                f"onto the axis (cfg.extra_slot_axis)"
                if moved
                else "axis = the clip's own numbering (statement as is)"
            ),
            flush=True,
        )
    return ContactSensorCfg, _cp, _mdp, _xe, plan, K, Ks, axis, cells
