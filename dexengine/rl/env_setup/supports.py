"""Support geometry of the scene (aerial discs, the desk support, per-slot pedestals) and the scene support builder."""

from __future__ import annotations

import json
import os

from dexengine.rl.env_setup.dyn_pairs import build_dyn_pairs
from dexengine.rl.env_setup.extra_slots import build_extra_slots
from dexengine.rl.env_setup.float_settle import build_float_settle
from dexengine.rl.env_setup.objects import (
    AERIAL_DISC_SLOTS,
    AERIAL_HIDDEN_Z,
    PEDESTAL_SLOTS_MAX,
    _slot_entity,
    build_all_obj_bodies,
)


def build_scene_supports(cfg, args_cli, ctx_label) -> None:
    """The one support-disc branch of the physics scene: the aerial scene (rule D20, scene_aerial) or the desk support
    (rule D26′, desk_discs stamp). Called by both the training cfg (rl.train.runner train_cfg_overrides) and the eval
    and replay render cfg (rl.eval _eval_cfg_overrides). Without this call on the eval side, aerial clips would
    replay in a scene with the desk standing and no discs, and the entities would split from the record (the judge
    scene; the collection stage 2 [REPLAY_MAP] entity set differs: support_disc_*)."""
    if int(getattr(args_cli, "scene_aerial", 0) or 0):
        build_aerial_scene(cfg, args_cli, ctx_label)  # aerial scene (rule D20): desk sunk + disc slots + reset event
    elif getattr(args_cli, "desk_discs", None) is not None:
        build_desk_support(cfg, args_cli, ctx_label)  # desk support (rule D26′): stamped disc slots + reset event
    build_all_obj_bodies(cfg, args_cli, ctx_label)  # static bodies of all-object collision (default off = does nothing)
    build_extra_slots(cfg, args_cli, ctx_label)  # extra object slots (DEXENGINE_EXTRA_SLOTS · default on · 0 = off)
    build_float_settle(
        cfg, args_cli, ctx_label
    )  # S-float-settle statement read by the pin (rl/tasks/scene/float_lower)
    build_dyn_pairs(cfg, args_cli, ctx_label)  # S-dyn-pairs statement read by the authoring (collision_dyn_pairs)


def build_aerial_scene(cfg, args_cli, ctx_label) -> None:
    """Aerial scene (rule D20): sink the desk prims (not deleted: sensors and filters reference DeskCollisionTop) and
    spawn K static disc slots per clip as kinematic cylinders. Slot k's radius differs per clip, so per-clip
    CylinderCfgs go in through MultiAssetSpawnerCfg (env_i -> asset[i % n], loader clip order, the same contract as
    multi-object), and the reset event `reset_support_discs` writes the position from the env's clip values. The
    values are owned by `args_cli.aerial_discs` (the sequence_dir entry collects the trees'
    `task_info.aerial_support.discs`). Marker `[AERIAL]`. Slots and the event come from `_build_disc_slots` (shared
    with the desk support).
    """
    seqs = [t for t in (args_cli.sequence or "").split(",") if t]  # loader order (sorted; the multi_object block sorts)
    discs_by_seq = getattr(args_cli, "aerial_discs", None) or {}
    moved = []
    for key in ("desk", "desk_collision_top"):
        ent = getattr(cfg.scene, key, None)
        if ent is not None:
            x, y, z = ent.init_state.pos
            ent.init_state.pos = (x, y, z + AERIAL_HIDDEN_Z)
            moved.append(f"{key}→z{z + AERIAL_HIDDEN_Z:+.1f}")
    for key in ("desk_leg_0", "desk_leg_1", "desk_leg_2", "desk_leg_3"):
        ent = getattr(cfg.scene, key, None)
        if ent is not None:
            x, y, z = ent.init_state.pos
            ent.init_state.pos = (x, y, z + AERIAL_HIDDEN_Z)
            moved.append(key)
    seq_dir = getattr(args_cli, "sequence_dir", None)
    names, _table, n_real = _build_disc_slots(cfg, seqs, discs_by_seq, "AERIAL", seq_dir)
    print(
        f"[AERIAL] {ctx_label}: aerial scene (rule D20) — "
        f"desk sunk ({', '.join(moved) or 'no desk'}) · "
        f"disc slots {len(names)} (clips {len(seqs)}, real discs {n_real}) · "
        f"reset event reset_support_discs registered",
        flush=True,
    )
    _build_support_slots(cfg, seqs, "AERIAL", seq_dir, aerial=True, n_disc=n_real, ctx_label=ctx_label)


def build_desk_support(cfg, args_cli, ctx_label) -> None:
    """Desk scene support (rule D26′): the desk stays and per-clip support disc slots spawn.
    The only values are the bake's stamp in the tree, `task_info[qpos_layout.DESK_SUPPORT_KEY].discs` (collected by the
    sequence_dir entry into `args_cli.desk_discs`); the judge, bake, gates and final guard measure collisions from the
    same stamp. Without any support no slots are made and only the marker prints (the env stays byte-identical).
    Marker `[DESK_SUPPORT]`."""
    seqs = [t for t in (args_cli.sequence or "").split(",") if t]
    discs_by_seq = getattr(args_cli, "desk_discs", None)
    if discs_by_seq is None:
        raise RuntimeError(
            "[DESK_SUPPORT] no args_cli.desk_discs — the sequence_dir entry did not collect the desk supports"
        )
    n_any = sum(1 for q in seqs if discs_by_seq.get(q))
    seq_dir = getattr(args_cli, "sequence_dir", None)
    if n_any == 0:
        print(f"[DESK_SUPPORT] {ctx_label}: 0 desk support discs (clips {len(seqs)}) — no disc slots made", flush=True)
        n_real = 0
    else:
        names, _table, n_real = _build_disc_slots(cfg, seqs, discs_by_seq, "DESK_SUPPORT", seq_dir)
        print(
            f"[DESK_SUPPORT] {ctx_label}: desk support (rule D26′) — disc slots {len(names)} (clips {len(seqs)}, "
            f"clips with supports {n_any}, real discs {n_real}) · reset event reset_support_discs registered",
            flush=True,
        )
    _build_support_slots(cfg, seqs, "DESK_SUPPORT", seq_dir, aerial=False, n_disc=n_real, ctx_label=ctx_label)


def _disc_stamp_statement(seqs: list, discs_by_seq: dict, tag: str, seq_dir) -> tuple:
    """Check the disc stamps against the current decision and set the scene's collision statement; returns (tis {clip:
    task_info}, pedestal, want). A disagreeing stamp, a bundle mixing discs and pedestals, or an unknown tree folder
    fails."""
    from dexengine.placement.coords import (
        core as _coords,  # owner of the disc stamp rule (collide · pedestal range)
    )

    # Rule D40 (disc-hand and disc-arm collisions off): the `collide` stamped on each disc (owner
    # coords.DISC_COLLIDE) is consumed; missing (a stamp older than D40) or different from the current rule fails
    # (re-bake). The statement goes to one place on the env cfg (`support_disc_collide`), and the robot-side physics is
    # built by the collision authoring that reads it (`collision_authoring.author_disc_collision`).
    # Pedestal decision: discs of the desk scene (DESK_SUPPORT) are pedestals (visible columns from the desk top to the
    # object floor), the aerial scene keeps discs. Pedestal collision partners live in one per-kind table,
    # coords.PEDESTAL_COLLIDE (where a main object rests = objects only; under a fixed object = visual only, collision
    # off); coords.disc_collide_of(kind) gives each disc's expected stamp. Every support is off with the robot (the
    # judge's arm guard filters placements where the arm passes through a column).
    tis = {}
    for q in seqs:
        if not discs_by_seq[q]:
            continue
        if not seq_dir:
            raise RuntimeError(
                f"[{tag}] support discs exist but the tree folder (args_cli.sequence_dir) is unknown — the "
                "disc owner side and the expected stamps (task_info.objects · dataset_name) cannot be read"
            )
        with open(os.path.join(str(seq_dir), q, "task_info.json"), encoding="utf-8") as _f:
            tis[q] = json.load(_f)
    peds = {q: _coords.pedestal_scene(tis[q], aerial=tag == "AERIAL") for q in tis}
    if len(set(peds.values())) > 1:
        raise RuntimeError(
            f"[{tag}] disc and pedestal clips mixed in one env {sorted(set(peds.values()))} — a bundle must "
            "share one support kind"
        )
    pedestal = next(iter(peds.values()), False)
    bad = sorted(
        {
            (q, str(d.get("kind")), str(d.get("collide")))
            for q in seqs
            for d in discs_by_seq[q]
            if d.get("collide") != _coords.disc_collide_of(tis[q], aerial=tag == "AERIAL", kind=d.get("kind"))
            or (pedestal and d.get("h") is None)
        }
    )
    if bad:
        raise RuntimeError(
            f"[{tag}] disc stamp collide differs from the current decision"
            f" ({dict(_coords.PEDESTAL_COLLIDE) if pedestal else _coords.DISC_COLLIDE!r}) or is "
            f"missing (pedestals need the height h too) — {len(bad)} clips: {bad[:3]} (re-bake)"
        )
    # Collision statement (read by collision_authoring and the audit): pedestal scene = the value of the column under a
    # main object (objects only, robot off); 'visual only' columns get prim collision off below (owner statement static,
    # no object filter target)
    want = _coords.PEDESTAL_OBJECTS if pedestal else _coords.DISC_COLLIDE
    return tis, pedestal, want


def _build_disc_slots(cfg, seqs: list, discs_by_seq: dict, tag: str, seq_dir=None) -> tuple:
    """Spawn K (=AERIAL_DISC_SLOTS) support disc slots per clip as kinematic cylinders and register the reset event;
    shared by aerial (rule D20) and desk (rule D26′). Slot k's radius differs per clip, so they go in through
    MultiAssetSpawnerCfg (env_i -> asset[i % n], loader clip order, the same contract as multi-object), and the reset
    event `reset_support_discs` writes the position from the env's clip values.
    Table row = [x, y, z (top, the stamp as is), r]. Discs stand from reset to the end of the episode (rule D40, no
    lifetime window). Too few slots fail. Returns (names, table, n_real)."""
    import isaaclab.sim as _su  # lazy import, Isaac runtime only (order as in the old block)
    from isaaclab.managers import EventTermCfg as _EventTerm

    from dexengine.placement.coords import (
        core as _coords,  # owner of the disc stamp rule (collide · owner side); numpy is fine after Kit
    )
    from dexengine.rl.env_setup.build import AERIAL_DISC_HEIGHT  # its owner is build.py (read there by path)
    from dexengine.rl.tasks.mdp import terms as _mdp

    if seqs != sorted(seqs):
        raise RuntimeError(
            f"[{tag}] --sequence is not sorted — the disc table's clip order contract (loader = sorted) broken"
        )
    missing = [q for q in seqs if q not in discs_by_seq]
    if missing:
        raise RuntimeError(
            f"[{tag}] {len(missing)} clips missing from the disc table: {missing[:3]} — "
            "the sequence_dir entry did not collect them"
        )
    K = AERIAL_DISC_SLOTS
    tis, pedestal, want = _disc_stamp_statement(seqs, discs_by_seq, tag, seq_dir)
    if pedestal:  # pedestals: up to one per main object + fixed object (support objects, static extras)
        K = max(K, max(len(discs_by_seq[q]) for q in seqs))
        if K > PEDESTAL_SLOTS_MAX:
            raise RuntimeError(f"[{tag}] pedestal slots {K} > cap {PEDESTAL_SLOTS_MAX}")
    over = {q: len(discs_by_seq[q]) for q in seqs if len(discs_by_seq[q]) > K}
    if over:
        raise RuntimeError(f"[{tag}] {len(over)} clips above {K} disc slots: {list(over.items())[:3]}")
    cfg.support_disc_collide = want
    cfg.support_disc_kinds = (
        dict(_coords.PEDESTAL_COLLIDE) if pedestal else None
    )  # for the marker: collision partners per kind
    visible = _coords.disc_visible(want)
    names = []
    table = []
    hs, n_real, n_seen = _slot_column_heights(
        cfg,
        seqs,
        discs_by_seq,
        tag,
        AERIAL_DISC_HEIGHT,
        K,
        _EventTerm,
        _coords,
        _mdp,
        _su,
        names,
        pedestal,
        table,
        tis,
        visible,
    )
    print(
        f"[{tag}] {'pedestal' if pedestal else 'disc'} collide={want}"
        f"{f' · per kind {cfg.support_disc_kinds} · visual only (collision off) {n_seen}/{n_real}' if pedestal else ''}"
        f" · visual = "
        f"{'visible' if visible else 'invisible'} (coords.disc_visible: render, camera) · column height "
        f"{min(hs) if hs else 0:.3f}~{max(hs) if hs else 0:.3f} m · from reset to the end",
        flush=True,
    )
    return names, table, n_real


def _slot_column_heights(
    cfg,
    seqs,
    discs_by_seq,
    tag,
    AERIAL_DISC_HEIGHT,
    K,
    _EventTerm,
    _coords,
    _mdp,
    _su,
    names,
    pedestal,
    table,
    tis,
    visible,
):
    """[clip][slot] column heights: the stamped height for pedestals (top minus desk)."""
    # [clip][slot] = column height: the stamped h for pedestals (top − desk top), the disc thickness for discs and empty
    # slots
    heights = []
    # rule D11″ (d): [clip][slot] = side of the main object the disc supports (owner coords.disc_owner_side); empty
    # slots None
    owner = []
    seen_only = []  # [clip][slot] = 'visual only' column (coords.PEDESTAL_SEEN): prim collision off
    for q in seqs:
        row = []
        own = []
        hrow = []
        srow = []
        ti = tis.get(q)
        for k in range(K):
            if k < len(discs_by_seq[q]):
                d = discs_by_seq[q][k]
                row.append([float(d["x"]), float(d["y"]), float(d["z"]), float(d["r"])])
                own.append(_coords.disc_owner_side(ti, d))
                hrow.append(max(float(d["h"]), AERIAL_DISC_HEIGHT) if pedestal else AERIAL_DISC_HEIGHT)
                srow.append(d.get("collide") == _coords.PEDESTAL_SEEN)
            else:
                row.append([0.0, 0.0, AERIAL_HIDDEN_Z, 0.001])
                own.append(None)
                hrow.append(AERIAL_DISC_HEIGHT)
                srow.append(False)
        table.append(row)
        owner.append(own)
        heights.append(hrow)
        seen_only.append(srow)
    hs, n_real = _record_disc_owner_and_reset(
        cfg, seqs, tag, AERIAL_DISC_HEIGHT, K, _EventTerm, _mdp, _su, names, table, visible, heights, owner, seen_only
    )
    n_seen = sum(1 for c in range(len(seqs)) for k in range(K) if seen_only[c][k])
    return hs, n_real, n_seen


def _record_disc_owner_and_reset(
    cfg, seqs, tag, AERIAL_DISC_HEIGHT, K, _EventTerm, _mdp, _su, names, table, visible, heights, owner, seen_only
):
    """Record the disc owner statement, register the support-disc reset event and count the real slots."""
    # Statement (read by the authoring `collision_authoring.author_disc_collision` and the RO audit): env i's clip = i %
    # clip count (static assignment)
    cfg.support_disc_owner = owner
    for k in range(K):
        name = f"support_disc_{k}"
        assets = [
            _su.CylinderCfg(
                radius=max(table[c][k][3], 0.001),
                height=heights[c][k],
                axis="Z",
                visible=visible and table[c][k][2] > -5.0,  # discs invisible (rule D40) · pedestals visible
                rigid_props=_su.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                collision_props=(
                    _su.CollisionPropertiesCfg(collision_enabled=False)
                    if seen_only[c][k]
                    else _su.CollisionPropertiesCfg()
                ),
                visual_material=_su.PreviewSurfaceCfg(diffuse_color=(0.85, 0.70, 0.30)),
            )
            for c in range(len(seqs))
        ]
        _slot_entity(cfg, name, f"SupportDisc{k}", assets)
        names.append(name)
    # Reset event: the same mode as the object reset ("reset"). The table goes in as params (the cfg cannot change after
    # the env is built).
    cfg.events.reset_support_discs = _EventTerm(
        func=_mdp.reset_support_discs,
        mode="reset",
        params={
            "disc_names": names,
            "table": table,
            "tag": tag,
            "height": AERIAL_DISC_HEIGHT,
            "heights": heights,
            "hidden_z": AERIAL_HIDDEN_Z,
            "visible": bool(visible),
        },
    )
    n_real = sum(1 for c in range(len(seqs)) for k in range(K) if table[c][k][2] > -5.0)
    hs = [heights[c][k] for c in range(len(seqs)) for k in range(K) if table[c][k][2] > -5.0]
    return hs, n_real


def _support_rows(seqs: list, seq_dir, aerial: bool) -> tuple[list, int]:
    """One entry per support object per clip [(support name, USD, scene pose xyz·wxyz, obj_id)] (coords.clip_supports
    bundle) and the count of clips with an old stamp.
    Reading = the stamp window of coords (the same stamp as the env discs; computed values before the bake fail) · pose
    =
    coords.support_scene_pose (stamped pose, coords.scene_z_offset) · USD = objects.hull.usd_file (the USD training
    spawns)."""
    from dexengine.data.objects import hull as _oh  # owner of the training USD paths
    from dexengine.placement.coords import core as _coords  # owner of stamp reading and support poses

    if not seq_dir:
        raise RuntimeError(
            "[SUPPORT] the tree folder (args_cli.sequence_dir) is unknown — the support stamps cannot be read"
        )
    rows, prestamp = [], 0
    for q in seqs:
        tree = os.path.join(str(seq_dir), q)
        with open(os.path.join(tree, "task_info.json"), encoding="utf-8") as f:
            ti = json.load(f)
        stamp = _coords.read_aerial_support(tree) if aerial else _coords.desk_support_discs(tree)
        if not aerial and not stamp["placed"]:  # the aerial reading already fails without a stamp
            raise RuntimeError(
                f"[SUPPORT] {q}: a tree before the bake (computed values instead of the support stamp, "
                f"coordinates before placement, source={stamp['source']}) — the env reads stamps only"
            )
        lst, old = _coords.clip_supports(ti, stamp["supports"], q)
        prestamp += int(old)
        z_off = float(_coords.scene_z_offset(tree)) if lst else 0.0
        by = {str(o.get("name")): o for o in ti.get("objects") or []}
        clip = []
        for s in lst:
            o = by.get(str(s["obj"]))
            if o is None or o.get("articulation"):
                raise RuntimeError(f"[SUPPORT] {q}: support {s['obj']!r} is not in objects or is articulated")
            usd = str(_oh.usd_file(str(ti["dataset_name"]), str(o["obj_id"]), False))
            if not os.path.isfile(usd):
                raise RuntimeError(f"[SUPPORT] {q}: support {s['obj']!r} has no USD: {usd} — bake the supports first")
            xyz, wxyz = _coords.support_scene_pose(s["pose"], z_off)
            clip.append(
                (str(s["obj"]), usd, [float(v) for v in (*xyz, *wxyz)], str(o["obj_id"]), str(ti["dataset_name"]))
            )
        rows.append(clip)
    return rows, prestamp


def _support_collider_statement(cfg, rows: list, K: int) -> None:
    """`cfg.support_colliders` (clip × slot: the support's collider shape, `collider_shape.shape_of` of a pinned
    support; None = no k-th support, the hidden stand-in keeps its parts), written only with a triangles cell (read by
    `collision_authoring.author_support_collision`)."""
    from dexengine.data.objects import collider_shape as _cs  # noqa: PLC0415 — owner of the collider shape

    cols = [[_cs.shape_of(r[k][4], _cs.SUPPORT, True) if len(r) > k else None for k in range(K)] for r in rows]
    if any(c == _cs.TRIANGLES for r in cols for c in r):
        cfg.support_colliders = cols


def _build_support_slots(cfg, seqs: list, tag: str, seq_dir, *, aerial: bool, n_disc: int, ctx_label: str) -> list:
    """Support objects (rule D42: a support object collides with hands, arms and objects): per clip, the stamped
    support objects become K (= the per-clip maximum, ≤ AERIAL_DISC_SLOTS) slots of kinematic bodies (gravity off)
    pinned at the first-frame pose; the same fixed slots as discs (`_slot_entity`), no lifetime; observations, rewards,
    terminations and the judge do not read these entities. A clip without a k-th support gets another clip's USD of the
    same slot at the hidden pose (each slot needs the same rigid prim structure for RigidObject to bind the same bodies
    per env). Collisions are built by the authoring that reads the statement `cfg.support_object_slots`
    (`collision_authoring.author_support_collision`, table collision_presets.SUPPORT_COLLIDE). Marker `[SUPPORT]
    n=<supports> kinds=disc:<discs>,support:<supports> prestamp=<clips with an old stamp>`, printed also at 0."""
    import isaaclab.sim as _su  # noqa: I001 — lazy import, Isaac runtime only
    from isaaclab.managers import EventTermCfg as _EventTerm
    from dexengine.rl.tasks.scene.object_pin import SUPPORT_OBJ_PREFIX
    from dexengine.rl.tasks.mdp import terms as _mdp

    rows, prestamp = _support_rows(seqs, seq_dir, aerial)
    K = max((len(r) for r in rows), default=0)
    if K > AERIAL_DISC_SLOTS:
        raise RuntimeError(
            f"[{tag}][SUPPORT] clips above {AERIAL_DISC_SLOTS} support slots: "
            f"{[(q, len(r)) for q, r in zip(seqs, rows, strict=True) if len(r) > AERIAL_DISC_SLOTS][:3]}"
        )
    names = []
    hidden = [0.0, 0.0, AERIAL_HIDDEN_Z, 1.0, 0.0, 0.0, 0.0]
    for k in range(K):
        stand_in = next(r[k][1] for r in rows if len(r) > k)
        assets = [
            _su.UsdFileCfg(
                usd_path=r[k][1] if len(r) > k else stand_in,
                visible=len(r) > k,
                rigid_props=_su.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            )
            for r in rows
        ]
        names.append(f"{SUPPORT_OBJ_PREFIX}{k}")
        _slot_entity(cfg, names[-1], f"SupportObj{k}", assets)
    if K:
        cfg.support_object_slots = K
        _support_collider_statement(cfg, rows, K)
        cfg.events.reset_support_objects = _EventTerm(
            func=_mdp.reset_support_objects,
            mode="reset",
            params={
                "names": names,
                "poses": [[r[k][2] if len(r) > k else hidden for k in range(K)] for r in rows],
                "assets": [[[r[k][3], r[k][1]] if len(r) > k else None for k in range(K)] for r in rows],
                "tag": tag,
            },
        )
    n = sum(len(r) for r in rows)
    kinds = sorted({s[0] for r in rows for s in r})
    print(
        f"[SUPPORT] n={n} kinds=disc:{n_disc},support:{n} prestamp={prestamp} · {ctx_label} {tag} · support slots {K} "
        f"(clips {len(seqs)} · support objects {kinds[:6]}) · kinematic first-frame pose · no lifetime · collision = "
        "collision_presets.SUPPORT_COLLIDE",
        flush=True,
    )
    return names
