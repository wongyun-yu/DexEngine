"""Ledger of the extra (non-manipulated) object slots: what a checkpoint was trained with, recorded and checked per
run."""

from __future__ import annotations

import os

from dexengine.core.errors import InputMissingError
from dexengine.rl.tasks import env_flags


def extra_slots_owner():
    """Owner of the on/off switch (name ENV · default DEFAULT · requested): dexengine/data/objects/extra_slots (the
    torch-free scene audit oracle reads the same)."""
    from dexengine.placement.coords import core as _coords  # noqa: F401, I001 — import kept for its side effect
    from dexengine.data.objects import extra_slots as _xs

    return _xs


def extra_slots_requested() -> bool:
    """Whether extra object slots are on: reads the owner extra_slots.requested (default on · 0 = off · unknown values
    fail)."""
    return extra_slots_owner().requested()


# Extra slot ledger = <run folder>/extra_slots.txt (run folder = the parent of checkpoints/, the same place as
# hand_recipe.txt). Written only when training with slots on (no file = off: checkpoints older than the ledger, or
# training with off stated). Every entry that reloads a checkpoint (continued training, fresh-process judge, render,
# rl.eval) compares it with the current value: extra slots change the scene, so loading with another value would
# measure the same checkpoint in another scene.
# Form (owner of reading, signatures and the 'same scene' interpretation = rl.launch.train_scene): first line = the
# value (1) · `rule<TAB>rule version` (extra_slots.RULE) · per clip `clip<TAB>name<TAB>spawn list signature` · per kind
# `collide<TAB>kind<TAB>collision partners` · `authored<TAB>value measured by the authoring` · per clip with brushed
# static extras the placement rule (place_rule) dropped from the spawn, `drop<TAB>name<TAB>object:reason,…` (a record;
# the collection render reads these names and does not draw them). A scene with the same on/off but a **different
# spawn list or collision partners** is another scene too (the value line `1` stays the same when the rule changes
# from 'every static extra' to 'receivers only'): an entry compares the value (before building the env,
# extra_slots_ledger_check) plus, per clip, the spawn list and that kind's collision partners (after building the env,
# build_extra_slots).
# An old ledger with only the first line has no spawn list: it counts as the same scene only when this run's clips have
# no extra the rule splits on (a static extra the hand never touched); otherwise it fails. A checkpoint **without a
# ledger** was trained without extra slots: it loads as off, but fails when the current rule stands slots with
# colliders (colliding with the robot or the main objects) on this run's clips (_scene_without_ledger).
EXTRA_SLOTS_LEDGER = "extra_slots.txt"
#: Extra slot statement this process built (written by build_extra_slots): {rule, clips: {clip: signature}, sensitive:
# {clip: count}}
_EXTRA_STATE: dict = {}
#: Checkpoint ledger path this process value-checked (extra_slots_ledger_check); read by the spawn list comparison
_EXTRA_LEDGER_SEEN: list = []
#: Checkpoints **without a ledger** this process loaded (trained without extra slots); read by the off-branch scene
# comparison
_EXTRA_NO_LEDGER: list = []
#: (args_cli, ctx_label) of a run that already passed the off branch: a ledger-less checkpoint loaded later is compared
# at once
_EXTRA_OFF_BUILT: list = []


def train_scene_owner():
    """Owner of the training scene comparison (ledger form, signatures, 'same scene' interpretation):
    rl.launch.train_scene (standard library; the queue tool's carry-over rule reads the same)."""
    from dexengine.placement.coords import core as _coords  # noqa: F401 — import kept for its side effect
    from dexengine.rl.launch import train_scene as _ts

    return _ts


def _ledger_read(path: str) -> dict:
    """Ledger file -> {value: first line, rule: rule version | None (old ledger), clips: {clip: signature}}; unknown
    lines fail."""
    try:
        return train_scene_owner().read_ledger(path)
    except ValueError as e:
        raise SystemExit(f"[EXTRA_SLOTS] {e}") from None


def _ledger_value(path: str) -> str:
    return _ledger_read(path)["value"]


def extra_slot_signature(slots: list) -> str:
    """Spawn list signature of one clip: `<moving/touched static/receiving static counts>:<first 12 chars of the sha1 of
    the name · object · kind list>`. `slots` = the slot rows of extra_slots.slot_rows (order = slot order)."""
    return train_scene_owner().signature(slots, extra_slots_owner().spawn_key(slots))


def _by_clip_name(sigs: dict, who: str) -> dict:
    """{clip name: spawn list signature}. Before joining by name both sides' keys go through the name rule owner
    `paths.clip_name` (the ledger has the training folder's names, this run the current folder's; if tree names `_pl_rt`
    and clip names split, the overlap would be 0 and the comparison silently skipped). Two names that collapse to one
    clip with different signatures fail."""
    from dexengine.core import paths as _paths  # the clip name rule owner

    out: dict = {}
    for name, sig in sigs.items():
        c = _paths.clip_name(name)
        if out.setdefault(c, sig) != sig:
            raise InputMissingError(
                f"[EXTRA_SLOTS] {who}: clip {c} has two spawn lists ({out[c]} · {sig} — name "
                f"{name!r} is another folder of the same clip) — cannot tell which scene"
            )
    return out


def _ledger_compare(path: str, state: dict, ctx: str) -> None:
    """Compare, per clip, the checkpoint ledger's spawn lists with the spawn lists this process built (`state` =
    _EXTRA_STATE). A difference fails: the same checkpoint is never measured in another scene. Clips missing from the
    ledger (clips the checkpoint never learned) are counted, not compared. Old ledger (no rule version): fails when this
    run's clips have extras the rule splits on."""
    rec = _ledger_read(path)
    if rec["rule"] is None:
        sens = sorted(c for c, n in state["sensitive"].items() if n)
        if sens:
            raise InputMissingError(
                f"[EXTRA_SLOTS] {ctx}: ledger {path} has no rule version · spawn list (old ledger — trained before "
                f"rule "
                f"{state['rule']}) · {len(sens)}/{len(state['clips'])} clips of this run have extras the rule splits "
                f"on "
                f"(untouched still extras — received · visual only), so the scene cannot be called the training scene: "
                f"{sens[:5]} — use a checkpoint trained under the current rule"
            )
        print(
            f"[EXTRA_SLOTS] spawn list check {path}: old ledger (no rule version) · this run's {len(state['clips'])} "
            f"clips have "
            f"0 extras the rule splits on -> training scene = current scene ({ctx})",
            flush=True,
        )
        return
    led_c, now_c = _by_clip_name(rec["clips"], f"ledger {path}"), _by_clip_name(state["clips"], "this run")
    both = sorted(set(led_c) & set(now_c))
    diff = [c for c in both if led_c[c] != now_c[c]]
    if diff:
        raise SystemExit(
            f"[EXTRA_SLOTS] {ctx}: spawn lists of {len(diff)}/{len(both)} clips differ from training (ledger {path} · "
            f"rule "
            f"{rec['rule']} -> now {state['rule']}): "
            + " · ".join(f"{c} {led_c[c]} → {now_c[c]}" for c in diff[:5])
            + " — the same checkpoint is never measured in another scene"
        )
    off = train_scene_owner().collide_changed(
        {**rec, "clips": led_c}, both, now_c, state["collide"], extra_slots_owner().KINDS
    )
    from dexengine.rl.tasks.scene import collision_presets as _cp

    if off and _cp.all_obj_collision():  # cells the switch changed on purpose; the spawn list comparison above stays
        print(
            f"{_cp.ALL_OBJ_TAG} ledger check: collision partners of slot kinds {off} differ from training (switch) · "
            f"ledger {path} ({ctx})",
            flush=True,
        )
        off = []
    if off:
        # a kind with no ledger row is a missing input (rc 12), as when the text said so; a changed partner is not
        exc = InputMissingError if any(k not in rec["collide"] for k in off) else SystemExit
        raise exc(
            f"[EXTRA_SLOTS] {ctx}: collision partners of slot kinds {off} differ from training (ledger {path}): "
            + " · ".join(
                f"{k} training [{rec['collide'].get(k, 'no row')}] → now [{state['collide'].get(k)}]" for k in off
            )
            + " — the same checkpoint is never measured under other physics"
        )
    print(
        f"[EXTRA_SLOTS] spawn list check {path}: same for {len(both)} clips (spawn list · that kind's collision "
        f"partners) · "
        f"clips missing from the ledger {len(set(now_c) - set(led_c))} (not compared) · rule ledger {rec['rule']} · "
        f"now "
        f"{state['rule']} ({ctx})",
        flush=True,
    )


def _authored_line(authored: dict | None) -> str:
    """The value the collision authoring measured (the return of collision_authoring.author_extra_collision, a filter
    read-back after the env is built) as one ledger line. A run that stood slots without an authoring record, or with a
    robot-filtered slot count different from the statement (slots of kinds that do not collide with the robot), fails:
    the ledger's collision partner line must be what the authoring really set."""
    ts, kinds = train_scene_owner(), extra_slots_owner().KINDS
    stood = any(ts.sig_kinds(s, kinds) for s in _EXTRA_STATE["clips"].values())
    if not authored or not authored.get("slots"):
        if stood:
            raise InputMissingError(
                f"[EXTRA_SLOTS] the run stood slots but has no collision authoring record "
                f"({authored!r}) — no collision partners go into the ledger without proof that "
                "the authoring ran"
            )
        return "slots=0"
    # every kind of this process's collision table (base and scene-rule kinds; the empty place has no table row)
    table, hits = _EXTRA_STATE["collide"], set(_EXTRA_STATE["robot_kinds"])
    want = sum(n for k, n in authored["kinds"].items() if k in table and k not in hits)
    if int(authored["robot_off"]) != want:
        raise SystemExit(
            f"[EXTRA_SLOTS] slots the authoring turned robot-off {authored['robot_off']} ≠ slots of kinds that do "
            f"not collide with the robot {want} (kinds {authored['kinds']} · table {_EXTRA_STATE['collide']})"
        )
    return (
        f"slots={authored['slots']} envs={authored['envs']} object_pairs={authored['pairs']} "
        f"robot_off={authored['robot_off']} kinds=" + ",".join(f"{k}:{v}" for k, v in sorted(authored["kinds"].items()))
    )


def extra_slots_ledger_record(run_dir: str, authored: dict | None = None) -> str | None:
    """Training start: with slots on, write the ledger in the run folder (value · rule version · per-clip spawn list
    signatures · collision partners per kind · the authoring's measured value; an existing ledger must agree on value,
    spawn lists and collision partners, new clips are added). With slots off nothing is written.
    On while this process built no slots fails (no ledger without knowing the spawn list).
    authored = the collision authoring's return (the env's `_coll_author_info["extra"]`), see _authored_line."""
    if not extra_slots_requested():
        return None
    if not _EXTRA_STATE:
        raise SystemExit(
            "[EXTRA_SLOTS] slots on but this process built no extra slots (build_extra_slots) — no ledger "
            "without knowing the spawn list"
        )
    line = _authored_line(authored)
    path = os.path.join(run_dir, EXTRA_SLOTS_LEDGER)
    clips, drops = dict(_EXTRA_STATE["clips"]), dict(_EXTRA_STATE["drops"])
    if os.path.isfile(path):
        rec = _ledger_read(path)
        if rec["value"] != "1":
            raise SystemExit(
                f"[EXTRA_SLOTS] ledger {path} = {rec['value']!r} but slots are on now — "
                "the same run name is never continued in another scene"
            )
        _ledger_compare(path, _EXTRA_STATE, "ledger write (continuing the same run name)")
        clips = {**rec["clips"], **clips}
        drops = {**{c: t for c, t in rec["drops"].items() if c not in _EXTRA_STATE["clips"]}, **drops}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(
            "1\n"
            + f"rule\t{_EXTRA_STATE['rule']}\n"
            + "".join(f"clip\t{c}\t{sig}\n" for c, sig in sorted(clips.items()))
            + "".join(f"drop\t{c}\t{txt}\n" for c, txt in sorted(drops.items()))
            + "".join(f"collide\t{k}\t{v}\n" for k, v in sorted(_EXTRA_STATE["collide"].items()))
            + f"authored\t{line}\n"
        )
    os.replace(tmp, path)
    print(
        f"[EXTRA_SLOTS] ledger written {path}: 1 · rule {_EXTRA_STATE['rule']} · spawn list signatures of "
        f"{len(clips)} clips · "
        f"brushed-object lines dropped from the spawn {len(drops)} · "
        f"collision partners per kind {_EXTRA_STATE['collide']} · authoring {line}",
        flush=True,
    )
    return path


def extra_slots_ledger_check(ckpt_path: str | None, ctx: str) -> str | None:
    """Entries that load a checkpoint (continued training, judge, render, rl.eval): the value is owned by the ledger
    next
    to the checkpoint (none = off). A launch without the input (DEXENGINE_EXTRA_SLOTS) loads the ledger value into the
    env and prints a marker, so judge, re-judge, replay and augmentation entries get the same scene without carrying
    the training env. A stated value that disagrees with the ledger fails before the env is built.
    An on ledger records its path: after the env is built build_extra_slots compares the spawn lists (_ledger_compare;
    at once here when called after the build)."""
    if not ckpt_path:
        return None
    path = train_scene_owner().ledger_of(ckpt_path)
    rec = _ledger_value(path) if os.path.isfile(path) else "0"
    if rec not in ("0", "1"):
        raise SystemExit(f"[EXTRA_SLOTS] {ctx}: ledger {path} = {rec!r} — 0|1 only")
    env = extra_slots_owner().ENV
    if os.environ.get(env, "").strip() == "":
        os.environ[env] = (
            rec  # an old checkpoint without a ledger = trained off -> off (so the default on does not slip in)
        )
        how = "read from the ledger"
    else:
        how = "stated"
    cur = "1" if extra_slots_requested() else "0"
    if rec != cur:
        raise InputMissingError(
            f"[EXTRA_SLOTS] {ctx}: checkpoint {ckpt_path} was trained with extra slots {rec}, now "
            f"{cur} (ledger {path} — none = off) · leave the input out to follow the ledger"
        )
    if cur == "1":
        print(f"[EXTRA_SLOTS] ledger check {path}: trained 1 = now 1 ({how} · {ctx})", flush=True)
        if path not in _EXTRA_LEDGER_SEEN:
            _EXTRA_LEDGER_SEEN.append(path)
        if _EXTRA_STATE:
            _ledger_compare(path, _EXTRA_STATE, ctx)
    elif not os.path.isfile(
        path
    ):  # no ledger = trained without extra slots; the off branch compares with the current scene
        if ckpt_path not in _EXTRA_NO_LEDGER:
            _EXTRA_NO_LEDGER.append(ckpt_path)
        for built in _EXTRA_OFF_BUILT:
            _scene_without_ledger(*built)
    return path


def _extra_slots_marker(_xs, _cp, rows: list, seqs: list, K: int, routed: dict, ctx_label: str) -> None:
    """Marker `[EXTRA_SLOTS]`, two lines: slot count · kinds · shares outside the slots · collision table, and the spawn
    count distribution per clip."""
    n = sum(len(r) for r in rows)
    n_st = sum(bool(x["static"]) for r in rows for x in r)
    n_kind = {k: sum(x["kind"] == k for r in rows for x in r) for k in _xs.KINDS}
    dist: dict = {}
    for r in rows:
        dist[_xs.spawn_key(r)] = dist.get(_xs.spawn_key(r), 0) + 1
    print(
        f"[EXTRA_SLOTS] n={n} K={K} clips={sum(1 for r in rows if r)}/{len(seqs)} · following the reference "
        f"{n - n_st} · "
        f"static spawns {n_st} (pinned at frame 0) · kinds "
        + " ".join(f"{k}={v}" for k, v in n_kind.items())
        + f" (of which hand-touched receivers {sum(_xs.hand_receiver(x) for r in rows for x in r)}) · outside the "
        f"slots " + " ".join(f"{c}={v}({_xs.ROUTED[c]})" for c, v in routed.items()) + f" · {ctx_label} · rule "
        f"dexengine/data/objects/extra_slots.py({_xs.RULE} · K_MAX {_xs.K_MAX}) · kinematic · collision "
        f"(per kind — "
        f"collision_presets.EXTRA_KIND_COLLIDE) {_cp.extra_collide_text()} · "
        f"objects {sorted({x['obj_id'] for r in rows for x in r})[:6]}",
        flush=True,
    )
    print(f"[EXTRA_SLOTS] {_xs.spawn_dist_text(dist)}", flush=True)


def _extra_collide_tables() -> dict:
    """{slot kind: collision partner table}; owner collision_presets.extra_collide (the table the authoring reads)."""
    from dexengine.rl.tasks.scene import collision_presets as _cp

    return {k: _cp.extra_collide(k) for k in extra_slots_owner().KINDS}


def _colliding_kinds(tables: dict) -> set:
    """Kinds whose table stands a collider (they collide with the robot or a main object, the things the policy moves).
    Rule owner = collision_presets.table_has_collider (a pure rule over one table line, read through the registry window
    paths.load_registry, the same window as the training scene comparison rl.launch.train_scene); the caller passes
    the tables (_extra_collide_tables)."""
    from dexengine.placement.coords import core as _coords  # noqa: F401 — import kept for its side effect
    from dexengine.rl.tasks.scene.collision_presets import table_has_collider as rule

    return {k for k, t in tables.items() if rule(t)}


def _scene_without_ledger(args_cli, ctx_label) -> None:
    """An off run that loaded a ledger-less checkpoint (trained without extra slots): compare with the slots the current
    rule stands on this run's clips (interpretation owner train_scene.changed; collision partners read from
    collision_presets.extra_collide). A clip where a slot with a collider stands (a kind colliding with the robot or a
    main object, _colliding_kinds) fails: that clip's canonical scene is not the scene the checkpoint learned (a
    retraining target). Slots that collide with nothing keep the same scene. An off run that loaded no checkpoint has no
    training scene to compare with.
    A run that does not enter through a tree folder cannot stand slots (build_extra_slots): only the marker prints."""
    if not _EXTRA_NO_LEDGER:
        return
    _ts, _xs = train_scene_owner(), extra_slots_owner()
    seqs = [t for t in (args_cli.sequence or "").split(",") if t]
    seq_dir = getattr(args_cli, "sequence_dir", None)
    if not seqs or not seq_dir:
        print(
            f"[EXTRA_SLOTS] no scene check — not a tree folder entry (an entry that cannot stand slots · ledger-less "
            f"checkpoints "
            f"{len(_EXTRA_NO_LEDGER)}) · {ctx_label}",
            flush=True,
        )
        return
    # Runs on the preserved trees of a frozen generation (a named roster, dexengine.core.layout.NAMED_ROSTERS): that
    # tree is the generation's scene and carries no extra object information (objects[]); the generation trained without
    # extras and this run is off too, so training scene = current scene. The current rule's comparison reads base-layer
    # trees only (extra_slots.clip_rows) and dies on preserved trees (measured on a judge of preserved trees).
    # Only when the caller stated DEXENGINE_ROSTER=<that name> and seq_dir is that roster's preserved trees root (the
    # same condition as the launcher's roster guard, owner paths.named_roster_of) is the comparison skipped (marker).
    # Everything else is compared below.
    from dexengine.core import paths as _paths  # owner of the frozen generations' preserved roots

    _ro = env_flags.get("DEXENGINE_ROSTER").strip()
    if _ro and _paths.named_roster_of(str(seq_dir)) == _ro:
        print(
            f"[EXTRA_SLOTS] no scene check — preserved trees of the frozen generation roster {_ro} (no extra "
            f"information · "
            f"that generation = trained without extras · now off) · ledger-less checkpoints {len(_EXTRA_NO_LEDGER)} · "
            f"clips {len(seqs)} · {ctx_label}",
            flush=True,
        )
        return
    tables = _extra_collide_tables()
    hits, coll = _colliding_kinds(tables), {k: _ts.partners(t) for k, t in tables.items()}
    bad, n_inert = {}, 0
    for q in seqs:
        now = _ts.clip_now(os.path.join(str(seq_dir), q))
        if now is None:
            raise InputMissingError(
                f"[EXTRA_SLOTS] {ctx_label}: scene check — the tree of clip {q} is not in {seq_dir}"
            )
        why = _ts.changed(None, q, now[0], now[1], hits, coll, tuple(_xs.KINDS))
        if why:
            bad[q] = why
        n_inert += bool(now[0]) and not why
    if bad:
        raise InputMissingError(
            f"[EXTRA_SLOTS] {ctx_label}: checkpoint {_EXTRA_NO_LEDGER[0]} has no extra-slot ledger (trained without "
            f"extras) · the current rule {_xs.RULE} places slots colliding with the robot · main object in "
            f"{len(bad)}/{len(seqs)} clips of this run: "
            + " · ".join(f"{q}({w})" for q, w in sorted(bad.items())[:5])
            + " — training scene ≠ current scene · measure those clips with a checkpoint retrained under the "
            "current rule"
        )
    print(
        f"[EXTRA_SLOTS] scene check passed — ledger-less checkpoint (trained without extras) · slots colliding with "
        f"the "
        f"robot or a main object ({'/'.join(sorted(hits))}) 0 on {len(seqs)} clips · clips with only "
        f"non-colliding slots {n_inert} · rule {_xs.RULE} · "
        f"{ctx_label}",
        flush=True,
    )
