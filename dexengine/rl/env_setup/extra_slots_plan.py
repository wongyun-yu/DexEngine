"""Kit-free plan of the extra object slots: per clip, the rows that take a slot (with their training USD, held spans and
collider shape), and this process's spawn-list statement, the value a checkpoint's extra-slot ledger is compared with.

Two callers run these functions: the scene build (`env_setup.extra_slots.build_extra_slots`, after Kit) and the trainer
preflight (`rl.cli_entry.preflight`, before Kit). A clip the scene build refuses (a PART_UNMERGED share, a slot without
its USD, more than K_MAX slots, a spawn list that differs from the checkpoint's ledger) therefore fails the preflight
with the same error. The module imports no Isaac Lab, Omniverse, USD or torch (tests/rl_tasks/test_extra_slots_plan.py).
"""

from __future__ import annotations

import dataclasses
import json
import os

from dexengine.data.objects import collider_shape as _cs  # owner of each slot body's collider shape
from dexengine.data.objects import extra_slots as _xs  # owner of the rule and the slot count
from dexengine.data.objects import extra_slots_variant as _xv  # placement variant -> source tree
from dexengine.data.objects import hull as _oh  # owner of the training USD paths
from dexengine.placement.coords import core as _coords  # noqa: F401 — import kept for its side effect
from dexengine.placement.scene import place_rule as _pr  # statement of the brushed static extras dropped from the spawn
from dexengine.rl.env_setup import extra_slots_ledger as _el  # the ledger: process state, signatures, comparison


@dataclasses.dataclass
class SlotPlan:
    """The extra-slot rows of a run; clip order = the sorted roster of the tree folder entry."""

    seqs: list  # clip names
    rows: list = dataclasses.field(default_factory=list)  # per clip: its slot rows + usd · held · shape
    routed: dict = dataclasses.field(default_factory=dict)  # shares outside the slots by class (extra_slots.ROUTED)
    sensitive: dict = dataclasses.field(default_factory=dict)  # clip -> extras the rule splits on (old-ledger check)
    dropped: dict = dataclasses.field(default_factory=dict)  # clip -> brushed static extras the placement rule dropped
    dsets: set = dataclasses.field(default_factory=set)  # dataset names of the clips
    # per clip: the slot rows the ledger signature is taken from (a placement variant: its source's)
    sig_rows: list = dataclasses.field(default_factory=list)

    def rule(self) -> str:
        """The rule version the ledger and the marker carry (`extra_slots.rule_version` over the run's datasets)."""
        return _xs.rule_version(sorted(self.dsets))


def plan_rows(seq_dir, sequence: str | None) -> SlotPlan:
    """Read the extra-slot rows of every clip of a tree folder entry (`--sequence_dir` and the comma-separated
    `--sequence`). Prints the collider statement per clip (`[TRI_COLL] statement`, the value stage 2f's ledger
    carries)."""
    seqs = [t for t in (sequence or "").split(",") if t]
    if not seqs or not seq_dir or seqs != sorted(seqs):
        raise RuntimeError(
            f"[EXTRA_SLOTS] clips {len(seqs)} · tree folder {seq_dir!r} · sorted {seqs == sorted(seqs)} — "
            f"slots are built only from the sorted roster of the tree folder entry (--sequence_dir; loader "
            f"order contract) · other entries state {_xs.ENV}=0 (default on)"
        )
    plan = SlotPlan(seqs, routed=dict.fromkeys(_xs.ROUTED, 0))
    fps = {q: _plan_clip(plan, os.path.join(str(seq_dir), q), q) for q in seqs}
    print(  # the collider statement per clip (collider_shape.statement_fp; stage 2f's ledger carries the same value)
        f"[TRI_COLL] statement clips {len(fps)} · " + " ".join(f"{q}={fp}" for q, fp in fps.items()), flush=True
    )
    return plan


def _plan_clip(plan: SlotPlan, tree: str, q: str) -> str:
    """Add clip q's slot rows to `plan`; return the clip's collider statement fingerprint."""
    with open(os.path.join(tree, "task_info.json"), encoding="utf-8") as f:
        ti = json.load(f)
    plan.dsets.add(str(ti.get("dataset_name") or ""))
    slots, out = _xs.slot_rows(tree, ti)  # above K_MAX fails here; shares outside the slots are counted
    _split_static_extras(plan, out, ti, tree, q)
    plan.rows.append([_slot_row(tree, ti, q, r) for r in slots])
    # ledger signature rows: a placement variant's are its source's (the classification is the source's; a brushed
    # extra the placement rule dropped at the variant's place is not in the signature — extra_slots_variant)
    src = _xv.source_dir(tree, ti)
    plan.sig_rows.append(_xs.slot_rows(src)[0] if src else slots)
    return _cs.statement_fp(str(ti["dataset_name"]), slots)


def _slot_row(tree: str, ti: dict, q: str, r: dict) -> dict:
    """One slot row with its training USD (which must exist), its held spans and its collider shape."""
    usd = str(_oh.usd_file(str(ti["dataset_name"]), r["obj_id"], False))
    if not os.path.isfile(usd):
        raise RuntimeError(f"[EXTRA_SLOTS] {q}: extra {r['name']} ({r['obj_id']}) has no USD: {usd}")
    try:  # input of the reference line (the spans a person held); unknown when missing
        held = _xs.held_intervals(tree, ti, r["name"])
    except ValueError as e:
        held = None
        print(f"[EXTRA_TRACE] {q}: {r['name']} held span unknown — {e}", flush=True)
    return dict(r, usd=usd, held=held, shape=_cs.row_shape(str(ti["dataset_name"]), r))


def _split_static_extras(plan: SlotPlan, out: dict, ti: dict, tree: str, q: str) -> None:
    """Extras the rule splits on (static, never touched by a hand: receivers, visual only, hand-touched receivers),
    read by the old-ledger comparison; the brushed static extras the placement rule dropped; the outside-slot shares."""
    _cr = _xs.clip_rows(tree, ti)
    plan.sensitive[q] = sum(
        1
        for r in _cr
        if r["cls"] == _xs.VISUAL_ONLY
        or (r["cls"] == _xs.STATIC_SPAWN and r["why"] == _xs.RECEIVER)
        or _xs.hand_receiver(r)
    )
    if _gone := _pr.spawn_dropped(_cr):  # brushed static extras the placement rule dropped from this clip's scene
        plan.dropped[q] = _gone
    for c, names in out.items():
        plan.routed[c] += len(names)


def collide_state(tables: dict) -> dict:
    """Collision partner tables -> the two cells of this process's statement (collide = ledger line form · robot_kinds =
    kinds that collide with the robot)."""
    ts = _el.train_scene_owner()
    return {"collide": {k: ts.partners(t) for k, t in tables.items()}, "robot_kinds": sorted(ts.robot_kinds(tables))}


def spawn_state(plan: SlotPlan, rule: str) -> dict:
    """This process's spawn-list statement {rule, clips: {clip: signature}, sensitive, collide, robot_kinds, drops}:
    what the ledger write records (`extra_slots_ledger_record`) and the checkpoint ledger comparison reads. A placement
    variant's signature is its source's (`sig_rows`)."""
    return {
        "rule": rule,
        "clips": {q: _el.extra_slot_signature(r) for q, r in zip(plan.seqs, plan.sig_rows, strict=True)},
        "sensitive": dict(plan.sensitive),
        **collide_state(_el._extra_collide_tables(_xs.kinds_present(plan.rows))),
        "drops": {q: _el.train_scene_owner().drop_text(d) for q, d in plan.dropped.items()},
    }


def record_spawn_list(plan: SlotPlan, rule: str, ctx_label: str) -> None:
    """Set this process's spawn-list statement, print the dropped extras and compare the statement with every
    checkpoint ledger this process value-checked (`_ledger_compare`: a different spawn list fails)."""
    _el._EXTRA_STATE.clear()
    _el._EXTRA_STATE.update(spawn_state(plan, rule))
    print(
        f"{_pr.TAG} brushed static extras dropped from the spawn: clips {len(plan.dropped)}/{len(plan.seqs)} · "
        f"objects {sum(len(d) for d in plan.dropped.values())} — kept by name in the drop lines of the ledger "
        f"{_el.EXTRA_SLOTS_LEDGER} ({ctx_label})",
        flush=True,
    )
    for led in _el._EXTRA_LEDGER_SEEN:
        _el._ledger_compare(led, _el._EXTRA_STATE, ctx_label)


def slots_off(args_cli, ctx_label: str) -> None:
    """Extra slots off (`DEXENGINE_EXTRA_SLOTS=0`, stated or read from a checkpoint ledger): the marker, and the scene
    check of a ledger-less checkpoint this process loaded (`_scene_without_ledger`)."""
    print(f"[EXTRA_SLOTS] off ({_xs.ENV}=0 stated or from the ledger) · {ctx_label}", flush=True)
    _el._EXTRA_OFF_BUILT.append((args_cli, ctx_label))
    _el._scene_without_ledger(args_cli, ctx_label)
