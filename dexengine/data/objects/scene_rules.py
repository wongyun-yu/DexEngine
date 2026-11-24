"""The scene-rule passes of the extra-object slot rule (extra_slots.clip_rows runs them after its own passes).

Scene rules are on for every dataset by default; a dataset leaves one only by naming it, with a reason, in its
`Profile.scene_rules_off` (rules registry scope `default:scene_rules`). Off for a dataset = the rows of the rule before
(byte-identical). Each rule changes the rows of one class only:
  S-carried     carried SLOT rows: carrier, interval and relative pose; attached to the carrier (kind carried_attached)
  S-touch-gate  touched static spawns: kind touched_solid (collides with the main objects); a recorded overlap above
                the true-shape value states `overlap_drop` and drops the clip (touch_gate)
  S-envelope    visual-only still extras next to the recorded motion: static spawn `why=envelope`, kind envelope
                (envelope), with the slot cap that drops envelope rows first
  S-true-shape  no row change: the collider shape of the pinned still kinds (triangles, collider_shape.shape_of), read
                by the env authoring and every ruler
  S-dyn-pairs   still rows that collide with the main objects and move together with one at frame-0 overlap: attached
                to it (kind carried_attached) or `together` no_path (dyn_pairs.together_pass); the both-dynamic pairs
                are a statement per tree (dyn_pairs.dyn_pairs), no row change
The rule version a run carries (`rule_version`) names the rules on for its datasets.
"""

from __future__ import annotations

import numpy as np

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import dyn_pairs as DP
from dexengine.data.objects import envelope as EV
from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import touch_gate as TG

CARRY_RULE = "S-carried"
#: S-robot-static: the robot collides with the still extras in training (owner collision_presets.extra_collide).
ROBOT_RULE = "S-robot-static"
#: Rule → the suffix it adds to the slot rule version when on for a dataset of the run (fixed order).
VERSIONS = {
    CARRY_RULE: "SC-2026-11-19-attached",
    TG.RULE_ID: "TG-2026-11-19-artic",
    EV.RULE_ID: "EV-2026-11-19-artic",
    CS.RULE_ID: "TS-2026-11-20-triangles",
    ROBOT_RULE: "RS-2026-11-20-robot-on",
    "S-float-settle": "FS-2026-11-20-settle",
    DP.RULE_ID: "DP-2026-11-21-scope",  # extra pairs together at 5 mm / 5°; main pairs separated by F0-1d first
}


def on(rid: str, ds: str) -> bool:
    """Is scene rule `rid` on for dataset `ds` (rules registry `rule_on`)? A dataset outside the registry has no
    profile, so every scene rule is off for it."""
    from dexengine.data import rules as RR  # noqa: PLC0415

    return ds in RR.ALL and RR.rule_on(rid, ds)


def kinds_on(ds: str) -> tuple:
    """The scene-rule kinds (extra_slots.SCOPED_KINDS order) whose rule is on for dataset `ds`."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    rule_of = {
        XS.KIND_ATTACHED: CARRY_RULE,
        XS.KIND_TOUCHED_SOLID: TG.RULE_ID,
        XS.KIND_ENVELOPE: EV.RULE_ID,
    }
    return tuple(k for k in XS.SCOPED_KINDS if on(rule_of[k], ds))


def rule_version(base: str, datasets) -> str:
    """`base`, joined with the suffix of every scene rule on for some dataset of the run (VERSIONS order)."""
    tail = [v for rid, v in VERSIONS.items() if any(on(rid, str(d)) for d in datasets)]
    return "+".join([base, *tail])


def apply(tree_dir: str, ti: dict, q: np.ndarray, rows: list[dict], drop, kinds_only: bool = False) -> None:
    """Run the scene-rule passes that are on for the tree's dataset, in place (module doc). `drop(o)` = the placement
    rule's reason for still extra o. kinds_only=True (extra_slots.clip_rows): the envelope pass records no overlap and
    the touch gate stops a pair at its first frame over the true-shape overlap — the classes and kinds are the same."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415 — class and kind vocabulary

    ds = str(ti.get("dataset_name") or "")
    if any(r.get("why") == "carried" for r in rows) and on(CARRY_RULE, ds):
        carry_pass(tree_dir, ti, q, rows)
    if on(TG.RULE_ID, ds):
        touch_pass(tree_dir, ti, q, rows, kinds_only=kinds_only)
    if on(EV.RULE_ID, ds):
        EV.envelope_pass(tree_dir, ti, q, rows, drop, measure_overlap=not kinds_only)
        if gone := EV.cap(rows, XS.K_MAX):
            print(
                f"[EXTRA_SLOTS] {EV.RULE_ID} slot cap {XS.K_MAX}: envelope rows left visual only, farthest first "
                f"{gone}",
                flush=True,
            )
    if on(DP.RULE_ID, ds):
        DP.together_pass(tree_dir, ti, q, rows)


def touch_pass(tree_dir, ti: dict, q: np.ndarray, rows: list[dict], kinds_only: bool = False) -> None:
    """S-touch-gate: every touched static spawn (kind touched) gets the gate statement (row `touch_gate`), kind
    touched_solid when all its pairs pass, and `overlap_drop` when a pair is deeper than the true-shape overlap.
    kinds_only: a pair stops at its first frame over the overlap (touch_gate.gate_row) — the same kind."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    mains = TG.main_objects(ti)
    for r in rows:
        if r.get("cls") == XS.STATIC_SPAWN and r.get("kind") == XS.KIND_TOUCHED:
            g = TG.gate_row(tree_dir, ti, q, r, mains, kinds_only=kinds_only)
            r["touch_gate"] = g
            if g["on"]:
                r["kind"] = XS.KIND_TOUCHED_SOLID
            if g["drop"]:
                r["overlap_drop"] = g["drop"]  # the clip is dropped (collider_shape.overlap_drop, step 2f)


def carry_pass(tree_dir: str, ti: dict, q: np.ndarray, rows: list[dict]) -> None:
    """S-carried: every carried row is attached to its carrier for the whole clip — kinematic, at the simulated carrier
    pose × the stated relative pose. The carry relation comes from `XG.carry_of` (candidates = main objects and the
    moved extra slots a hand moved). Row field `carry` = {carrier, carrier_role main|slot, carrier_side (main), start,
    end (tree frames, the measured carry interval), rel (7), dev_m, dev_deg}; kind carried_attached. No carrier: kind
    moving (the world-recorded reference: carrier unknown) with row field `carry_miss`
    (extra_slots_geom.CARRY_MISS). Measured on the 151 main-carrier pairs of four hands: the stated relative pose holds
    the reference rider path within median 1.2 mm / 1.7° and at most 19.0 mm / 9.4° over the whole clip."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    objs = {str(o["name"]): o for o in ti.get("objects") or []}
    idle = set(ti.get("idle_sides") or [])  # a main object of an idle side has no robot hand: not a carrier
    mains = [
        n
        for n, o in objs.items()
        if o.get("role") == "main"
        and o.get("mesh_dir")
        and o.get("qpos_start") is not None
        and o.get("side") not in idle
    ]
    slots = [
        r["name"] for r in rows if r["cls"] == XS.SLOT and r.get("why") != "carried" and objs[r["name"]].get("mesh_dir")
    ]
    for r in rows:
        if r["cls"] != XS.SLOT or r.get("why") != "carried" or not objs[r["name"]].get("mesh_dir"):
            continue
        rel = XG.carry_of(tree_dir, ti, q, r["name"], mains + slots)
        if rel is None:  # carrier unknown: keeps following the world-recorded reference (kind moving)
            r["carry_miss"] = XG.carry_miss(tree_dir, ti, q, r["name"], mains + slots)
            continue
        main = rel["carrier"] in mains
        r["carry"] = {
            "carrier": rel["carrier"],
            "carrier_role": "main" if main else "slot",
            "carrier_side": str(objs[rel["carrier"]].get("side")) if main else None,
            "start": int(rel["start"]),
            "end": int(rel["end"]),
            "rel": [round(v, 6) for v in rel["rel"]],
            "dev_m": rel["dev_m"],
            "dev_deg": rel["dev_deg"],
        }
        r["kind"] = XS.KIND_ATTACHED
    if line := carry_marker(tree_dir, rows):
        print(line, flush=True)


def carry_marker(tree_dir, rows: list[dict]) -> str:
    """`[CARRY_ATTACHED] <clip>: <rider> on <carrier> (frames a–b) · … · whole clip at the stated relative pose`
    ('' without carry)."""
    import os  # noqa: PLC0415

    got = [
        f"{r['name']} on {r['carry']['carrier']} (frames {r['carry']['start']}–{r['carry']['end']})"
        for r in rows
        if r.get("carry")
    ]
    if not got:
        return ""
    clip = os.path.basename(str(tree_dir).rstrip("/"))
    return f"[CARRY_ATTACHED] {clip}: " + " · ".join(got) + " · whole clip at the stated relative pose"
