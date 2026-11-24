"""Extra-slot rows of the collision truth table (`collision_check`): RE = robot <-> extra slot, OE = task object <->
extra slot.

Independent oracle: the expectation is read from `collision_presets.extra_collide(kind)` (env i uses the kinds of
clip i % n_clips; an empty slot = robot ON, task objects OFF); the measurement is the stage: FilteredPairs (either
direction, rigid-body sub-prims included) and the collision groups (`collision_groups.GroupIndex`); robot <-> slot is
the groups alone, a FilteredPairs one fails (`collision_groups.refuse_filtered`). A pair is OFF when
every rigid-body pair is cut, ON when none is, partial otherwise. Values are `<kind>:<value>` per slot. A mismatch
raises. Not part of the cross-env comparison (kinds differ per env). Needs only pxr.
"""

from __future__ import annotations

import re

ROWS = ("RE", "OE")
_PAREN = re.compile(r"\([^)]*\)")


def slot_paths(scene, env_root: str) -> list[str]:
    """Extra-slot prim roots of one env."""
    from dexengine.rl.tasks.scene.collision_bodies import slot_prim_paths
    from dexengine.rl.tasks.scene.extra_slots import PREFIX

    return slot_prim_paths(scene, env_root, PREFIX, "COLL_AUDIT")


def _anc(path: str, roots: set) -> str | None:
    p = path
    while p and p != "/":
        if p in roots:
            return p
        p = p.rsplit("/", 1)[0]
    return None


def filter_pairs(stage, env_root: str, slots: list, others: set) -> set:
    """(slot root, partner body) for every FilteredPairs entry in the env with one end under a slot and the other
    under a body of `others`."""
    from pxr import Usd, UsdPhysics

    sl, out = set(slots), set()
    for p in Usd.PrimRange(stage.GetPrimAtPath(env_root)):
        if not p.HasAPI(UsdPhysics.FilteredPairsAPI):
            continue
        rel = UsdPhysics.FilteredPairsAPI(p).GetFilteredPairsRel()
        for t in rel.GetTargets() if rel else []:
            a, b = p.GetPath().pathString, t.pathString
            for x, y in ((a, b), (b, a)):
                s, o = _anc(x, sl), _anc(y, others)
                if s and o:
                    out.add((s, o))
    return out


def group_pairs(stage, slots: list, others: set) -> set:
    """(slot root, partner body) for every pair the stage's collision groups cut."""
    from dexengine.rl.tasks.scene.collision_groups import GroupIndex

    ix = GroupIndex(stage)
    return {(s, o) for s in slots for o in others if ix.blocked(s, o)}


def rows(slots: list, kinds_row, robot: set, obj: set, ex_filt: set) -> dict:
    """{'RE': ..., 'OE': ...}: `<kind>:<value>` per slot ('?' when no kind was stated)."""
    out = {}
    for row, bodies in (("RE", sorted(robot)), ("OE", sorted(obj))):
        vals = []
        for k, sl in enumerate(slots):
            kind = "?" if kinds_row is None or k >= len(kinds_row) else (kinds_row[k] or "empty")
            n = sum(1 for b in bodies if (sl, b) in ex_filt)
            v = "ON" if n == 0 else f"OFF(filter {n})" if n == len(bodies) else f"partial(filter {n} / {len(bodies)})"
            vals.append(f"{kind}:{v}")
        out[row] = " ".join(vals)
    return out


def fields(scene, env_root: str, kinds_row, robot: set, obj: set) -> dict:
    """Fields added to one audit row: {'extra': {RE, OE}, 'extra_slots': n_slots}."""
    slots = slot_paths(scene, env_root)
    if not slots:
        return {"extra": dict.fromkeys(ROWS, "-(no slots)"), "extra_slots": 0}
    from dexengine.rl.tasks.scene.collision_groups import refuse_filtered

    refuse_filtered(env_root, "extra slot", filter_pairs(scene.stage, env_root, slots, set(robot)))
    ex = filter_pairs(scene.stage, env_root, slots, set(obj)) | group_pairs(scene.stage, slots, set(robot) | set(obj))
    return {"extra": rows(slots, kinds_row, set(robot), set(obj), ex), "extra_slots": len(slots)}


def expected(kinds_row) -> dict:
    """Stated kinds -> expected RE / OE values (without the parenthesized evidence)."""
    from dexengine.rl.tasks.scene.collision_presets import extra_collide

    re_, oe = [], []
    for kind in kinds_row:
        t = extra_collide(kind) if kind is not None else {"hand": "ON", "arm": "ON", "objects": "OFF"}
        if t["hand"] != t["arm"]:
            raise RuntimeError(f"[COLL_AUDIT] kind {kind}: hand {t['hand']} != arm {t['arm']}; RE needs one value")
        re_.append(f"{kind or 'empty'}:{t['hand']}")
        oe.append(f"{kind or 'empty'}:{t['objects']}")
    return {"RE": " ".join(re_), "OE": " ".join(oe)}


def norm(v: str) -> str:
    """A row value without its parenthesized evidence."""
    return " ".join(_PAREN.sub("", v).split())


def check(res: list[dict], idx: list[int], extra_kinds) -> None:
    """Print RE / OE and compare them with the stated kinds; raise on a mismatch or when slots exist without kinds."""
    if not any(r.get("extra_slots") for r in res):
        return
    for row in ROWS:
        for i, r in zip(idx, res, strict=True):
            print(f"  {row}    env_{i}: {r['extra'][row]}", flush=True)
    if not extra_kinds:
        raise RuntimeError("[COLL_AUDIT] extra slots exist but no slot kinds (cfg.extra_slot_kinds) reached the audit")
    bad = []
    for i, r in zip(idx, res, strict=True):
        want = expected(extra_kinds[i % len(extra_kinds)])
        bad += [
            f"env_{i}.{row}: measured {r['extra'][row]!r} != expected {want[row]!r}"
            for row in ROWS
            if norm(r["extra"][row]) != want[row]
        ]
    if bad:
        raise RuntimeError(
            "[COLL_AUDIT] extra-slot collision differs from collision_presets.extra_collide: " + " | ".join(bad[:4])
        )
    print(f"[COLL_AUDIT] extra slots RE / OE = collision_presets.extra_collide for env {idx}", flush=True)
