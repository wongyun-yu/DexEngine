"""Extra-slot classification of a placement variant (`<clip>_pvNN`, record.augment.place_gen) = the source clip's.

Why: a variant is a rigid transform of the whole scene (every object and the active hand reference) on the desk
plane, so the relations between objects (motion, hand contact, receiving, below the work surface) are the source's.
The classification is a property of the source clip: a variant takes the source tree's rows, so the scene the
checkpoint learned, the judge scene and the record scene stay one scene per clip whatever a measurement taken again on
the variant would say. Source tree = the
variant's sibling in the same layer folder (augment.scene shadow_dir links the source entries there), measured with
the source geometry and the source trajectory.

Only properties of the placement are measured at the variant's place: the placement rule (place_rule.drop_why: a
brushed still extra overlapping the robot's footprint or off the desk) is a per-placement verdict, so when the source
row or the variant's own row was dropped for such a reason (extra_slots.PLACE_DROPPED) that object takes the variant's
own row: dropped where the variant violates it, standing in the variant where only the source violated it. The
checkpoint ledger comparison uses the source rows' signature (rl.env_setup.extra_slots_plan).

When the own classification (measured again on the variant geometry) differs from the one used, one marker line is
printed per tree and nothing dies:
  [EXTRA_SLOTS_VARIANT] <variant>: own classification != source <source tree> — <name> own <cls:why:kind> -> used …
Input contract (no silent pass): a variant name needs task_info.aug_place.src_tree = the source tree name and the
source tree as a sibling; a source that is itself a variant, an aug_place record on a non-variant name, or extra
objects that differ between variant and source die (ValueError, the form of clip_rows' other input defects).
The ledger comparison (rl.env_setup.extra_slots_ledger._ledger_compare) also folds variant names to the source clip
name (source_clip) and dies when the source has no ledger row.
Name rules: core.paths (clip_name, variant_source, tree_name). Tests: tests/data/test_extra_slots_variant.py.
"""

from __future__ import annotations

import os

MARK = "[EXTRA_SLOTS_VARIANT]"
_SAID: set = set()  # variant trees (real paths) whose marker was printed; once per process


def source_clip(name: str) -> str:
    """Tree or clip name -> the source clip name (no suffix; a non-variant gives its own clip name)."""
    from dexengine.core import paths as P  # noqa: PLC0415 — owner of the name rules

    return P.variant_source(P.clip_name(name))


def is_variant(name: str) -> bool:
    from dexengine.core import paths as P  # noqa: PLC0415

    return source_clip(name) != P.clip_name(name)


def source_dir(tree_dir, ti: dict) -> str | None:
    """The source tree folder (sibling) of a variant tree; None for a non-variant. Name and aug_place record that
    disagree die."""
    from dexengine.core import paths as P  # noqa: PLC0415

    d = os.path.normpath(str(tree_dir))
    name, ap = os.path.basename(d), ti.get("aug_place")
    if not is_variant(name):
        if ap is not None:
            raise ValueError(
                f"{MARK} {d}: not a variant name but an aug_place record ({ap.get('src_tree')!r}) — cannot tell "
                "source from variant"
            )
        return None
    want = P.tree_name(source_clip(name))
    if is_variant(want):
        raise ValueError(f"{MARK} {d}: the source name {want} is a variant too — there is no variant of a variant")
    got = (ap or {}).get("src_tree")
    if got != want:
        raise ValueError(f"{MARK} {d}: the variant name's source {want} != aug_place.src_tree {got!r}")
    src = os.path.join(os.path.dirname(d), want)
    if not os.path.isfile(os.path.join(src, "task_info.json")):
        raise ValueError(f"{MARK} {d}: no source tree {src} — a variant's classification is read from its source")
    return src


def _key(r: dict) -> str:
    return f"{r['cls']}:{r.get('why') or '-'}:{r.get('kind') or '-'}"


def rows(tree_dir, ti: dict, q, strict: bool, src: str, own_fn, kinds_only: bool = False) -> list[dict]:
    """A variant's rows = the source rows (classification); only objects the placement rule dropped take the variant's
    own row; a marker when they differ from the own classification. own_fn = extra_slots._own_rows (the body measuring
    a tree on its own geometry: used to record the variant's own classification and to read the placement verdict)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415 — class names and the source rows

    base = XS.clip_rows(src, None, None, strict, kinds_only)
    own = {r["name"]: r for r in own_fn(str(tree_dir), ti, q, False, kinds_only)}
    if set(own) != {r["name"] for r in base}:
        raise ValueError(f"{MARK} {tree_dir}: extra objects {sorted(own)} != source {sorted(r['name'] for r in base)}")

    def placed(r: dict) -> bool:  # a row the placement rule dropped
        return r["cls"] == XS.STATIC and r.get("why") in XS.PLACE_DROPPED

    out = [dict(own[r["name"]] if placed(r) or placed(own[r["name"]]) else r) for r in base]
    diff = [f"{r['name']} own {_key(own[r['name']])} -> used {_key(r)}" for r in out if _key(own[r["name"]]) != _key(r)]
    real = os.path.realpath(str(tree_dir))
    if diff and real not in _SAID:
        _SAID.add(real)
        print(
            f"{MARK} {os.path.basename(real)}: own classification != source {os.path.basename(src)} — "
            + " · ".join(diff)
            + " (placement augment does not change object relations: the classification is the source's, only the "
            "placement rule is the variant's)",
            flush=True,
        )
    return out
