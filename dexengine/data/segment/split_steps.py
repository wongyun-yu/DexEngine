"""D36 split steps: the units of this hand's timeline (compute_units), the registry writer (write_d36) and the
two canonical-chain links (define_step = D36, tail_step = D41); driven by segment/split.py."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from dexengine.core import paths as _P
from dexengine.data import rules as RR
from dexengine.data.segment.build import (
    apply_table,
    reslice_stale,
)
from dexengine.data.segment.common import (
    HAND,
    RULE_NAME,
    _base_emb,
    _own_registry_path,
    _registry_path,
)
from dexengine.data.segment.rule import (
    CUT_STATS,
    drop_contained,
    segments_of,
    unit_key,
)
from dexengine.data.segment.tail import (
    d36_diff,
    d41_in_scope,
    static_artic_own,
    tail_main,
)


def compute_units(ds: str, emb: str, hand_dir: str, min_seconds: float, nested: bool) -> tuple[dict, int, dict]:
    """Parent → [{name,t0,t1}] (on this hand's base timeline).
    Returns (table, count skipped as nested candidates, parent → names of wholly contained pieces dropped).
    Must be hand-independent, which --verify-hand measures. Contained-piece removal (drop_contained) runs **after** the
    1-second threshold (pieces dropped by the threshold are already gone)."""
    out: dict[str, list[dict]] = {}
    contained: dict[str, list[str]] = {}
    nested_skipped = 0
    CUT_STATS.clear()
    root = _base_emb(ds, emb, hand_dir)
    names = sorted(os.listdir(root)) if root.is_dir() else []
    for clip in names:
        d = root / clip
        if not (d / "task_info.json").is_file():
            continue
        ti = json.loads((d / "task_info.json").read_text())
        segs = segments_of(ti)
        if not segs:
            continue
        if re.fullmatch(r".+_t\d+", clip) and not nested:
            nested_skipped += 1
            continue
        fps = float((ti.get("case") or {}).get("fps") or 30.0)
        T = int((ti.get("case") or {}).get("T") or 0) or int(np.load(d / "0/trajectory_kinematic.npz")["qpos"].shape[0])
        minf = int(min_seconds * fps)
        units = []
        for i, sg in enumerate(segs):
            t0, t1 = max(0, sg["t0"]), min(T, sg["t1"])
            if t1 - t0 < minf:
                continue
            units.append(
                {
                    "name": f"{clip}_t{sg.get('i', i)}",
                    "t0": t0,
                    "t1": t1,
                    "left": sg.get("left"),
                    "right": sg.get("right"),
                }
            )
        units, gone = drop_contained(units)
        if gone:
            contained[clip] = [u["name"] for u in gone]
        if units:
            out[clip] = units
    c = CUT_STATS
    print(
        f"[CUT_RULE] D36 v6 {ds}/{emb}({hand_dir}): switches to non-main objects = only when moved"
        f"(case.move_runs) · parents with such runs {c.get('parents', 0)} · runs {c.get('nonmain_segs', 0)}"
        f"(kept for motion {c.get('moved_kept', 0)} · filled {c.get('filled', 0)}) · "
        f"objects dropped from grasp switching {c.get('used_dropped', 0)}",
        flush=True,
    )
    return out, nested_skipped, contained


def write_d36(reg: Path, table: dict, contained: dict, min_seconds: float, nested: bool) -> bool:
    """Single owner of writing the registry's D36 statement (--define · define_step); a hand-independent layer written
    only by the reference hand (this hand). The D41 overlay (tail_rule) is not included; for a registry whose windows
    changed, the overlay is measured again by the reference hand's tail_step. Equal content is not rewritten (mtime
    unchanged, tree_io.write_json_if_changed). Returns whether it wrote."""
    reg.parent.mkdir(parents=True, exist_ok=True)
    from dexengine.data import tree_io as _TIO

    return _TIO.write_json_if_changed(
        reg,
        {
            "rule": RULE_NAME,
            "reference_hand_dir": HAND,
            "min_seconds": min_seconds,
            "nested": nested,
            "parents": table,
            # history (not statement): pieces this rule dropped; evidence for the difference proof and cleanup of
            # existing trees
            "contained_removed": contained,
        },
    )


def _own_windows(ds: str, emb: str, r: dict, r0: dict | None, ref: str) -> tuple:
    """(parents owned by per-hand windows, the comparison without them). Per-hand windows in a D41-scope dataset
    fail (the overlay rule does not know them)."""
    own = (
        static_artic_own(ds, emb, set(r["diff"]) | set(r["only_ref_present"]) | set(r["only_this"]), ref) if r0 else []
    )
    if own and d41_in_scope(ds):
        raise SystemExit(
            f"[D36_STEP] ✗ {ds}/{emb}({HAND}): {len(own)} per-hand windows in a D41-scope dataset — "
            "the overlay rule does not know per-hand windows yet"
        )
    if own:
        ow = set(own)
        r = dict(r, **{k: [c for c in r[k] if c not in ow] for k in ("diff", "only_ref_present", "only_this")})
    return own, r


def _verify_other_hand(
    ds: str,
    emb: str,
    head: str,
    r0: dict | None,
    stale_rule: bool,
    differ: list,
    own: list,
    table: dict,
    ref: str,
    r: dict,
) -> int:
    """A hand other than the reference only compares (rc 2 on a difference); when it agrees, its per-hand window file
    is written (owned parents) or removed (none)."""
    bad = r0 is None or stale_rule or bool(differ)
    if not bad:
        from dexengine.data import tree_io as _TIO

        ownf = _own_registry_path(ds, emb)
        if own:
            _TIO.write_json_if_changed(
                ownf,
                {
                    "rule": RULE_NAME,
                    "hand_dir": HAND,
                    "reference_hand_dir": ref,
                    "why": "per-hand window — the parent window that the static-articulation rule (STATIC_ARTIC) "
                    "loaded for this hand only",
                    "parents": {c: table.get(c, []) for c in own},
                },
            )
        elif ownf.is_file():
            # with no owned parents there is no per-hand window (rebuilding the reference hand with the static rule
            # lets the shared windows win again)
            ownf.unlink()
    print(
        f"{head} → {'✗ differs -- the reference hand runs stage1.split_apply first' if bad else 'verified same'}",
        flush=True,
    )
    for c in differ[:5]:
        print(
            f"    {c}: registry {unit_key(r['ref'].get(c, []))} vs this hand {unit_key(table.get(c, []))}",
            flush=True,
        )
    return 2 if bad else 0


def _redefine(
    ds: str,
    emb: str,
    label: str,
    head: str,
    reg,
    table: dict,
    contained,
    min_seconds: float,
    nested: bool,
    nested_skipped,
    differ: list,
    r: dict,
) -> int:
    """Reference hand with a changed table: rewrite the D36 statement and re-slice the pieces whose windows changed
    (rc 2 when not every stale piece was re-sliced)."""
    write_d36(reg, table, contained, min_seconds, nested)
    rs = reslice_stale(ds, emb, table, write=True, label=label)
    print(
        f"{head} → redefined · parents {len(table)} · pieces {sum(len(v) for v in table.values())} · "
        f"re-sliced {rs['resliced']}/{rs['stale']}" + (f" · nested skipped {nested_skipped}" if nested_skipped else ""),
        flush=True,
    )
    for c in differ[:5]:
        print(f"    {c}: old registry {unit_key(r['ref'].get(c, []))} → {unit_key(table.get(c, []))}", flush=True)
    return 2 if rs["resliced"] != rs["stale"] else 0


def define_step(ds: str, emb: str, label: str, min_seconds: float, nested: bool) -> int:
    """D36 link of the canonical chain (called by stage1.split_apply **before** --apply): re-measure the current rule's
    table on this hand's timeline (compute_units) and compare it with the registry's D36 statement (without the D41
    part) and rule version (RULE_NAME). If the piece rule or the case timeline changed without redefining the
    registry, --apply would cut pieces with old windows.
    Reference hand (the registry's reference_hand_dir; without a registry the canonical axis hand
    data.rules.AXIS_HAND):
      on a difference, redefine (write_d36, the same write as --define) and re-slice existing pieces whose windows
      changed (reslice_stale --label).
    Other hands: compare only; differing shared parent windows or parents only in the registry but present here fail
      (the reference hand's link runs first). Parents only in this hand are only counted (--apply cuts registry parents
      only; such a parent getting no pieces stays visible in the marker).
    Only the version stale (any hand): when this hand re-measures **every** registry parent under the current rule and
      they match (windows differ · only in registry · absent here · only here all 0), only the version name is rewritten
      (restamp_rule; reference hand, parents and D41 overlay unchanged). If any cannot be measured, other hands fail and
      the reference hand redefines.
    Marker [D36_STEP] (state = same · version updated · redefined · verified same · ✗). rc 0 · 2."""
    reg = _registry_path(ds, emb)
    table, nested_skipped, contained = compute_units(ds, emb, HAND, min_seconds, nested)
    r0 = json.loads(reg.read_text()) if reg.is_file() else None
    ref = r0.get("reference_hand_dir") if r0 else _P.hand_spec_dir(RR.AXIS_HAND)
    r = d36_diff(ds, emb, r0["parents"] if r0 else {}, table)
    stale_rule = bool(r0) and r0.get("rule") != RULE_NAME
    # Parents for which the static-articulation rule (STATIC_ARTIC) applied only in this hand are owned by per-hand
    # windows:
    #   left out of the shared comparison and written to the per-hand registry (only when not the reference hand)
    own, r = _own_windows(ds, emb, r, r0, ref)
    differ = r["diff"] + r["only_ref_present"]
    changed = r0 is None or stale_rule or bool(differ) or bool(r["only_this"])
    head = (
        f"[D36_STEP] {ds}/{emb}({HAND}): reference hand {ref} · parents here {len(table)} · windows differ "
        f"{len(r['diff'])} · registry only {len(r['only_ref_present'])} · this hand only {len(r['only_this'])} · "
        f"rule {'same' if not stale_rule else repr((r0 or {}).get('rule', ''))[:40] + ' != current'}"
        + (" · no registry" if r0 is None else "")
        + (f" · per-hand windows (STATIC_ARTIC) {len(own)}" if own else "")
    )
    proven = r0 is not None and not differ and not r["only_this"] and not r["absent"]
    if stale_rule and proven:
        from dexengine.data import tree_io as _TIO

        _TIO.write_json_if_changed(reg, dict(r0, rule=RULE_NAME))
        print(
            f"{head} → rule updated (same content: all {len(r['common'])} parents re-measured with this hand; "
            "reference hand and overlays unchanged)",
            flush=True,
        )
        return 0
    if ref != HAND:
        return _verify_other_hand(ds, emb, head, r0, stale_rule, differ, own, table, ref, r)
    if not changed:
        print(f"{head} → same (not rewritten)", flush=True)
        return 0
    return _redefine(ds, emb, label, head, reg, table, contained, min_seconds, nested, nested_skipped, differ, r)


def tail_step(ds: str, emb: str, label: str) -> int:
    """D41 link of the canonical chain (called by stage1.split_apply after --apply). Reference hand: write the overlay
    (tail_main), re-slice pieces whose windows changed when the registry changed (reslice_stale) and make new pieces
    (apply_table) · other hands: compare the overlay (the same function as --verify-tail). Out of scope or no registry:
    skipped, marker only. Marker [D41_STEP]. rc 0 · 2."""
    reg = _registry_path(ds, emb)
    if not d41_in_scope(ds) or not reg.is_file():
        print(
            f"[D41_STEP] {ds}/{emb}({HAND}): skipped -- {'outside the D41 scope' if reg.is_file() else 'no registry'}"
        )
        return 0
    before = reg.read_bytes()
    ref = json.loads(before).get("reference_hand_dir")
    if ref != HAND:
        rc = tail_main(ds, emb, reg, write=False)
        print(f"[D41_STEP] {ds}/{emb}({HAND}): verified against reference hand {ref} rc {rc}")
        return rc
    rc = tail_main(ds, emb, reg, write=True)
    if rc or reg.read_bytes() == before:
        print(
            f"[D41_STEP] {ds}/{emb}({HAND}): reference-hand overlay rc {rc} · registry "
            f"{'unchanged' if not rc else 'not written'}"
        )
        return rc
    table = json.loads(reg.read_text())["parents"]
    rs = reslice_stale(ds, emb, table, write=True, label=label)
    ap = apply_table(ds, emb, table, purge_unregistered=True)
    print(
        f"[D41_STEP] {ds}/{emb}({HAND}): reference-hand overlay written · re-sliced {rs['resliced']}/{rs['stale']} · "
        f"new pieces {ap['made']} · window mismatch {ap['bad']} · unregistered quarantined {ap['purged']}"
    )
    return 2 if ap["bad"] or rs["resliced"] != rs["stale"] else 0
