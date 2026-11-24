"""The D41 aerial tail cut and the per-hand registry view (the shared D36 table plus this hand's overlays)."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from dexengine.core.errors import input_missing
from dexengine.data import rules as RR  # noqa: E402
from dexengine.data.segment.common import (  # noqa: F401
    D41_RULE,
    HAND,
    TAIL_RULE_NAME,
    _base_dir,
    _base_emb,
    _own_registry_path,
    _tc,
)
from dexengine.data.segment.rule import unit_key  # noqa: F401

#: D41 result counts (printed by the overlay run's marker [D41_TAIL]):
#: cut|none|whole|no_main|piece_missing|window_mismatch
TAIL_STATS: dict[str, int] = {}


def _has_static_artic(ds: str, emb: str, parent: str, hand_dir: str) -> bool | None:
    """Whether the generation tag of that hand's parent tree has a STATIC_ARTIC piece · None without a tree.
    The generation's owner is dexengine.data.contact.fingerprint.generation_of alone."""
    from dexengine.data.contact import fingerprint as SF  # noqa: PLC0415

    f = _base_emb(ds, emb, hand_dir) / parent / "task_info.json"
    if not f.is_file():
        return None
    return f"{SF.GEN_STATIC_KEY}=" in SF.generation_of(json.loads(f.read_text()), SF.LAYER_ALL)


def static_artic_own(ds: str, emb: str, parents, ref_hand: str) -> list[str]:
    """Owner decision for per-hand windows: the parent's window belongs to this hand only when this hand's parent tree
    has STATIC_ARTIC and the reference hand's same parent tree does not. Without a reference-hand tree, or with
    STATIC_ARTIC in both, the shared registry owns it (rebuilding the reference hand with the static rule lets the
    shared windows win again)."""
    if ref_hand == HAND:
        return []
    return sorted(
        c for c in parents if _has_static_artic(ds, emb, c, HAND) and _has_static_artic(ds, emb, c, ref_hand) is False
    )


def effective_table(ds: str, emb: str, shared: dict) -> dict:
    """Table this hand cuts = the shared registry table overlaid with the per-hand window registry. Every place that
    cuts from the registry (apply · re-slice · refresh · compare) reads this (one reading path).
    An empty per-hand entry means this hand does not cut that parent."""
    own = _own_registry_path(ds, emb)
    if not own.is_file():
        return shared
    o = json.loads(own.read_text())
    if o.get("hand_dir") != HAND:
        raise SystemExit(f"[split] ✗ per-hand window registry {own.name}: hand {o.get('hand_dir')!r} ≠ {HAND}")
    t = dict(shared)
    for c, us in o["parents"].items():
        if us:
            t[c] = us
        else:
            t.pop(c, None)
    return t


def _parents(ds: str, emb: str, hand_dir: str) -> list[str]:
    root = _base_emb(ds, emb, hand_dir)
    if not root.is_dir():
        return []
    return sorted(
        c for c in os.listdir(root) if not re.fullmatch(r".+_t\d+", c) and (root / c / "task_info.json").is_file()
    )


def tail_cut(d: Path, ti: dict, clip: str, units: list[dict], T: int, prev: dict) -> tuple[list[dict], str]:
    """D41 aerial tail: in the **last candidate window** (the last piece, or the whole clip without pieces) of a parent
    in an aerial scene (dexengine.data.candidate_rules.scene_of), cut off the end run in which the main object has
    neither contact nor support until the end; in the air nothing holds the object up, so the reference cannot be
    followed. The decision belongs to coords.aerial_tail_start (the same function as the detector and the bake stamp).
    Measured on: the parent without pieces, else the base of the last piece (the window's main object and sidecar are
    there). Pieces already cut (window ≠ D36 window) are kept only when the previous registry's (`prev`) D41 record
    comes
    from the same D36 window (idempotent redefinition); otherwise window_mismatch (--define does not write when that
    count is nonzero).
    A parent without pieces becomes a one-window piece `<parent>_t0` (the parent drops out of candidates via D36
    split_parent).
    The 1-second threshold (min_seconds) is not applied here; shortened windows are dropped with a reason by D12-d
    (never
    silently left as the parent).
    Returns (pieces, result cut|none|whole|no_main|piece_missing|window_mismatch); whole = the whole window has no
    support (a tail cut cannot fix it; left as is)."""
    from dexengine.data.support import gap as SG  # noqa: PLC0415
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — owner of support and contact (placement layer)

    src, sti, off, n = d, ti, 0, T
    last = max(units, key=lambda u: u["t1"]) if units else None
    if last is not None:
        src = d.parent / last["name"]
        if not (src / "task_info.json").is_file():
            return units, "piece_missing"  # piece not built yet; measured when redefined after --apply
        sti = json.loads((src / "task_info.json").read_text())
        win = [last["t0"], last["t1"]]
        if sti.get("segment_range") != win:
            old = next((u for u in prev.get(clip, []) if u["name"] == last["name"]), None)
            tc = (old or {}).get("tail_cut") or {}
            if old and tc.get("t1_before") == last["t1"] and sti.get("segment_range") == [old["t0"], old["t1"]]:
                return [dict(u, t1=old["t1"], tail_cut=tc) if u is last else u for u in units], "cut"
            print(
                input_missing(
                    f"[D41] ✗ {src.name}: piece window {sti.get('segment_range')} ≠ D36 window {win}, and "
                    f"it does not match the previous registry's D41 record({tc or 'none'}) either — cannot "
                    "measure (definition refused; fix the pieces first)"
                ),
                file=sys.stderr,
            )
            return units, "window_mismatch"
        off, n = last["t0"], last["t1"] - last["t0"]
    mains = [o for o in sti.get("objects") or [] if o.get("role") == "main" and o.get("qpos_start") is not None]
    if not mains:
        return units, "no_main"  # no main object; first-round candidates drop it separately (nothing to support)
    q = np.load(src / "0/trajectory_kinematic.npz")["qpos"]
    k = C.aerial_tail_start(
        src, sti, q, SG.main_names(sti), 0, n
    )  # objects measured = the same main-object set as the dexengine.data.support.gap gaps
    if k is None:
        return units, "none"
    if k <= 0:
        return units, "whole"
    cut = {"rule": "D41", "t1_before": off + n}
    if last is not None:
        return [dict(u, t1=off + k, tail_cut=cut) if u is last else u for u in units], "cut"
    side = {o.get("side"): str(o["name"]) for o in mains}
    return [
        {
            "name": f"{clip}_t0",
            "t0": 0,
            "t1": k,
            "left": side.get("left"),
            "right": side.get("right"),
            "tail_cut": dict(cut, created=True),
        }
    ], "cut"


def strip_tail(units: list[dict]) -> list[dict]:
    """Strip the D41 part from registry pieces, leaving only the D36 statement: shortened windows go back to their
    original end and one-window pieces D41 made are removed."""
    out = []
    for u in units:
        tc = u.get("tail_cut") or {}
        if tc.get("created"):
            continue
        out.append({k: v for k, v in u.items() if k != "tail_cut"} | ({"t1": tc["t1_before"]} if tc else {}))
    return out


def tail_overlay(ds: str, emb: str, prev: dict) -> dict:
    """D41: a new table keeping the registry's D36 statement (`prev` without its D41 part) **as is** and adding tail
    cuts
    (tail_cut) only to aerial-path parents in this hand's base. D36 is not recomputed from the timeline (D36
    redefinition is separate; if this overlay changed D36 windows, pieces outside the D41 decision would be rejudged).
    Rows of parents this hand lacks stay as they are."""
    TAIL_STATS.clear()
    out = {c: us for c, us in ((c, strip_tail(us)) for c, us in prev.items()) if us}
    for clip in _parents(ds, emb, HAND):  # pieces are not parents; measured on the parent's last window
        d = _base_dir(ds, emb, clip)
        ti = json.loads((d / "task_info.json").read_text())
        if _tc().scene_of(ds, ti) != "aerial":
            continue
        T = int((ti.get("case") or {}).get("T") or 0) or int(np.load(d / "0/trajectory_kinematic.npz")["qpos"].shape[0])
        units, how = tail_cut(d, ti, clip, out.get(clip, []), T, prev)
        TAIL_STATS[how] = TAIL_STATS.get(how, 0) + 1
        if units:
            out[clip] = units
    return out


def tail_diff(ds: str, emb: str, parents: dict) -> tuple[dict, list[str], int]:
    """D41 comparison (re-measured overlay table, this hand's parents differing from the registry: before the overlay,
    the parents that should be cut, count of this hand's parents)."""
    table, here, k = tail_overlay(ds, emb, parents), set(_parents(ds, emb, HAND)), unit_key
    diff = sorted(c for c in set(parents) | set(table) if c in here and k(parents.get(c, [])) != k(table.get(c, [])))
    return table, diff, len(here)


def d36_diff(ds: str, emb: str, ref: dict, table: dict) -> dict:
    """D36 comparison: registry statement (without the D41 part) ↔ this hand's timeline table (compute_units). Parents
    this hand lacks are no mismatch."""
    ref = {c: us for c, us in ((c, strip_tail(us)) for c, us in ref.items()) if us}
    only_ref, here = sorted(set(ref) - set(table)), set(_parents(ds, emb, HAND))
    common, present = sorted(set(ref) & set(table)), [c for c in only_ref if c in here]
    return {
        "ref": ref,
        "common": common,
        "diff": [c for c in common if unit_key(ref[c]) != unit_key(table[c])],
        "only_ref_present": present,
        "only_this": sorted(set(table) - set(ref)),
        "absent": len(only_ref) - len(present),
    }


def tail_main(ds: str, emb: str, reg: Path, write: bool) -> int:
    """--define-tail / --verify-tail for one dataset and emb. Marker [D41_TAIL].
    rc 0 · 2 (cannot measure · not the reference hand · comparison differs)."""
    if not reg.is_file():
        print(input_missing(f"[split] ✗ {ds}/{emb}: no registry {reg} — run D36 --define first"), file=sys.stderr)
        return 2
    r0 = json.loads(reg.read_text())
    table, diff, n_here = tail_diff(ds, emb, r0["parents"])
    ts = TAIL_STATS
    n_cut = sum(1 for us in table.values() for u in us if "tail_cut" in u)
    print(
        f"[D41_TAIL] {ds}/{emb}({HAND}): aerial-path parents {sum(ts.values())} · cut {ts.get('cut', 0)} · "
        f"no tail {ts.get('none', 0)} · whole window unsupported {ts.get('whole', 0)} · "
        f"no main object {ts.get('no_main', 0)} · piece missing {ts.get('piece_missing', 0)} · "
        f"window mismatch {ts.get('window_mismatch', 0)} · windows cut {n_cut}",
        flush=True,
    )
    if ts.get("window_mismatch") or ts.get("piece_missing"):
        print(
            f"[split] ✗ {ds}/{emb}: parents where D41 cannot be measured — "
            f"{'not writing' if write else 'cannot compare'}"
            f" (differs from the registry; fix the pieces first: --reslice-stale)",
            file=sys.stderr,
        )
        return 2
    if write:
        if r0.get("reference_hand_dir") != HAND:
            print(
                f"[split] ✗ {ds}/{emb}: the registry's reference hand is {r0.get('reference_hand_dir')} — "
                f"hand {HAND} does not write (compare with --verify-tail)",
                file=sys.stderr,
            )
            return 2
        from dexengine.data import tree_io as _TIO  # noqa: PLC0415

        wrote = _TIO.write_json_if_changed(reg, dict(r0, parents=table, tail_rule=TAIL_RULE_NAME))
        print(
            f"[split] {ds}/{emb} D41 overlay → {reg.name}({'written' if wrote else 'same content — not rewritten'}): "
            f"parents {len(table)} · pieces {sum(len(v) for v in table.values())} · windows cut {n_cut}"
        )
        return 0
    print(
        f"[split] {ds}/{emb} D41 verify ({HAND} vs registry): this hand's parents {n_here} · differing {len(diff)} → "
        f"{'same' if not diff else '✗ differs'}"
    )
    for c in diff[:5]:
        print(f"    {c}: registry {unit_key(r0['parents'].get(c, []))} vs this hand {unit_key(table.get(c, []))}")
    return 2 if diff else 0


def d41_in_scope(ds: str) -> bool:
    """Whether D41 applies to the dataset; the scope belongs to the rule table (data.rules.rule_datasets, empty
    tuple = all)."""
    scope = RR.rule_datasets(D41_RULE)
    return not scope or ds in scope


def d41_pending(ds: str, emb: str, reg_obj: dict) -> bool:
    """The registry has no D41 overlay yet while this hand's base has aerial-path parents
    (dexengine.data.candidate_rules.scene_of): pieces made from that registry would be candidates with untrimmed tails
    (the overlay is the reference hand's stage1.split_apply tail step)."""
    if not d41_in_scope(ds) or reg_obj.get("tail_rule") == TAIL_RULE_NAME:
        return False
    for clip in _parents(ds, emb, HAND):
        ti = json.loads((_base_dir(ds, emb, clip) / "task_info.json").read_text())
        if _tc().scene_of(ds, ti) == "aerial":
            return True
    return False
