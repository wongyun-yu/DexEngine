"""Carried cells of the per-chunk status ledger: the cross-repair residue (read from each adopted tree's xr report),
the per-side gate depth comparison, and their summary. ledger.py folds the states and writes the rows.
"""

import collections
import json
import os

from dexengine.core.errors import InputMissingError
from dexengine.placement.ledger_states import GATE_DEPTH_LEDGER, GATE_DEPTH_SOURCE

#: Cross-repair residue cells (read by the ledger's --xr-summary).
XR_COLUMNS = ("xr_on_pen_mm", "xr_on_pen_f0")


#: Cells adopted rows carry after the fingerprint — cross-repair residue, then the per-side gate comparison
#: (COLUMNS here and status_fold.LEDGER_COLS carry them in this order).
CARRY_COLUMNS = XR_COLUMNS + GATE_DEPTH_LEDGER


#: The cross-repair report inside a tree (written by cross_repair).
XR_REPORT = "0/xr_report.json"


def xr_on_values(tree, sets: tuple) -> tuple[str, str]:
    """The two cross-repair cells of one adopted clip = max max_depth_mm over the tree report's after_hand[sets on] ·
    whether any of them has pen_at_frame0. A missing report or a set outside the report's sets (params.sets) dies.
    No set on → '-'."""
    p = os.path.join(str(tree), XR_REPORT)
    try:
        with open(p, encoding="utf-8") as fh:
            rep = json.load(fh)
    except (OSError, ValueError) as e:
        raise SystemExit(
            f"[status] ✗ XR on but the cross-repair report of an adopted clip is unreadable: {p} ({type(e).__name__})"
        ) from None
    ah, said = rep.get("after_hand"), (rep.get("params") or {}).get("sets")
    if not isinstance(ah, dict) or not isinstance(said, list) or not set(sets) <= set(said) & set(ah):
        raise InputMissingError(
            f"[status] ✗ the cross-repair report's sets {said} · after_hand {sorted(ah or {})} "
            f"lack the sets on {list(sets)} — collision_presets.XR_SET_ROW and "
            f"cross_repair.SET_REPAIR disagree: {p}"
        )
    if not sets:
        return "-", "-"
    return (
        f"{max(float(ah[s]['max_depth_mm']) for s in sets):.2f}",
        str(int(any(bool(ah[s]["pen_at_frame0"]) for s in sets))),
    )


def xr_adopted_cells(new: list, xr: str, tree_root: str, preset: str, sets_on, tree_of) -> dict:
    """Cells {clip: (depth, first frame)} of this run's adopted rows `new` — xr = the --xr statement ('on' · 'off' ·
    '' = none). sets_on(preset) = the sets-on rule (collision_presets.xr_sets_on) · tree_of(clip) = tree folder name
    (paths.tree_name). Dies when there are adopted rows but no statement, or 'on' without a tree root and preset, or
    the preset gives no sets."""
    if new and not xr:
        raise InputMissingError(
            f"[status] ✗ {len(new)} adopted rows in this run but no --xr on|off — the cross-repair cells cannot be set"
        )
    if not new:
        return {}
    if xr == "on":
        if not (tree_root and preset):
            raise InputMissingError("[status] ✗ --xr on needs --tree-root and --coll-preset (no default)")
        try:
            sets = sets_on(preset)
        except (KeyError, ValueError) as e:
            raise InputMissingError(
                f"[status] ✗ cannot resolve the cross-repair sets on from training preset {preset!r}: {e}"
            ) from e
        out = {c: xr_on_values(os.path.join(tree_root, tree_of(c)), sets) for c in new}
        print(
            f"[XR_ON_PEN] xr=on · training preset {preset} → hand sets on {list(sets)} · {len(new)} adopted rows"
            " (rule collision_presets.xr_sets_on · report after_hand)"
        )
        return out
    print(f"[XR_ON_PEN] xr=off · {len(new)} adopted rows '-'")
    return dict.fromkeys(new, ("-", "-"))


def carry_cells(row) -> tuple[str, ...]:
    """The carried cells of one ledger row (CARRY_COLUMNS order · missing cell = empty) — the fold carries them."""
    return tuple(row.get(k) or "" for k in CARRY_COLUMNS)


def gate_depth_cells(adopted, fresh, gate_sides, old) -> dict:
    """Per-side gate comparison cells of the adopted rows (GATE_DEPTH_LEDGER order): with a gate table
    (gate_sides = {clip: {side: {cell: value}}}) that side's cell text as is (source GATE_DEPTH_SOURCE; main()
    enforced both side rows) · without one, '-' for this run's adopted rows · rows only from --from keep their old
    cells (old). Marker [GATE_DEPTH] = where the cells came from."""
    out, n = {}, collections.Counter()
    for c in adopted:
        if gate_sides is not None:
            out[c], k = tuple(gate_sides[c][s][g] for g, s in GATE_DEPTH_SOURCE), "gate table"
        elif c in fresh:
            out[c], k = ("-",) * len(GATE_DEPTH_SOURCE), "no gate '-'"
        else:
            out[c] = old.get(c) or ("",) * len(GATE_DEPTH_SOURCE)
            k = "old ledger" if any(out[c]) else "old ledger (no cells)"
        n[k] += 1
    print(
        f"[GATE_DEPTH] per-side gate comparison cells (hand ↔ own object · hand ↔ support, first frame) on "
        f"{len(adopted)} adopted rows — "
        + " · ".join(f"{k} {v}" for k, v in sorted(n.items()))
        + " (recorded only · not filtered)"
    )
    return out


def xr_summary(rows) -> str:
    """One summary line of the cross-repair cells of ledger rows (the --xr-summary CLI · cells XR_COLUMNS)."""
    ad = [r for r in rows if r["status"] == "ADOPTED"]
    v = [r.get(XR_COLUMNS[0]) for r in ad]
    mm = [float(x) for x in v if x not in (None, "", "-")]
    return (
        f"[XR_ON_PEN] cross-repair residue (hand sets on in training) adopted {len(ad)} · measured {len(mm)} · "
        f"off '-' {v.count('-')} · no record {sum(1 for x in v if x in (None, ''))} · residue > 0 "
        f"{sum(1 for x in mm if x > 0)} · first frame {sum(1 for r in ad if r.get(XR_COLUMNS[1]) == '1')} · "
        f"max {max(mm, default=0.0):.2f} mm (recorded only · not filtered)"
    )
