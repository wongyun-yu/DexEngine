"""Rest interval definition, window slicing and object identity; dependency-free single owner.

Why separate: rest intervals are built by `support_audit` (INTAKE, MANO layer), sliced into windows by the base adapter
(`_segNN` pieces) and the D36 split (`_tN` pieces), and read by `coords` (D20 aerial disc · D26′ support disc). All
three must share the same thresholds, slicing and object identity: copying without slicing would leave a piece's
intervals in parent frame numbers, and resolving an interval's object as "the hand's main object" (after D23 re-roling)
would put the disc under a different object than the one measured. No numpy, so any layer (base adapter, split,
coords, auditor) reads it at no cost.
"""

from __future__ import annotations

#: Rest-interval thresholds, applied by support_audit. This is the one owner of the values.
#:   V_REST      smoothed (0.3 s) object origin speed threshold (m/s)
#:   Z_STAB      threshold on the object mesh bottom z range over placed samples (m).
#:               coords uses the same value for "does the declared bottom exist in this tree".
#:   REST_MIN_S  minimum rest/placement duration (s)
#:   TIP_FREE    placed = fingertip–object surface min distance at least this (m)
V_REST = 0.03
Z_STAB = 0.01
REST_MIN_S = 0.5
TIP_FREE = 0.025

#: An interval's `side` is the **object slot** (`<side>_object`) at INTAKE (MANO) time, not the hand's main object:
#: the object support_audit measured with `ti[f"{side}_object_mesh_dir"]` · `qpos_obj_{side}`, named `f"{side}_object"`
#: in base trees (base_layer.geom.build_objects). D23 re-roling changes role/side only, never names.
REST_OBJECT_FMT = "{side}_object"


def slice_rest_intervals(rest_intervals, a: int, b: int, off: int = 0) -> list:
    """Slice the rest-interval list to the window [a, b): intervals outside the window are dropped and boundaries
    clipped (returned in window frames).

    `off` is the MANO→kin offset (base adapter `mano_to_kin_offset`, measured 0). The D36 split is already in base
    frames, so off=0. `_segNN` (base_layer.stale.support_for) and `_tN` (segment.split) call this one function. side and
    bottom_z are carried over as is: bottom_z is the median bottom over placed samples, so a sliced window keeps the
    same object and bottom."""
    out = []
    for r in rest_intervals or []:
        if not (isinstance(r, dict) and "start" in r and "end" in r):
            continue
        s0 = max(int(r["start"]) - off - a, 0)
        e0 = min(int(r["end"]) - off - a, b - a)
        if e0 > s0:
            out.append({**r, "start": s0, "end": e0})
    return out


def rest_object_name(iv: dict, ti: dict) -> str | None:
    """Rest interval → name of the object it measured (name in base objects[]). None when unresolved; the caller counts
    and skips it (no fallback).

    Rule: name = `<side>_object`. When it is not in objects[], check whether the base adapter's duplicate merge
    (`merged_duplicate`) removed it, and if so move to the kept name with the same index (as in grab, where one object
    arrives in two slots and becomes one body after the merge). Otherwise (no object) None."""
    side = iv.get("side")
    if side not in ("right", "left"):
        return None
    name = REST_OBJECT_FMT.format(side=side)
    names = {o.get("name") for o in (ti.get("objects") or [])}
    if name in names:
        return name
    md = ti.get("merged_duplicate") or {}
    removed, kept = list(md.get("removed") or []), list(md.get("kept") or [])
    if name in removed:
        k = removed.index(name)
        if k < len(kept) and kept[k] in names:
            return kept[k]
    return None
