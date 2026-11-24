"""Desk height rule: placed height · midpoint · not lowered. Dependency-free single owner (thresholds, branch names,
decision, marker).

It applies at one place, the rest-estimate branch of `plane.apply_policy` (cls table · table_conflict). Only the rule
lives here; RULE_ID names it in every stamp and marker.

  ① desk height = the height the object rested at (rest estimate table_z).
  ② when the object bottom's lowest point over **all frames** is more than DESK_MID_START_M below it, lower the desk
     to between the two (DESK_MID_FRACTION, the midpoint; slightly embedded, slightly floating).
  ③ more than DESK_MID_STOP_M below: do not lower (a tracking error or a wrong placed height); datasets with a measured
     desk surface use that surface (z 0), others keep the placed height.
Clips where the desk is in place but the object passes through the top are not removed by this rule; the first-round
candidate rule (dexengine.data.candidate_rules.OBJ_BELOW_M) removes them.
The branch names and the `why` text are stored in INTAKE and in trees (desk_branch, support.why) and stay as written.
"""

from __future__ import annotations

#: Number in the rule table and decision log; printed by the [DESK_RULE] marker.
RULE_ID = "D47"
DESK_MID_START_M = 0.010
DESK_MID_STOP_M = 0.020
DESK_MID_FRACTION = 0.5
#: Branch names, used by the run-log marker [DESK_RULE], the why prefix and the INTAKE field desk_branch.
DESK_REST, DESK_MID, DESK_PLANE, DESK_KEPT = "rest height", "middle", "measured plane", "not lowered"
DESK_BRANCHES = (DESK_REST, DESK_MID, DESK_PLANE, DESK_KEPT)


def desk_height(tz: float, low, has_plane: bool) -> tuple:
    """(desk_z, branch). tz = placed height · low = lowest point of the object bottom over all frames (None = not
    measured → placed height) · has_plane = whether the dataset has a measured desk surface (z 0). Boundaries are
    'more than' (strict): a drop exactly at the threshold takes the lower branch. The drop is read rounded to 0.001 mm
    (records are rounded to 0.1 mm, so subtraction error never crosses the boundary)."""
    tz = float(tz)
    drop = None if low is None else round(tz - float(low), 6)
    if drop is None or not drop > DESK_MID_START_M:
        return tz, DESK_REST
    if not drop > DESK_MID_STOP_M:
        return round(tz - DESK_MID_FRACTION * drop, 4), DESK_MID
    if has_plane:
        return 0.0, DESK_PLANE
    return tz, DESK_KEPT


def desk_fields(tz: float, low, has_plane: bool) -> dict:
    """Record fields of the rest-estimate branch. desk_branch stays in INTAKE only (outside the fields copied into
    trees,
    base_layer.common.SUPPORT_COPIED). Branches that do not move the desk (placed height · not lowered) create no fields
    that reach the tree (desk_z · desk_z_est · why), so those trees look as if the rule did not exist."""
    dz, br = desk_height(tz, low, has_plane)
    if br in (DESK_REST, DESK_KEPT):
        return {"desk_branch": br}
    how = "lowered to the middle" if br == DESK_MID else "not lowered; measured desk plane (z 0)"
    return {
        "desk_branch": br,
        "desk_z": dz,
        "desk_z_est": tz,
        "why": f"{br}: object lowest point {low:.3f} below the rest height {tz:.3f} by {(tz - low) * 1000:.0f} mm → "
        f"{how}",
    }


def low_all_frames(v, o):
    """Lowest z (m) of the object bottom over all frames, from mesh sample vertices v (n,3) and poses o (T,7 = position
    +
    wxyz). Finite frames only, else None. Only the third row (z) of the rotation matrix is used (same normalisation as
    support_audit.qmat). numpy is imported on call only."""
    import numpy as np

    ok = np.isfinite(o).all(1)
    if not ok.any():
        return None
    q = o[ok, 3:7]
    w, x, y, z = (q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)).T
    r3 = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1)
    return float(((v @ r3.T).min(0) + o[ok, 2]).min())


def desk_rule_mark(ds: str, recs: list, has_plane: bool) -> str:
    """Run-log marker: how many clips took each branch. Counts the desk_branch apply_policy wrote into the records (no
    recomputation). 'outside the rule' = clips outside the rest-estimate branch (no placement · official 0 · plane · no
    desk · error)."""
    n = dict.fromkeys(DESK_BRANCHES, 0)
    out = 0
    for r in recs:
        b = (r.get("policy") or {}).get("desk_branch")
        if b is None:
            out += 1
        else:
            n[b] += 1
    return (
        f"[DESK_RULE] {ds} {RULE_ID} " + " · ".join(f"{b} {n[b]}" for b in DESK_BRANCHES) + f" · outside the rule {out}"
        f" (lowering starts {DESK_MID_START_M} m · limit {DESK_MID_STOP_M} m · fraction {DESK_MID_FRACTION}"
        f" · lowest point = all frames · measured surface {'yes' if has_plane else 'no'})"
    )
