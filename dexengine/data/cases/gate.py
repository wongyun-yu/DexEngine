"""'Not a handover' gate of handover cases: the rule behind the case names BI_HANDOVER(_PLUS).

A unit that share_kind (hand_object_assign) calls 'handover' (staggered holds · overlap ≤ TH_HANDOVER_MAX_OV_S · the
receiving hand releases later) is still not a handover in the case name when nothing passes from hand to hand; it is
then counted as share kind mixed. Any one of four conditions rules it out:
  ① start gap of the two holds ≤ TH_START_GAP_S                 held together, one hand releases first
  ② the giving hand holds until TH_GIVER_END_S before the end    the other hand joins at the end
  ③ the unit is an articulated part (articulation · articulation_group)   two hands split frame and drawer, bottle and
      lid
  ④ zero overlap of the two holds                                one hand grabs after the other let go
The assignment (rules.ASSIGN · share_kind) is unchanged; only the case name goes through this gate.
Used by cases.taxonomy (when a case is decided; record field handover_gate, marker [HANDOVER_GATE]) and by readers that
re-evaluate stored case records under the current rule (regate). No dependencies. Tests: tests/data/test_case_gate.py.
"""

from __future__ import annotations

MARK = "[HANDOVER_GATE]"
TH_START_GAP_S = 0.2  # ① start gap of the two holds at most this
TH_GIVER_END_S = 0.25  # ② last release of the giving hand within this of the clip end
#: Handover case name → name when the gate fires (cases.taxonomy decision table: one mixed unit → BI_SHARED_MIXED ·
#: two units with a non-handover among them → BI_SHARED_PLUS).
RENAME = {"BI_HANDOVER": "BI_SHARED_MIXED", "BI_HANDOVER_PLUS": "BI_SHARED_PLUS"}


def _overlap(a: list, b: list) -> float:
    return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def unit_gate(runs: dict, t_end_s: float, articulated: bool) -> list[str]:
    """One unit: runs = {'right': [[start s, end s], …], 'left': …} (timeline grasps, case.grasp_s form).
    Returns the reasons it is not a handover (empty = handover). When either hand has no interval it cannot be measured
    and the single 'no interval' reason comes back (kept as a handover; the caller prints the marker)."""
    r, lf = runs.get("right") or [], runs.get("left") or []
    if not (r and lf):
        return ["no interval to measure"]
    giver = r if r[0][0] < lf[0][0] else lf  # on equal start times, the left hand (same rule as share_kind)
    why = []
    if abs(r[0][0] - lf[0][0]) <= TH_START_GAP_S:
        why.append(f"① two-hand grasp start gap ≤ {TH_START_GAP_S} s")
    if giver[-1][1] >= t_end_s - TH_GIVER_END_S:
        why.append(f"② giver hand does not release until {TH_GIVER_END_S} s before the end")
    if articulated:
        why.append("③ articulated part")
    if _overlap(r, lf) <= 0:
        why.append("④ zero overlap")
    return why


def art_units(members: dict, objects: list) -> set:
    """Articulated units: units with at least one member object carrying articulation or articulation_group
    (members = hand_object_assign.units_of)."""
    art = {str(o["name"]) for o in objects if o.get("articulation") or o.get("articulation_group")}
    return {u for u, ns in members.items() if art & set(ns)}


def gate(units: list, grasp_s: dict, t_end_s: float, art: set) -> dict:
    """Handover units → {unit: reasons} (units with reasons only). The 'no interval' reason is not 'not a handover'
    (is_out)."""
    out = {}
    for u in units:
        why = unit_gate(grasp_s.get(u) or {}, t_end_s, u in art)
        if why:
            out[u] = why
    return out


def is_out(why: list) -> bool:
    """Whether a reason list means 'not a handover'; a lone 'no interval' reason does not (handover kept, marker
    only)."""
    return bool(why) and why != ["no interval to measure"]


def regate(case: dict, members: dict, objects: list) -> tuple[str, dict]:
    """Stored case record (task_info.case) → (case name under the current rule, {unit: reasons}). A non-handover case
    returns its name as is and {}. Handover units = the used units whose share kind (case.share) is handover; when
    there are none (the assignment's shared unit produced the handover), all used units (a unit with one hand's
    intervals only stays a handover through the 'no interval' reason)."""
    name = str(case.get("case") or "")
    base, plus, tail = name.partition("+")
    if base not in RENAME:
        return name, {}
    share, used = case.get("share") or {}, list(case.get("used") or [])
    units = [u for u in used if (share.get(u) or {}).get("kind") == "handover"] or used
    g = gate(units, case.get("grasp_s") or {}, float(case["T"]) / float(case["fps"]), art_units(members, objects))
    out = {u: w for u, w in g.items() if is_out(w)}
    return (RENAME[base] + plus + tail if out else name), g
