"""The D36 cutting rule: where a clip is cut into pieces (MANO contact timeline and sustained grasp switches)."""

from __future__ import annotations

from dexengine.data.cases import (
    assign_version as AR,  # noqa: E402 — single entry for reading assignment records (main object = per_side unit)
)
from dexengine.data.segment import (
    grasp_runs as GR,  # noqa: E402 — owner of grasp runs and the sustained-grasp rule (D36)
)
from dexengine.data.segment.common import HAND, _tc  # noqa: F401


def grasp_switch_cuts(case: dict, used: list) -> list[tuple[int, int]]:
    """Places where a hand switches the object it holds: [(end of previous grasp, start of next grasp)] (frames). Short
    grasp switches the timeline (case.timeline, the main object per hand) misses are cut by the same criterion ('a hand
    switches to another object'). Only grasps of objects handled and moved (case.used, owned by cases.taxonomy) count
    (touching an object that is not moved is no switch). When the two grasps overlap (the same hand holds both objects)
    there is no place to cut, so no cut. `used` = the targets after the moved-only rule (cut_rule: non-main objects only
    with remaining moved runs).
    Only sustained grasps count (assists and brushes are no switch): ① a grasp lying wholly inside the same hand's grasp
    of another object = holding together (assist), dropped ② grasps shorter than the piece minimum
    (`dexengine.data.candidates.MIN_DURATION_S`) = brushes that cannot get their own piece, dropped. Consecutive grasps
    of the same
    object left after that are merged into one."""
    fps = float(case.get("fps") or 30.0)
    if not any(iv for per in (case.get("grasp_s") or {}).values() for iv in (per or {}).values()):
        return []  # no grasp record, no switch (the same place as the timeline rule's 'one interval')
    if "used" not in case:
        raise ValueError(
            "no case.used stamp — grasp switch cuts look only at objects handled and moved (cases.taxonomy used)"
        )
    used = set(used)
    cuts = set()
    for side in ("left", "right"):
        runs = GR.sustained_runs([r for r in GR.grasp_runs(case, side, fps) if r[2] in used], fps, _tc().MIN_DURATION_S)
        for (_a1, b1, o1), (a2, _b2, o2) in zip(runs, runs[1:], strict=False):
            if o1 != o2 and b1 < a2:
                cuts.add((b1, a2))
    return sorted(cuts)


def _label(case: dict, t0: int, t1: int) -> dict:
    """Object each hand held longest in the piece window (None without one); the piece's left/right label."""
    fps = float(case.get("fps") or 30.0)
    out = {}
    for side in ("left", "right"):
        best = max(
            ((min(b, t1) - max(a, t0), o) for a, b, o in GR.grasp_runs(case, side, fps) if min(b, t1) > max(a, t0)),
            default=(0, None),
        )
        out[side] = best[1]
    return out


#: Moved-only rule: a place where a hand's 'handled object' (case.timeline) switches to a **non-main object** is a
#: boundary only when that object really moved in that interval. Motion = case.move_runs (hand_object_assign.move_runs:
#: carried ∧ body motion ∨ joint drive; no new thresholds). Main objects = per-hand units of the assignment record
#: (dexengine.data.cases.assign_version.record). Without an overlapping motion run the interval takes the value of that
#: hand's nearest remaining interval in time and equal neighbours merge. Such objects are also dropped from grasp switch
#: cuts (only main objects
#: ∪
#: objects with remaining moved runs). Switches to main objects and other rules are unchanged. Counts = [CUT_RULE]
#: marker (compute_units).
CUT_STATS: dict[str, int] = {}


def _cut_inputs(ti: dict) -> tuple[set, dict | None]:
    """Inputs of the moved-only rule: (set of main-object units, case.move_runs or None).
    dexengine.data.cases.assign_version.record fails without an assignment record or with another version. A missing
    motion layer fails in cut_rule (only when non-main object intervals exist)."""
    per = AR.record(ti, HAND).get("per_side") or {}
    return {(v or {}).get("unit") for v in per.values()} - {None}, (ti.get("case") or {}).get("move_runs")


def _moved(move: dict, side: str, unit: str, t0: int, t1: int) -> bool:
    """Whether any motion run of that hand and unit overlaps [t0, t1)."""
    return any(a < t1 and t0 < b for a, b in ((move.get(unit) or {}).get(side) or []))


def _fill(tl: list, side: str, keep) -> list:
    """Replace the hand's timeline values with keep(interval index, value)=False by the nearest remaining interval value
    in time (None without one)."""
    vals = [s.get(side) for s in tl]
    ok = [i for i, v in enumerate(vals) if v is not None and keep(i, v)]
    out = []
    for i, v in enumerate(vals):
        if v is None or keep(i, v):
            out.append(v)
        elif not ok:
            out.append(None)
        else:
            j = min(ok, key=lambda j: (tl[i]["t0"] - tl[j]["t1"]) if j < i else (tl[j]["t0"] - tl[i]["t1"]))
            out.append(vals[j])
    return out


def cut_rule(ti: dict) -> tuple[list[dict], list]:
    """(timeline, grasp switch targets used) after the moved-only rule. Without non-main object intervals in the
    timeline, the case as is."""
    case = ti.get("case") or {}
    tl = case.get("timeline") or []
    used = list(case.get("used") or [])
    hands = [h for h in ("left", "right") if any(h in s for s in tl)]
    if not any(s.get(h) for s in tl for h in hands) or (len(tl) < 2 and len(set(used)) < 2):
        return tl, used  # nothing to cut (no boundaries, no grasp switches); inputs are not read
    mains, move = _cut_inputs(ti)
    nonmain = [(i, h, s[h]) for i, s in enumerate(tl) for h in hands if s.get(h) and s[h] not in mains]
    if not nonmain:
        return tl, used
    if move is None:
        raise ValueError(
            "no case.move_runs stamp — the moved-only piece rule (D36 v6) measures with this layer whether a "
            "non-main object moved · rerun stage 1 step 3 (cases/taxonomy.py --write)"
        )
    keep = {(i, h): _moved(move, h, u, int(tl[i]["t0"]), int(tl[i]["t1"])) for i, h, u in nonmain}
    newv = {h: _fill(tl, h, lambda i, v, h=h: v in mains or keep.get((i, h), False)) for h in hands}
    ntl: list[dict] = []
    for i, s in enumerate(tl):
        seg = {"t0": s["t0"], "t1": s["t1"], **{h: newv[h][i] for h in hands}}
        if ntl and all(ntl[-1].get(h) == seg.get(h) for h in hands):
            ntl[-1]["t1"] = seg["t1"]
        else:
            ntl.append(seg)
    moved_units = {u for i, h, u in nonmain if keep[(i, h)]}
    out_used = [u for u in used if u in mains or u in moved_units]
    for k, v in (
        ("parents", 1),
        ("nonmain_segs", len(nonmain)),
        ("moved_kept", sum(keep.values())),
        ("filled", sum(not x for x in keep.values())),
        ("used_dropped", len(used) - len(out_used)),
    ):
        CUT_STATS[k] = CUT_STATS.get(k, 0) + v
    return ntl, out_used


def segments_of(ti: dict) -> list[dict]:
    """Intervals to split (timeline boundaries `_timeline_segments`, then further cut at grasp switches
    `grasp_switch_cuts`).
    Empty when there is nothing to split. When a grasp switch (end e of previous grasp, start s of next) lies inside a
    piece, that piece becomes [t0, s] · [e, t1] (the gap is shared by both, the same rule as timeline boundaries).
    The left/right of split pieces = the object each hand held longest in that window.
    Piece numbers (`i`, name `_t<i>`) never change: timeline pieces keep their number, the first part of a split piece
    keeps the original number and later parts take numbers after the timeline piece count in turn (so names of pieces
    already built never point at another window)."""
    case = ti.get("case") or {}
    tl, used = cut_rule(ti)
    units = [dict(u, i=i) for i, u in enumerate(_timeline_segments(ti, tl))]
    cuts = grasp_switch_cuts(case, used)
    T = int(case.get("T") or 0)
    if not cuts or (not units and T <= 0):
        return units
    units = units or [{"t0": 0, "t1": T - 1, "left": None, "right": None, "i": 0}]
    free = len(units)
    for e, s in cuts:
        nxt = []
        for u in units:
            if u["t0"] < e and s < u["t1"]:
                nxt += [dict(u, t1=s, **_label(case, u["t0"], s)), dict(u, t0=e, i=free, **_label(case, e, u["t1"]))]
                free += 1
            else:
                nxt.append(u)
        units = nxt
    return units if len(units) > 1 else []


def _timeline_segments(ti: dict, tl: list) -> list[dict]:
    """Intervals at timeline boundaries (the original D36 rule). Empty when there is nothing to split.

    **Split criterion**: a hand switching to *another object*. Passing the same object (handover) or alternating
    shared/solo (BI_SHARED_MIXED) is **not** split: both hands get that object's reward and the idle hand is held at
    zero contact, so it trains as one episode (the same handling as case 3). The decision therefore looks at whether
    **the set of objects an interval uses** changes, not at the (left, right) pair.

    **Boundaries**: a *gap nobody holds* between intervals is **shared by both sides**; the earlier interval extends to
    the next grasp start and the later one starts at the previous grasp end, so motion is never cut off at the cut.
    `tl` = the timeline after the moved-only rule (cut_rule).
    """
    case = ti.get("case") or {}
    if len(tl) < 2:
        return []
    used = [frozenset(x for x in (s.get("left"), s.get("right")) if x) for s in tl]
    # neighbouring intervals with the same object set merge (a change of sharing pattern alone is no boundary)
    merged: list[dict] = []
    for s, u in zip(tl, used, strict=True):
        if merged and merged[-1]["objs"] == u:
            merged[-1]["t1"] = int(s["t1"])
            continue
        merged.append(
            {"t0": int(s["t0"]), "t1": int(s["t1"]), "objs": u, "left": s.get("left"), "right": s.get("right")}
        )
    if len(merged) < 2:
        return []
    fps = float(case.get("fps") or 30.0)
    grasp = case.get("grasp_s") or {}
    spans: list[tuple[int, int]] = [
        (int(round(float(iv[0]) * fps)), int(round(float(iv[1]) * fps)))
        for per in grasp.values()
        for side in ("left", "right")
        for iv in per.get(side) or []
    ]
    spans.sort()
    for i, m in enumerate(merged):
        if i + 1 < len(merged):  # backwards: extend until the next grasp starts
            nxt = [a for a, _b in spans if a >= m["t1"]]
            m["t1"] = min(int(merged[i + 1]["t1"]), nxt[0] if nxt else m["t1"])
        if i > 0:  # forwards: pull back to where the previous grasp ended
            prv = [b for _a, b in spans if b <= m["t0"]]
            m["t0"] = max(int(merged[i - 1]["t0"]), prv[-1] if prv else m["t0"])
    return [{"t0": m["t0"], "t1": m["t1"], "left": m["left"], "right": m["right"]} for m in merged]


def drop_contained(units: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop pieces lying **wholly inside another piece** of the same parent: (kept, dropped). Hand-independent (windows
    only).

    Pieces overlapping each other is **intended** (the gap between grasps is shared by both sides). But in an A→B→A
    pattern, when the middle piece after extension lies wholly inside the last one (e.g. taco
    `cut_knife_plate__20231020_233` t1 67–138 ⊂ t2 67–256), it is a pure duplicate of the same motion: judged and baked
    twice and entering the training set twice.
    Rule: u lies in v ⇔ v.t0 ≤ u.t0 ∧ u.t1 ≤ v.t1 (u ≠ v). Of two pieces with the **same** window the lower number
    stays.
    Containment is transitive, so every dropped piece lies in some kept piece. Piece names (`_t<i>`) are interval
    numbers and never change; kept pieces keep their names and windows (no mismatch with trees already built).
    """
    keep, drop = [], []
    for i, u in enumerate(units):
        inside = any(
            j != i and v["t0"] <= u["t0"] and u["t1"] <= v["t1"] and ((v["t0"], v["t1"]) != (u["t0"], u["t1"]) or j < i)
            for j, v in enumerate(units)
        )
        (drop if inside else keep).append(u)
    return keep, drop


def unit_key(us: list[dict]) -> list[tuple]:
    return [(u["name"], u["t0"], u["t1"]) for u in us]  # registry comparison key, shared by D36 · D41
