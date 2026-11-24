"""ASSIGN-2: hand–object assignment rule R and its validity gate.

The version (dexengine.data.cases.assign_version.ASSIGN2_VERSION) covers this module, the frame-input module
(assign_r_frames) and the input reader (assign_r_inputs) together; bump it when any of them changes.
The ASSIGN-1 measurement (hand_object_assign) is left untouched; this module takes its result, compares it with R and
chooses. Pure functions only: assign_r_inputs reads the source statements and builds the R inputs.

Rule R (spec rule_v3_1, FULL version of the reference implementation dryrun_v3_1; function names kept, plus the TACO
and rider clauses):
  hand points   5 fingertips + a palm proxy (wrist + TH_PALM_ALPHA·(mean fingertip − wrist)).
                Material points = the hand points in the object (part) local frame (K1).
  frame layer   prox = fingertips ≥ 2 ∨ palm ∨ (small object ∧ 1 fingertip) · touch = 1 fingertip ∨ palm ·
                over a window ±TH_WIN_S/2: object net displacement Dobj · slip Drel · hand net displacement Dtip ·
                with = touch ∧ moving (Dobj ≥ TH_MOVE_MPS·window) ∧ Drel < TH_COMOVE·Dobj (pushing included) ·
                weak = prox ∧ moving ∧ ¬with ∧ Dtip ≥ TH_HAND_K·Dobj
                       ∧ (no 'with' of the other hand ∨ own 'with' ≥ TH_RESCUE_S) ·
                lone = prox ∧ moving ∧ other hand farther than 5 cm ∧ not a tracking jump · fix = prox ∧ ¬moving ·
                anything else = passing (dropped). There are no rotation or surface-point (witness) clauses; those were
                the two sources of over-attribution.
  rider         in frames where two objects (different units) move together: when their relative pose stays within
                TH_RIDER_POS_M · TH_RIDER_DEG, the carrier moves at least TH_RIDER_MOVE_M, the carrier has some hand's
                prox ∧ with (strong) for at least TH_WMIN_S and the rider has no such hand, the rider's motion belongs
                    to
                the carrier, not to a hand: its with/weak/lone frames are cleared and it leaves the main-object
                    candidates.
  unit layer    E = engaged (with ∨ weak ∨ lone ∨ fix) interval total (≥ TH_RUN_S) · M = longest moved (with ∨ weak ∨
                lone) run · selection score = (engaged after the overlap and cross discounts) × (TH_CARRY_W if M ≥
                    TH_WMIN_S).
  hand layer    active = Σ E ≥ TH_ACT_S ∧ some candidate (E ≥ TH_RUN_S) → maximum by score, M, name.
                The representative name follows the ASSIGN-1 rule.
  source layer  OakInk2 = the stage's per-hand object lists, when the measurement supports them (E ≥ TH_ACCEPT_E_S ∧
                score ≥ TH_ACCEPT_RATIO × the top measured score).
                TACO = tool and target from the source file name: when both hands touch (touch ≥ TH_TOUCH_S), the tool
                hand gets the tool and the other hand the target; otherwise as v3.1 (accepted when E ≥ TH_ACCEPT_E_S).
                Tool and target are always scene objects (they are part of the gate's requirement).
Validity gate (the `gate` field of the ASSIGN-2 record):
  scene     = moving objects spawned by ASSIGN-1 = the ASSIGN-1 main-object names (per object; a part group spawns only
              its representative part).
  required  = R main-object names ∪ TACO tool and target (riders excluded).
              Moving = the object's window span ≥ hand_object_assign.TH_STATIC_SUPPORT_M (1 cm).
  valid     = every moving requirement is in the scene → per_side stays ASSIGN-1. Still requirements missing from the
              scene = static_required (candidates to add at render time; the reader drops objects already spawned as
              supports by tree role).
  invalid   = some requirement is missing → per_side = R. Moving requirements that R itself could not make a main object
              (r_unspawned) are kept as a marker. static_required is then the still requirements missing from the
              rebaked scene (R main objects); R main objects are spawned already and are not added.
"""

from __future__ import annotations

import re

import numpy as np

from dexengine.data.cases import (
    # frame inputs of R (hand points, objects, hand–object pairs), under the same version (ASSIGN2_VERSION)
    assign_r_frames as ARF,
)
from dexengine.data.cases import (
    # ASSIGN-1 measurement and shared thresholds (palm, fingertip count, small object, gaps, smoothing, 1 cm)
    hand_object_assign as H,
)
from dexengine.data.cases.assign_version import ASSIGN2_VERSION as VERSION

# Thresholds owned by R (unchanged values plus changes); shared thresholds come from hand_object_assign.
TH_MOVE_MPS = 0.03  # moving: window net displacement ≥ this speed × window length
TH_COMOVE = 0.5  # with: slip < this ratio × object net displacement
TH_HAND_K = 0.7  # weak: hand net displacement ≥ this ratio × object net displacement
TH_CARRY_W = 2.0  # selection score: weight of a carrying hand
TH_WMIN_S = 0.3  # minimum carry time for the carry weight, tag carry, cross discount and rider evidence
TH_RUN_S = 0.3  # minimum engaged interval and minimum engagement of a candidate unit
TH_ACT_S = 0.5  # active hand: engaged total
TH_RESCUE_S = 0.3  # weak rescue: own 'with' time
TH_OVL_M_S = (
    0.5  # overlap discount: frames on other units while handling a unit carried at least this long leave the score
)
TH_FAR_M = 0.05  # lone: distance to the other hand
TH_ACCEPT_E_S = 0.5  # source acceptance: minimum engaged time
TH_ACCEPT_RATIO = 0.5  # source acceptance: ratio to the top measured score
TH_TOUCH_S = 0.3  # TACO two-hand touch (one fingertip ∨ palm; total of intervals ≥ TH_RUN_S)
TH_OAK_OVERLAP_S = 0.3  # OakInk2 stage ↔ clip window overlap
OAK_HZ = 120.0  # OakInk2 source frame rate (unit of the program_info intervals)
TH_RIDER_POS_M = 0.01  # rider: upper bound of the relative position change
TH_RIDER_DEG = 5.0  # rider: upper bound of the relative rotation change
TH_RIDER_MOVE_M = 0.03  # rider: minimum distance the carrier moved along
RULE = "R=v3.1+taco2+rider"


def thresholds() -> dict:
    """All thresholds of R (this module + the frame-input module)."""
    return {**ARF.thresholds(), **{k: v for k, v in globals().items() if k.startswith("TH_")}}


def _tot(mask, fps: float, run_s: float = TH_RUN_S) -> float:
    return sum(e - s for s, e in H.runs(mask, H.frames_of(run_s, fps), H.frames_of(H.TH_GAP_CLOSE_S, fps))) / fps


def _longest(mask, fps: float) -> float:
    r = H.runs(mask, 1, H.frames_of(H.TH_GAP_CLOSE_S, fps))
    return max((e - s for s, e in r), default=0) / fps


# Decision: hand layer and source layer.
def classify(F: dict) -> dict:
    """Frame classes (with/weak/lone/fix/engaged/moved), both hands together (motion attribution).
    strong = prox ∧ with (rider evidence)."""
    T, fps, hands = F["T"], F["fps"], F["hands"]
    cls = {}
    for (s, n), A in F["pair"].items():
        with np.errstate(invalid="ignore"):
            moving = A["DobjC"] >= TH_MOVE_MPS * A["span"]
            co = A["DrelC"] < TH_COMOVE * A["DobjC"]
            hm = A["DtipC"] >= TH_HAND_K * A["DobjC"]
        cls[(s, n)] = {"moving": moving, "with": A["touch"] & moving & co, "strong": A["prox"] & moving & co, "hm": hm}
    for (s, n), A in F["pair"].items():
        c = cls[(s, n)]
        others_with = np.zeros(T, bool)
        far = np.ones(T, bool)
        for s2 in hands:
            if s2 != s and (s2, n) in cls:
                others_with |= cls[(s2, n)]["with"]
                far &= F["pair"][(s2, n)]["dmin"] > TH_FAR_M
        prox, mv, w = A["prox"], c["moving"], c["with"]
        own_with = w.sum() / fps
        allow = np.ones(T, bool) if own_with >= TH_RESCUE_S else ~others_with
        weak = prox & mv & ~w & c["hm"] & allow
        lone = prox & mv & ~w & ~weak & far & ~F["obj"][n]["jump"]
        c.update(weak=weak, lone=lone, fix=prox & ~mv)
        _settle(c)
    return cls


def _settle(c: dict) -> None:
    """Rebuild engaged and moved from the fields, with the same definition after rider stripping."""
    c.update(eng=c["with"] | c["weak"] | c["lone"] | c["fix"], moved=c["with"] | c["weak"] | c["lone"])


def riders(F: dict, cls: dict) -> dict:
    """{rider: (carrier, mask of the frames moving together)}; see 'rider' in the module docstring. Objects of the same
    unit (part group) are not compared."""
    fps, hands, ob = F["fps"], F["hands"], F["obj"]
    unit_of = {n: u for u, ms in F["members"].items() for n in ms}
    strong = {
        n: max((float(cls[(s, n)]["strong"].sum()) / fps for s in hands if (s, n) in cls), default=0.0) for n in ob
    }
    out = {}
    for a in ob:
        if strong[a] >= TH_WMIN_S:
            continue
        for b in ob:
            if a == b or unit_of[a] == unit_of[b] or strong[b] < TH_WMIN_S:
                continue
            co = (ob[a]["speed"] >= TH_MOVE_MPS) & (ob[b]["speed"] >= TH_MOVE_MPS) & ~ob[a]["jump"] & ~ob[b]["jump"]
            if not co.any():
                continue
            pb = ob[b]["po"][co]
            if float((pb.max(0) - pb.min(0)).max()) < TH_RIDER_MOVE_M:
                continue
            Rb, Ra = ob[b]["Ro"][co], ob[a]["Ro"][co]
            rel_p = np.einsum("tji,tj->ti", Rb, ob[a]["po"][co] - pb)
            rel_R = np.einsum("tji,tjk->tik", Rb, Ra)
            if float(np.linalg.norm(rel_p - rel_p[0], axis=1).max()) > TH_RIDER_POS_M:
                continue
            if float(ARF.rot_deg(np.einsum("ji,tjk->tik", rel_R[0], rel_R)).max()) > TH_RIDER_DEG:
                continue
            out[a] = (b, co)
            break
    return out


def strip_riders(F: dict, cls: dict, rid: dict) -> None:
    """Clear hand attribution (with/weak/lone) on a rider's co-moving frames; that motion belongs to the carrier."""
    for a, (_b, co) in rid.items():
        for s in F["hands"]:
            c = cls.get((s, a))
            if c is None:
                continue
            for k in ("with", "weak", "lone"):
                c[k] = c[k] & ~co
            _settle(c)


def unit_layer(F: dict, cls: dict) -> dict:
    T, fps, hands, units, members = F["T"], F["fps"], F["hands"], F["units"], F["members"]
    U = {}
    for s in hands:
        for u in units:
            mem = [n for n in members[u] if (s, n) in cls]
            z = np.zeros(T, bool)
            eng, moved, fix, prox, wth, tch = z.copy(), z.copy(), z.copy(), z.copy(), z.copy(), z.copy()
            for n in mem:
                tch |= F["pair"][(s, n)]["touch"]
                eng |= cls[(s, n)]["eng"]
                moved |= cls[(s, n)]["moved"]
                fix |= cls[(s, n)]["fix"]
                wth |= cls[(s, n)]["with"]
                prox |= F["pair"][(s, n)]["prox"]
            U[(s, u)] = {
                "eng": eng,
                "moved": moved,
                "fix": fix,
                "prox": prox,
                "E": _tot(eng, fps),
                "M": _longest(moved, fps),
                "W": wth.sum() / fps,
                "touch": tch,
                "touch_s": _tot(tch, fps, TH_TOUCH_S),
            }
    for s in hands:
        so = next((h for h in hands if h != s), None)
        cm = np.zeros(T, bool)
        for u in units:
            if U[(s, u)]["M"] >= TH_OVL_M_S:
                cm |= U[(s, u)]["eng"]
        for u in units:
            m = U[(s, u)]["eng"].copy()
            if U[(s, u)]["M"] < TH_OVL_M_S:
                m &= ~cm
            if so and U[(s, u)]["M"] < TH_WMIN_S and U[(so, u)]["M"] >= TH_WMIN_S:
                busy = np.zeros(T, bool)
                for v in units:
                    if v != u:
                        busy |= U[(s, v)]["eng"] & ~U[(so, v)]["eng"]
                m &= ~(U[(s, u)]["fix"] & U[(so, u)]["eng"] & busy)
            U[(s, u)]["Esel"] = _tot(m, fps)
    return U


def score(U: dict, s: str, u: str) -> float:
    x = U[(s, u)]
    return x["Esel"] * (TH_CARRY_W if x["M"] >= TH_WMIN_S else 1.0)


def measured(F: dict, U: dict, rider_units: set) -> dict:
    out = {}
    for s in F["hands"]:
        etot = sum(U[(s, u)]["E"] for u in F["units"])
        cand = [u for u in F["units"] if U[(s, u)]["E"] >= TH_RUN_S and u not in rider_units]
        if etot >= TH_ACT_S and cand:
            rk = sorted(cand, key=lambda u: (score(U, s, u), U[(s, u)]["M"], u), reverse=True)
            out[s] = {"unit": rk[0], "rank": rk, "etot": etot}
        else:
            out[s] = {"unit": None, "rank": [], "etot": etot}
    return out


def oak_native(
    task: str,
    seg_start: int,
    offset: int,
    T: int,
    fps: float,
    units: list,
    members: dict,
    obj_id_of: dict,
    entries: list,
    rid_of,
    root_of,
) -> dict:
    """OakInk2 per-hand N_s (that hand's object list for the stages whose hand interval overlaps the clip window by at
    least TH_OAK_OVERLAP_S) and the matching units. entries = (key, info) list of program_info (key order = interval
    order) · rid_of/root_of = part identity from oakink2.parts."""
    import ast  # noqa: PLC0415

    m = re.match(r"oakink2_(.+?)__scene_(\d+)_([0-9a-f]{12})_s(\d+)$", task)
    if not m:
        raise ValueError(
            f"OakInk2 clip name {task!r}: cannot read the interval number (the measurement is not used alone "
            "without the source layer)"
        )
    k = int(m.group(4))
    if k >= len(entries):
        raise ValueError(
            f"OakInk2 {task}: interval number {k} outside the {len(entries)} program_info intervals — "
            "source statement and clip disagree"
        )

    def ivs(key):
        lh, rh = ast.literal_eval(key)
        return {"left": lh, "right": rh}

    key, info = entries[k]
    iv = ivs(key)
    a0 = min(x[0] for x in iv.values() if x)
    w0 = a0 + (seg_start + offset) * OAK_HZ / fps
    w1 = w0 + T * OAK_HZ / fps
    out = {"window": [round(w0), round(w1)]}
    rid_m = {u: {rid_of(str(obj_id_of[n])) for n in members[u]} for u in units}
    for s, lk in (("left", "obj_list_lh"), ("right", "obj_list_rh")):
        N = set()
        for k2, inf2 in entries:
            iv2 = ivs(k2)
            if iv2[s] and inf2.get(lk) and min(w1, iv2[s][1]) - max(w0, iv2[s][0]) >= TH_OAK_OVERLAP_S * OAK_HZ:
                N |= {rid_of(x) for x in inf2[lk]}
        exact = [u for u in units if rid_m[u] & N]
        if not exact and N:
            roots = {root_of(x) for x in N} - {None}
            exact = [u for u in units if {root_of(x) for x in rid_m[u]} & roots]
        out[s] = {"N": sorted(N), "units": exact}
    return out


def _nid(s) -> str:
    return re.sub(r"^(left|right|raw|supp|extra)_", "", str(s).lower()).replace("@", "_")


def taco_native(roles: dict, units: list, members: dict, obj_id_of: dict) -> dict:
    """TACO source statement: roles = {'tool': mesh id, 'target': mesh id} (source file name, assign_r_inputs.raw_roles)
    → units."""

    def unit_of_id(x):
        hit = [u for u in units if any(_nid(obj_id_of[n]) == _nid(x) for n in members[u])]
        return hit[0] if hit else None

    tu, gu = unit_of_id(roles["tool"]), unit_of_id(roles["target"])
    if tu is None or gu is None:
        raise ValueError(
            f"TACO source tool/target {roles} not among the tree objects {sorted(obj_id_of.values())} — "
            "source statement and tree disagree"
        )
    return {"tool": roles["tool"], "target": roles["target"], "tool_unit": tu, "target_unit": gu}


def resolve_native(F: dict, U: dict, meas: dict, rider_units: set) -> dict:
    """Source layer, followed only when the measurement supports it (TACO: the two-hand touch clause first). Returns {s:
    (unit, src)}."""
    hands = F["hands"]
    res = {s: (meas[s]["unit"], "MEAS") for s in hands}
    nat = F["nat"]
    if "oak" in nat:
        for s in hands:
            o = nat["oak"].get(s) or {}
            mu = meas[s]["unit"]
            cand = [u for u in (o.get("units") or []) if u not in rider_units]
            if not o.get("N") or not cand:
                continue
            if mu in cand:
                res[s] = (mu, "NAT_AGREE")
                continue
            b = max(cand, key=lambda u: (score(U, s, u), U[(s, u)]["E"], u))
            sm = score(U, s, mu) if mu else 0.0
            if U[(s, b)]["E"] >= TH_ACCEPT_E_S and score(U, s, b) >= TH_ACCEPT_RATIO * sm:
                res[s] = (b, "NAT_ACCEPT")
    elif "taco" in nat and len(hands) == 2:
        t = nat["taco"]
        tu, gu = t["tool_unit"], t["target_unit"]
        th = max(hands, key=lambda s: (score(U, s, tu), U[(s, tu)]["touch_s"], s == "right"))
        gh = next(s for s in hands if s != th)
        if U[(th, tu)]["touch_s"] >= TH_TOUCH_S and U[(gh, gu)]["touch_s"] >= TH_TOUCH_S:
            res[th], res[gh] = (tu, "TACO_TOOL2"), (gu, "TACO_TGT2")  # each hand touches = two roles = two hands
        else:
            if U[(th, tu)]["E"] >= TH_ACCEPT_E_S:
                res[th] = (tu, "TACO_TOOL")
            if U[(gh, gu)]["E"] >= TH_ACCEPT_E_S:
                res[gh] = (gu, "TACO_TGT")
    return res


def rep_name(F: dict, s: str, u: str) -> str:
    """Representative part, same rule as ASSIGN-1 (hold time = total of (prox ∨ old joint drive) intervals ≥ 0.3 s →
    frame count; ties = first in objects[] order)."""
    mem = [n for n in F["members"][u] if (s, n) in F["pair"]]

    def key(n):
        m = F["pair"][(s, n)]["prox"] | F["pair"][(s, n)]["drive_old"]
        return (_tot(m, F["fps"], 0.3), int(m.sum()))

    return max(mem, key=key)


def decide(F: dict) -> dict:
    """Decision of rule R: {"per_side": {s: {active, main, unit, tag, carry_s, hold_s, src}}, "riders": {object:
    carrier},
    "U": unit layer}."""
    cls = classify(F)
    rid = riders(F, cls)
    strip_riders(F, cls, rid)
    rider_units = {u for u, ms in F["members"].items() if ms and all(n in rid for n in ms)}
    U = unit_layer(F, cls)
    meas = measured(F, U, rider_units)
    res = resolve_native(F, U, meas, rider_units)
    per = {}
    for s in F["hands"]:
        u, src = res[s]
        if u is None:
            per[s] = {
                "active": False,
                "main": None,
                "unit": None,
                "tag": "idle",
                "carry_s": 0.0,
                "hold_s": 0.0,
                "src": src,
            }
        else:
            per[s] = {
                "active": True,
                "main": rep_name(F, s, u),
                "unit": u,
                "tag": "carry" if U[(s, u)]["M"] >= TH_WMIN_S else "hold_static",
                "carry_s": round(U[(s, u)]["M"], 2),
                "hold_s": round(U[(s, u)]["E"], 2),
                "src": src,
            }
    return {"per_side": per, "riders": {a: b for a, (b, _co) in rid.items()}, "U": U}


# Validity gate.
def gate(hands: list, a1_per_side: dict, r_per_side: dict, span: dict, task_objects: list, riders_: dict) -> dict:
    """Validity gate (see the module docstring); pure function. span = {object name: window span (m)} ·
    task_objects = TACO tool and target names ([] if none)."""
    scene = {a1_per_side[s]["main"] for s in hands if (a1_per_side.get(s) or {}).get("main")}
    rmain = {r_per_side[s]["main"] for s in hands if (r_per_side.get(s) or {}).get("main")}
    req = (rmain | set(task_objects)) - set(riders_)
    lost = sorted(n for n in req if n not in span)
    if lost:
        raise KeyError(f"validity gate: no span for required objects {lost} — inputs disagree with objects[]")
    moving = sorted(n for n in req if span[n] >= H.TH_STATIC_SUPPORT_M)
    missing = [n for n in moving if n not in scene]
    after = (
        rmain if missing else scene
    )  # scene to render; when invalid, the rebaked scene = R main objects (spawned already)
    static_required = sorted(n for n in req if span[n] < H.TH_STATIC_SUPPORT_M and n not in after)
    verdict = "invalid" if missing else ("valid_static" if static_required else "valid")
    return {
        "rule": RULE,
        "verdict": verdict,
        "scene": sorted(scene),
        "task_objects": sorted(set(task_objects)),
        "moving_required": moving,
        "missing": missing,
        "static_required": static_required,
        "r_unspawned": sorted(n for n in missing if n not in rmain),
        "riders": dict(sorted(riders_.items())),
        "r_per_side": {
            s: [r_per_side[s]["unit"], r_per_side[s]["main"], r_per_side[s]["tag"], r_per_side[s]["src"]] for s in hands
        },
    }


def _r_shared_kind(A1: dict, U: dict, r_per: dict, hands: list, fps: float):
    """Shared unit of R and its kind. The kind is the ASSIGN-1 timeline share kind when that unit has one (ASSIGN-1 owns
    the timeline); otherwise share_kind runs on the R engaged intervals (both hands have them, since the unit is main
    for both)."""
    mains = [r_per[s]["unit"] for s in hands if r_per[s]["unit"]]
    shared = mains[0] if len(mains) == 2 and mains[0] == mains[1] else None
    if shared is None:
        return None, None
    kind = ((A1.get("share") or {}).get(shared) or {}).get("kind")
    if kind is None:
        mn, gap = H.frames_of(TH_RUN_S, fps), H.frames_of(H.TH_GAP_CLOSE_S, fps)
        gr, gl = (H.runs(U[(s, shared)]["eng"], mn, gap) for s in ("right", "left"))
        if not (gr and gl):
            raise ValueError(f"R shared unit {shared} but one hand has no engaged interval — inconsistent computation")
        kind = H.share_kind(gr, gl, fps)["kind"]
    return shared, kind


def assign2(A1: dict, F: dict, task_objects: list) -> dict:
    """ASSIGN-2 record (rules.ASSIGN). A1 = hand_object_assign.assign result · F = R inputs (built by the case stage).
    Valid → the ASSIGN-1 decision fields as is; invalid → R's per_side, share and object tags (quality fields are
    measurements and stay ASSIGN-1)."""
    hands = list(F["hands"])
    R = decide(F)
    span = {n: F["obj"][n]["span"] for n in F["obj"]}
    g = gate(hands, A1["per_side"], R["per_side"], span, task_objects, R["riders"])
    rec = {k: A1[k] for k in H.RECORD_KEYS}
    rec["version"] = VERSION
    if g["verdict"] == "invalid":
        per = {s: {k: v for k, v in R["per_side"][s].items() if k != "src"} for s in hands}
        shared, kind = _r_shared_kind(A1, R["U"], R["per_side"], hands, F["fps"])
        main_tag: dict = {}
        for s in hands:
            n = per[s]["main"]
            if n:
                main_tag[n] = "carry" if "carry" in (main_tag.get(n), per[s]["tag"]) else "hold_static"
        old = A1["obj_tags"]
        tags = {
            n: main_tag.get(n) or (old[n] if old[n] in ("static_support", "background") else "background") for n in old
        }
        rec.update(per_side=per, shared=shared, share_kind=kind, obj_tags=tags)
    rec["gate"] = g
    return rec
