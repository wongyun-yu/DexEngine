"""Hand–object assignment standard (A1): which hand handles which object, and when.

Pure computation on the arrays the case stage (cases.taxonomy, stage 1 step 3) has read; no file reads. The result is
written to base as `rules.ASSIGN`, and the roles stage (object_roles, step 4) applies it as is. Inputs are
hand-independent: MANO fingertip/palm ↔ object surface distances (contact_all sidecar), MANO fingertip speed and the
object trajectory. The robot hand is never read, so one base gives the same assignment for every hand.

Layers (frame → interval → per-hand summary):
  contact         one fingertip < threshold ∨ palm < TH_PALM_M. Information only (per-hand contact fraction, input of
                  the 4b quality step); not used for the assignment.
  hold            fingertips ≥ TH_GRASP_MIN_TIPS ∨ palm ∨ one fingertip on a small object (longest side
                  < TH_SMALL_OBJ_M).
  carry           hold ∧ (the surface points near the fingertips move with the object: surface speed ≥ TH_OBJ_STILL_MPS
                  ∧ ‖v_tip − v_surface‖ < TH_CARRY_REL·‖v_surface‖ ∨ in-place rotation > TH_ROT_DPS)
                  ∨ joint drive (joint speed > TH_ARTIC_DRIVE_RADPS while a fingertip touches).
                  Speeds are smoothed over TH_SMOOTH_S. There is no absolute speed floor and no 'still object' pass.
  timeline grasp  hold ∧ (relative < max(TH_TIMELINE_COMOVE_MPS, ½·v) ∨ v < TH_OBJ_STILL_MPS) ∨ joint drive.
                  Only the timeline (D36 input), active hands, used objects and share kind read it, so the piece
                  registry stays stable. Not used to pick the main object.
  motion          carry ∧ body motion ∨ joint drive; read only by the split rule (move_runs).
Assignment (per hand):
  active hand     timeline-grasp total ≥ TH_ACTIVE_MIN_S. An inactive hand is idle (no main object, rest pose).
  carry time      longest carried interval on the unit (gaps shorter than TH_GAP_CLOSE_S are bridged).
  hold time       total of hold (∨ joint drive) intervals ≥ TH_GRASP_MIN_S.
  main object     the unit with the longest carry time ≥ TH_CARRY_MIN_S (ties: hold time, then name), tag carry.
                  Otherwise the unit held longest, tag hold_static (holding something still is also a task). No held
                  unit: idle. For a part group the name is the part this hand held longest (interval total,
                  then frames).
  assisting hand  in a unit whose share kind is assist, the hand that handled it less (carry time, then hold time)
                  loses it as a candidate, but only when its carry time on that unit is below TH_CARRY_MIN_S (a
                  carrying hand is never dropped). No candidates left: idle.
  shared          both hands have the same main unit. Kind = share kind measured on the timeline grasps (sim ·
                  handover · mixed · parts · assist); when one hand has no timeline grasp on the unit, the same rule
                  (share_kind) runs on the two hands' hold intervals, so a kind always exists.
Object tags (information; roles and slots do not change): carry · hold_static (main object of some hand; carry if both)
  · static_support (not a main object, held by no hand, total motion over the window < TH_STATIC_SUPPORT_M)
  · background.
Quality inputs (read by 4b): per-hand contact fraction · no contact · contactless motion (the object moves ≥ MOVE_M
while no hand touches it).
Fails when it cannot apply: a part group whose base is not in objects[] (ghost unit) · inputs that disagree with
objects[] · an active hand with zero held units.
"""

from __future__ import annotations

import numpy as np

from dexengine.data.cases.assign_version import (
    ASSIGN_VERSION as VERSION,  # version; in a dependency-free module so the stage table reads it without numpy
)
from dexengine.data.objects.scene_requirement import (
    MOVE_M,  # contactless-motion threshold (scene requirement statement, dependency-free module)
)

# Thresholds (owned here; cases.taxonomy imports them).
TH_GRASP_MIN_S = 0.3  # minimum length of hold and timeline-grasp intervals
TH_GRASP_MIN_TIPS = 2  # hold = 2+ fingertips (pinch); 1 fingertip = contact
TH_TIMELINE_COMOVE_MPS = 0.08  # comove floor of the timeline grasp (timeline only; carry has none)
TH_OBJ_STILL_MPS = 0.03  # carry: lower bound of surface-point speed · timeline grasp: still-object pass
TH_PALM_M = 0.03  # palm centre ↔ surface < 3 cm = the palm supports a large object (evidence of hold)
TH_SMALL_OBJ_M = 0.08  # small object (longest side < 8 cm): 1 fingertip counts as hold
TH_USED_MIN_S = 0.5  # used object = some hand's timeline-grasp total ≥ 0.5 s
TH_TIMELINE_MIN_S = 0.5  # timeline: assignment pieces shorter than this merge into a neighbour (flicker)
TH_GAP_CLOSE_S = 0.10  # gaps shorter than this are bridged into one interval
TH_ACTIVE_MIN_S = 0.5  # active hand = timeline-grasp total ≥ 0.5 s
TH_SIM_STAGGER_S = (
    0.3  # start and end differences of the two holds both below this → picked up and put down together (sim)
)
TH_HANDOVER_MAX_OV_S = 1.5  # overlap of staggered holds ≤ this → handover, above → mixed
# D27: a joint turning at least this fast under a touching fingertip is handled by that hand (even with a still body)
TH_ARTIC_DRIVE_RADPS = 0.05
TH_CARRY_MIN_S = 0.3  # main object: units whose longest carried interval is at least this
TH_CARRY_REL = 0.5  # carry: ‖v_tip − v_surface‖ < 0.5·‖v_surface‖
TH_ROT_DPS = 20.0  # carry: in-place rotation while held (object angular speed)
TH_SMOOTH_S = 0.2  # velocity smoothing window
TH_STATIC_SUPPORT_M = 0.01  # object tag static_support: upper bound of total motion over the window


class GhostUnit(ValueError):
    """The base of a part group (articulation_group.base) is not in objects[]; no timeline or assignment on a ghost
    name."""


def thresholds() -> dict:
    """Threshold table for the run marker and the census (values owned by this file)."""
    return {k: v for k, v in globals().items() if k.startswith("TH_")}


def runs(mask, min_len: int, gap: int) -> list:
    """bool (T,) → [(s, e)) intervals. Gaps of length ≤ gap are bridged; intervals shorter than min_len are dropped."""
    mask = np.asarray(mask, bool)
    if len(mask) == 0 or not mask.any():
        return []
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(int), [0]])))
    merged: list = []
    for s, e in zip(idx[0::2], idx[1::2], strict=True):
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(int(s), int(e)) for s, e in merged if e - s >= min_len]


def frames_of(sec: float, fps: float) -> int:
    return max(1, int(round(sec * fps)))


def smooth_velocity(p, fps: float) -> tuple:
    """(T,3) positions → (smoothed speed (T,), smoothed velocity (T,3)). The window (TH_SMOOTH_S) is capped at the
    signal length."""
    p = np.asarray(p, float)
    k = max(1, min(frames_of(TH_SMOOTH_S, fps), len(p)))
    v = np.gradient(p, axis=0) * fps
    v = np.array([np.convolve(v[:, i], np.ones(k) / k, mode="same") for i in range(3)]).T
    return np.linalg.norm(v, axis=1), v


def units_of(objects: list) -> tuple:
    """(unit list, unit → member objects). A part group is one unit (base = articulation_group.base).
    A base missing from objects[] raises GhostUnit."""
    names = [str(o["name"]) for o in objects]
    group_of = {}
    for o in objects:
        g = o.get("articulation_group") or {}
        if g.get("base"):
            if str(g["base"]) not in names:
                raise GhostUnit(
                    f"object {o['name']}: part-group base {g['base']!r} is not in objects[] ({names}) — ghost unit"
                )
            group_of[str(o["name"])] = str(g["base"])
    units = list(dict.fromkeys(group_of.get(n, n) for n in names))
    return units, {u: [n for n in names if group_of.get(n, n) == u] for u in units}


def _union(T: int, masks: list):
    out = np.zeros(T, bool)
    for m in masks:
        out |= np.asarray(m, bool)
    return out


def _longest_s(mask, fps: float) -> float:
    r = runs(mask, 1, frames_of(TH_GAP_CLOSE_S, fps))
    return max((e - s for s, e in r), default=0) / fps


def _total_s(rr: list, fps: float) -> float:
    return sum(e - s for s, e in rr) / fps


def share_kind(gr: list, gl: list, fps: float) -> dict:
    """Share kind of a unit held by both hands (timeline-grasp intervals gr, gl): sim · handover · assist · mixed.
    Part groups (parts) are split by the caller (_shares)."""
    T = max([e for _, e in gr + gl] + [0])
    mr, ml = np.zeros(T, bool), np.zeros(T, bool)
    for s, e in gr:
        mr[s:e] = True
    for s, e in gl:
        ml[s:e] = True
    ov = (mr & ml).sum() / fps
    first_r, first_l, last_r, last_l = gr[0][0], gl[0][0], gr[-1][1], gl[-1][1]
    d_start, d_end = abs(first_r - first_l) / fps, abs(last_r - last_l) / fps
    giver, receiver = ("right", "left") if first_r < first_l else ("left", "right")
    last_giver, last_recv = (last_r, last_l) if giver == "right" else (last_l, last_r)
    lesser_s = min(_total_s(gr, fps), _total_s(gl, fps))
    if d_start < TH_SIM_STAGGER_S and d_end < TH_SIM_STAGGER_S:
        kind = "sim"  # picked up together, put down together
    elif ov <= TH_HANDOVER_MAX_OV_S and last_recv > last_giver and lesser_s >= TH_USED_MIN_S:
        kind = "handover"  # staggered holds, the receiving hand holds to the end = handover
    elif lesser_s < TH_USED_MIN_S or (ov <= TH_HANDOVER_MAX_OV_S and last_recv <= last_giver):
        kind = "assist"  # one hand helps briefly, then the first hand takes it back
    else:
        kind = "mixed"  # long joint hold, one hand starts or ends earlier
    return {
        "kind": kind,
        "overlap_s": round(ov, 2),
        "d_start_s": round(d_start, 2),
        "d_end_s": round(d_end, 2),
        "order": f"{giver[0].upper()}->{receiver[0].upper()}",
        "right_s": round(_total_s(gr, fps), 2),
        "left_s": round(_total_s(gl, fps), 2),
    }


def _held_per_frame(ugrasp: dict, used: list, s: str, T: int, fps: float):
    """One hand's held unit per frame: the used unit with the longest timeline-grasp total among those it holds at
    that frame; empty frames take the nearest held frame's unit (all None when the hand holds nothing)."""
    cur = np.array([None] * T, dtype=object)
    for u in used:
        for a, b in ugrasp[(s, u)]:
            for t in range(a, b):
                if cur[t] is None or _total_s(ugrasp[(s, u)], fps) > _total_s(ugrasp[(s, cur[t])], fps):
                    cur[t] = u
    held = [i for i in range(T) if cur[i] is not None]
    if held:
        held = np.array(held)
        for t in range(T):
            if cur[t] is None:
                cur[t] = cur[held[np.abs(held - t).argmin()]]
    return cur


def timeline(ugrasp: dict, used: list, hands: list, T: int, fps: float) -> list:
    """Per-interval assignment table (D36 input): per frame, the used unit the hand holds (several: the one with the
    longest timeline-grasp interval). Empty frames take the nearest held unit (None if there is none); pieces shorter
    than TH_TIMELINE_MIN_S merge into a neighbour (flicker = contact)."""
    per_frame = {s: _held_per_frame(ugrasp, used, s, T, fps) for s in hands}
    tl, t0 = [], 0
    for t in range(1, T + 1):
        if t == T or any(per_frame[s][t] != per_frame[s][t - 1] for s in hands):
            tl.append({"t0": t0, "t1": t, **{s: per_frame[s][t0] for s in hands}})
            t0 = t
    min_len, changed = frames_of(TH_TIMELINE_MIN_S, fps), True
    while changed and len(tl) > 1:
        changed = False
        for i, seg in enumerate(tl):
            if seg["t1"] - seg["t0"] < min_len:
                j = i - 1 if i > 0 else i + 1
                tl[j] = {**tl[j], "t0": min(tl[j]["t0"], seg["t0"]), "t1": max(tl[j]["t1"], seg["t1"])}
                tl.pop(i)
                changed = True
                break
        merged = [tl[0]]
        for seg in tl[1:]:
            if all(seg.get(s) == merged[-1].get(s) for s in hands):
                merged[-1] = {**merged[-1], "t1": seg["t1"]}
            else:
                merged.append(seg)
        tl = merged
    return tl


def move_runs(T: int, fps: float, hands: list, objects: list, masks: dict, obj_pos: dict, obj_rot_dps: dict) -> dict:
    """Motion layer {unit: {hand: [[a, b]]}}: body = smoothed speed ≥ TH_OBJ_STILL_MPS ∨ angular speed > TH_ROT_DPS (no
    new threshold)."""
    units, members = units_of(objects)
    mn, gap = frames_of(TH_GRASP_MIN_S, fps), frames_of(TH_GAP_CLOSE_S, fps)
    out: dict = {}
    for u in units:
        body = _union(
            T,
            [
                (smooth_velocity(obj_pos[n], fps)[0] >= TH_OBJ_STILL_MPS) | (np.asarray(obj_rot_dps[n]) > TH_ROT_DPS)
                for n in members[u]
            ],
        )
        out[u] = {
            s: [
                [a, b]
                for a, b in runs(
                    (_union(T, [masks[(s, n)]["carry"] for n in members[u]]) & body)
                    | _union(T, [masks[(s, n)]["drive"] for n in members[u]]),
                    mn,
                    gap,
                )
            ]
            for s in hands
        }
    return out


def _unit_layers(T: int, fps: float, hands: list, units: list, members: dict, masks: dict) -> dict:
    """(hand, unit) → {contact, hold_runs (hold ∨ joint drive), carry_s, grasp_runs (timeline)}; a part group sums its
    members."""
    mn, gap = frames_of(TH_GRASP_MIN_S, fps), frames_of(TH_GAP_CLOSE_S, fps)
    out = {}
    for u in units:
        for s in hands:
            mm = [masks[(s, n)] for n in members[u]]
            gm = np.zeros(T, bool)
            for m in mm:
                for a, b in runs(m["grasp_tl"], mn, gap):
                    gm[a:b] = True
            out[(s, u)] = {
                "contact": _union(T, [m["contact"] for m in mm]),
                "hold_runs": runs(_union(T, [m["hold"] | m["drive"] for m in mm]), mn, gap),
                "carry_s": _longest_s(_union(T, [m["carry"] for m in mm]), fps),
                "grasp_runs": runs(gm, mn, gap),
            }
    return out


def _shares(L: dict, hands: list, used: list, members: dict, masks: dict, fps: float, T: int) -> dict:
    """Share kind per used unit (when both hands have timeline-grasp intervals).
    In a part group, parts when the two hands mainly hold different parts."""
    share = {}
    if len(hands) != 2:
        return share
    mn, gap = frames_of(TH_GRASP_MIN_S, fps), frames_of(TH_GAP_CLOSE_S, fps)
    for u in used:
        gr, gl = L[("right", u)]["grasp_runs"], L[("left", u)]["grasp_runs"]
        if not (gr and gl):
            continue
        if len(members[u]) > 1:
            g = {(s, n): _total_s(runs(masks[(s, n)]["grasp_tl"], mn, gap), fps) for s in hands for n in members[u]}
            dom = {s: max(members[u], key=lambda n, s=s: g[(s, n)]) for s in hands}  # both hands (checked above)
            if dom["right"] != dom["left"] and all(g[(s, dom[s])] >= TH_USED_MIN_S for s in hands):
                k = share_kind(gr, gl, fps)
                share[u] = {
                    "kind": "parts",
                    "overlap_s": k["overlap_s"],
                    "right_part": dom["right"],
                    "left_part": dom["left"],
                    "right_s": k["right_s"],
                    "left_s": k["left_s"],
                }
                continue
        share[u] = share_kind(gr, gl, fps)
    return share


def _pick(s: str, hands: list, units: list, L: dict, share: dict, fps: float) -> tuple:
    """(unit, tag) of one hand: the unit with the longest carry time ≥ TH_CARRY_MIN_S (ties: hold time, then name)
    → else the longest hold time (hold_static) → else (None, idle).
    Assisting hand: in an assist unit, the hand that handled it less (carry time, then hold time) loses it as a
    candidate when its carry time on that unit is below TH_CARRY_MIN_S; no candidates left → idle."""
    other = [h for h in hands if h != s]

    def hold_s(side, u):
        return _total_s(L[(side, u)]["hold_runs"], fps)

    def helper_out(u):
        if (share.get(u) or {}).get("kind") != "assist" or not other:
            return False
        mine, theirs = L[(s, u)], L[(other[0], u)]
        lesser = (theirs["carry_s"], hold_s(other[0], u)) > (mine["carry_s"], hold_s(s, u))
        return lesser and mine["carry_s"] < TH_CARRY_MIN_S

    names = sorted(units)
    carry = [u for u in names if L[(s, u)]["carry_s"] >= TH_CARRY_MIN_S and not helper_out(u)]
    if carry:
        return max(carry, key=lambda u: (L[(s, u)]["carry_s"], hold_s(s, u))), "carry"
    held = [u for u in names if L[(s, u)]["hold_runs"] and not helper_out(u)]
    if held:
        return max(held, key=lambda u: hold_s(s, u)), "hold_static"
    return None, "idle"


def _span_m(p) -> float:
    """Largest per-axis range of (T,3) positions (total motion over the window)."""
    p = np.asarray(p, float)
    return float((p.max(0) - p.min(0)).max()) if len(p) else 0.0


def _check_inputs(hands: list, names: list, masks: dict, obj_pos: dict) -> None:
    """Every (hand, object) mask and every object position is present, else KeyError (inputs disagree with
    objects[])."""
    lost = [f"{s}:{n}" for s in hands for n in names if (s, n) not in masks] + [n for n in names if n not in obj_pos]
    if lost:
        raise KeyError(
            f"(hand:object)/object {lost} missing from the assignment inputs — inputs disagree with objects[]"
        )


def _main_of(
    s: str,
    hands: list,
    units: list,
    members: dict,
    L: dict,
    share: dict,
    hold: dict,
    masks: dict,
    fps: float,
    hand_total: dict,
) -> tuple:
    """(unit, tag, main object name) of one active hand; the name is the unit member this hand held longest (hold
    interval total, then frames). An active hand with no held unit is an inconsistent computation (ValueError)."""
    if not any(L[(s, u)]["hold_runs"] for u in units):
        raise ValueError(
            f"active hand {s} (timeline-grasp total {hand_total[s]:.2f} s) with zero held units — "
            "inconsistent computation"
        )
    u, tag = _pick(s, hands, units, L, share, fps)
    name = (
        None
        if u is None
        else max(
            members[u],
            key=lambda n, s=s: (
                _total_s(hold[(s, n)], fps),
                int((masks[(s, n)]["hold"] | masks[(s, n)]["drive"]).sum()),
            ),
        )
    )
    return u, tag, name


def _shared_unit(per_side: dict, hands: list, share: dict, L: dict, fps: float) -> tuple:
    """(shared unit, share kind): the unit both hands chose as main, or (None, None)."""
    mains = [per_side[s]["unit"] for s in hands if per_side[s]["unit"]]
    shared = mains[0] if len(mains) == 2 and mains[0] == mains[1] else None
    kind = (share.get(shared) or {}).get("kind") if shared else None
    if shared and kind is None:  # one hand has no timeline grasp on the unit — measure on the two hands' hold intervals
        kind = share_kind(L[("right", shared)]["hold_runs"], L[("left", shared)]["hold_runs"], fps)["kind"]
    return shared, kind


def _obj_tags(per_side: dict, hands: list, names: list, hold: dict, obj_pos: dict) -> dict:
    """Object tags: carry · hold_static (main object of some hand; carry if both) · static_support · background."""
    main_tag: dict = {}
    for s in hands:
        n = per_side[s]["main"]
        if n:
            main_tag[n] = "carry" if "carry" in (main_tag.get(n), per_side[s]["tag"]) else "hold_static"
    held_any = {n for (_s, n), rr in hold.items() if rr}
    return {
        n: main_tag.get(n)
        or ("static_support" if n not in held_any and _span_m(obj_pos[n]) < TH_STATIC_SUPPORT_M else "background")
        for n in names
    }


#: Fields written to base `rules.ASSIGN`; the rest (units · timeline · share …) goes into the case record
#: (task_info.case).
RECORD_KEYS = ("version", "per_side", "shared", "share_kind", "obj_tags", "quality")


def assign(T: int, fps: float, hands: list, objects: list, masks: dict, obj_pos: dict) -> dict:
    """Assignment of one tree; pure function (no file reads).
    masks[(hand, object)] = frame_layers.hand_object_masks(...) · obj_pos[object] = (T,3) object position ·
    objects = task_info.objects (names, part groups).
    Returns the case fields (units · active · used · touched · share · timeline · …) + the assignment fields
    (RECORD_KEYS)."""
    hands = list(hands)
    units, members = units_of(objects)
    names = [str(o["name"]) for o in objects]
    _check_inputs(hands, names, masks, obj_pos)
    mn, gap = frames_of(TH_GRASP_MIN_S, fps), frames_of(TH_GAP_CLOSE_S, fps)
    L = _unit_layers(T, fps, hands, units, members, masks)
    grasp = {(s, n): runs(masks[(s, n)]["grasp_tl"], mn, gap) for s in hands for n in names}
    hold = {(s, n): runs(masks[(s, n)]["hold"] | masks[(s, n)]["drive"], mn, gap) for s in hands for n in names}
    hand_total = {s: sum(_total_s(L[(s, u)]["grasp_runs"], fps) for u in units) for s in hands}
    hand_frac = {s: max((float(L[(s, u)]["contact"].mean()) for u in units), default=0.0) for s in hands}
    active = [s for s in hands if hand_total[s] >= TH_ACTIVE_MIN_S]
    used = [u for u in units if any(_total_s(L[(s, u)]["grasp_runs"], fps) >= TH_USED_MIN_S for s in hands)]
    touched = [u for u in units if u not in used and any(L[(s, u)]["contact"].any() for s in hands)]
    share = _shares(L, hands, used, members, masks, fps, T)
    per_side = {}
    for s in hands:
        if s not in active:
            per_side[s] = {"active": False, "main": None, "unit": None, "tag": "idle", "carry_s": 0.0, "hold_s": 0.0}
            continue
        u, tag, name = _main_of(s, hands, units, members, L, share, hold, masks, fps, hand_total)
        per_side[s] = {
            "active": True,
            "main": name,
            "unit": u,
            "tag": tag,
            "carry_s": round(L[(s, u)]["carry_s"], 2) if u else 0.0,
            "hold_s": round(_total_s(L[(s, u)]["hold_runs"], fps), 2) if u else 0.0,
        }
    shared, kind = _shared_unit(per_side, hands, share, L, fps)
    tags = _obj_tags(per_side, hands, names, hold, obj_pos)
    moved = max((_span_m(obj_pos[n]) for n in names), default=0.0)
    none = all(v == 0.0 for v in hand_frac.values())
    return {
        "version": VERSION,
        "units": units,
        "part_groups": {u: m for u, m in members.items() if len(m) > 1},
        "active": active,
        "used": used,
        "touched": touched,
        "hand_total_s": {s: round(v, 2) for s, v in hand_total.items()},
        "hand_frac": hand_frac,  # before rounding; the case flag touch_only compares it with its threshold
        "contact_frac": {n: {s: round(float(masks[(s, n)]["contact"].mean()), 3) for s in hands} for n in names},
        "grasp_s": {
            n: {s: [[round(a / fps, 2), round(b / fps, 2)] for a, b in grasp[(s, n)]] for s in hands} for n in names
        },
        "share": share,
        "timeline": timeline({k: v["grasp_runs"] for k, v in L.items()}, used, hands, T, fps),
        "artic_drive": sorted(f"{s}:{n}" for s in hands for n in names if masks[(s, n)]["drive"].any()),
        "per_side": per_side,
        "shared": shared,
        "share_kind": kind,
        "obj_tags": tags,
        "quality": {
            "contact_frac": {s: round(v, 3) for s, v in hand_frac.items()},
            "no_contact": bool(none and moved < MOVE_M),
            "moved_without_contact": bool(none and moved >= MOVE_M),
            "object_moved_m": round(moved, 4),
        },
    }


def record(a: dict) -> dict:
    """assign() result → base `rules.ASSIGN` fields."""
    return {k: a[k] for k in RECORD_KEYS}
