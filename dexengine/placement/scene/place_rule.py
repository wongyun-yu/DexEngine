"""Put-down rule — where objects are put in a candidate placement (thresholds · put-down places · the verdict).

Rule (desk scenes, every dataset)
  ① The **outline** (top view = the convex hull of the vertices) of every main object at its **first place** (frame 0)
     and at every **put-down place** (where it stops after the hand puts it down — put_frames) must not overlap the
     robot's standing spot (coords.robot_keepout), and its **area** outside the desk top must be ≤
     static_desk.OVERHANG_MAX_FRAC. Poses while carried are not checked.
  ② The fixed still objects the training scene spawns (coords.fixed_spawned — support objects · still extras) obey the
     same two conditions.
  ③ The judge's robot-side band (`--desk-shrink-robot`, on the main objects' position points) is not this rule.
  ④ If the only offenders are **grazed still objects** (touched for less than GRAZE_MAX_S, receiving no main object),
     the placement stays and those objects are dropped from the spawn (extra_slots reads the same formula drop_why).
     If an object that must be placed offends (a main object · a support object · one that receives a main object · a
     receiver extra · one touched for GRAZE_MAX_S or longer), the placement is rejected; no passing placement → the
     clip is rejected (ledger_states.PLACE_NO_SPOT).
Touch time   the case's contact stamp (case.contact_frac): per hand, contact fraction × window length, summed.
Put-down     the last frame is a put-down place only when the object has stopped (moved < PUT_STILL_M over the last
             PUT_STILL_S — end_still) and is either low (within PUT_LOW_M of its own lowest height) or high and released
             (floor ≥ desk + PUT_HIGH_M and not held over the end window — the same end window and "holds" as the end
             pedestal). The stop time is counted on the **pre-retime clock** (end_window_start), so the judge (before
             retiming) and the audit (after) see the same end place.
Measure      footprint = outline samples (static_fixed._outline — hull vertices + edge samples). Overlap = a sample
             inside the robot disc (static_fixed.keepout_hit); off desk = the outline area outside the desk box
             (static_desk.overhang_frac — exact, invariant under rigid motion). The outline is taken once in capture
             coordinates before the placement and moved rigidly per candidate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TAG = "[PLACE_RULE]"
#: A still object touched for less than this (s) that receives no main object is "grazed".
GRAZE_MAX_S = 0.3
#: Put-down place (last frame) — "low": the object's reference point is within this (m) of its own lowest height.
PUT_LOW_M = 0.02
#: Put-down place (last frame) — "high": a not-low end is a put-down place only when its floor is at least this (m)
#: above the desk and the hand has released it. Differs from the end pedestal's coords.HIGH_REST_M (1 cm).
PUT_HIGH_M = 0.02
#: The object counts as put down only if its reference point moved less than PUT_STILL_M (m) over the last
#: PUT_STILL_S (s).
PUT_STILL_M = 0.005
PUT_STILL_S = 0.2
#: Tolerance (m · quaternion components) under which two poses of a frame count as the same (seen once).
SAME_POSE_TOL = 1e-3
#: Names of the violated conditions (reject reason · spawn-drop reason · stale reason).
KEEPOUT, OFF_DESK = "robot_keepout", "off_desk"
WHYS = (KEEPOUT, OFF_DESK)
#: Placement roles:
#:   aug   = a placement candidate (default — judge · augmentation cells · spawn drop · staleness): both conditions
#:   canon = the canonical cell of a judge cell group (the original placement): exempt from OFF_DESK only ·
#:           KEEPOUT still applies
ROLES = ("aug", "canon")
#: Kinds of put-down place.
START, REST, END = "start", "rest", "end"
#: Stale-reason prefix (dexengine.placement.train_ready --stale-out).
STALE_REASON = "D48:placed"


class PlaceRuleError(RuntimeError):
    """The rule cannot be applied — never silently read as "no violation"."""


def _coords():
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — robot spot · desk box · placement formula

    return C


def touch_s(ti: dict, name: str) -> float:
    """Seconds a hand touched still object `name` — the case contact stamp case.contact_frac[object][hand] (fraction
    of frames) × the window length (case.T / case.fps), summed over hands (time both hands touch counts twice — this
    narrows what is read as grazed). No stamp → raises."""
    case = ti.get("case") or {}
    cf = case.get("contact_frac")
    if not isinstance(cf, dict) or not case.get("T") or not case.get("fps"):
        raise PlaceRuleError(
            f"{ti.get('clip_id')}: no case.contact_frac · T · fps stamp — without the touch time "
            f"grazed objects cannot be told apart"
        )
    return sum(float(v) for v in (cf.get(name) or {}).values()) * float(case["T"]) / float(case["fps"])


def grazed(seconds: float, receives: bool) -> bool:
    """A grazed still object — touched for less than GRAZE_MAX_S and receiving no main object (a support object
    receives by statement)."""
    return not receives and float(seconds) < GRAZE_MAX_S


def end_window_start(tree_dir, T: int, back: int) -> int:
    """The tree frame `back` frames (PUT_STILL_S) before the last frame, counted on the **pre-retime clock**. A baked
    tree has a retime map (`retime_src_idx` in the tree trajectory: tree frame → fractional pre-retime frame): the
    first tree frame `back` frames before the last on that map. A tree without one (the judge's shadow · base):
    `last - back`. Why: the bake slows a clip down so the robot can follow — on the tree clock the same motion would be
    "still moving" for the judge and "stopped" for the audit. A map whose length differs from the trajectory raises."""
    last = int(T) - 1
    f = Path(tree_dir) / "0" / "trajectory_kinematic.npz"
    if not f.is_file():
        raise PlaceRuleError(
            f"{tree_dir}: no tree trajectory file — the end place's clock (retime map or not) is unknown"
        )
    with np.load(f) as z:
        src = np.asarray(z["retime_src_idx"], float) if "retime_src_idx" in z.files else None
    if src is None:
        return max(0, last - int(back))
    if len(src) != int(T):
        raise PlaceRuleError(f"{tree_dir}: retime map {len(src)} frames ≠ trajectory {T} frames")
    return int(min(last, np.searchsorted(src, float(src[last]) - float(back), side="left")))


def still_frames(tree_dir, ti: dict) -> int:
    """Frames of the end stop window (PUT_STILL_S) at task_info.ref_dt. Missing → raises."""
    dt = float(ti.get("ref_dt") or 0.0)
    if not dt > 0.0:
        raise PlaceRuleError(f"{tree_dir}: task_info.ref_dt is missing — cannot measure the stop time of a put-down")
    return max(1, int(round(PUT_STILL_S / dt)))


def still_start(tree_dir, ti: dict, T: int) -> int:
    """First tree frame of the end stop window — still_frames counted back on the pre-retime clock
    (end_window_start). Read by put_frames (END) and coords.end_window (end pedestals)."""
    return end_window_start(tree_dir, T, still_frames(tree_dir, ti))


def end_still(q, s: int, f_still: int) -> bool:
    """ "Still at the end": the object origin (qpos columns s..s+2) moved less than PUT_STILL_M between the end stop
    window's first frame (`f_still` = still_start) and the last frame. Read by put_frames (END) and
    coords.end_pedestals."""
    return float(np.linalg.norm(q[len(q) - 1, s : s + 3] - q[int(f_still), s : s + 3])) < PUT_STILL_M


def put_frames(tree_dir, ti: dict, q) -> list:
    """[(main object, tree frame, kind)] — first place (START · frame 0) · middle of each rest interval (REST) · last
    frame (END — only when stopped and low or released, see the module docstring). The same pose is seen once. The
    frames come from coords.main_rest_frames."""
    C = _coords()
    last = len(q) - 1
    f_still = still_start(tree_dir, ti, len(q))
    out, seen, end = [], {}, None
    for o, f in sorted(C.main_rest_frames(tree_dir, ti, q), key=lambda t: (str(t[0]["name"]), t[1])):
        s = int(o["qpos_start"])
        if f == 0:
            kind = START
        elif f == last:
            z = q[:, s + 2]
            if not end_still(q, s, f_still):
                continue  # still moving at the end — not a put-down place
            if float(z[last]) > float(z.min()) + PUT_LOW_M:  # not low — only a high, released end counts
                if end is None:  # (placed vertices · end window · desk plane — as for the end pedestal)
                    end = (
                        C._placers(tree_dir, ti, q)[0],
                        C.end_window(tree_dir, ti, q),
                        float(C._qpos_layout_mod().desk_z_of(ti, str(tree_dir))[0]),
                    )
                placed, win, dz = end
                if float(placed(o, last)[:, 2].min()) - dz < PUT_HIGH_M or C._held_in_interval(
                    tree_dir, ti, str(o["name"]), win[1], win[2] - 1
                ):
                    continue  # not high · or a hand still holds it up
            kind = END
        else:
            kind = REST
        pose = np.asarray(q[f, s : s + 7], float)
        if any(np.abs(pose - p).max() < SAME_POSE_TOL for p in seen.get(o["name"], ())):
            continue
        seen.setdefault(o["name"], []).append(pose)
        out.append((o, int(f), kind))
    return out


def footprint(V_xy) -> np.ndarray:
    """Vertex xy (N,2) → footprint (outline samples (M,2) — module docstring "Measure")."""
    from dexengine.placement.scene import static_fixed as SF  # noqa: PLC0415 — outline samples · robot-spot overlap

    return SF._outline(V_xy)


def hit(outline_cap, place, keepout, box, role: str = "aug") -> str:
    """The condition one footprint violates under a placement (capture dx · dy · yaw · pivot — the bake's placement
    formula · None = the tree's own place) → KEEPOUT (overlaps the robot spot) · OFF_DESK (off-desk area > threshold)
    · '' (passes). The judge's final check, the spawn drop and the staleness all use this (the judge mask's grids
    static_fixed.grid_keepout_ok · static_desk.grid_overhang_ok pre-filter it). role ∈ ROLES — canon skips OFF_DESK
    (KEEPOUT still applies · an unknown role raises)."""
    if role not in ROLES:
        raise PlaceRuleError(f"placement role {role!r} — only {ROLES}")
    from dexengine.placement.scene import static_desk as SD  # noqa: PLC0415 — off-desk measure · threshold
    from dexengine.placement.scene import static_fixed as SF  # noqa: PLC0415

    C = _coords()
    xy = C.capture_to_scene(outline_cap if place is None else C.place_objects(outline_cap, *place))
    if SF.keepout_hit(xy, keepout):
        return KEEPOUT
    if role == "canon":
        return ""
    return OFF_DESK if SD.overhang_frac(xy, box=box) > SD.OVERHANG_MAX_FRAC else ""


def in_scope(tree_dir, ti: dict) -> bool:
    """A desk scene (an aerial scene has no desk and no robot-side edge — out of the rule)."""
    return not (_coords().scene_mode(tree_dir) == "aerial" or ti.get("scene") == "aerial")


def clip(tree_dir, ti: dict | None = None, q=None, arm: str | None = None, margin_m: float = 0.0) -> dict:
    """The put-down rule context of one tree (pre-placement shadow · baked tree) → {on, placed, keepout, box,
    mains: [{name, frame, kind, outline}] (capture xy outline), fixed: [static_fixed.columns rows — must · touch_s ·
    xy_cap (outline)], dropped: [(name, reason)]}. fixed = the fixed objects the env spawns (coords.fixed_spawned) — on
    a baked tree the grazed still extras that violated the rule there are already out (extra_slots drops them by
    drop_why) and their names come in `dropped` (the env list's statement, not re-measured). Aerial scenes: on False,
    empty lists. arm = the arm that decides the robot spot (the judge passes its `--arm`; else coords.robot_keepout's
    order). margin_m = extra radius (m) added to the robot disc — only the reselect judge passes it; the post-bake
    check and the audit use 0 (the rule)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415 — the clip rows
    from dexengine.placement.scene import static_fixed as SF  # noqa: PLC0415 — fixed object rows

    if not (np.isfinite(margin_m) and margin_m >= 0):
        raise PlaceRuleError(f"{tree_dir}: robot spot margin {margin_m!r} is not a finite value ≥ 0")
    C = _coords()
    tree_dir = Path(tree_dir)
    if ti is None:
        ti = json.loads((tree_dir / "task_info.json").read_text(encoding="utf-8"))
    out = {
        "on": False,
        "placed": C.is_baked_tree(ti),
        "keepout": None,
        "box": None,
        "mains": [],
        "fixed": [],
        "dropped": [],
    }
    if not in_scope(tree_dir, ti):
        return out
    objs = [o for o in ti.get("objects") or [] if o.get("mesh_dir") and o.get("qpos_start") is not None]
    if not objs:
        return out
    if q is None:
        q = np.load(tree_dir / "0/trajectory_kinematic.npz")["qpos"]
    placed, _ = C._placers(tree_dir, ti, q)
    cx, cy, r = C.robot_keepout(tree_dir, arm)
    out.update(on=True, keepout=(cx, cy, r + float(margin_m)), box=C.desk_bounds_scene_for(tree_dir, margin=0.0))
    for o, f, kind in put_frames(tree_dir, ti, q):
        out["mains"].append(
            {"name": str(o["name"]), "frame": f, "kind": kind, "outline": footprint(placed(o, f)[:, :2])}
        )
    if any(o.get("role") != "main" for o in objs):
        dz = float(C._qpos_layout_mod().desk_z_of(ti, str(tree_dir))[0])
        mains_v = [placed(o, 0) for o in objs if o.get("role") == "main"]
        rows = XS.clip_rows(str(tree_dir), ti, q, strict=False)  # once: the fixed rows and the drops read it
        out["fixed"] = SF.columns(tree_dir, ti, q, dz, mains_v, rows, receives=True)
        out["dropped"] = spawn_dropped(rows)
    return out


def spawn_dropped(rows: list) -> list:
    """The grazed still extras this rule dropped from the spawn on a tree [(name, reason)] — read from its env list
    `rows` (extra_slots.clip_rows: STATIC ∧ reason ∈ WHYS). A pre-placement tree gives []."""
    return [(str(r["name"]), str(r["why"])) for r in rows if r.get("cls") == "STATIC" and r.get("why") in WHYS]


def verdict(pc: dict, place=None) -> tuple:
    """The verdict at that placement → (reject reason, grazed objects to drop [(name, reason)]). Reject reason = the
    first condition violated by an object that must be placed, `<main|fixed>:<object>@<kind|f0>:<KEEPOUT|OFF_DESK>`,
    else ''. A violating grazed still object leaves the placement as is and goes on the drop list."""
    if not pc["on"]:
        return "", []
    for m in pc["mains"]:
        if why := hit(m["outline"], place, pc["keepout"], pc["box"]):
            return f"main:{m['name']}@{m['kind']}:{why}", []
    drops = list(pc["dropped"])  # baked tree: already dropped by the env list (that place's statement)
    for o in pc["fixed"]:
        why = hit(o["xy_cap"], place, pc["keepout"], pc["box"])
        if why and o["must"]:
            return f"fixed:{o['name']}@f0:{why}", []
        if why:
            drops.append((str(o["name"]), why))
    return "", drops


def drop_why(tree_dir, ti: dict, q, o: dict, V=None) -> str:
    """Why still spawn candidate `o` (a still extra of objects[]) is dropped from the spawn on a baked desk tree —
    KEEPOUT · OFF_DESK when it is grazed and violates a condition at that tree's place, else ''. A pre-placement tree
    (no scene place) · aerial · no mesh → ''. Read by extra_slots (the env list) — the same formula as the verdict's
    drop list."""
    C = _coords()
    if not C.is_baked_tree(ti) or not o.get("mesh_dir") or not in_scope(tree_dir, ti):
        return ""
    placed, _ = C._placers(tree_dir, ti, q)
    if V is None:
        V = placed(o, 0)
    why = hit(footprint(V[:, :2]), None, C.robot_keepout(tree_dir), C.desk_bounds_scene_for(tree_dir, margin=0.0))
    if not why:
        return ""
    return why if grazed(touch_s(ti, str(o["name"])), C.receives_main(tree_dir, ti, q, V, placed)) else ""


def keepout_short_m(pc: dict, place=None) -> float:
    """How deep (m) the objects that must be placed (every main-object place · must-place still objects) reach into the
    robot disc — 0 = not at all. The same points (outline samples) and disc as `hit`."""
    if not pc["on"]:
        return 0.0
    C = _coords()
    cx, cy, r = pc["keepout"]
    deep = 0.0
    for oc in [m["outline"] for m in pc["mains"]] + [o["xy_cap"] for o in pc["fixed"] if o["must"]]:
        xy = C.capture_to_scene(oc if place is None else C.place_objects(oc, *place))
        deep = max(deep, float(r - np.hypot(xy[:, 0] - cx, xy[:, 1] - cy).min()))
    return deep


def baked_check(tree_dir, ti: dict | None = None, arm: str | None = None) -> tuple:
    """The verdict on one baked tree → (reject reason | '', depth into the robot spot m — 0 when it passes). **The one
    post-bake check**: the bake chain's final check (--bake-check) and the audit's staleness (stale_reason →
    baked_reject) both call it. Margin 0 (the rule). arm = the arm that decides the robot spot (the bake passes its
    own; else clip's order)."""
    pc = clip(tree_dir, ti, arm=arm)
    why, _drops = verdict(pc)
    return why, (keepout_short_m(pc) if why else 0.0)


def baked_reject(tree_dir, ti: dict | None = None, arm: str | None = None) -> str:
    """Only the reject reason of baked_check (the audit calls it)."""
    return baked_check(tree_dir, ti, arm=arm)[0]


def stale_reason(tree_dir, ti: dict | None = None) -> str:
    """dexengine.placement.train_ready --stale-out reason — `<STALE_REASON>:<verdict reason>` when a baked tree violates
    the rule (baked_reject — a placement the judge now rejects → rejudge), else ''."""
    why = baked_reject(tree_dir, ti)
    return f"{STALE_REASON}:{why}" if why else ""


def marker_clip(clip_name: str, pc: dict) -> str:
    """One line per clip — what the rule looks at (printed even when 0)."""
    if not pc["on"]:
        return f"{TAG} {clip_name} not a desk scene — no check"
    kinds = {k: sum(m["kind"] == k for m in pc["mains"]) for k in (START, REST, END)}
    must = [o["name"] for o in pc["fixed"] if o["must"]]
    graze = [o["name"] for o in pc["fixed"] if not o["must"]]
    gone = [f"{n}:{w}" for n, w in pc["dropped"]]
    return (
        f"{TAG} {clip_name} main-object places {len(pc['mains'])} (start {kinds[START]} · rest {kinds[REST]} · end "
        f"{kinds[END]}) · must-place still objects {len(must)} ({','.join(must) or '-'}) · grazed still objects "
        f"{len(graze)} ({','.join(graze) or '-'}) · grazed objects dropped from the spawn {len(gone)} "
        f"({','.join(gone) or '-'}) · robot-spot overlap 0 · off-desk area ≤ threshold · grazed < {GRAZE_MAX_S} s"
    )
