"""Rule 1d, the f0 frame-0 separation: a dynamic main object that starts inside a body it collides with in training is
moved out of it by the least translation that frees the pair, over the f0 object repair's window and fade.

WHY. A main object recorded inside a collision-on body (a receiver it rests in, a touched still that passed the touch
gate, an envelope still, a support) starts with a PhysX penetration that pushes it out at the spawn hold's release. The
touch gate keeps such a pair colliding only when the recorded overlap is at most TRUE_SHAPE_OVERLAP_M. This rule takes
the start overlap out, and physics handles the rest.

THE RULE (per tree, at frame 0; on with the f0 object repair, repair_obj.obj_repair_on)
  bodies    every role=main object (dynamic: its cooked parts, or an articulated object's links, touch_gate.body_of),
            every role=support object (pinned at frame 0) and every extra row the training scene spawns whose objects
            cell is ON (collision_presets.extra_collide(kind, full=False, ds)["objects"]): a still at its frame-0 pose,
            a slot at each frame's pose, each in the shape physics collides (collider_shape.row_body · body)
  pair      a main object and a body that is not dynamic in training, deeper than CLEAR_TOL_M into each other at
            frame 0 (collider_shape.pair_depth, read both ways). Not a pair whose two bodies move together over the
            whole clip (data/objects/dyn_pairs.together, class EXTRA): S-dyn-pairs makes it one body (the still rides
            the main object) or drops it (together_no_path), so this rule never moves it
  main pair two dynamic bodies (main ↔ main) deeper than CLEAR_TOL_M at frame 0 (dyn_pairs.f0_pairs) that do not
            move together (class MAIN): separated here, it keeps its collision. The mover is a rigid one of the two
            whose move makes no new overlap (`_new_overlaps`), the smaller move first, then name order. A main pair
            no such move frees (deeper than CAP_M, no move up to CAP_M, both articulated, every move makes a new
            overlap) is KEEP_OFF: S-dyn-pairs keeps it off (and its visual drop); no hold, no drop, not in the key
  move      the least translation of the main object along hand_clear.DIRS, in STEP_M steps up to CAP_M, after which
            the pair is at most CLEAR_TOL_M deep. At the least step, the direction with the smallest remaining depth
            wins (then DIRS order). The move holds over frames 0 … window_frames()−1 and fades to 0 over OBJ_FADE_FRAMES
            (repair_obj: the same window and fade as the hand). Kinematic bodies never move
  verdict   per tree, the worst of its pairs (drop > hold > separate; no pair or KEEP_OFF only = clean)
    drop      a pair deeper than CAP_M, or one no move up to CAP_M frees
    hold      the rule does not decide which move: the mover is articulated · one mover in two or more pairs · the
              moves make a new overlap (a pair of a moved object with any body, the other main object included,
              deeper than CLEAR_TOL_M and deeper than before, on any moved row) or a deeper desk penetration (f0's
              desk plane)
    separate  every pair has its move and nothing new overlaps
  A body without a measurable shape is not measured, and the marker names it.

OUTPUT. `applied(tree, qpos)` gives the rows every f0 reader sees (repair_side._qpos): the moves added to the main
objects' position columns when the tree separates, else the rows as they are. A tree that already carries its written
move (repair_obj_write: F0_OBJ_WINDOW.json field SEP_FIELD) is read as it is. The writer writes the same rows in place,
and the rows' hash covers them (the judge key). A drop or a hold holds the clip (HOLD_OBJ with reason "1d drop: …" or
"1d hold(<why>): …"): it is not judged, and it is counted. Key: `key_line` (shadow_key, through
repair.obj_acted_part), "" for a tree without a pair. Marker `[F0_SEP]`, one line per tree.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dexengine.data.objects import dyn_pairs as DP
from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import hand_clear as HCL
from dexengine.data.objects import scene_requirement as SR
from dexengine.placement.judge.f0_key import OBJ_WINDOW_MARKER
from dexengine.retarget.f0 import phrases as PH
from dexengine.retarget.f0 import repair_obj as RO
from dexengine.retarget.f0.repair_sep_items import (  # noqa: F401 — moved definitions, re-exported (the same objects)
    Item,
    _depth_within,
    _first,
    bodies,
    depth,
    scene_rows,
)

RULE_ID = "F0-1d-separation"
MARK = "[F0_SEP]"
STEP_M = 0.0005  # step of the move search (m)
CAP_M = SR.TRUE_SHAPE_OVERLAP_M  # the largest move (m): the one true-shape overlap constant
TOL_M = HCL.CLEAR_TOL_M  # a pair at most this deep is free (the hand_clear tolerance)
STEPS = np.arange(1, int(round(CAP_M / STEP_M)) + 1) * STEP_M
SEP_FIELD = "sep"  # F0_OBJ_WINDOW.json field of a written move
CLEAN, SEPARATE, HOLD, DROP = "clean", "separate", "hold", "drop"
APPLIED = "applied"  # the tree carries its written move
KEEP_OFF = "pair_off"  # a main pair this rule does not free: S-dyn-pairs keeps its collision off (module doc)
MAIN_KIND = "main"  # the kind of a main pair's partner
_RANK = {CLEAN: 0, APPLIED: 0, SEPARATE: 1, HOLD: 2, DROP: 3}
_MEMO: dict = {}
DYNAMIC = DP.DYNAMIC_ROLES  # the Item classes dynamic in training (`dynamic`): an Item's class is its role


def dynamic(it: Item) -> bool:
    """Is the body dynamic in training? The main objects are: env_setup.object_mass.object_rigid_props spawns them with
    kinematic_enabled=False, and an articulated main object is a free articulation (env_setup.objects, rule D1:
    fix_root_link=False). Stills, slots and supports are kinematic or pinned."""
    return it.cls in DYNAMIC


def _partner(p: Item) -> bool:
    """Is the body a partner of this rule (a collision-on body that is not dynamic in training)?"""
    return not dynamic(p)


def least_move(m: Item, p: Item, q, poses: dict, d0: float) -> tuple | None:
    """(step m, direction index, depth after m) of the least move of m out of p at frame 0 (module doc), None when no
    move up to CAP_M frees the pair. The depth falls by at most the move (a point's distance to a surface is 1-Lipschitz
    in its position), so the steps below d0 − TOL_M − STEP_M (one step of slack for the ruler's nearest-face search)
    cannot free the pair and are not measured. A direction is free when its depth is at most TOL_M: `_depth_within`
    gives that depth exactly, and stops reading a direction that exceeds it."""
    for s in STEPS[d0 - TOL_M - STEP_M <= STEPS]:
        got = [(_depth_within(m, p, q, poses, 0, TOL_M, ma=s * u), i) for i, u in enumerate(HCL.DIRS)]
        free = [(d, i) for d, i in got if d is not None]
        if free:
            d, i = min(free)
            return float(s), int(i), float(d)
    return None


def _pairs(items: list, q, poses: dict) -> list:
    """[(mover, partner, depth at frame 0)] of the pairs deeper than TOL_M: each main object against every body that
    is not dynamic and does not move together with it (both are S-dyn-pairs', module doc)."""
    out = []
    for m in items:
        if not dynamic(m):
            continue
        for p in items:
            if p is m or not _partner(p) or DP.together(q, m.qs, p.qs, DP.EXTRA):
                continue
            d = depth(m, p, q, poses, 0)
            if d > TOL_M:
                out.append((m, p, d))
    return out


def _judge_pair(m: Item, p: Item, d0: float, q, poses: dict) -> dict:
    """One pair's verdict (module doc): drop · hold (articulated mover) · separate, with its move."""
    row = {
        "mover": m.name,
        "partner": p.name,
        "kind": p.kind,
        "cls": p.cls,
        "side": m.side,
        "d0_mm": round(d0 * 1000.0, 2),
        "digests": [m.digest, p.digest],
    }
    if d0 > CAP_M:
        return {**row, "verdict": DROP, "reason": "cap", "why": f"deeper than {CAP_M * 1000:g} mm"}
    mv = least_move(m, p, q, poses, d0)
    if mv is None:
        return {**row, "verdict": DROP, "reason": "cap", "why": f"no move <= {CAP_M * 1000:g} mm frees it"}
    s, i, d = mv
    row.update(move_m=s, move_mm=round(s * 1000.0, 2), dir=i, after_mm=round(d * 1000.0, 2), qs=m.qs)
    if m.art is not None:
        return {**row, "verdict": HOLD, "reason": "articulated mover", "why": "the mover is articulated"}
    return {**row, "verdict": SEPARATE}


def _main_pairs(items: list, ti: dict, tree: Path, q, poses: dict) -> list:
    """[(a, b, depth at frame 0)] of the main pairs (module doc), by dyn_pairs.f0_pairs over the main objects that are
    measured here (an Item), without the pairs that move together (class MAIN). None on a dataset S-dyn-pairs is off
    for (that rule states those pairs)."""
    from dexengine.data.objects import scene_rules as SC  # noqa: PLC0415 — the per-dataset switch

    if not SC.on(DP.RULE_ID, str(ti["dataset_name"])):
        return []
    by = {it.name: it for it in items if dynamic(it)}
    mains = [o for o in DP.mains_of(ti) if str(o["name"]) in by]
    out = []
    for oa, ob, d0, _b in DP.f0_pairs(str(ti["dataset_name"]), tree, q, mains):
        a, b = by[str(oa["name"])], by[str(ob["name"])]
        if not DP.together(q, a.qs, b.qs, DP.MAIN):
            out.append((a, b, float(d0)))
    return out


def _main_row(m: Item, p: Item, d0: float, **kw) -> dict:
    head = {"kind": MAIN_KIND, "cls": "main", "d0_mm": round(d0 * 1000.0, 2), "mover": m.name, "partner": p.name}
    return {**head, "side": m.side, "digests": [m.digest, p.digest], **kw}


def _judge_main(a: Item, b: Item, d0: float, items: list, q, poses: dict, desk_z: float) -> dict:
    """One main pair's verdict (module doc: main pair): SEPARATE with its mover and move, or KEEP_OFF with why."""
    if d0 > CAP_M:
        return _main_row(a, b, d0, verdict=KEEP_OFF, reason="cap", why=f"deeper than {CAP_M * 1000:g} mm")
    pairs = ((a, b), (b, a))
    cands = sorted(
        (mv[0], m.name, m, p, mv) for m, p in pairs if m.art is None and (mv := least_move(m, p, q, poses, d0))
    )
    if not cands:
        both = a.art is not None and b.art is not None
        why = "both articulated" if both else f"no move <= {CAP_M * 1000:g} mm frees it"
        return _main_row(a, b, d0, verdict=KEEP_OFF, reason="cap", why=why)
    w, news = weights(len(q)), []
    for s, _n, m, p, (_s, i, d) in cands:
        if not (new := _new_overlaps(items, q, poses, {m.name: w[:, None] * (s * HCL.DIRS[i])[None]}, desk_z)):
            after = round(d * 1000.0, 2)
            return _main_row(
                m, p, d0, verdict=SEPARATE, move_m=s, move_mm=round(s * 1000.0, 2), dir=i, after_mm=after, qs=m.qs
            )
        news.append(f"{m.name}: {new[0]}")
    return _main_row(
        a, b, d0, verdict=KEEP_OFF, reason="new overlap", why="every move makes a new overlap: " + " · ".join(news)
    )


def freed_main_pairs(pl: dict) -> set:
    """{frozenset({mover, partner})} of the main pairs a separating plan frees (dyn_pairs.separated reads it)."""
    freed = [p for p in pl["pairs"] if p["kind"] == MAIN_KIND and p["verdict"] == SEPARATE]
    return {frozenset((p["mover"], p["partner"])) for p in freed} if pl["verdict"] == SEPARATE else set()


def _several(pairs: list) -> None:
    """A separable pair whose main object is in two or more pairs is held (the order of the moves is a choice)."""
    seen: dict = {}
    for p in pairs:
        seen[p["mover"]] = seen.get(p["mover"], 0) + 1
    for p in pairs:
        if p["verdict"] == SEPARATE and seen[p["mover"]] > 1:
            p.update(verdict=HOLD, reason="several pairs", why=f"{p['mover']} is in {seen[p['mover']]} pairs")


def weights(T: int) -> np.ndarray:
    """(≤ window + fade,) share of the move per frame: 1 over the window, then the fade to 0 (repair_obj)."""
    n, B = RO.window_frames(), RO.OBJ_FADE_FRAMES
    return np.array([1.0] * n + [1.0 - (k + 1) / (B + 1) for k in range(B)])[:T]


def _lowest(m: Item, q, poses: dict, f: int, move=None) -> float:
    return min(float((r.points @ M[:3, :3].T + M[:3, 3])[:, 2].min()) for r, M in m.world(q, poses[m.name], f, move))


def _new_overlaps(items: list, q, poses: dict, rows: dict, desk_z: float) -> list:
    """['what'] of the pairs and desk penetrations the moves make (module doc: hold, main pair), over every moved
    row."""
    by = {it.name: it for it in items}
    new = []
    for mn, mrows in sorted(rows.items()):
        m = by[mn]
        for b in items:
            if b is m or (b.name in rows and b.name < mn):
                continue
            for f in range(len(mrows)):
                mb = rows[b.name][f] if b.name in rows else None
                d1 = depth(m, b, q, poses, f, mrows[f], mb)
                if d1 > TOL_M and d1 > depth(m, b, q, poses, f):
                    new.append(f"{mn}x{b.name}({b.kind}) frame {f} {d1 * 1000:.2f}mm")
                    break
        for f in range(len(mrows)):
            z1 = _lowest(m, q, poses, f, mrows[f])
            if desk_z - z1 > TOL_M and z1 < _lowest(m, q, poses, f):
                new.append(f"{mn}xdesk frame {f} {(desk_z - z1) * 1000:.2f}mm")
                break
    return new


def _written(tree: Path) -> bool:
    mk = Path(tree) / OBJ_WINDOW_MARKER
    return mk.is_file() and SEP_FIELD in json.loads(mk.read_text())


def _unmeasured(tree: Path, ti: dict, on: bool) -> tuple | None:
    """(verdict, why) of a tree the rule does not measure (switch off · move already written · legacy · no pair
    possible), else None."""
    if not on:
        return CLEAN, "switch off (DEXENGINE_F0_OBJ_REPAIR=0)"
    if _written(tree):
        return APPLIED, f"the tree carries its move ({OBJ_WINDOW_MARKER} {SEP_FIELD})"
    if not isinstance(ti.get("objects"), list):
        return CLEAN, "legacy tree (no objects[])"
    posed = [o for o in ti["objects"] if o.get("qpos_start") is not None]
    if len(posed) < 2 or not any(o.get("role") == "main" for o in posed):
        return CLEAN, "no pair possible (fewer than two posed objects, or no main object)"
    return None


def _plan(tree: Path, ti: dict, q: np.ndarray, on: bool) -> dict:
    out = {"tree": tree.name, "verdict": CLEAN, "why": "", "pairs": [], "lost": [], "new": [], "moves": {}}
    skip = _unmeasured(tree, ti, on)
    if skip is not None:
        return {**out, "verdict": skip[0], "why": skip[1]}
    from dexengine.placement.coords.core import desk_height, scene_z_offset  # noqa: PLC0415 — f0's desk plane

    items, lost = bodies(tree, ti, q, str(ti["dataset_name"]))
    poses = {it.name: XG.pose_mats(q, it.qs) for it in items}
    desk_z = desk_height() - scene_z_offset(tree)
    pairs = [_judge_pair(m, p, d, q, poses) for m, p, d in _pairs(items, q, poses)]
    pairs += [_judge_main(a, b, d, items, q, poses, desk_z) for a, b, d in _main_pairs(items, ti, tree, q, poses)]
    acting = [p for p in pairs if p["verdict"] != KEEP_OFF]
    _several(acting)
    out.update(pairs=pairs, lost=lost, verdict=max((p["verdict"] for p in acting), key=_RANK.get, default=CLEAN))
    if out["verdict"] != SEPARATE:
        return out
    w = weights(len(q))
    rows = {p["mover"]: w[:, None] * (p["move_m"] * HCL.DIRS[p["dir"]])[None] for p in acting}
    new = _new_overlaps(items, q, poses, rows, desk_z)
    if new:
        return {**out, "verdict": HOLD, "new": new}
    moves = {p["mover"]: {"qs": p["qs"], "rows": rows[p["mover"]], "partner": p["partner"]} for p in acting}
    return {**out, "moves": moves}


def plan(tree, q: np.ndarray | None = None) -> dict:
    """The tree's separation (module doc), memoized on the tree path and the content of its qpos rows and task_info:
    {verdict, why, pairs, lost, new, moves {mover: {qs, rows (F,3), partner}}}."""
    tree = Path(tree)
    if q is None:
        q = np.load(tree / "0/trajectory_kinematic.npz", allow_pickle=True)["qpos"]
    ti_b = (tree / "task_info.json").read_bytes()
    on = RO.obj_repair_on()
    key = (str(tree), hashlib.sha1(np.ascontiguousarray(q).tobytes()).hexdigest(), hashlib.sha1(ti_b).hexdigest(), on)
    if key not in _MEMO:
        if len(_MEMO) >= 256:
            _MEMO.clear()
        _MEMO[key] = _plan(tree, json.loads(ti_b), np.asarray(q), on)
    return _MEMO[key]


def applied(tree, q: np.ndarray) -> np.ndarray:
    """The qpos rows every f0 reader sees: `q` with the moves of a separating tree, else `q` itself."""
    pl = plan(tree, q)
    if pl["verdict"] != SEPARATE:
        return q
    out = np.array(q, dtype=np.float64, copy=True)
    for mv in pl["moves"].values():
        F = len(mv["rows"])
        out[:F, mv["qs"] : mv["qs"] + 3] += mv["rows"]
    return out


def moved(pl: dict) -> dict | None:
    """The moves to write ({mover: {qs, rows, partner}}) of a separating tree, else None."""
    return pl["moves"] if pl["verdict"] == SEPARATE else None


def held_status(pl: dict) -> str | None:
    """The SKIP_OBJ status of a dropped or held tree (repair_obj.refuse's form; shadow_tree.held_row reads it), else
    None."""
    v = pl["verdict"]
    if v not in (HOLD, DROP):
        return None
    held = [p for p in pl["pairs"] if p["verdict"] == v]
    why = [f"{p['mover']}x{p['partner']}({p['kind']}) {p['d0_mm']}mm {p['why']}" for p in held]
    why += [f"new overlap {x}" for x in pl["new"]]
    reasons = sorted({p["reason"] for p in held} | ({"new overlap"} if pl["new"] else set()))
    head = "1d drop" if v == DROP else f"1d hold({','.join(reasons)})"
    return f"{PH.SKIP_OBJ}: {head}: " + " · ".join(why)


def held_side(pl: dict, sides: list) -> str:
    """The side a held tree's line names: the first held pair's mover side when checked, else the first side."""
    v = pl["verdict"]
    side = next((p["side"] for p in pl["pairs"] if p["verdict"] == v and p["side"] in sides), None)
    return side or (sides[0] if sides else next((p["side"] for p in pl["pairs"] if p["side"]), "right"))


def marker(seq: str, pl: dict) -> str:
    """The `[F0_SEP]` line of a tree: verdict, each pair (depth, move or reason), new overlaps, bodies not measured."""
    ps = []
    for p in pl["pairs"]:
        tail = f"move {p['move_mm']}mm dir {p['dir']}" if p["verdict"] == SEPARATE else p.get("why", "")
        ps.append(f"{p['mover']}x{p['partner']}({p['kind']}) {p['d0_mm']}mm {p['verdict']}: {tail}")
    extra = (f" new {pl['new']}" if pl["new"] else "") + (f" not measured {pl['lost']}" if pl["lost"] else "")
    why = f" ({pl['why']})" if pl["why"] else ""
    return f"{MARK} {seq} {pl['verdict']}{why} pairs {len(pl['pairs'])} [{' | '.join(ps)}]{extra}"


def policy_line() -> str:
    """The values that decide the separation, as the f0 cache key carries them (the ruler by its identity)."""
    from dexengine.data.objects.sdf_grid import ruler_id  # noqa: PLC0415 — the ruler's identity owner

    return (
        f"{RULE_ID}|step={STEP_M!r}|cap={CAP_M!r}|tol={TOL_M!r}|dirs={len(HCL.DIRS)}|window={RO.window_frames()}"
        f"|fade={RO.OBJ_FADE_FRAMES}|ruler={ruler_id()[:12]}"
    )


def key_line(pl: dict) -> str:
    """The separation's part of the f0 key (repair.obj_acted_part): "" for a tree without a pair, else the policy, the
    verdict and each pair (bodies, digests, verdict, move)."""
    acting = [p for p in pl["pairs"] if p["verdict"] != KEEP_OFF]  # a kept-off main pair changes no row
    if not acting:
        return ""
    ps = ";".join(
        f"{p['mover']}>{p['partner']}:{p['kind']}:{'/'.join(p['digests'])}:{p['verdict']}:{p.get('move_m')}:"
        f"{p.get('dir')}"
        for p in acting
    )
    return f"sep|{policy_line()}|{pl['verdict']}|{ps}|new={len(pl['new'])}"
