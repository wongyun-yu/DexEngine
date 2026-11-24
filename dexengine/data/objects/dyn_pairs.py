"""Rule S-dyn-pairs: frame-0 pairs of a body dynamic in training with a body it collides with.

  dynamic     the role=main objects (DYNAMIC_ROLES, `dynamic`): every other body is kinematic or pinned in training
  pair        two bodies deeper than hand_clear.CLEAR_TOL_M into each other at frame 0 on the shapes physics
              collides: a main object's collision body (touch_gate.body_of) against the other main object's, a still
              row's shape (touch_gate.static_body) or a support's (collider_shape.body) — the f0 1d reading
  together    the second body's pose relative to the first stays within the pair class's values of its frame-0 value
              at every frame (`together`): the two are one rigid body. Two classes: a main object with a
              still extra row or a support (EXTRA: EXTRA_TOGETHER_POS_M · EXTRA_TOGETHER_ROT_DEG) and two main objects
              (MAIN: MAIN_TOGETHER_POS_M · MAIN_TOGETHER_ROT_DEG)

A pair that moves together is one body (② attach, `together_pass`, a row pass of scene_rules.apply):
  still row (objects cell ON) together with one main object: attached to it for the whole clip — row `carry`, kind
              carried_attached, the S-carried path (kinematic at the simulated main pose × the relative pose, no
              collision with the main objects). No 1d move and no drop: a recorded overlap with that main object
              (touch_gate or envelope `overlap_drop`) moves to `together.overlap_drop`.
  no path     riding is not exact, the clip drops (NO_PATH_WORD, NO_PATH reasons): the main object rests on the still
              (carry_class.rests_on: their collision goes off), the still meets another main object (frame-0 overlap
              or a recorded relation), the main's side is idle (no carrier entity), a support partner (not a row), or
              two main objects (both dynamic; a main object rides nothing).
A both-dynamic pair that does not move together (`dyn_pairs`, a statement per tree):
  separate    the F0-1d frame-0 separation frees it (retarget/f0/repair_sep, the owner of the move: one rigid main
              object moved by at most its cap, nothing new overlapping); its collision stays on, no visual reading
  pair_off    any other: that pair's collision is off (collision_presets.dyn_pair_cells); no object moves
  drop        a pair_off pair whose recorded overlap on the visual meshes (dyn_pair_check: deepest frames on the
              collision shapes first) is deeper than scene_requirement.TRUE_SHAPE_OVERLAP_M in some frame
              (dyn_pair_check.DROP_WORD)
  Every pair is measured: a body that cannot be read (no collision body, no visual mesh) raises, there is no hold.
Marker `[DYN_PAIRS]` per tree (`dyn_marker`); the per-run line is the scene statement's (env_setup.dyn_pairs).

After cross repair (`after_cross_repair`, gate J-gate-dynpairs-xr). The statement above reads the base rows; the last
step that moves objects is the stage-2 cross repair (placement/repair/cross_repair: a hand carries its object with the
wrist, up to its cap), and the scene reads the rows after it (`off_pairs` on the tree's rows). The rule is a rule of the
state the data ends in, so the gate reads a pair again on the final rows when cross repair
  turned it off   no frame-0 pair before cross repair (its collision was on) and one after: the deeper-than-the-rule
                  branch — its collision is off (the scene's statement) and it drops when its visual overlap on the
                  final rows is deeper than the true-shape overlap in some frame; it is never separated again
  deepened it     a pair before and after whose collision-shape depth grew by more than XR_DEEPER_TOL_M at some
                  frame cross repair changed: the visual reading on the final rows replaces the exclusion step's
An off pair cross repair did not deepen keeps the exclusion step's reading; a pair cross repair freed keeps its
collision (the scene's statement) and is counted. Marker `[DYN_PAIRS_XR]` per tree that has such a pair.
"""

from __future__ import annotations

import numpy as np

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import touch_gate as TG

RULE_ID = "S-dyn-pairs"
#: Roles dynamic in training: env_setup.object_mass.object_rigid_props spawns every role=main object with
#: kinematic_enabled=False (an articulated main object is a free articulation, rule D1); stills, slots and supports are
#: kinematic or pinned. The one statement of 'dynamic in training' for the data rules (f0 1d reads it).
DYNAMIC_ROLES = ("main",)
#: 'Moves together': the relative pose stays within these of its frame-0 value at every frame (m · °), per pair class.
#: EXTRA (a main object with a still extra row or a support): 5 mm / 5°, the knee of the measured relative motion of
#: those pairs (the pairs with a frame-0 overlap > 1 mm: together 26 at 2/2 · 82 at 5/5 · 89 at 10/10; envelope 18 ·
#: 57 · 61). MAIN (two main objects): 2 mm / 2° (no knee: 0 · 5 · 14 of 730 pairs at 2 · 5 · 10).
EXTRA, MAIN = "extra", "main"
EXTRA_TOGETHER_POS_M = 0.005
EXTRA_TOGETHER_ROT_DEG = 5.0
MAIN_TOGETHER_POS_M = 0.002
MAIN_TOGETHER_ROT_DEG = 2.0
DROP, NO_PATH_V, ATTACHED = "drop", "no_path", "attached"
SEPARATE_V = "separate"  # a both-dynamic pair the F0-1d separation frees (its collision stays on)
#: `after_cross_repair`: why a pair is read again on the final rows, and the growth of a pair's collision-shape depth at
#: a frame (m) that counts as 'cross repair deepened it' — one micrometre, above the float rounding of a re-written row
#: (frames cross repair left equal are not measured).
XR_TURNED_OFF, XR_DEEPENED = "turned off by cross repair", "deepened by cross repair"
XR_DEEPER_TOL_M = 1e-6
NO_PATH_WORD = "together_no_path"  # exclusion reason word of a pair that moves together without an exact one-body path
NO_PATH = {
    "main": "both dynamic: a main object rides nothing (the carry path rides extra slot rows)",
    "rests": "the main object rests on it: riding the main turns their collision off",
    "unread": "no cooked parts to read whether the main object rests on it",
    "other": "it also meets another main object",
    "idle": "the main object's side is idle: no carrier entity (S-carried R1)",
    "support": "a support object is not an extra slot row",
}


def dynamic(o: dict) -> bool:
    """Is task_info object `o` dynamic in training (DYNAMIC_ROLES)?"""
    return o.get("role") in DYNAMIC_ROLES


def together_values(pair: str) -> tuple[float, float]:
    """(m, °) of 'moves together' for pair class `pair` (EXTRA | MAIN); any other class raises."""
    if pair == EXTRA:
        return EXTRA_TOGETHER_POS_M, EXTRA_TOGETHER_ROT_DEG
    if pair == MAIN:
        return MAIN_TOGETHER_POS_M, MAIN_TOGETHER_ROT_DEG
    raise ValueError(f"[DYN_PAIRS] unknown pair class {pair!r} (one of {EXTRA!r}, {MAIN!r})")


def together(q: np.ndarray, sa: int, sb: int, pair: str) -> dict | None:
    """{rel (7: the mean pose of the body at qpos column sb in the frame of the one at sa), dev_m, dev_deg} when that
    relative pose stays within the values of pair class `pair` (`together_values`) of its frame-0 value at every
    frame, else None."""
    pos_m, rot_deg = together_values(pair)
    R = np.linalg.inv(XG.pose_mats(q, sa)) @ XG.pose_mats(q, sb)
    dm, dd = XG.pose_dev(R, R[0])
    if dm.max() > pos_m or dd.max() > rot_deg:
        return None
    rel = [round(v, 6) for v in XG.pose7(XG.mean_pose(R))]
    return {"rel": rel, "dev_m": round(float(dm.max()), 4), "dev_deg": round(float(dd.max()), 2)}


def shape_depth(ba, bb, q: np.ndarray, Ta: np.ndarray, Tb: np.ndarray, f: int = 0) -> float:
    """Depth (m) at frame f of two moving bodies (touch_gate.Body at poses (T,4,4)) on their collision shapes:
    collider_shape.pair_depth over their link pairs, a bounding-sphere test first."""
    best = 0.0
    for ra, La in ba.links(q[f]):
        Ma = Ta[f] @ La
        for rb, Lb in bb.links(q[f]):
            Mb = Tb[f] @ Lb
            if np.linalg.norm(Ma[:3, :3] @ ra.center + Ma[:3, 3] - Mb[:3, :3] @ rb.center - Mb[:3, 3]) <= (
                ra.radius + rb.radius
            ):
                best = max(best, CS.pair_depth(ra, rb, np.linalg.inv(Ma) @ Mb))
    return best


def pinned_depth(ds: str, tree_dir, m: dict, ruler, T0: np.ndarray, q: np.ndarray) -> float:
    """Frame-0 depth (m) of main object `m` into a pinned body (ruler at pose T0; its ruler reads first, the
    touch_gate order); 0 for a main object without a collision body."""
    body, _why = TG.body_of(ds, m, tree_dir)
    if body is None or ruler is None:
        return 0.0
    Tm = XG.pose_mats(q, int(m["qpos_start"]))[0]
    inv = np.linalg.inv(T0)
    return max(CS.pair_depth(ruler, sh, inv @ Tm @ L) for sh, L in body.links(q[0]))


def mains_of(ti: dict) -> list:
    """The dynamic objects of a tree that have a mesh and a pose column."""
    return [o for o in ti.get("objects") or [] if dynamic(o) and o.get("mesh_dir") and o.get("qpos_start") is not None]


def _no_path(ds: str, ti: dict, q: np.ndarray, m: dict, o: dict, r: dict, others: bool) -> str:
    """The NO_PATH key of riding still row r (object o) on main object m, '' when riding is exact (module doc);
    `others` = the still overlaps another main object at frame 0."""
    from dexengine.data.objects import carry_class as CC  # noqa: PLC0415 — owner of 'rests on'

    name = str(m["name"])
    rel = {r.get("recv_by"), *((r.get("touch_gate") or {}).get("pairs") or {}), (r.get("envelope") or {}).get("by")}
    if others or rel - {None, name}:
        return "other"
    if m.get("side") in set(ti.get("idle_sides") or []):
        return "idle"
    rests = CC.rests_on(ds, m, o, q)
    return "unread" if rests is None else "rests" if rests else ""


def together_pass(tree_dir, ti: dict, q: np.ndarray, rows: list[dict]) -> None:
    """② attach on the still rows that collide with the main objects (module doc): row `together` = {main, f0_m,
    verdict attached | no_path, why}; attached rows get `carry` and kind carried_attached."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.data.objects import hand_clear as HCL  # noqa: PLC0415 — the free-pair tolerance
    from dexengine.rl.tasks.scene import collision_presets as CP  # noqa: PLC0415 — the objects cell

    ds = str(ti.get("dataset_name") or "")
    objs = {str(o["name"]): o for o in ti.get("objects") or []}
    mains = mains_of(ti)
    for r in rows:
        o = objs.get(r["name"]) or {}
        if r.get("cls") != XS.STATIC_SPAWN or not r.get("kind") or not o.get("mesh_dir"):
            continue
        if CP.extra_collide(r["kind"], full=False, ds=ds)["objects"] != "ON":
            continue
        T0 = XG.pose_mats(q, int(o["qpos_start"]))[0]
        tog = [(m, t) for m in mains if (t := together(q, int(m["qpos_start"]), int(o["qpos_start"]), EXTRA))]
        ruler = TG.static_body(ds, o, r)[0] if tog else None
        hits = [(m, t, d) for m, t in tog if (d := pinned_depth(ds, tree_dir, m, ruler, T0, q)) > HCL.CLEAR_TOL_M]
        if not hits:
            continue
        m, t, d = hits[0]
        others = any(x is not m and pinned_depth(ds, tree_dir, x, ruler, T0, q) > HCL.CLEAR_TOL_M for x in mains)
        why = _no_path(ds, ti, q, m, o, r, others)
        r["together"] = {"main": str(m["name"]), "f0_m": round(d, 4), "verdict": NO_PATH_V if why else ATTACHED}
        r["together"]["why"] = NO_PATH[why] if why else "one body: rides the main object for the whole clip"
        if not why:
            r["carry"] = {"carrier": str(m["name"]), "carrier_role": "main", "carrier_side": str(m.get("side"))}
            r["carry"].update(start=0, end=len(q) - 1, **t)
            r["kind"] = XS.KIND_ATTACHED
            if (r.get("overlap_drop") or {}).get("partner") == m["name"]:  # one body: its overlap with m drops nothing
                r["together"]["overlap_drop"] = r.pop("overlap_drop")


def _support_pairs(ds: str, tree_dir, ti: dict, q: np.ndarray, mains: list) -> list:
    """No-path entries of the supports that move together with a main object they overlap at frame 0."""
    from dexengine.data.objects import hand_clear as HCL  # noqa: PLC0415
    from dexengine.data.objects import hull as OH  # noqa: PLC0415

    out = []
    for s in (o for o in ti.get("objects") or [] if o.get("role") == "support" and o.get("qpos_start") is not None):
        tog = [m for m in mains if together(q, int(m["qpos_start"]), int(s["qpos_start"]), EXTRA)]
        if not tog:
            continue
        try:
            ruler = CS.body(ds, str(s["obj_id"]), CS.shape_of(ds, CS.SUPPORT, True))
        except OH.HullMissing:
            continue
        T0 = XG.pose_mats(q, int(s["qpos_start"]))[0]
        for m in tog:
            if (d := pinned_depth(ds, tree_dir, m, ruler, T0, q)) > HCL.CLEAR_TOL_M:
                out.append({"a": str(m["name"]), "b": str(s["name"]), "kind": "support", "f0_m": round(d, 4)})
                out[-1].update(verdict=NO_PATH_V, why=NO_PATH["support"])
    return out


def _dyn_pair(ds: str, tree_dir, oa: dict, ob: dict, q: np.ndarray, d0: float, shapes: tuple, freed: set) -> dict:
    """The verdict of one frame-0 pair of two dynamic bodies (module doc); `shapes` = their collision bodies, which
    order the frames the visual reading reads (deepest first; it stops at the first one above the cap); `freed` =
    `separated` of the tree."""
    from dexengine.data.objects import dyn_pair_check as DC  # noqa: PLC0415 — the visual ruler and verdict
    from dexengine.rl.tasks.scene import collision_presets as CP  # noqa: PLC0415 — the pair-off meaning

    sa, sb = int(oa["qpos_start"]), int(ob["qpos_start"])
    p = {"a": str(oa["name"]), "b": str(ob["name"]), "f0_m": round(d0, 4)}
    if together(q, sa, sb, MAIN):
        return {**p, "verdict": NO_PATH_V, "why": NO_PATH["main"]}
    if frozenset((p["a"], p["b"])) in freed:
        return {**p, "verdict": SEPARATE_V, "why": "freed at frame 0 by the F0-1d separation; its collision stays on"}
    reading, v = _visual(ds, tree_dir, oa, ob, q, shapes)
    p.update(reading)
    if DC.verdict(v) == DC.DROP:
        return {**p, "verdict": DROP, "why": "recorded visual overlap above the true-shape overlap"}
    return {**p, "verdict": CP.DYN_PAIR_OFF, "why": "both dynamic, overlapping at frame 0"}


def _visual(ds: str, tree_dir, oa: dict, ob: dict, q: np.ndarray, shapes: tuple) -> tuple[dict, float]:
    """The visual reading of one pair on rows `q` (dyn_pair_check): ({visual_m, visual_frame, visual_f0_m}, the depth
    unrounded — the verdict's input); frames are read deepest first on the collision bodies `shapes` and the reading
    stops at the first one above the cap."""
    from dexengine.data.objects import dyn_pair_check as DC  # noqa: PLC0415 — the visual ruler

    ra, rb = DC.ruler(ds, oa, tree_dir), DC.ruler(ds, ob, tree_dir)
    Ta, Tb = XG.pose_mats(q, int(oa["qpos_start"])), XG.pose_mats(q, int(ob["qpos_start"]))
    v, f = DC.depth_over(ra, rb, q, Ta, Tb, np.argsort(-_depths(shapes, q, oa, ob), kind="stable"))
    f0 = DC.depth_over(ra, rb, q, Ta, Tb, [0])[0]
    return {"visual_m": round(v, 4), "visual_frame": f, "visual_f0_m": round(f0, 4)}, v


def _depths(shapes: tuple, q: np.ndarray, oa: dict, ob: dict) -> np.ndarray:
    """(T,) the pair's depth on its collision bodies `shapes` at every frame of rows `q` (m; 0 = apart)."""
    Ta, Tb = XG.pose_mats(q, int(oa["qpos_start"])), XG.pose_mats(q, int(ob["qpos_start"]))
    return np.asarray([shape_depth(*shapes, q, Ta, Tb, f) for f in range(len(q))])


def f0_pairs(ds: str, tree_dir, q: np.ndarray, mains: list):
    """The frame-0 pairs of the dynamic objects `mains`: (oa, ob, d0, (body a, body b)) for every pair deeper than
    hand_clear.CLEAR_TOL_M at frame 0 on their collision shapes, by name order. A main object without its collision
    body (touch_gate.body_of) raises: its pairs cannot be read. The one pair generator of the statement and scene."""
    from dexengine.data.objects import hand_clear as HCL  # noqa: PLC0415

    got = []
    for o in sorted(mains, key=lambda x: str(x["name"])):
        b, why = TG.body_of(ds, o, tree_dir)
        if b is None:
            raise ValueError(
                f"[DYN_PAIRS] {tree_dir}: {o['name']} has no collision body ({why}) — its pairs cannot be read"
            )
        got.append((o, b, XG.pose_mats(q, int(o["qpos_start"]))))
    for i, (oa, ba, Ta) in enumerate(got):
        for ob, bb, Tb in got[i + 1 :]:
            if (d0 := shape_depth(ba, bb, q, Ta, Tb)) > HCL.CLEAR_TOL_M:
                yield oa, ob, d0, (ba, bb)


def separated(tree_dir, q: np.ndarray) -> set:
    """{frozenset({a, b})} of the both-dynamic pairs the F0-1d frame-0 separation frees in this tree (owner
    retarget/f0/repair_sep: its plan on these rows). Empty when that rule is off, when the tree already carries its
    written move or when the tree does not separate (held, dropped). Read by the exclusion step's statement
    (`dyn_pairs`, before f0 writes the move); the scene reads the rows (`off_pairs`): a written move leaves no pair."""
    from dexengine.retarget.f0 import repair_sep as RS  # noqa: PLC0415 — the owner of the frame-0 move

    return RS.freed_main_pairs(RS.plan(tree_dir, q))


def off_pairs(tree_dir, ti: dict, q: np.ndarray) -> list:
    """[{a, b, f0_m}] of the both-dynamic pairs the scene turns off: `f0_pairs` of these rows that do not move together
    (the frame-0 part of `dyn_pairs` without the visual reading; its drop verdicts are the exclusion step's). The scene
    reads the tree's rows as f0 wrote them: a pair the F0-1d separation freed is no pair there (its move is written),
    and rows without the move keep the pair off. Read by the scene statement (env_setup.dyn_pairs) and the scene
    manifest. [] on a dataset that opts out."""
    from dexengine.data.objects import scene_rules as SC  # noqa: PLC0415

    ds = str(ti.get("dataset_name") or "")
    if not SC.on(RULE_ID, ds):
        return []
    return [
        {"a": str(oa["name"]), "b": str(ob["name"]), "f0_m": round(d0, 4)}
        for oa, ob, d0, _b in _off(ds, tree_dir, q, mains_of(ti))
    ]


def _off(ds: str, tree_dir, q: np.ndarray, mains: list) -> list:
    """The pairs the scene turns off on rows `q`, whole: [(oa, ob, d0, shapes)] — `f0_pairs` that do not move together
    (the one definition `off_pairs` and `after_cross_repair` read)."""
    return [
        (oa, ob, d0, sh)
        for oa, ob, d0, sh in f0_pairs(ds, tree_dir, q, mains)
        if not together(q, int(oa["qpos_start"]), int(ob["qpos_start"]), MAIN)
    ]


def after_cross_repair(tree_dir, ti: dict, q_pre: np.ndarray, q_post: np.ndarray, q_final: np.ndarray) -> dict:
    """S-dyn-pairs on the rows the data ends with (module doc 'After cross repair'; gate J-gate-dynpairs-xr).

    q_pre · q_post = the tree's rows before and after cross repair on one time axis (before retiming); q_final = the
    tree's rows (what the scene reads). Returns {"read": [{a, b, f0_pre_m, f0_m, why, visual_m, visual_frame,
    visual_f0_m, verdict pair_off|drop}], "freed": [{a, b, f0_pre_m}], "drop": (word, text) | None}: `read` = the pairs
    read again on the final rows (why XR_TURNED_OFF | XR_DEEPENED), `freed` = the pairs off before cross repair and
    free after it (their collision is on). A dataset that opts out states nothing. Prints `[DYN_PAIRS_XR]` when a pair
    was read or freed."""
    from dexengine.data.objects import dyn_pair_check as DC  # noqa: PLC0415 — the visual ruler and verdict
    from dexengine.data.objects import scene_rules as SC  # noqa: PLC0415 — the per-dataset switch
    from dexengine.rl.tasks.scene import collision_presets as CP  # noqa: PLC0415 — the pair-off meaning

    st = {"read": [], "freed": [], "drop": None}
    ds = str(ti.get("dataset_name") or "")
    if not SC.on(RULE_ID, ds):
        return st
    if len(q_pre) != len(q_post):
        raise ValueError(
            f"[DYN_PAIRS_XR] {tree_dir}: rows before / after cross repair {len(q_pre)} / {len(q_post)} frames"
        )
    mains = mains_of(ti)
    key = lambda oa, ob: frozenset((str(oa["name"]), str(ob["name"])))  # noqa: E731
    pre = {key(oa, ob): d0 for oa, ob, d0, _sh in _off(ds, tree_dir, q_pre, mains)}
    fin = _off(ds, tree_dir, q_final, mains)
    for oa, ob, d0, sh in fin:
        d_pre = pre[key(oa, ob)] if key(oa, ob) in pre else float(_depths(sh, q_pre[:1], oa, ob)[0])
        p = {"a": str(oa["name"]), "b": str(ob["name"]), "f0_pre_m": round(d_pre, 4)}
        if key(oa, ob) not in pre:
            p["why"] = XR_TURNED_OFF
        elif _deepened(sh, q_pre, q_post, oa, ob):
            p["why"] = XR_DEEPENED
        else:
            continue  # cross repair did not deepen it: the exclusion step's reading stands
        reading, v = _visual(ds, tree_dir, oa, ob, q_final, sh)
        p.update(f0_m=round(d0, 4), **reading, verdict=DROP if DC.verdict(v) == DC.DROP else CP.DYN_PAIR_OFF)
        st["read"].append(p)
    done = {key(oa, ob) for oa, ob, _d, _s in fin}
    st["freed"] = [
        {"a": a, "b": b, "f0_pre_m": round(d, 4)} for k, d in pre.items() if k not in done for a, b in [sorted(k)]
    ]
    got = [
        f"{p['a']}×{p['b']}: {p['why']}, visual overlap above the true-shape overlap"
        for p in st["read"]
        if p["verdict"] == DROP
    ]
    st["drop"] = (DC.DROP_WORD, " · ".join(got)) if got else None
    if line := xr_marker(tree_dir, st):
        print(line, flush=True)
    return st


def _deepened(shapes: tuple, q_pre: np.ndarray, q_post: np.ndarray, oa: dict, ob: dict) -> bool:
    """Did cross repair deepen the pair: its collision-shape depth grew by more than XR_DEEPER_TOL_M at some frame
    (q_pre → q_post, one time axis)? Only the frames cross repair changed are measured (a row equal before and after
    has the same depth); none changed = False."""
    moved = np.flatnonzero((q_pre != q_post).any(axis=1))
    if not len(moved):
        return False
    grew = _depths(shapes, q_post[moved], oa, ob) - _depths(shapes, q_pre[moved], oa, ob)
    return bool(grew.max() > XR_DEEPER_TOL_M)


def xr_marker(tree_dir, st: dict) -> str:
    """`[DYN_PAIRS_XR] <clip>: a×b f0 0.0 → 1.2 mm (turned off by cross repair) · visual ≤ 5.0 mm (frame 40) → pair_off
    · c×d f0 3.1 mm → free (freed by cross repair)` ('' without such a pair)."""
    import os  # noqa: PLC0415

    from dexengine.data.objects import dyn_pair_check as DC  # noqa: PLC0415

    cells = [
        f"{p['a']}×{p['b']} f0 {p['f0_pre_m'] * 1e3:.1f} → {p['f0_m'] * 1e3:.1f} mm ({p['why']}) · visual "
        f"{DC.reading(p['visual_m'])} (frame {p['visual_frame']}) → {p['verdict']}"
        for p in st["read"]
    ] + [f"{p['a']}×{p['b']} f0 {p['f0_pre_m'] * 1e3:.1f} mm → free (freed by cross repair)" for p in st["freed"]]
    clip = os.path.basename(str(tree_dir).rstrip("/"))
    return f"[DYN_PAIRS_XR] {clip}: " + " · ".join(cells) if cells else ""


def dyn_pairs(tree_dir, ti: dict, q: np.ndarray, rows: list) -> dict:
    """The statement of one tree (module doc): {rule, pairs: [{a, b, f0_m, verdict separate|pair_off|drop|no_path, why,
    visual_m, visual_frame, visual_f0_m}], together: [{a, b, kind, f0_m, verdict attached|no_path, why}], drop,
    no_path: (word, text) | None}. `rows` = the tree's extra_slots.clip_rows, the caller's (measured once per tree
    and passed down, never read again here). The visual readings are dyn_pair_check.screened. A dataset that opts out
    states nothing. Prints the marker."""
    from dexengine.data.objects import dyn_pair_check as DC  # noqa: PLC0415
    from dexengine.data.objects import scene_rules as SC  # noqa: PLC0415 — the per-dataset switch

    ds = str(ti.get("dataset_name") or "")
    st = {"rule": RULE_ID, "pairs": [], "together": [], "drop": None, "no_path": None}
    if not SC.on(RULE_ID, ds):
        return st
    mains = mains_of(ti)
    found = list(f0_pairs(ds, tree_dir, q, mains))
    freed = separated(tree_dir, q) if found else set()
    st["pairs"] = [_dyn_pair(ds, tree_dir, oa, ob, q, d0, sh, freed) for oa, ob, d0, sh in found]
    for r in rows:
        if tg := r.get("together"):
            st["together"].append({"a": tg["main"], "b": r["name"], "kind": r["kind"], "f0_m": tg["f0_m"]})
            st["together"][-1].update(verdict=tg["verdict"], why=tg["why"])
    st["together"] += _support_pairs(ds, tree_dir, ti, q, mains)
    every = st["pairs"] + st["together"]
    for v, word, key in ((DROP, DC.DROP_WORD, "drop"), (NO_PATH_V, NO_PATH_WORD, "no_path")):
        got = [f"{p['a']}×{p['b']}: {p['why']}" for p in every if p["verdict"] == v]
        st[key] = (word, " · ".join(got)) if got else None
    if line := dyn_marker(tree_dir, st):
        print(line, flush=True)
    return st


def dyn_marker(tree_dir, st: dict) -> str:
    """`[DYN_PAIRS] <clip>: a×b f0 3.2 mm · visual ≤ 5.0 mm (frame 40) → pair_off · …` ('' without a pair; the visual
    reading as dyn_pair_check.reading)."""
    import os  # noqa: PLC0415

    from dexengine.data.objects import dyn_pair_check as DC  # noqa: PLC0415

    cells = []
    for p in st["pairs"] + st["together"]:
        vis = f" · visual {DC.reading(p['visual_m'])} (frame {p['visual_frame']})" if "visual_m" in p else ""
        cells.append(f"{p['a']}×{p['b']} f0 {p['f0_m'] * 1e3:.1f} mm{vis} → {p['verdict']}")
    clip = os.path.basename(str(tree_dir).rstrip("/"))
    return f"[DYN_PAIRS] {clip}: " + " · ".join(cells) if cells else ""


def scene_pair(tree_dir, ti: dict, q: np.ndarray) -> dict | None:
    """The one pair-off pair a scene cuts, as {a, b, f0_m, sides: {name: side}} (`off_pairs` of the tree), or None.
    The scene cuts its right main object against its left one (collision_dyn_pairs), so more than one pair, or a pair
    that is not one right and one left main object, raises. Read by the scene statement (env_setup.dyn_pairs) and the
    delivered-output statement (record/store/merge_dyn_pairs: the pair's objects are the object_<side> columns)."""
    pairs = off_pairs(tree_dir, ti, q)
    clip = str(tree_dir).rstrip("/").rsplit("/", 1)[-1]
    if len(pairs) > 1:
        raise ValueError(f"[DYN_PAIRS] {clip}: {len(pairs)} pair-off pairs {pairs} — a scene has one per clip")
    if not pairs:
        return None
    side = {str(o["name"]): o.get("side") for o in ti.get("objects") or []}
    p = pairs[0]
    on = [side.get(p["a"]), side.get(p["b"])]
    if set(on) != {"left", "right"}:
        raise ValueError(
            f"[DYN_PAIRS] {clip}: pair {p['a']}×{p['b']} on sides {on} — the scene cuts the right object against the"
            " left one"
        )
    return {**p, "sides": {p["a"]: side[p["a"]], p["b"]: side[p["b"]]}}


#: Exclusion-step row sides (placement/repair/hand_clear_stage ledger): the visual-overlap drops of the both-dynamic
#: pairs, and the together pairs without an exact one-body path; each its own reason word.
PAIR_SIDE, TOGETHER_SIDE = "dyn_pairs", "dyn_together"


def stage_rows(tree_dir, ti: dict, q: np.ndarray, rows: list, qpos_sha: str, version: str, statuses: tuple) -> list:
    """The exclusion step's two rows of one clip (statuses = (keep, drop) of that step; `rows` = its extra_slots rows,
    the attach verdicts on them): side PAIR_SIDE DROP when a both-dynamic pair drops (reason dyn_pair_check
    .DROP_WORD), side TOGETHER_SIDE DROP when a together pair has no path (NO_PATH_WORD); `rows` field = a×b:verdict
    per pair of that side."""
    st = dyn_pairs(tree_dir, ti, q, rows=rows)
    clip = str(tree_dir).rstrip("/").rsplit("/", 1)[-1]
    out = []
    for side, key, ps in ((PAIR_SIDE, "drop", st["pairs"]), (TOGETHER_SIDE, "no_path", st["together"])):
        out.append(
            {
                "clip": clip,
                "side": side,
                "status": statuses[1] if st[key] else statuses[0],
                "rows": " ".join(f"{p['a']}x{p['b']}:{p['verdict']}" for p in ps),
                "qpos_sha": qpos_sha,
                "version": version,
                "why": (st[key] or ("", ""))[1],
            }
        )
    return out
