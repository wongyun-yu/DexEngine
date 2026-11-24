"""The support statement of a tree: compute it (aerial · desk), stamp it into a baked tree's task_info, and read it back
(the one reader every post-bake consumer uses); the tree → robot-world z offset."""

from __future__ import annotations

import numpy as np

from dexengine.placement.coords.contacts import frame_toucher
from dexengine.placement.coords.discs import pedestal_scene, rest_discs
from dexengine.placement.coords.frames import _paths, desk_height, scene_mode
from dexengine.placement.coords.objects import _placers, _qpos_layout_mod, _tree_key, is_baked_tree
from dexengine.placement.coords.pedestals import end_pedestals, extra_rest_pedestals, static_pedestals
from dexengine.placement.coords.support_kind import (
    clip_supports,
    scene_support_poses,
    stands_supported,
    start_support,
    support_object_verts,
)
from dexengine.placement.coords.thresholds import DESK_DISC_KEYS, FLOOR_Z, HIGH_REST_M, PEDESTAL_COLLIDE, PEDESTAL_RULE

#: Computed aerial statements by tree key (path + mtimes).
_AERIAL_CACHE: dict = {}
#: Trees whose "old bake generation, no desk stamp" marker was printed.
_DESK_SUPPORT_SAID: set = set()
#: Trees whose `[DESK_Z]` marker was printed.
_DESK_Z_SAID: set = set()


def compute_aerial_support(
    tree_dir, ti: dict | None = None, q=None, src_idx="tree", height: tuple | None = None
) -> dict:
    """**Compute** the support statement of a deskless (aerial) scene from the tree's data. Callers: the bake stamp
    (`stamp_scene`) and pre-bake consumers (the judge on the f0 shadow tree) only — post-bake consumers read the stamp
    (`read_aerial_support`). Never reads a previous stamp: the `scene`/`aerial_support` keys are ignored.
    Returns:
      z_ref  reference-plane z in capture coordinates (floor of the first rest disc, else the floor of the first main
             object at frame 0 — right hand first) · source "disc"|"obj0"
      z_off  world z offset = desk_height() − z_ref (the reference plane lands at world z = DESK_HEIGHT)
      discs  the rest discs start_support keeps (rest_discs minus the ones a support object holds) + its start discs,
             all with world z added and the stamp fields (kind · collide); every disc stands from reset to the end
      excluded  start-support exclusions (start_support) — the gate reads them from the stamp
    z_ref reads the rest discs **before** the support-object filter (the plane is the first resting floor, whatever
    carries it). `height` = (z_ref, source) decided earlier — a **re-stamp** passes the bake's first stamp: the robot
    was solved against that height; the discs are still computed from this trajectory. A tree's qpos is in that tree's
    own frames (post-placement for a baked tree), so the disc xy are scene coordinates relative to that tree."""
    import json as _json
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    override = ti is not None or q is not None or height is not None
    key = _tree_key(tree_dir)
    if not override and key in _AERIAL_CACHE:
        return _AERIAL_CACHE[key]
    if ti is None:
        ti = _json.loads((tree_dir / "task_info.json").read_text())
    ti = {k: v for k, v in ti.items() if k not in ("scene", "aerial_support")}
    if q is None:
        q = np.load(tree_dir / "0/trajectory_kinematic.npz")["qpos"]
    stats: dict = {}
    rest = rest_discs(tree_dir, ti, q, stats=stats, src_idx=src_idx, require_free_frame=True)
    _placed, _main_for = _placers(tree_dir, ti, q)
    if height is not None:
        z_ref, src = float(height[0]), str(height[1])
    elif rest:
        z_ref, src = rest[0]["bottom_z_capture"], "disc"
    else:
        o = _main_for("right") or _main_for("left")
        if o is None:
            raise ValueError(
                f"{tree_dir}: aerial reference plane — no main object and no rest disc — cannot decide "
                f"the reference plane"
            )
        z_ref, src = float(_placed(o, 0)[:, 2].min()), "obj0"
    sup = start_support(tree_dir, ti, q, rest, aerial=True, src_idx=src_idx)
    if sup["on_support"]:
        stats["on_support"] = len(sup["on_support"])
    discs = sup["rest"] + sup["discs"]
    off = desk_height() - z_ref
    for d in discs:
        d["z"] = d["bottom_z_capture"] + off
    out = {
        "z_ref": z_ref,
        "z_off": off,
        "source": src,
        "discs": discs,
        "n_rest": len(sup["rest"]),
        "n_start": len(sup["discs"]),
        "excluded": sup["excluded"],
        "supports": sup["supports"],
        "skipped": stats,
    }
    if not override:
        _AERIAL_CACHE[key] = out
    return out


def aerial_tail_start(tree_dir, ti: dict, q, objs, t0: int = 0, t1: int | None = None) -> int | None:
    """The earliest frame from which, to the end of the base window [t0, t1), some object in `objs` is neither touched
    (`frame_toucher`) nor carried (`stands_supported` — its own disc of `compute_aerial_support` or a support object);
    None = every object is supported at the window end. Measured on the parent base timeline so a re-cut piece gives
    the same answer. A value equal to t0 means the whole window is unsupported — a tail cut cannot repair it."""
    t1 = len(q) if t1 is None else min(int(t1), len(q))
    discs = compute_aerial_support(tree_dir, ti=ti, q=q)["discs"]
    placed, _ = _placers(tree_dir, ti, q)
    sv = support_object_verts(tree_dir, ti, q)
    by_name = {str(o.get("name")): o for o in ti.get("objects") or []}
    starts = []
    for name in objs:
        touch = frame_toucher(tree_dir, ti, q, str(name), lambda b: b)
        mine = [d for d in discs if str(d.get("obj")) == str(name)]
        k = t1
        carry, o = bool(mine) or bool(sv), by_name[str(name)]
        while k > t0 and not (touch(k - 1) or (carry and stands_supported(placed(o, k - 1), mine, sv))):
            k -= 1
        if k < t1:
            starts.append(k)
    return min(starts) if starts else None


def stamp_supports_match(tree_dir, ti: dict, stamped) -> int:
    """At the stamp: the support objects the stamp carries (the env spawns these) must equal the tree's stated support
    objects (the judge measures these — scene_support_poses). A difference raises: a stated support no main stands on,
    or a desk tree whose stamp placed none (the repair is to spawn the judge's set, never to drop the tree). Returns
    the count."""
    import pathlib as _pl

    want = set(scene_support_poses(ti, np.load(_pl.Path(tree_dir) / "0/trajectory_kinematic.npz")["qpos"][:1]))
    got = {str(s["obj"]) for s in stamped or []}
    if got != want:
        raise ValueError(
            f"{tree_dir}: stamped support objects {sorted(got)} ≠ role-stated supports {sorted(want)} — "
            "the env and the judge would carry different supports; spawn the judge's set in the env "
            "instead of dropping the tree"
        )
    return len(want)


def stamp_scene(tree_dir, keep_height: bool = False) -> dict:
    """Write the **support stamp** into a baked tree — the only writer of the stamp. Every step that writes a tree
    trajectory (the bake · cross repair) calls this right after, so the stamp is always computed from the final
    trajectory.
      table  : the desk supports under `qpos_layout.DESK_SUPPORT_KEY` (an empty list when there are none — no key =
               an old bake generation).
      aerial : the aerial supports (discs · reference plane) and scene=aerial.
    Both carry the exclusion list `excluded` (the gate reads it — start_support_verdict). Computed from the tree's own
    trajectory and task_info (compute_* never read an existing stamp). Returns the written statement.
    keep_height: a **re-stamp** after the bake (retime · coarse path · cross repair) passes True — the aerial height
    (z_ref · z_off) keeps the bake's first stamp (the robot was judged at that height); the discs are recomputed from
    the final trajectory. Without a first stamp it raises."""
    import json as _json
    import pathlib as _pl

    p = _pl.Path(tree_dir) / "task_info.json"
    ti = _json.loads(p.read_text())
    # the scene is the env (DEXENGINE_SCENE) checked against the tree record — a re-stamp with an empty env follows
    # the scene the bake decided
    if scene_mode(tree_dir) != "aerial":
        dsup = compute_desk_support(tree_dir, ti=ti)
        n_sup = stamp_supports_match(tree_dir, ti, dsup["supports"])
        ti[desk_support_key()] = dsup
        p.write_text(_json.dumps(ti, indent=1))
        print(
            f"[DESK_SUPPORT] stamped {_pl.Path(tree_dir).name}: discs {dsup['n']} · support objects "
            f"{len(dsup['supports'])} · excluded={len(dsup['excluded'])} · supports = role statement {n_sup}",
            flush=True,
        )
        return dsup
    height = None
    if keep_height:
        prev = ti.get("aerial_support")
        if not isinstance(prev, dict) or prev.get("z_ref") is None:
            raise ValueError(
                f"{tree_dir}: a height-keeping re-stamp but no first stamp (aerial_support.z_ref) — the "
                f"bake stamp comes first"
            )
        height = (float(prev["z_ref"]), str(prev.get("source") or ""))
    sup = compute_aerial_support(tree_dir, ti=ti, height=height)
    n_sup = stamp_supports_match(tree_dir, ti, sup["supports"])
    ti["scene"] = "aerial"
    ti["aerial_support"] = {
        "z_ref": sup["z_ref"],
        "z_off": sup["z_off"],
        "source": sup["source"],
        "discs": sup["discs"],
        "excluded": sup["excluded"],
        "supports": sup["supports"],
        "skipped": sup.get("skipped") or {},
    }
    p.write_text(_json.dumps(ti, indent=1))
    print(
        f"[SCENE] stamped {_pl.Path(tree_dir).name}: scene=aerial discs={len(sup['discs'])} (rest {sup['n_rest']} · "
        f"start {sup['n_start']}) support objects {len(sup['supports'])} z_ref={sup['z_ref']:+.4f}"
        f"({sup['source']}{' · height kept' if keep_height else ''}) excluded={len(sup['excluded'])} "
        f"supports = role statement {n_sup}",
        flush=True,
    )
    return ti["aerial_support"]


def read_aerial_support(tree_dir) -> dict:
    """**Read** the stamped aerial support statement (`task_info.aerial_support`, written by `stamp_scene`) — the
    numbers the env spawns. Read through the env's own reader (`qpos_layout.read(...).aerial`, uncached — a re-bake of
    the same path must not serve old numbers), so placement and env cannot read two different things.
    No stamp → raises: a recompute here would differ from the env."""
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    lay = _qpos_layout_mod().read(str(tree_dir), use_cache=False)
    a = lay.aerial if getattr(lay, "scene", "table") == "aerial" else None
    if not a or a.get("z_off") is None or a.get("z_ref") is None:
        raise ValueError(
            f"{tree_dir}: a baked aerial tree without the task_info.aerial_support stamp — a different "
            f"bake generation; not replaced by a recompute (it would differ from the env)"
        )
    discs = list(a.get("discs") or [])
    return {
        "z_ref": float(a["z_ref"]),
        "z_off": float(a["z_off"]),
        "source": str(a.get("source")),
        "discs": discs,
        "n_rest": sum(1 for d in discs if d.get("kind") != "start"),
        "n_start": sum(1 for d in discs if d.get("kind") == "start"),
        "excluded": a.get("excluded"),
        "supports": a.get("supports"),
        "stamped": True,
    }


def aerial_support(tree_dir) -> dict:
    """The aerial support statement for any consumer — one dispatch rule:
    stamped tree (task_info.scene == "aerial")          → read_aerial_support (the numbers the env uses)
    baked tree (placement_v2) without the stamp          → raises (a different bake generation)
    pre-bake tree (base layer, f0 shadow staging)        → compute_aerial_support"""
    import json as _json
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    ti = _json.loads((tree_dir / "task_info.json").read_text())
    if ti.get("scene") == "aerial":
        return read_aerial_support(tree_dir)
    if is_baked_tree(ti):
        raise ValueError(
            f"{tree_dir}: a baked tree asked as an aerial scene without the scene/aerial_support stamp — "
            f"a different bake generation (no recompute)"
        )
    return compute_aerial_support(tree_dir)


def desk_support_key() -> str:
    """The task_info key of the desk-scene support statement (`qpos_layout.DESK_SUPPORT_KEY`, the env reader's)."""
    return str(_qpos_layout_mod().DESK_SUPPORT_KEY)


def _desk_discs(sup, endp, xped, off):
    """The desk-scene disc rows: every support and pedestal disc at its desk-frame height, as a column from what it
    stands on (desk top or floor)."""
    discs = []
    foot = {"desk": desk_height(), "floor": FLOOR_Z}
    for d in sup["rest"] + sup["discs"] + endp + xped:
        z = d["bottom_z_capture"] + off
        row = {k: d[k] for k in DESK_DISC_KEYS if k not in ("z", "collide")}
        # a pedestal: a column from what it stands on (the desk top · the floor off the desk box) up to the object's
        # floor — what it collides with is its kind's (PEDESTAL_COLLIDE)
        base = d.get("base", "desk")
        discs.append(
            row | {"z": z, "h": round(z - foot[base], 4), "base": base, "collide": PEDESTAL_COLLIDE[d["kind"]]}
        )
    return discs


def compute_desk_support(tree_dir, ti: dict | None = None, q=None, stats: dict | None = None) -> dict:
    """**Compute** the desk-scene support statement — only the bake stamp (`stamp_scene`), the staleness check and
    audits call it. An existing stamp (the key) is never read.
      rest discs     rest_discs(require_unheld) whose declared floor is at least HIGH_REST_M above the desk, minus
                     the ones a support object carries (start_support → rest_on_support). Discs stand from reset to
                     the end.
      start supports start_support(aerial=False) — main objects that start unheld at least HIGH_REST_M above the desk.
      pedestals      in the pedestal rule's scope only (pedestal_scene): fixed-object pedestals, moved-extra rest
                     pedestals and end pedestals (end_pedestals); out of scope the objects-only discs.
    Desk scenes only (an aerial tree raises). World z = capture floor + (DESK_HEIGHT − desk_z). A clip excluded at the
    first pass never reaches this. Returns {discs: [DESK_DISC_KEYS], excluded: [{obj,side,why,t_s}], supports, rule,
    n}."""
    import json as _json
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    if ti is None:
        ti = _json.loads((tree_dir / "task_info.json").read_text())
    if scene_mode(tree_dir) == "aerial" or ti.get("scene") == "aerial":
        raise ValueError(f"{tree_dir}: asked for the desk supports of an aerial tree — aerial_support owns those")
    ti = {k: v for k, v in ti.items() if k != desk_support_key()}
    dz = _qpos_layout_mod().desk_z_of(ti, str(tree_dir))[0]  # the env's · judge's desk plane (no statement raises)
    if q is None:
        q = np.load(tree_dir / "0/trajectory_kinematic.npz")["qpos"]
    off = desk_height() - float(dz)
    rest = []
    for d in rest_discs(tree_dir, ti, q, stats=stats, require_unheld=True):
        gap = (d["bottom_z_decl"] if d.get("bottom_z_decl") is not None else d["bottom_z_capture"]) - float(dz)
        if gap >= HIGH_REST_M:
            rest.append({**d, "gap_m": round(gap, 4)})
    sup = start_support(tree_dir, ti, q, rest, aerial=False)
    if sup["on_support"] and stats is not None:
        stats["on_support"] = stats.get("on_support", 0) + len(sup["on_support"])
    for d in sup["discs"]:
        d["gap_m"] = round(d["bottom_z_capture"] - float(dz), 4)
    if not pedestal_scene(ti, aerial=False):  # out of the pedestal rule's scope: the objects-only discs
        discs = [
            {k: d[k] for k in DESK_DISC_KEYS if k != "z"} | {"z": d["bottom_z_capture"] + off}
            for d in sup["rest"] + sup["discs"]
        ]
        return {
            "discs": discs,
            "excluded": sup["excluded"],
            "supports": sup["supports"],
            "rule": "D26′·D40",
            "n": len(discs),
        }
    placed, _ = _placers(tree_dir, ti, q)
    mains_v = [
        placed(o, 0)
        for o in ti.get("objects") or []
        if o.get("role") == "main" and o.get("mesh_dir") and o.get("qpos_start") is not None
    ]
    xped = static_pedestals(tree_dir, ti, q, float(dz), mains_v)
    xped += extra_rest_pedestals(tree_dir, ti, q, float(dz), stats)
    endp = end_pedestals(tree_dir, ti, q, float(dz), sup["rest"] + sup["discs"], placed, stats)
    for d in endp + xped:
        d["gap_m"] = round(d["bottom_z_capture"] - float(dz), 4)
    discs = _desk_discs(sup, endp, xped, off)
    return {
        "discs": discs,
        "excluded": sup["excluded"],
        "supports": sup["supports"],
        "rule": f"D26′·D40·{PEDESTAL_RULE}",
        "n": len(discs),
    }


def desk_support_discs(tree_dir) -> dict:
    """The desk-scene supports — the one reader for consumers (judge · bake chain · gate · final guard · render).
      stamped (baked tree)                   → the stamp as is · placed=True (already in tree coordinates)
      baked tree without the stamp (old)     → 0 supports · placed=True · marker `[DESK_SUPPORT] … → 0 supports`
      pre-bake tree (base · f0 shadow)       → compute_desk_support · placed=False (moves with the candidate placement)
    Returns {discs, excluded, supports, placed, source}; excluded / supports are None on an old stamp without them.
    An aerial tree raises (aerial_support owns it)."""
    import json as _json
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    ti = _json.loads((tree_dir / "task_info.json").read_text())
    if ti.get("scene") == "aerial":
        raise ValueError(f"{tree_dir}: asked for desk supports on an aerial tree — aerial_support owns those")
    key = desk_support_key()
    if key in ti:
        # read through the env's reader (qpos_layout.read → desk_support, uncached)
        lay = _qpos_layout_mod().read(str(tree_dir), use_cache=False)
        ds_ = lay.desk_support or {}
        return {
            "discs": list(ds_.get("discs") or []),
            "excluded": ds_.get("excluded"),
            "supports": ds_.get("supports"),
            "placed": True,
            "source": "stamp",
        }
    if is_baked_tree(ti):
        if str(tree_dir) not in _DESK_SUPPORT_SAID:
            _DESK_SUPPORT_SAID.add(str(tree_dir))
            print(f"[DESK_SUPPORT] {tree_dir.name}: no stamp (old bake generation) → 0 supports", flush=True)
        return {"discs": [], "excluded": None, "supports": None, "placed": True, "source": "absent_old_bake"}
    sup = compute_desk_support(tree_dir, ti)
    return {
        "discs": sup["discs"],
        "excluded": sup["excluded"],
        "supports": sup["supports"],
        "placed": False,
        "source": "computed",
    }


def start_support_verdict(tree_dir) -> list:
    """The start-support exclusions **stamped** on a baked tree — the gate's reading, through the stamp readers
    (aerial → read_aerial_support · table → desk_support_discs). A stamp without `excluded` (an older bake) raises: the
    gate cannot say whether the start is supported. Returns [{obj, side, why, t_s}] ([] = supported)."""
    import json as _json
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    ti = _json.loads((tree_dir / "task_info.json").read_text())
    if not is_baked_tree(ti):
        raise ValueError(f"{tree_dir}: the start-support verdict is read only from a baked tree's stamp")
    ex = (read_aerial_support(tree_dir) if ti.get("scene") == "aerial" else desk_support_discs(tree_dir))["excluded"]
    if ex is None:
        raise ValueError(
            f"{tree_dir}: the support stamp has no start-support verdict (excluded) — an older bake generation"
        )
    return list(ex)


def support_slot_objects(tree_dir, ti: dict, clip: str) -> list:
    """The object name of env support-object slot k (index k) — the same stamp and grouping (clip_supports) the env's
    support slots use. Stamp reader by scene (task_info.scene == "aerial" → read_aerial_support · else
    desk_support_discs). The collection form reads it to pair recorded slots with objects. A pre-bake tree (computed,
    placed False) raises, as in the env."""
    stamp = read_aerial_support(tree_dir) if ti.get("scene") == "aerial" else desk_support_discs(tree_dir)
    if not stamp.get("stamped", stamp.get("placed")):
        raise ValueError(
            f"[SUPPORT] {clip}: pre-bake tree (computed supports instead of a stamp · "
            f"source={stamp.get('source')}) — the env reads stamps only"
        )
    return [str(s["obj"]) for s in clip_supports(ti, stamp["supports"], clip)[0]]


def scene_z_offset(tree_dir) -> float:
    """Tree (capture) coordinates → robot-world z offset: world z = capture z + offset.

    A base-layer tree declares `task_info.support.desk_z` (the desk plane in capture coordinates), so offset =
    DESK_HEIGHT − desk_z (the rule is qpos_layout.desk_z_of — a tree without a desk-plane statement raises; TACO stage-1
    staging reads 0). Aerial trees use the aerial reference plane (`aerial_support`: stamp after the bake, computed
    before). The marker `[DESK_Z]` is printed once per tree."""
    import json as _json
    import pathlib as _pl

    tree_dir = _pl.Path(tree_dir)
    ti = _json.loads((tree_dir / "task_info.json").read_text())
    if scene_mode(tree_dir) == "aerial":
        # deskless scene — support.desk_z is not used. Reference plane = the floor of the first rest disc (else the
        #   object's starting floor) → world DESK_HEIGHT.
        sup_a = aerial_support(tree_dir)
        key = str(tree_dir)
        if key not in _DESK_Z_SAID:
            _DESK_Z_SAID.add(key)
            print(
                f"[DESK_Z] {tree_dir.name}: z_ref={sup_a['z_ref']:+.4f} → z_off={sup_a['z_off']:.4f} "
                f"(aerial:{sup_a['source']} discs={sup_a['n_rest']} "
                f"{'stamped' if sup_a.get('stamped') else 'computed'})",
                flush=True,
            )
        return float(sup_a["z_off"])
    desk_z, src = _paths().qpos_layout.desk_z_of(ti, str(tree_dir))  # the rule is qpos_layout's (same as training)
    off = desk_height() - desk_z
    key = str(tree_dir)
    if key not in _DESK_Z_SAID:
        _DESK_Z_SAID.add(key)
        print(f"[DESK_Z] {tree_dir.name}: desk_z={desk_z:+.4f} → z_off={off:.4f} ({src})", flush=True)
    return off
