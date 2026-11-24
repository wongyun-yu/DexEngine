"""Rule S-float-settle: a free main object that rests above its support is set onto it for the rest window; a reference
whose object would jump more than FLOAT_SETTLE_MAX_M at the first touch is dropped.

Statement (`statement`), per main object of a tree, measured on the shapes the training scene collides
(touch_gate.body_of: cooked parts, an articulated object's links at the frame's joint values):
  first_touch  the first tree frame a hand touches that object or comes within contact distance of it
               (placement.coords.contacts.frame_toucher · frame_near, the object's own contact sidecar, in base frames
               mapped to the tree by coords._base_to_tree); the rest window is the frames before it
  rest_frame   the last frame of the rest window (first_touch − 1; 0 when touched from the start; the last frame when
               never touched)
  gap_m        the object's lowest point at rest_frame − the highest collision-on support under its lowest patch
               (points within PATCH_M of the bottom, footprint padded by PATCH_PAD_M): the desk top, the object's own
               support discs (standing from reset to the end, rule D40), and every body the object collides with in
               training (other main objects, supports, still or moving extras whose objects cell is ON) with points
               under the footprint at most UNDER_M above the bottom
  support      what that highest support is: `desk` · `disc` · the body's name
  desk_gap_m   the same lowest point − the desk or own-disc top alone (what the training pin can see); an object within
               FLOAT_REST_M of it rests there, and the other bodies are not read
Verdict (first that applies):
  stale_base     the object's obj_id differs from its base tree's (coords.contacts.base_mismatch: the base was rebuilt
                 after placement): not measured; the tree is listed for regeneration (`regenerate`, REGEN_WORD)
  held           touched, or a hand within contact distance, from the first base frame: no rest window
  unmeasured     the object has no cooked collision shape
  unsupported    no collision-on support under it at all (an aerial object without its disc)
  over_off_body  a collision-off body stands under it higher than its support (it rests on a body it passes through)
  rest           gap_m ≤ FLOAT_REST_M
  over_body      floating above another body (support is a body): reported, not lowered
  settle         FLOAT_REST_M < gap_m ≤ FLOAT_SETTLE_MAX_M: the training pin lowers it onto the desk or disc during
                 the rest window (rl/tasks/scene/float_lower.py)
  gap_m > FLOAT_SETTLE_MAX_M, by the object's motion over the rest window (`moved_m`: largest displacement from frame
  0) — no hand is within contact distance of it there (first_touch reads either hand):
  held           still, and held over the start window by the start-support rule (coords.start_gaps): the spawn
                 hold keeps it until the hand takes it; not lowered (`start_held`)
  support        still and not held at the start: the start-support rule's case, whose rest disc the bake writes
                 (once stamped the object rests on its own disc); not lowered, and the env refuses it unstamped
  carried        moving with another object of the tree that touches it (extra_slots_geom.carry_of over the rest
                 window, the carrier in `support`): it rides on that object; not lowered
  drop           moving on its own: the tree is dropped (`dropped`), the case listed with its numbers
Off for a dataset (scene_rules.on) = no statement (every reader keeps its old path). Marker `[FLOAT_SETTLE]`.
"""

from __future__ import annotations

import numpy as np

from dexengine.data.objects import collider_shape as CS
from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import scene_requirement as SR
from dexengine.data.objects import touch_gate as TG

RULE_ID = "S-float-settle"
VERDICTS = (
    "stale_base",
    "held",
    "unmeasured",
    "unsupported",
    "over_off_body",
    "rest",
    "over_body",
    "settle",
    "support",
    "carried",
    "drop",
)
DROP_WORD = "float_settle_jump"
MARK = "[FLOAT_SETTLE]"
PATCH_M = 0.010  # the lowest patch: points within this of the object's bottom
PATCH_PAD_M = 0.005  # the patch footprint (x·y box) is padded by this
UNDER_M = 0.002  # a body's point under the footprint counts as support up to this above the bottom
DESK, DISC = "desk", "disc"


def _world(body, T: np.ndarray, qrow) -> np.ndarray:
    """(N,3) world points of a Body at pose T (4x4) and joint values qrow."""
    out = [sh.points @ (T @ L)[:3, :3].T + (T @ L)[:3, 3] for sh, L in body.links(qrow)]
    return np.concatenate(out)


def _first_touch(tree_dir, ti: dict, q: np.ndarray, name: str) -> tuple[int, int | None]:
    """(tree frame, base frame) of the first touch of object `name`: the first base frame its contact reads touched
    (coords.frame_toucher) or a hand is within contact distance of it (coords.frame_near: a hand that already holds it
    while the contact flags lag), mapped to the tree through the base → tree frame map (coords._base_to_tree: the f0
    pad and the retime map). Never touched: (the clip length, None) — the whole clip is its rest window."""
    from dexengine.placement.coords import contacts as CT  # noqa: PLC0415 — owner of the distance reading
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — owner of touch and the frame map

    b2t = C._base_to_tree(tree_dir, ti, len(q))
    touch, near = C.frame_toucher(tree_dir, ti, q, name, b2t), CT.frame_near(tree_dir, ti, name)
    fb = next((b for b in range(C._base_T(ti) or len(q)) if touch(b) or near(b)), None)
    return (len(q), None) if fb is None else (int(b2t(fb)), fb)


def _desk_tops(ti: dict, name: str, P: np.ndarray) -> dict:
    """{support: top z} of the desk and the object's own discs under its points P (coords.disc_covers: the footprint
    test of coords.on_own_disc). A disc stands from reset to the end of the episode (rule D40: the stamp's start/end
    is where it was measured, not a lifetime), so it supports the object at any frame where it is under it."""
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — owner of the disc footprint test

    s = ti.get("support") or {}
    dz = s.get("desk_z")
    if dz is None or s.get("desk_mode") in (None, "none"):
        return {}
    tops = {DESK: float(dz)}
    for d in (ti.get("desk_support") or {}).get("discs") or []:
        if d.get("obj") == name and C.disc_covers(P, d):
            tops[DISC] = max(tops.get(DISC, -np.inf), float(dz) + float(d.get("h", d.get("gap_m"))))
    return tops


def scene_bodies(tree_dir, ti: dict, q: np.ndarray, clip_rows: list[dict]) -> dict:
    """{name: (Body | None, objects cell 'ON' | 'OFF', pinned, poses (T,4,4))} of every body the training scene spawns
    besides the main objects: supports and the slotted extras (`clip_rows` = extra_slots.clip_rows of the tree,
    strict=False)."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.rl.tasks.scene import collision_presets as CP  # noqa: PLC0415 — owner of the training cells

    ds = str(ti.get("dataset_name") or "")
    objs = {str(o["name"]): o for o in ti.get("objects") or [] if o.get("qpos_start") is not None}
    rows = {r["name"]: r for r in clip_rows}
    out = {}
    for n, o in objs.items():
        r = rows.get(n)
        if o.get("role") == "support":
            parts = TG.parts_of(ds, o)
            out[n] = (TG.Body([CS.Parts(parts)]) if parts else None, "ON", True, XG.pose_mats(q, int(o["qpos_start"])))
        elif r is not None and r.get("kind") and r.get("cls") in XS.SLOTTED:
            b, _why = TG.static_body(ds, o, r) if r["cls"] == XS.STATIC_SPAWN else (None, "")
            if b is not None:
                b = TG.Body([b])
            elif (parts := TG.parts_of(ds, o)) is not None:
                b = TG.Body([CS.Parts(parts)])
            cell = CP.extra_collide(r["kind"], full=False, ds=ds)["objects"]
            out[n] = (b, cell, r["cls"] == XS.STATIC_SPAWN, XG.pose_mats(q, int(o["qpos_start"])))
    return out


def _under(P: np.ndarray, lo: np.ndarray, hi: np.ndarray, bot: float) -> float | None:
    m = np.all((P[:, :2] >= lo) & (P[:, :2] <= hi), axis=1) & (P[:, 2] <= bot + UNDER_M)
    return float(P[m, 2].max()) if m.any() else None


def _supports(name, P, f, ti, q, mains, others) -> tuple[dict, dict]:
    """({support: top} collision-on, {body: top} collision-off) under the lowest patch of points P at frame f."""
    bot = float(P[:, 2].min())
    patch = P[P[:, 2] <= bot + PATCH_M]
    lo, hi = patch[:, :2].min(0) - PATCH_PAD_M, patch[:, :2].max(0) + PATCH_PAD_M
    on, off = _desk_tops(ti, name, P), {}
    bodies = {n: (b, "ON", False, T) for n, (b, T) in mains.items() if n != name}
    bodies |= {n: (b, cell, pin, T) for n, (b, cell, pin, T) in others.items()}
    for n, (b, cell, pin, T) in bodies.items():
        if b is None:
            continue
        top = _under(_world(b, T[0] if pin else T[f], q[f]), lo, hi, bot)
        if top is not None:
            (on if cell == "ON" else off)[n] = top
    return on, off


def _verdict(row: dict, on: dict, off: dict) -> str:
    sup = max(on.values()) if on else -np.inf
    if not on:
        return "unsupported"
    if any(t > sup + SR.FLOAT_REST_M for t in off.values()):
        return "over_off_body"
    if row["gap_m"] <= SR.FLOAT_REST_M:
        return "rest"
    if row["support"] not in (DESK, DISC):
        return "over_body"
    return "settle" if row["gap_m"] <= SR.FLOAT_SETTLE_MAX_M else "drop"


def statement(tree_dir, ti: dict, q: np.ndarray, rows_of) -> dict:
    """{main object name: {first_touch, rest_frame, gap_m, support, verdict, side}} (module doc); {} when the rule is
    off for the tree's dataset. rows_of() = the tree's extra_slots.clip_rows (strict=False), read once and only when a
    main object is not resting on the desk (the caller says where the rows come from)."""
    from dexengine.data.objects import scene_rules as SC  # noqa: PLC0415 — the rule switch

    ds = str(ti.get("dataset_name") or "")
    if not SC.on(RULE_ID, ds):
        return {}
    mains = {}
    for o in TG.main_objects(ti):
        b, _why = TG.body_of(ds, o, tree_dir)
        mains[str(o["name"])] = (b, XG.pose_mats(q, int(o["qpos_start"])), o)
    from dexengine.placement.coords import contacts as CT  # noqa: PLC0415 — owner of the base identity

    stale = CT.base_mismatch(tree_dir, ti)
    out, others = {}, None
    for n, (b, T, o) in mains.items():
        if n in stale:
            out[n] = {"first_touch": None, "rest_frame": None, "side": str(o.get("side") or ""), "gap_m": None}
            out[n] |= {"support": None, "desk_gap_m": None, "verdict": "stale_base", "ids": list(stale[n])}
            continue
        ft, fb = _first_touch(tree_dir, ti, q, n)
        f = min(max(ft - 1, 0), len(q) - 1)
        row = {"first_touch": ft, "rest_frame": f, "side": str(o.get("side") or ""), "gap_m": None, "support": None}
        row["desk_gap_m"] = None
        if fb == 0 or b is None:
            out[n] = row | {"verdict": "held" if fb == 0 else "unmeasured"}
            continue
        P = _world(b, T[f], q[f])
        desk = _desk_tops(ti, n, P)
        if desk:
            row["desk_gap_m"] = round(float(P[:, 2].min()) - max(desk.values()), 5)
        if desk and row["desk_gap_m"] <= SR.FLOAT_REST_M:
            on, off = desk, {}  # resting on the desk or its disc (module doc)
        else:
            others = scene_bodies(tree_dir, ti, q, rows_of()) if others is None else others
            on, off = _supports(n, P, f, ti, q, {k: (v[0], v[1]) for k, v in mains.items()}, others)
        if on:
            row["support"] = max(on, key=on.get)
            row["gap_m"] = round(float(P[:, 2].min()) - on[row["support"]], 5)
        out[n] = row | {"verdict": _verdict(row, on, off)}
        if out[n]["verdict"] == "drop":
            out[n] |= _high_float(tree_dir, ti, q, o, f)
    return out


def inputs(tree_dir, ti: dict) -> list:
    """The data-root files `statement` opens besides the tree's own and the objects' USDs (none when the rule is off
    for the tree's dataset): the contact readers' files (coords.contacts.read_inputs — the base tree's task_info, the
    contact_base and contact_all sidecars and their fingerprint inputs), the shapes of the bodies it measures
    (touch_gate.body_inputs of every posed object: main objects, supports, slotted extras), the object files the
    placed vertices and an articulated object's base part are read from (coords.object_files) and, with an
    articulated main object, the MANO keypoints frame_toucher measures the fingertips from. Read by the bundle input
    statement (run_inputs)."""
    from dexengine.data.objects import scene_rules as SC  # noqa: PLC0415 — the rule switch

    ds = str(ti.get("dataset_name") or "")
    if not SC.on(RULE_ID, ds):
        return []
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data.contact import fingerprint as SF  # noqa: PLC0415 — owner of the MANO keypoint path
    from dexengine.placement.coords import contacts as CT  # noqa: PLC0415 — owner of the contact readers
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — owner of the object file declaration

    out = CT.read_inputs(tree_dir, ti)
    for o in ti.get("objects") or []:
        if o.get("qpos_start") is not None and o.get("obj_id"):
            out += TG.body_inputs(ds, o, tree_dir)
    out += [C.input_path(ds, p) for p in C.object_files(ti)]
    if any(o.get("articulation") for o in TG.main_objects(ti)):
        out.append(SF.mano_file(P.main_root(), ti))  # frame_toucher's MANO keypoints
    return list(dict.fromkeys(out))


def _high_float(tree_dir, ti: dict, q: np.ndarray, o: dict, f: int) -> dict:
    """Class of a float above the cap by its motion over the rest window [0, f] (module doc): {verdict, moved_m} and
    the carrier's name in `support` when it is carried. A still one is the start-support rule's case
    (coords.start_gaps): held over the start window → held (the spawn hold keeps it until the hand takes it, and the
    bake writes no support), otherwise support (the bake writes its rest disc; an own disc in the stamp makes it rest
    before this is reached)."""
    st = int(o["qpos_start"])
    p = q[: f + 1, st : st + 3]
    moved = float(np.linalg.norm(p - p[0], axis=1).max())
    if moved <= SR.FLOAT_REST_M:
        from dexengine.placement.coords import core as C  # noqa: PLC0415 — owner of the start-support rule

        g = next((x for x in C.start_gaps(tree_dir, ti, q, aerial=False) if x["obj"] == str(o.get("name"))), None)
        held = bool(g and g["held"])
        return {"verdict": "held" if held else "support", "moved_m": round(moved, 5), "start_held": held}
    cands = [
        str(x["name"])
        for x in ti.get("objects") or []
        if x.get("qpos_start") is not None and x.get("name") != o.get("name")
    ]
    rel = XG.carry_of(tree_dir, ti, q[: f + 1], str(o["name"]), cands) if cands else None
    if rel:
        return {"verdict": "carried", "moved_m": round(moved, 5), "support": rel["carrier"]}
    return {"verdict": "drop", "moved_m": round(moved, 5)}


def dropped(st: dict) -> tuple[str, str] | None:
    """(DROP_WORD, description) when a main object of the statement would jump above the cap, else None — the clip's
    exclusion verdict."""
    bad = {n: r for n, r in st.items() if r["verdict"] == "drop"}
    if not bad:
        return None
    why = " · ".join(
        f"{n} gap {r['gap_m'] * 1000:.1f} mm over {r['support']} at frame {r['rest_frame']} > cap "
        f"{SR.FLOAT_SETTLE_MAX_M * 1000:.0f} mm, moving {r.get('moved_m', 0) * 1000:.1f} mm with no hand near "
        "and no carrier"
        for n, r in sorted(bad.items())
    )
    return DROP_WORD, why


#: Reason word of a tree that must be placed again (`regenerate`).
REGEN_WORD = "obj_id_differs_from_base"


def regenerate(st: dict) -> tuple[str, str] | None:
    """(REGEN_WORD, description) when an object of the statement is stale against its base, else None."""
    bad = {n: r for n, r in st.items() if r["verdict"] == "stale_base"}
    if not bad:
        return None
    why = " · ".join(f"{n} {r['ids'][0]} in the tree vs {r['ids'][1]} in its base" for n, r in sorted(bad.items()))
    return REGEN_WORD, why


#: Side column of the exclusion step's float row (placement/repair/hand_clear_stage: one row per clip, like its
#: objects row).
ROW_SIDE = "float_settle"


def stage_row(
    tree_dir, ti: dict, q: np.ndarray, clip_rows: list[dict], qpos_sha: str, version: str, statuses: tuple
) -> dict:
    """The exclusion step's row of one clip (statuses = (keep, drop) of that step; clip_rows = the tree's
    extra_slots.clip_rows, strict=False): DROP when `dropped`, the statement of every main object in `rows`."""
    st = statement(tree_dir, ti, q, lambda: clip_rows)
    d = regenerate(st) or dropped(st)
    return {
        "clip": str(tree_dir).rstrip("/").rsplit("/", 1)[-1],
        "side": ROW_SIDE,
        "status": statuses[1] if d else statuses[0],
        "rows": " ".join(
            f"{n}:{r['verdict']}:{r['gap_m']}:{r['support']}:{r['first_touch']}" for n, r in sorted(st.items())
        ),
        "qpos_sha": qpos_sha,
        "version": version,
        "why": (d or ("", ""))[1],
    }


def marker(tree: str, st: dict) -> str:
    """`[FLOAT_SETTLE]` line of one tree: main objects by verdict and the largest settle (mm)."""
    by = {v: sum(r["verdict"] == v for r in st.values()) for v in VERDICTS}
    settle = [r["gap_m"] * 1000 for r in st.values() if r["verdict"] in ("settle", "support", "carried", "drop")]
    top = f" · largest {max(settle):.1f} mm" if settle else ""
    return f"{MARK} {tree} mains {len(st)} {by}{top} · cap {SR.FLOAT_SETTLE_MAX_M * 1000:.0f} mm"
