"""Scene object measurement, reduction half: a window of the measurement reduced to per-object scalars
(reduce_window) and the tree's objects[] keyed by raw object (tree_scene), as the record's render-only object list
reads them. The measurement half and the module contract are in readers/raw/common.py."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from dexengine.data.objects.scene_requirement import (
    JUMP_M,
    MOVE_M,
    RECV_M,
    ROT_DEG,
    TOUCH_M,
    TOUCH_MIN_S,
    touched,
)
from dexengine.data.readers.raw import touch as TP

SURF_SPACING = 0.003


def object_ids(o: dict) -> list[str]:
    """The raw object ids one objects[] entry stands for — the tree's statement, read here once (tree_scene · the
    audit's slot_spawn). A merged-mesh asset (`fixed_merge`: official parts of one instance joined into one rigid body,
    obj_id = the asset id, not a raw id) states the parts it covers; any other entry is its own obj_id. A fixed_merge
    that names another asset or no part cannot be read — it dies."""
    fm = o.get("fixed_merge")
    if fm is None:
        return [o["obj_id"]]
    parts = fm.get("parts") if isinstance(fm, dict) else None
    if not parts or fm.get("asset") != o["obj_id"]:
        raise ValueError(f"objects[] {o.get('name')}: fixed_merge {fm!r} does not state the parts of {o['obj_id']!r}")
    return [str(x) for x in parts]


def tree_scene(ti: dict, key_of) -> dict:
    """{rid: {roles, spawned}} from a tree's objects[] under the spawn rule in the module doc. An entry spawns every
    raw object it stands for (object_ids — a merged-mesh asset = its parts, roles marked `:merged`), and a spawned
    articulated main spawns its two link parts (base_id · moving_id → artic_part) and the parts fixed to a link
    (articulation.parts.fixed[].id — one rigid body with that link → artic_fixed)."""
    idle = set(ti.get("idle_sides") or [])
    rows: dict = {}

    def ent(rid):
        return rows.setdefault(rid, {"roles": [], "spawned": ""})

    for o in ti.get("objects") or []:
        role, side, art = o.get("role"), o.get("side"), o.get("articulation")
        sp = ""
        if role == "main" and side not in idle:
            sp = "main_artic" if art else "main"
        elif role == "support" and not art:
            sp = "support"
        tag = f"{role}:{side}" + (":artic" if art else "") + (":merged" if o.get("fixed_merge") is not None else "")
        for rid in dict.fromkeys(key_of(x) for x in object_ids(o)):
            e = ent(rid)
            e["roles"].append(tag)
            if sp and not e["spawned"]:
                e["spawned"] = sp
        if role == "main" and side not in idle and isinstance(art, dict):
            parts = art.get("parts") or {}
            linked = [(parts.get(k), "artic_part") for k in ("base_id", "moving_id")]
            for pid, how in linked + [(f["id"], "artic_fixed") for f in parts.get("fixed") or []]:
                if pid and not ent(key_of(pid))["spawned"]:
                    ent(key_of(pid))["spawned"] = how
    return rows


@lru_cache(maxsize=256)
def surface(mesh_path: str):
    import trimesh
    from scipy.spatial import cKDTree

    m = trimesh.load(mesh_path, force="mesh")
    n = int(np.clip(m.area / SURF_SPACING**2, 3000, 60000))
    p, _ = trimesh.sample.sample_surface(m, n, seed=0)
    p = np.vstack([p, m.vertices]).astype(np.float64)
    return p, cKDTree(p), p.mean(0), p.min(0), p.max(0)


@lru_cache(maxsize=256)
def hull(mesh_path: str):
    """Facet planes (k, 4) of the convex hull of the mesh's surface samples in its own frame (a point x is inside when
    every plane gives n·x + d < 0); None for a mesh spanning less than a volume (rank < 3: no inside)."""
    from scipy.spatial import ConvexHull, QhullError

    p = surface(mesh_path)[0]
    try:
        return ConvexHull(p).equations
    except QhullError:
        if np.linalg.matrix_rank(p - p.mean(0)) < 3:
            return None
        raise


def rot_angle_deg(R0: np.ndarray, Rs: np.ndarray) -> np.ndarray:
    c = (np.einsum("ij,tij->t", R0, Rs) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def _medfilt(x: np.ndarray, k: int = 3) -> np.ndarray:
    if len(x) < k:
        return x
    pad = np.concatenate([x[:1].repeat(k // 2, 0), x, x[-1:].repeat(k // 2, 0)])
    return np.median(np.stack([pad[i : i + len(x)] for i in range(k)]), axis=0)


def _surf_dist(st: dict, a: str, b: str, ia: int, ib: int, cap: int = 1200) -> tuple[float, np.ndarray]:
    """min distance of a's surface (pose ia) to b's surface (pose ib) and a's centroid in b's frame."""
    oa, ob = st["objs"][a], st["objs"][b]
    Pa = surface(oa["mp"])[0]
    Pa = Pa[:: max(1, len(Pa) // cap)]
    pw = Pa @ oa["R"][ia].T + oa["t"][ia]
    loc = (pw - ob["t"][ib]) @ ob["R"][ib]
    return float(surface(ob["mp"])[1].query(loc, k=1)[0].min()), loc.mean(0)


def reduce_window(st: dict, w0: int, w1: int, act: dict, declared: set) -> dict:
    """Scalars per raw object over the window [w0, w1) (raw frames). Touch (the verdict) = the six points
    (TP.window_touch: pt_frac_* · pt_tip_*_mm · pt_palm_*_mm) — touched_any = either hand's. Hand-surface contact —
    surf_*_s · surf_dmin_*_mm · surf_touched · surf_receiver = hand surface <= st["touch_m"] when a dataset module
    sets it (hand joints instead of a MANO surface), else TOUCH_M, for >= TOUCH_MIN_S: a reference column for an
    object in the tree, the intake's statement for one the tree does not hold (classify)."""
    tm = st.get("touch_m", TOUCH_M)
    fr = st["frames"]
    sel = (fr >= w0) & (fr < w1)
    n_win = int(sel.sum())
    out = {}
    for rid, o in st["objs"].items():
        r = {"rid": rid, "mesh": int(o["mesh"]), "declared": int(rid in declared)}
        ok = sel & o["have"] if o["have"].size else np.zeros(0, bool)
        r["n_samples"] = int(ok.sum())
        if not o["mesh"] or ok.sum() < 2:
            r.update(
                pt_frac_lh=0.0,
                pt_frac_rh=0.0,
                surf_lh_s=0.0,
                surf_rh_s=0.0,
                move_m=0.0,
                rot_deg=0.0,
                jump=0,
                measurable=0,
            )
            out[rid] = r
            continue
        idx = np.flatnonzero(ok)
        c = _medfilt(o["c"][idx])
        step = float(np.linalg.norm(np.diff(o["c"][idx], axis=0), axis=1).max())
        r.update(TP.window_touch(o, idx, n_win))
        r["pt_touch_s"] = TP.touch_s(r, n_win, st["dt"])
        for s in ("lh", "rh"):
            r.update(
                {
                    f"surf_{s}_s": round(float((o["dist"][s][idx] <= tm).sum() * st["dt"]), 3),
                    f"surf_dmin_{s}_mm": round(float(np.min(o["dist"][s][idx])) * 1e3, 1),
                }
            )
        r.update(
            move_m=round(float(np.linalg.norm(c - c[0], axis=1).max()), 4),
            rot_deg=round(float(rot_angle_deg(o["R"][idx[0]], o["R"][idx]).max()), 1),
            step_m=round(step, 4),
            jump=int(step >= JUMP_M),
            measurable=1,
            _i0=int(idx[0]),
            _i1=int(idx[-1]),
        )
        out[rid] = r
    for r in out.values():  # touch counts whatever the program says about the hand (scene_requirement: any hand)
        r["touched_any"] = TP.touched(r)
        r["touched_active"] = TP.touched(r, [s for s in ("lh", "rh") if act[s]])  # report only
        r["surf_touched"] = int(touched({"lh": r["surf_lh_s"], "rh": r["surf_rh_s"]}, TOUCH_MIN_S))
        r["_surf_active"] = int(touched({s: r[f"surf_{s}_s"] for s in ("lh", "rh") if act[s]}, TOUCH_MIN_S))
        r["moved"] = int(r["measurable"] and (r["move_m"] >= MOVE_M or r["rot_deg"] >= ROT_DEG))
    mv = {
        k: [x for x, r in out.items() if r["moved"] and r[k] and not r["jump"]] for k in ("touched_any", "surf_touched")
    }
    recv = {"touched_any": _receivers(st, out, mv["touched_any"])}
    recv["surf_touched"] = (
        recv["touched_any"] if mv["surf_touched"] == mv["touched_any"] else _receivers(st, out, mv["surf_touched"])
    )
    for rid, r in out.items():  # a receiver of a mover — the mover's touch read each way (classify picks one)
        r["receiver"], r["surf_receiver"] = recv["touched_any"].get(rid, ""), recv["surf_touched"].get(rid, "")
    return out


def _receivers(st: dict, out: dict, movers: list) -> dict:
    """{rid: "geom:<mover>"} — the unmoved measurable objects one of `movers` comes to rest on / in (scene_requirement:
    receiver): the mover's surface farther than RECV_M at its first window sample and within it (or its centroid
    inside the object's convex hull, in the object's frame) at its last."""
    got = {}
    for rid, r in out.items():
        if r["moved"] or not r["measurable"]:
            continue
        o = st["objs"][rid]
        lo, hi = surface(o["mp"])[3], surface(o["mp"])[4]
        for m in movers:
            if m == rid:
                continue
            i0, i1 = out[m]["_i0"], out[m]["_i1"]
            if not (o["have"][i0] and o["have"][i1]):
                continue
            d0, _ = _surf_dist(st, m, rid, i0, i0)
            d1, cm = _surf_dist(st, m, rid, i1, i1)
            H = hull(o["mp"]) if np.all((cm > lo) & (cm < hi)) else None  # the box holds the hull: screen first
            inside = H is not None and bool(np.all(H[:, :3] @ cm + H[:, 3] < 0))
            if d0 > RECV_M and (d1 <= RECV_M or inside):
                got[rid] = f"geom:{m}"
                break
    return got
