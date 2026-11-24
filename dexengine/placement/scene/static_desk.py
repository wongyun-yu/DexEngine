"""Desk check of the still objects added at render time (gate verdict `valid_static`): rule · thresholds · measure.

Why      a clip the assignment gate marks valid_static keeps the needed still objects (static_required) out of the
         training scene and adds them only at render time (static_add). The judge's desk check looks at the scene
         objects (coords.desk_mask_slots) only, so a placement could push such a still object off the desk.
Readers  the judge (desk mask → grid_overhang_ok · final check → overhang_frac on footprints moved by the placement
         formula coords.place_objects), the staleness check (dexengine.placement.train_ready --stale-out → assess ·
         stale_reason) and the put-down rule (place_rule). Marker `[DESK_STATIC] <clip> …` (printed even when 0).
Objects  required(base_ti) = the clip's **base** record rules.ASSIGN.gate.static_required (a tree or f0 shadow may carry
         an older record, so the base tree is read — coords.sidecar_base_tree). Names to add = static_add.add_names
         (minus the tree's supports) · poses = static_add.rows (tree frame 0 — the render's function). Moving ones
         (whole-tree span ≥ threshold) are refused by the render, so they are left out here and listed.
Geometry surface samples = fingerprint.surface_path (surface_points.npz · 2048 points) — the same body frame as the
         render USD's visual mesh, numpy-only, already among the judge inputs (coords.setup_inputs).
Box      desk_box(tree) = coords.desk_bounds_scene_for(tree, margin 0) — the desk top. Not the main objects' judge box
         (2 cm edge margin · robot-side trim): still objects are visual only, so only "does it overhang" is asked.
Overhang the **area** of the top-view outline (convex hull of the scene xy points) outside the box (overhang_frac —
         exact, invariant under rigid motion). > OVERHANG_MAX_FRAC = overhang.
Float    float = lowest surface z − desk height. > FLOAT_MAX_M = floating. The placement does not change z, so a
         rejudge cannot fix it: it does not block the judge.
Float rule (render recipe) float_rule — a floating object that neither hand comes within HAND_NEAR_M of over the clip
         is lowered (render pose z −= float); one a hand approaches is dropped from the render (lowering it would move
         the place of the action). Hand distance = ref_hand_geom (reference hand link meshes ↔ object mesh, original
         pose). Marker `[STATIC_FLOAT] <tree> <object> lowered|dropped …` + one summary line. Switched on only by
         static_add.rows(float_rule=True); needs mujoco.

Area geometry (no policy — thresholds and the box are the caller's):
  hull(xy)                   convex hull vertices (counter-clockwise) · degenerate (a line · a point) → None
  out_frac(h, box)           area fraction of a hull outside the box — exact (the hull clipped by the box's 4 edges)
  lattice(xy, n)             about n uniform grid points inside the hull (outside fraction ≈ outside area fraction)
  points_out_frac(xy, box)   outline area fraction outside the box (degenerate: the point fraction)
  grid_out_ok(...)           per cell offset, points_out_frac ≤ threshold — estimated on the lattice, cells near the
                             threshold re-measured exactly
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TAG = "[DESK_STATIC]"
#: Overhang threshold — the fraction of the top-view outline area outside the desk top above which it overhangs.
OVERHANG_MAX_FRAC = 0.10
#: Area samples of the judge's grid mask (grid_out_ok) · cells whose estimate is within this many sample rows of the
#: threshold are re-measured exactly.
AREA_POINTS_N = 4096
AREA_BAND_ROWS = 2.0
#: Float threshold (m) — lowest point higher than this above the desk = floating (float rule). Not a placement matter.
FLOAT_MAX_M = 0.02
#: Staleness measurement margin — 2 surface samples (2/2048): absorbs the difference between the judge's measure
#: (shadow + placement formula) and the baked tree's.
STALE_MARGIN_FRAC = 2.0 / 2048.0
#: Stale-reason prefix (dexengine.placement.train_ready --stale-out, 4th column).
STALE_REASON = "C896:static_desk"
#: Float rule hand threshold (m) — only floating objects neither hand comes within this of over the clip are lowered.
HAND_NEAR_M = 0.01
FLOAT_TAG = "[STATIC_FLOAT]"
#: Float rule results — keep (not floating · moving · aerial) · lowered · dropped.
FLOAT_KEEP, FLOAT_LOWERED, FLOAT_DROPPED = "keep", "lowered", "dropped"


class StaticDeskError(RuntimeError):
    """The rule cannot be applied — never silently read as 0 objects."""


def _coords():
    from dexengine.placement.coords import core as C  # noqa: PLC0415 — box · scene transform · desk height · base tree

    return C


# ── area geometry ─────────────────────────────────────────────────────────────────────────────────────────────


def hull(xy) -> np.ndarray | None:
    """Convex hull vertices (counter-clockwise) of xy points (N,2) · degenerate (a line · a point — no area) → None."""
    from scipy.spatial import ConvexHull  # noqa: PLC0415

    xy = np.asarray(xy, float)
    try:
        return xy[ConvexHull(xy).vertices]
    except Exception:  # noqa: BLE001 — degenerate
        return None


def poly_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def clip_half(p: np.ndarray, axis: int, bound: float, keep_ge: bool) -> np.ndarray:
    """Clip convex polygon p by a half-plane (axis coordinate ≥ bound or ≤ bound) — Sutherland–Hodgman."""
    out = []
    for a, b in zip(p, np.roll(p, -1, axis=0), strict=True):
        ia = a[axis] >= bound if keep_ge else a[axis] <= bound
        ib = b[axis] >= bound if keep_ge else b[axis] <= bound
        if ia:
            out.append(a)
        if ia != ib:
            out.append(a + (bound - a[axis]) / (b[axis] - a[axis]) * (b - a))
    return np.asarray(out, float).reshape(-1, 2)


def out_frac(h: np.ndarray, box) -> float:
    """Area fraction of hull h (counter-clockwise vertices) outside the box — the hull clipped by the box's 4 edges."""
    (x0, x1), (y0, y1) = box
    inside = h
    for axis, bound, ge in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        if len(inside) < 3:
            break
        inside = clip_half(inside, axis, float(bound), ge)
    return float(min(1.0, max(0.0, 1.0 - poly_area(inside) / poly_area(h))))


def lattice(xy, n: int) -> np.ndarray:
    """A uniform grid (M,2) filling the convex hull of xy points (N,2) · spacing = sqrt(hull area / n). Degenerate →
    the points themselves."""
    xy = np.asarray(xy, float)
    h = hull(xy)
    if h is None:
        return xy
    step = float(np.sqrt(poly_area(h) / n))
    lo, hi = h.min(0), h.max(0)
    gx, gy = np.arange(lo[0] + step / 2, hi[0], step), np.arange(lo[1] + step / 2, hi[1], step)
    P = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    a, e = h, np.roll(h, -1, axis=0) - h
    cr = e[None, :, 0] * (P[:, None, 1] - a[None, :, 1]) - e[None, :, 1] * (P[:, None, 0] - a[None, :, 0])
    P = P[(cr >= 0.0).all(1)]
    return P if len(P) else h.mean(0)[None, :]


def points_out_frac(xy, box) -> float:
    """Area fraction of the convex hull of xy points outside the box — invariant under rigid motion (the same object
    and placement measure the same anywhere). A degenerate hull (a line · a point) gives the point fraction."""
    (x0, x1), (y0, y1) = box
    xy = np.asarray(xy, float)
    h = hull(xy)
    if h is None:
        x, y = xy[:, 0], xy[:, 1]
        return float(1.0 - np.mean((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)))
    return out_frac(h, box)


def grid_out_ok(ww, gx, gy, box, max_frac: float, n: int, band_rows: float) -> np.ndarray:
    """ww (N,2) points · cell offsets gx (X,) · gy (Y,) → (X,Y) bool: per cell points_out_frac(ww + (gx, gy)) ≤
    max_frac. Cheap: estimated by the point fraction of a uniform sample inside the hull (lattice · about n points —
    one sample row ≈ 1/sqrt(n) of the area), then cells whose estimate is within band_rows rows of the threshold are
    re-measured exactly (out_frac) — the result equals the exact measure."""
    (x0, x1), (y0, y1) = box
    h = hull(ww)
    pts = lattice(ww, n)
    ix = ((pts[:, 0:1] + gx[None, :]) >= x0) & ((pts[:, 0:1] + gx[None, :]) <= x1)  # (N,X)
    iy = ((pts[:, 1:2] + gy[None, :]) >= y0) & ((pts[:, 1:2] + gy[None, :]) <= y1)  # (N,Y)
    inside = ix.T.astype(np.float64) @ iy.astype(np.float64)  # (X,Y) points inside
    frac = 1.0 - inside / float(len(pts))
    ok = frac <= max_frac
    if h is not None:
        for i, j in zip(*np.nonzero(np.abs(frac - max_frac) <= band_rows / np.sqrt(n)), strict=True):
            ok[i, j] = out_frac(h + np.array([gx[i], gy[j]]), box) <= max_frac
    return ok


# ── the rule ──────────────────────────────────────────────────────────────────────────────────────────────────


def desk_box(tree=None) -> tuple[tuple[float, float], tuple[float, float]]:
    """The desk box — ((x0, x1), (y0, y1)) in scene coordinates."""
    return _coords().desk_bounds_scene_for(tree, margin=0.0)


def required(base_ti: dict) -> tuple[str, list[str]]:
    """(gate verdict, static_required) of a base record. A record without a gate → ('-', [])."""
    from dexengine.data.cases import assign_version as AR  # noqa: PLC0415 — rules key · verdict vocabulary

    rec = (base_ti.get("rules") or {}).get(AR.ASSIGN_KEY)
    verdict = AR.gate_of(rec)
    if verdict in (AR.GATE_NO, AR.GATE_NONE):
        return verdict, []
    g = rec["gate"]
    if verdict not in AR.GATE_VERDICTS:
        raise StaticDeskError(f"gate verdict {verdict!r} is outside the vocabulary {AR.GATE_VERDICTS}")
    return verdict, [str(n) for n in g.get("static_required") or []]


def surface(ds: str, obj: dict) -> np.ndarray:
    """Body-frame surface samples (N,3) of an object. Missing → raises."""
    from dexengine.core import paths as P  # noqa: PLC0415
    from dexengine.data.contact import fingerprint as SF  # noqa: PLC0415 — surface sample place

    f = SF.surface_path(P.main_root(), ds, obj)
    if not f.is_file():
        raise StaticDeskError(f"{obj.get('name')}({obj.get('obj_id')}) has no surface samples: {f}")
    with np.load(f) as z:
        return np.asarray(z["points"], dtype=np.float64)


def _cap_points(pose, pts: np.ndarray) -> np.ndarray:
    """Body points → capture coordinates (7-pose x·y·z · wxyz)."""
    from scipy.spatial.transform import Rotation as R  # noqa: PLC0415

    p = np.asarray(pose, float)
    return R.from_quat([p[4], p[5], p[6], p[3]]).apply(pts) + p[:3]


def footprints(tree_dir, ti: dict | None = None, base_ti: dict | None = None) -> dict:
    """The still objects to add at a tree's (base · f0 shadow · baked) frame 0 → {verdict, add, moving, objs}.
    objs = [{name, obj_id, xy_cap (N,2), z_scene (N,), span_m}] — moving objects stay out of objs, named in moving."""
    from dexengine.placement.scene import static_add as SA  # noqa: PLC0415 — names · poses (the render's function)

    tree_dir = Path(tree_dir)
    if ti is None:
        ti = json.loads((tree_dir / "task_info.json").read_text(encoding="utf-8"))
    if base_ti is None:
        bt = _coords().sidecar_base_tree(tree_dir, ti)
        base_ti = json.loads((Path(bt) / "task_info.json").read_text(encoding="utf-8"))
    verdict, req = required(base_ti)
    out = {"verdict": verdict, "add": [], "moving": [], "objs": []}
    if not req:
        return out
    names = SA.add_names(req, ti)
    out["add"] = list(names)
    if not names:
        return out
    z_off = float(_coords().scene_z_offset(tree_dir))
    ds = str(ti["dataset_name"])
    for r in SA.rows(tree_dir, ti, names, scene=False, usd=False):
        if r["moving"]:
            out["moving"].append(f"{r['name']}:span={r['span_m'] * 1000:.0f}mm")
            continue
        o = next(o for o in ti["objects"] if str(o["name"]) == r["name"])
        P3 = _cap_points(r["pose"], surface(ds, o))
        out["objs"].append(
            {
                "name": r["name"],
                "obj_id": r["obj_id"],
                "xy_cap": P3[:, :2],
                "z_scene": P3[:, 2] + z_off,
                "span_m": float(r["span_m"]),
            }
        )
    return out


def overhang_frac(xy_scene: np.ndarray, *, box) -> float:
    """The area fraction of the top-view outline of scene xy points outside the box (exact — points_out_frac)."""
    return points_out_frac(xy_scene, box)


def float_m(z_scene: np.ndarray) -> float:
    """Lowest surface point − desk height (m)."""
    return float(z_scene.min() - _coords().desk_height())


def grid_overhang_ok(ww: np.ndarray, gx: np.ndarray, gy: np.ndarray, *, box) -> np.ndarray:
    """Judge mask — ww (N,2) scene xy (already rotated) · cell offsets gx (X,) · gy (Y,) → (X,Y) bool: per cell
    overhang_frac(ww + (gx, gy)) ≤ OVERHANG_MAX_FRAC — the same value as the final check (grid_out_ok)."""
    return grid_out_ok(ww, gx, gy, box, OVERHANG_MAX_FRAC, AREA_POINTS_N, AREA_BAND_ROWS)


def assess(tree_dir, ti: dict | None = None, base_ti: dict | None = None) -> dict:
    """A baked tree's (placed world) frame 0 — {verdict, add, moving, scene, per: [(name, obj_id, overhang, float_m)],
    overhang_max, float_max, bad_overhang, bad_float}. An aerial scene (no desk): scene='aerial', no check."""
    fp = footprints(tree_dir, ti, base_ti)
    mode = _coords().scene_mode(tree_dir)
    res = {k: fp[k] for k in ("verdict", "add", "moving")}
    res.update(scene=mode, per=[], overhang_max=None, float_max=None, bad_overhang=False, bad_float=False)
    if mode != "table" or not fp["objs"]:
        return res
    C = _coords()
    box = desk_box(tree_dir)
    for o in fp["objs"]:
        ov = overhang_frac(C.capture_to_scene(o["xy_cap"]), box=box)
        res["per"].append((o["name"], o["obj_id"], ov, float_m(o["z_scene"])))
    res["overhang_max"] = max(p[2] for p in res["per"])
    res["float_max"] = max(p[3] for p in res["per"])
    res["bad_overhang"] = res["overhang_max"] > OVERHANG_MAX_FRAC
    res["bad_float"] = res["float_max"] > FLOAT_MAX_M
    return res


def stale_reason(res: dict) -> str:
    """dexengine.placement.train_ready --stale-out reason — overhang only (floating cannot be fixed by a rejudge). ''
    when no overhang. The threshold gets STALE_MARGIN_FRAC: the judge measures on the f0 shadow + placement formula, the
    staleness on the baked tree's f0 — two values of the same placement can differ by one sample. Without the margin a
    clip the judge accepted just below the threshold would be flagged again and rejudged with the same placement,
    forever."""
    if not res.get("bad_overhang") or res["overhang_max"] <= OVERHANG_MAX_FRAC + STALE_MARGIN_FRAC:
        return ""
    worst = max(res["per"], key=lambda p: p[2])
    return f"{STALE_REASON}:{worst[0]}:{worst[2] * 100:.0f}%"


def float_rule(tree_dir, ti: dict, rows: list[dict]) -> list[dict]:
    """The float rule (module docstring) — per static_add.rows row (original pose · tree frame 0 capture) {name,
    action, float_m, dz_m, hand_min_m, hand_frame, hand_side}. Moving rows and aerial scenes keep."""
    from dexengine.placement.scene import ref_hand_geom as RH  # noqa: PLC0415 — reference hand geometry (mujoco)

    C = _coords()
    tree_dir = Path(tree_dir)
    mode = C.scene_mode(tree_dir)
    out, ref = [], None
    z_off, desk = float(C.scene_z_offset(tree_dir)), float(C.desk_height())
    for r in rows:
        dec = {
            "name": r["name"],
            "action": FLOAT_KEEP,
            "float_m": None,
            "dz_m": 0.0,
            "hand_min_m": None,
            "hand_frame": -1,
            "hand_side": "",
        }
        out.append(dec)
        if mode != "table" or r["moving"]:
            continue
        o = next(o for o in ti["objects"] if str(o["name"]) == r["name"])
        P = _cap_points(r["pose"], surface(str(ti["dataset_name"]), o))
        dec["float_m"] = fl = float(P[:, 2].min() + z_off - desk)
        if fl <= FLOAT_MAX_M:
            continue
        ref = ref or RH.RefHands(tree_dir, ti)
        dmin, k, side = ref.min_dist(P if r.get("surf") else ref.object_points(r["name"], r["pose"]))
        dec.update(hand_min_m=dmin, hand_frame=k, hand_side=side)
        if dmin <= HAND_NEAR_M:
            dec.update(action=FLOAT_DROPPED)
        else:
            dec.update(action=FLOAT_LOWERED, dz_m=fl)
        print(
            f"{FLOAT_TAG} {tree_dir.name} {r['name']} {dec['action']} float {fl * 1000:.1f}mm · lowered "
            f"{dec['dz_m'] * 1000:.1f}mm · hand min {dmin * 1000:.1f}mm (threshold {HAND_NEAR_M * 1000:.0f}mm · "
            f"frame {k} · "
            f"{side or '-'})",
            flush=True,
        )
    n = {a: sum(d["action"] == a for d in out) for a in (FLOAT_LOWERED, FLOAT_DROPPED)}
    print(
        f"{FLOAT_TAG} {tree_dir.name} scene {mode} · objects {len(out)} · lowered {n[FLOAT_LOWERED]} · dropped "
        f"{n[FLOAT_DROPPED]} · float threshold {FLOAT_MAX_M * 1000:.0f}mm · hand threshold "
        f"{HAND_NEAR_M * 1000:.0f}mm",
        flush=True,
    )
    return out
