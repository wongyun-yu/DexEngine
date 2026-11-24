"""Rule S-envelope: a still extra next to where the objects move stands in the scene and collides with them.

Why: an untouched still extra that receives nothing is visual only (no body in training), so a main object the
policy moves passes through it unseen. The objects that can meet it are the ones near the recorded motion.

Measurement (the interaction envelope): the movers are the bodies the scene moves — the main objects and the extra
slots that move (kind moving · carried_attached). Their vertices over every tree frame form the swept set;
a visual-only extra with a mesh whose frame-0 vertices come within `scene_requirement.ENVELOPE_M` of that set is in
the envelope (gap = the nearest vertex distance; the mover is recorded). The recorded hands are not movers: a still
extra only the hand passes stays visual only.
Rule, after the receiver passes (extra_slots.clip_rows):
  in the envelope and kept by the placement rule → STATIC_SPAWN `why=envelope`, kind envelope (objects ON, robot per
     S-robot-static; pinned at its frame-0 pose like every static spawn); a mover recorded deeper than
     scene_requirement.TRUE_SHAPE_OVERLAP_M into it on the shapes physics uses (`overlap`) drops the clip
     (`overlap_drop`: a real overlap is dropped, never shown as a body passing through)
  in the envelope but on the robot's footprint or off the desk (place_rule.drop_why — an untouched still extra is
     never handled, so the rule applies) → stays visual only, row `envelope.drop` says why
Slot cap (B6): when the clip's slots exceed extra_slots.K_MAX, envelope rows leave first, the farthest first (they
stay visual only, row `envelope.cap` true); the remaining overflow (no envelope row left) is the old error.
"""

from __future__ import annotations

import numpy as np

from dexengine.data.objects import extra_slots_geom as XG
from dexengine.data.objects import scene_requirement as SR

RULE_ID = "S-envelope"
WHY = "envelope"
MAX_MOVER_POINTS = 600  # vertices per mover kept (evenly spaced) — the swept set is dense in time already
MEMO: dict = {}


def _even(V: np.ndarray, n: int) -> np.ndarray:
    return V if len(V) <= n else V[np.linspace(0, len(V) - 1, n).astype(int)]


def swept(tree_dir, ti: dict, q: np.ndarray, movers: list[dict], placed) -> dict:
    """{mover name: (T·n, 3) vertices over every frame}. Rigid movers: body vertices × the frame poses; articulated:
    the placer per frame (joint value)."""
    out = {}
    for m in movers:
        if m.get("articulation"):
            out[str(m["name"])] = np.concatenate([_even(placed(m, f), MAX_MOVER_POINTS) for f in range(len(q))])
            continue
        T = XG.pose_mats(q, int(m["qpos_start"]))
        W0 = _even(placed(m, 0), MAX_MOVER_POINTS)
        B = (W0 - T[0, :3, 3]) @ T[0, :3, :3]  # body frame
        out[str(m["name"])] = (np.einsum("tij,nj->tni", T[:, :3, :3], B) + T[:, None, :3, 3]).reshape(-1, 3)
    return out


def gap(Vc: np.ndarray, sweep: dict) -> tuple[float, str] | None:
    """(nearest vertex distance ≤ ENVELOPE_M, mover) of a still extra's vertices Vc to the swept set; None = outside."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    d, lo, hi = SR.ENVELOPE_M, Vc.min(0), Vc.max(0)
    tree, best = None, None
    for name, W in sweep.items():
        m = W[np.all((lo - d <= W) & (hi + d >= W), axis=1)]
        if not len(m):
            continue
        if tree is None:
            tree = cKDTree(Vc)
        g = float(tree.query(m, distance_upper_bound=d)[0].min())
        if np.isfinite(g) and (best is None or g < best[0]):
            best = (g, name)
    return best


def envelope_pass(tree_dir, ti: dict, q: np.ndarray, rows: list[dict], drop, measure_overlap: bool = True) -> None:
    """Visual-only extras in the envelope become envelope static spawns (module doc). `drop(o)` = the placement rule's
    reason for still extra o ('' = stands). Rows are updated in place. measure_overlap=False: a new envelope spawn's
    overlap with the movers (`overlap`, a recorded measure and the clip drop) is not measured — its class and kind are
    the same."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415 — class and kind vocabulary

    objs = {str(o["name"]): o for o in ti.get("objects") or [] if o.get("qpos_start") is not None}
    cand = [r for r in rows if r["cls"] == XS.VISUAL_ONLY and objs.get(r["name"], {}).get("mesh_dir")]
    if not cand:
        return
    movers = [o for o in objs.values() if o.get("role") == "main" and o.get("mesh_dir")]
    movers += [objs[r["name"]] for r in rows if r["cls"] == XS.SLOT and objs.get(r["name"], {}).get("mesh_dir")]
    if not movers:
        return
    placed, _ = XG.geom(tree_dir, ti, q)
    sweep = swept(tree_dir, ti, q, movers, placed)
    for r in cand:
        g = gap(placed(objs[r["name"]], 0), sweep)
        if g is None:
            continue
        o = objs[r["name"]]
        why = drop(o) or no_shape(str(ti["dataset_name"]), o)
        r["envelope"] = {"gap_m": round(g[0], 4), "by": g[1], "drop": why, "cap": False}
        if not why:
            r.update(cls=XS.STATIC_SPAWN, still=r["why"], why=WHY, static=True, kind=XS.KIND_ENVELOPE)
            if measure_overlap:
                overlap(tree_dir, ti, q, r, o, movers)


def overlap(tree_dir, ti: dict, q: np.ndarray, r: dict, o: dict, movers: list[dict]) -> None:
    """The recorded overlap of every mover with an envelope body in the shape physics uses (touch_gate.pair_depth on
    collider_shape rulers): row `envelope.depth_m` (None with `envelope.unmeasured` when a body cannot be measured), and
    `overlap_drop` {partner, depth_m} above scene_requirement.TRUE_SHAPE_OVERLAP_M — the clip is dropped
    (collider_shape.overlap_drop, step 2f)."""
    from dexengine.data.objects import touch_gate as TG  # noqa: PLC0415

    ds = str(ti["dataset_name"])
    st, why = TG.static_body(ds, o, r)
    T0 = XG.pose_mats(q[:1], int(o["qpos_start"]))[0]
    deep, blind = [], [] if st is not None else [why]
    for m in movers if st is not None else []:
        b, _w = TG.body_of(ds, m, tree_dir)
        if b is None:
            blind.append(str(m["name"]))
        else:
            deep.append((TG.pair_depth(b, q, XG.pose_mats(q, int(m["qpos_start"])), st, T0), str(m["name"])))
    worst = max(deep, default=(0.0, ""))
    r["envelope"]["depth_m"] = None if blind else round(worst[0], 4)
    if blind:
        r["envelope"]["unmeasured"] = blind
    if worst[0] > SR.TRUE_SHAPE_OVERLAP_M:
        r["overlap_drop"] = {"partner": worst[1], "depth_m": round(worst[0], 4)}


def no_shape(ds: str, o: dict) -> str:
    """'no collision shape' when the envelope body would stand as convex parts (S-true-shape off for the dataset) and
    it has none; '' otherwise (a triangles body stands as its visual mesh)."""
    from dexengine.data.objects import collider_shape as CS  # noqa: PLC0415
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415
    from dexengine.data.objects import touch_gate as TG  # noqa: PLC0415

    if CS.shape_of(ds, XS.KIND_ENVELOPE, True) == CS.TRIANGLES:
        return ""
    return "" if TG.parts_of(ds, o) is not None else "no collision shape"


def cap(rows: list[dict], k_max: int) -> list[str]:
    """Slot cap (module doc): envelope rows leave, farthest first, until the slots fit k_max → the names that left."""
    from dexengine.data.objects import extra_slots as XS  # noqa: PLC0415

    over = sum(r["cls"] in XS.SLOTTED for r in rows) - k_max
    env = sorted((r for r in rows if r.get("why") == WHY), key=lambda r: -r["envelope"]["gap_m"])
    gone = []
    for r in env[: max(0, over)]:
        r.update(cls=XS.VISUAL_ONLY, why=r.pop("still"), static=False)
        r.pop("kind", None)
        r["envelope"]["cap"] = True
        gone.append(r["name"])
    return gone
