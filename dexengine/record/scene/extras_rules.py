"""Occlusion rule of the record render's extra objects (visual only): single owner of the thresholds, the
segment-to-hull test and the render session gate.

Rule 'drop an extra that occludes a moving object a lot': when, for any camera, the render frames in which the share
of camera -> target point segments passing through the extra's convex hull is >= OCC_BLOCK_FRAC add up to OCC_MIN_S
seconds or more, the extra is dropped with reason REASON in that episode only. Target points are counted as two
separate groups (so the many hand points do not bury the object share): hand = active hand link points · object =
moving objects (centres of the scene object slots object*: main objects and extras standing in the training scene). A
frame is occluded when either group's share is >= the threshold. Moved extras not in the training scene are already
not drawn (scene.extras moving), so they are not targets.
Two users call the same function (occ_rows): the render session gate (scene.extras_apply.apply; cameras = head left eye
per step + exo fixed; rollout state) and, outside Isaac, a given-camera occlusion measurement. numpy inside functions
only. Marker ``[EXTRAS_OCCLUDE] occlusion gate …``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the render session imports this module before AppLauncher: numpy inside functions only ([PRE_APP])
    import numpy as np

MARK = "[EXTRAS_OCCLUDE]"
# share of a frame's camera -> target segments passing the hull (at or above it the frame is occluded)
OCC_BLOCK_FRAC = 0.5
OCC_MIN_S = 1.0  # sum of occluded frames (seconds at the render fps)
REASON = "occlude"


class OccludeError(RuntimeError):
    """The occlusion rule cannot be applied: never passes silently."""


def rule_fields() -> dict:
    return {"occ_block_frac": OCC_BLOCK_FRAC, "occ_min_s": OCC_MIN_S}


def occ_min_frames(fps: float) -> int:
    """Occlusion threshold in frames (render fps; the caller passes the owner render.episode.FPS)."""
    return int(round(OCC_MIN_S * float(fps)))


def seg_hits_hull(E, Q, H):
    """Segments E → Q [R, 3] crossing the hull H [F, 4] (n·x + b ≤ 0)."""
    import numpy as np

    num, den = -(E @ H[:, :3].T + H[:, 3]), (Q - E) @ H[:, :3].T  # s·den ≤ num
    with np.errstate(divide="ignore", invalid="ignore"):
        s = num / den
    hi = np.where(den > 1e-12, s, np.inf).min(1).clip(max=1.0)
    lo = np.where(den < -1e-12, s, -np.inf).max(1).clip(min=0.0)
    return (lo <= hi) & np.all((np.abs(den) > 1e-12) | (num >= 0), axis=1)


def occ_rows(rows: list[dict], pts: dict, cams: dict, tg, fps: float, n_hand: int) -> dict:
    """Occlusion rule (single owner): per row the convex hull of its surface samples pts[name] (same coordinates as the
    target points) · cams {name: [3] | [F, 3]} · tg [F, M, 3] (target points per render frame: the first n_hand =
    hand, the rest = objects) -> row field occ_frames {camera: occluded frames} and (when any camera has >=
    occ_min_frames(fps)) reason occlude. Returns {camera: {name: occluded frames}}."""
    import numpy as np
    from scipy.spatial import ConvexHull

    out = {cn: {} for cn in cams}
    tg = np.asarray(tg, float)
    F, M = tg.shape[:2]
    if not rows or not F or not M:
        for r in rows:
            r["occ_frames"] = dict.fromkeys(cams, 0)
        return out
    if not 0 <= n_hand <= M:
        raise OccludeError(f"{MARK} hand points {n_hand} outside the target points {M}")
    groups = [g for g in (slice(0, n_hand), slice(n_hand, M)) if g.stop > g.start]
    Q = tg.reshape(-1, 3)
    for r in rows:
        P = np.asarray(pts[r["name"]], float)
        H, ctr = ConvexHull(P).equations, P.mean(0)
        rad = float(np.linalg.norm(P - ctr, axis=1).max())
        for cn, c in cams.items():
            E = np.repeat(np.broadcast_to(np.asarray(c, float), (F, 3)), M, 0)
            D = Q - E
            u = np.clip(((ctr - E) * D).sum(1) / np.maximum((D * D).sum(1), 1e-12), 0, 1)
            near = np.nonzero(np.linalg.norm(E + u[:, None] * D - ctr, axis=1) <= rad)[0]
            hit = np.zeros(len(Q), bool)
            hit[near] = seg_hits_hull(E[near], Q[near], H)
            hit = hit.reshape(F, M)
            out[cn][r["name"]] = int(np.any([hit[:, g].mean(1) >= OCC_BLOCK_FRAC for g in groups], 0).sum())
        r["occ_frames"] = {cn: out[cn][r["name"]] for cn in cams}
        if max(r["occ_frames"].values()) >= occ_min_frames(fps):
            r["reason"] = REASON
    return out


def gate(
    write, steps, cands: list[dict], pts: dict, origin: np.ndarray, shift_z: float, cam_fn, per: dict, fps: float
) -> None:
    """Occlusion gate (render session): per render frame (steps in order, repeated steps included) the cameras
    cam_fn() -> {name: world position [3]} of the written state · target points per['hand'][step] +
    per['object'][step] (world) -> occ_rows. Hull = the spawn place (pts + origin + shift_z)."""
    import numpy as np

    if not cands:
        return
    at: dict = {}
    for t in sorted(set(steps)):
        write(t)
        at[t] = {k: np.asarray(v, float).reshape(3) for k, v in cam_fn().items()}
    names = sorted(at[steps[0]])
    if any(sorted(v) != names for v in at.values()):
        raise OccludeError(f"{MARK} occlusion gate: camera names differ between steps")
    cams = {k: np.stack([at[t][k] for t in steps]) for k in names}
    hand = [np.asarray(per["hand"][t], float).reshape(-1, 3) for t in steps]
    if len({len(h) for h in hand}) != 1:
        raise OccludeError(f"{MARK} occlusion gate: hand point counts differ between steps")
    tg = np.stack(
        [
            np.concatenate([h, np.asarray(per["object"][t], float).reshape(-1, 3)])
            for h, t in zip(hand, steps, strict=True)
        ]
    )
    if not tg.shape[1]:
        raise OccludeError(f"{MARK} occlusion gate: 0 target points (no active hand links or object slots)")
    sh = origin + np.array([0.0, 0.0, shift_z])
    occ_rows(cands, {r["name"]: pts[r["name"]] + sh for r in cands}, cams, tg, fps, len(hand[0]))
    print(
        f"{MARK} occlusion gate · cameras {names} · render frames {len(steps)} · target points {tg.shape[1]} (hand "
        f"{len(hand[0])} · object {tg.shape[1] - len(hand[0])}, per group) · threshold "
        f"{OCC_BLOCK_FRAC} x {occ_min_frames(fps)} frames · {[(r['name'], r.get('occ_frames')) for r in cands]}",
        flush=True,
    )


# ---- overhang rule (always on) -------------------------------------------------------------------------------
# A static extra object whose top-down outline lies more than OVERHANG_MAX_FRAC outside the desk box is dropped with
# reason OVERHANG_REASON (same measure and threshold as the training-scene placement rule,
# placement.scene.static_desk). The decision order (off_desk, penetrate, then this rule) belongs to scene.extras.decide.
# Rows get OVERHANG_FIELD, meta rule.overhang_max_frac, and the marker
# ``[EXTRAS_OVERHANG] on · <clip> measured n · over m ...`` (printed even when nothing was measured).
OVERHANG_MARK = "[EXTRAS_OVERHANG]"
OVERHANG_REASON = "desk_overhang"
OVERHANG_FIELD = "f0_overhang"


def overhang_judge(row: dict, xy, box) -> None:
    """One row: store the overhang fraction of the scene xy samples (N, 2) and add the reason above the threshold."""
    from dexengine.placement.scene import static_desk as SD

    frac = float(SD.overhang_frac(xy, box=box))
    row[OVERHANG_FIELD] = round(frac, 4)
    if frac > SD.OVERHANG_MAX_FRAC:  # same predicate as the placement rule (unrounded value)
        row["reason"] = OVERHANG_REASON


def overhang_rule_fields() -> dict:
    from dexengine.placement.scene import static_desk as SD

    return {"overhang_max_frac": SD.OVERHANG_MAX_FRAC}


def overhang_marker(clip: str, rows: list[dict]) -> str:
    from dexengine.placement.scene import static_desk as SD

    seen = [r for r in rows if OVERHANG_FIELD in r]
    hit = [r for r in seen if r["reason"] == OVERHANG_REASON]
    return (
        f"{OVERHANG_MARK} on · {clip} measured {len(seen)} · over {len(hit)} (desk-outside area > "
        f"{SD.OVERHANG_MAX_FRAC}) " + " ".join(f"{r['name']}={r[OVERHANG_FIELD]}" for r in hit)
    ).rstrip()


def overhang_say(clip: str, rows: list[dict]) -> None:
    print(overhang_marker(clip, rows), flush=True)


def overhang_log_problems(text: str) -> list[str]:
    """Render session log: the overhang marker must be present."""
    has = any(ln.startswith(f"{OVERHANG_MARK} on") for ln in text.splitlines())
    return [] if has else [f"overhang marker '{OVERHANG_MARK} on' missing"]


if TYPE_CHECKING:
    import numpy as np


class RobotPtsError(RuntimeError):
    """Robot surface samples cannot be made."""


def _body_samples(body, xc, inv, sample_m):
    """Surface samples (body frame) of the visible render meshes under one rigid body (child bodies pruned)."""
    import numpy as np

    from pxr import Usd, UsdGeom, UsdPhysics

    from dexengine.placement.scene import ref_hand_geom as RH

    pts = []
    it = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
    for p in it:
        if p != body and p.HasAPI(UsdPhysics.RigidBodyAPI):
            it.PruneChildren()
            continue
        if not p.IsA(UsdGeom.Mesh):
            continue
        im = UsdGeom.Imageable(p)
        if im.ComputeVisibility() == UsdGeom.Tokens.invisible or im.ComputePurpose() in ("guide", "proxy"):
            continue
        m = UsdGeom.Mesh(p)
        V = np.asarray(m.GetPointsAttr().Get() or [], dtype=float)
        cnt = np.asarray(m.GetFaceVertexCountsAttr().Get() or [], dtype=int)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get() or [], dtype=int)
        if not len(V) or not len(cnt):
            continue
        M = np.array(xc.GetLocalToWorldTransform(p) * inv, dtype=float)  # row-vector form
        Vb = np.c_[V, np.ones(len(V))] @ M
        Vb = Vb[:, :3]
        F, s = [], 0
        for c in cnt:
            F += [(idx[s], idx[s + j], idx[s + j + 1]) for j in range(1, c - 1)]
            s += c
        pts.append(RH.mesh_samples(Vb, np.asarray(F, dtype=int), sample_m))
    return pts


def robot_local_points(raw_env, binds, sample_m: float, prim_path) -> list[tuple]:
    """Per robot articulation [(asset, body index, body-frame surface samples (N,3) spaced sample_m)] from the render
    meshes (visible Mesh). Body prim = the RigidBodyAPI prim under the articulation prim whose name is in body_names
    (exactly one, else dies). prim_path(raw_env, name) = the articulation prim path (owned by the caller)."""
    import numpy as np

    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    xc = UsdGeom.XformCache()
    out = []
    for b in binds:
        if b.kind != "artic" or b.name.startswith("object"):
            continue
        a = b.asset
        root = stage.GetPrimAtPath(prim_path(raw_env, b.name))
        if not root.IsValid():
            raise RobotPtsError(f"articulation {b.name}: prim missing")
        bodies: dict = {}
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                bodies.setdefault(p.GetName(), []).append(p)
        for bi, bn in enumerate(a.body_names):
            got = bodies.get(bn, [])
            if len(got) != 1:
                raise RobotPtsError(f"{b.name} body {bn}: {len(got)} prims (must be exactly one)")
            body = got[0]
            inv = xc.GetLocalToWorldTransform(body).GetInverse()
            pts = _body_samples(body, xc, inv, sample_m)
            if pts:
                out.append((a, bi, np.concatenate(pts)))
    if not out:
        raise RobotPtsError("0 robot render mesh samples: the rollout overlap cannot be measured")
    return out


def quat_apply(qwxyz: np.ndarray, P: np.ndarray) -> np.ndarray:
    import numpy as np

    w, v = qwxyz[0], qwxyz[1:]
    t = 2 * np.cross(v, P)
    return P + w * t + np.cross(v, t)
