"""Hand visibility in the head camera: per-frame head view (hands inside the left-eye image) and the per-episode
occlusion tally.

Head view (always on): per side, all hand link origins (render.track.track_points) are projected with the left-eye pose
of the render frame (Rig.pose("collect_ego_left"), framing.to_cam); inside = in front (z > 0) and inside the cropped
image (framing.inside, the intrinsics of the stored depth). in_view = inside fraction >= IN_VIEW_MIN_FRAC; occlusion is
ignored. File (render folder FILE -> dataset dataset_path(i)): frame_index, frame_step [F]; per side <side>.in_view
bool [F], <side>.in_frame_frac float32 [F], <side>.uv float32 [F, N, 2] (pixels, NaN behind the camera),
<side>.link_names [N]. Oracle: the active sides' inside fraction equals the occlusion tally's hand.in_frame.
Marker `[HEAD_VIEW] on ...` per render session (log_problems).

Occlusion tally: per rendered frame the hand points (active hand link origins) and object points are projected into
the left eye and compared with the rendered depth at that pixel: visible = inside and (depth invalid, i.e. nothing in
front, or depth >= point depth - margin). The margin is the body's own size, so self-occlusion is not counted: hand =
HAND_MARGIN_M, object = half diagonal of its visual bounds + OBJ_PAD_M. A measurement only; head pose selection does
not use it. Meta visibility{hand, object: {points, in_frame, visible}}; marker [COLLECT_VIS] (vis_log_problems).
numpy is imported inside functions (the render session imports this module before the AppLauncher).
"""

from __future__ import annotations

from pathlib import Path

from dexengine.core.errors import InputMissingError

MARK = "[HEAD_VIEW]"
CAMERA = "collect_ego_left"
IN_VIEW_MIN_FRAC = 0.5
FILE = "head_view.npz"
PATH = "head_view/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.npz"  # path inside the dataset (info.json)
# RULE lands in render meta and info.json
RULE = (
    f"hand link origins projected into {CAMERA} · in image = z > 0 ∧ inside the cropped image "
    f"(collect_frame.inside) · in_view = in-image fraction ≥ {IN_VIEW_MIN_FRAC} · occlusion ignored"
)


def log_problems(text: str) -> list[str]:
    """Render session log: the head-view marker must be present."""
    has = any(ln.startswith(f"{MARK} on") for ln in text.splitlines())
    return [] if has else [f"head view marker '{MARK} on' missing"]


def dataset_path(i: int) -> str:
    return PATH.format(episode_chunk=0, episode_index=i)


def project(P, pos, q, k: dict):
    """World points [N, 3] -> (inside bool [N], pixel uv [N, 2], NaN behind the camera)."""
    import numpy as np

    from dexengine.record.render import framing as CF

    pc = CF.to_cam(P, pos, q)
    ins = CF.inside(pc, k)
    uv = np.full((len(pc), 2), np.nan)
    front = pc[:, 2] > 0
    uv[front, 0] = k["fx"] * pc[front, 0] / pc[front, 2] + k["cx"]
    uv[front, 1] = k["fy"] * pc[front, 1] / pc[front, 2] + k["cy"]
    return ins, uv


class Recorder:
    """add() per render frame, write() at the end (render folder) -> render meta field."""

    def __init__(self, links: dict):
        self.links = {s: list(v) for s, v in links.items()}  # side -> link names (point order)
        if not self.links:
            raise InputMissingError(f"{MARK} no sides — hand links not found")
        self.ins = {s: [] for s in self.links}
        self.uv = {s: [] for s in self.links}

    def add(self, pts: dict, pos, q, k: dict) -> None:
        for s in self.links:
            P = pts[s]
            if len(P) != len(self.links[s]):
                raise SystemExit(f"{MARK} {s} points {len(P)} != links {len(self.links[s])}")
            ins, uv = project(P, pos, q, k)
            self.ins[s].append(ins)
            self.uv[s].append(uv)

    def oracle(self, active: list[str], tally_in_frame) -> float | None:
        """Inside fraction over the active sides' points (rounded like the tally's hand.in_frame); dies on mismatch."""
        import numpy as np

        n = sum(int(np.size(self.ins[s][0])) * len(self.ins[s]) for s in active if s in self.ins and self.ins[s])
        got = round(sum(int(np.sum(self.ins[s])) for s in active if s in self.ins) / n, 4) if n else None
        if got != tally_in_frame:
            raise SystemExit(f"{MARK} oracle — active hands inside {got} != occlusion tally {tally_in_frame}")
        return got

    def write(self, out: Path, steps: list[int]) -> dict:
        import numpy as np

        F = len(steps)
        arrs = {"frame_index": np.arange(F, dtype=np.int64), "frame_step": np.asarray(steps, np.int64)}
        summ = {}
        for s in self.links:
            ins = np.stack(self.ins[s]) if self.ins[s] else np.zeros((0, len(self.links[s])), bool)
            if ins.shape[0] != F:
                raise SystemExit(f"{MARK} {s} frames {ins.shape[0]} != render frames {F}")
            frac = ins.mean(axis=1).astype(np.float32)
            view = frac >= IN_VIEW_MIN_FRAC
            arrs[f"{s}.in_view"] = view
            arrs[f"{s}.in_frame_frac"] = frac
            arrs[f"{s}.uv"] = np.stack(self.uv[s]).astype(np.float32)
            arrs[f"{s}.link_names"] = np.asarray(self.links[s])
            summ[s] = {
                "links": len(self.links[s]),
                "in_view_frames": int(view.sum()),
                "in_view_frac": round(float(view.mean()), 4) if F else None,
            }
        np.savez_compressed(out / FILE, **arrs)
        meta = {
            "file": FILE,
            "camera": CAMERA,
            "frames": F,
            "rule": RULE,
            "in_view_min_frac": IN_VIEW_MIN_FRAC,
            "sides": summ,
        }
        print(
            f"{MARK} on · {CAMERA} {F} frames · "
            + " · ".join(f"{s} in view {v['in_view_frames']}/{F}" for s, v in summ.items())
            + f" · {out / FILE}",
            flush=True,
        )
        return meta


# ---- store (store.store, CPU) ----------------------------------------------------------------------------
def store(render_dir: Path, rmeta: dict, out: Path, i: int, clip: str) -> None:
    """Head-view file of the render folder -> dataset dataset_path(i); dies when the render meta lacks head_view."""
    import shutil

    if "head_view" not in rmeta:
        raise InputMissingError(f"{MARK} {clip}: no head_view in the render meta")
    dst = out / dataset_path(i)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(render_dir) / rmeta["head_view"]["file"], dst)


def info_fields(eps: list[dict]) -> dict:
    """info.json collect fields (only when an episode carries head_view)."""
    return {"head_view_path": PATH, "head_view_rule": RULE} if any("head_view" in e for e in eps) else {}


# ---- occlusion tally ----------------------------------------------------------------------------------------
VIS_MARK = "[COLLECT_VIS]"
HAND_MARGIN_M = 0.10
OBJ_PAD_M = 0.02


def vis_log_problems(text: str) -> list[str]:
    """Render session log: the occlusion tally marker must be present."""
    return [] if f"{VIS_MARK} " in text else [f"occlusion marker '{VIS_MARK}' missing"]


def visible(pc, depth, k: dict, margin):
    """Optical-frame points [N, 3], cropped depth [H, W] (m) -> (inside, visible) bool [N]; margin scalar | [N]."""
    import numpy as np

    from dexengine.record.render import framing as CF

    pc = np.asarray(pc, float)
    ins = CF.inside(pc, k)
    vis = np.zeros(len(pc), bool)
    if ins.any():
        z = pc[ins, 2]
        u = np.clip((k["fx"] * pc[ins, 0] / z + k["cx"]).astype(int), 0, depth.shape[1] - 1)
        v = np.clip((k["fy"] * pc[ins, 1] / z + k["cy"]).astype(int), 0, depth.shape[0] - 1)
        # pick the pixels first, then cast: the same float64 values as casting the whole image (a float32 -> float64
        # cast is exact per element) without a full-image copy per call (two calls per rendered frame)
        d = np.asarray(depth)[v, u].astype(float)
        m = np.broadcast_to(np.asarray(margin, float), (len(pc),))[ins]
        with np.errstate(invalid="ignore"):
            vis[ins] = ~np.isfinite(d) | (d <= 0) | (d >= z - m)
    return ins, vis


def object_half_diag(binds) -> list[float]:
    """Half diagonal of each object binding's visual bounds (USD local bounds, m), for object_margins."""
    import numpy as np

    import omni.usd
    from pxr import Usd, UsdGeom

    from dexengine.record.render import body as BODY

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    out = []
    for b in binds:
        if not b.name.startswith("object"):
            continue
        r = cache.ComputeLocalBound(stage.GetPrimAtPath(BODY._root_path(b))).ComputeAlignedRange()
        if r.IsEmpty():
            raise SystemExit(f"{VIS_MARK} object {b.name} has empty visual bounds")
        out.append(0.5 * float(np.linalg.norm(np.array(r.GetSize()))))
    return out


def object_boxes(binds) -> list[tuple]:
    """Local visual box (lo, hi)[3] m of each object binding, before the root prim pose (ComputeUntransformedBound),
    times the root prim's own scale (the physics root pose has no scale). Read by object_corners."""
    import numpy as np

    import omni.usd
    from pxr import Usd, UsdGeom

    from dexengine.record.render import body as BODY

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    out = []
    for b in binds:
        if not b.name.startswith("object"):
            continue
        prim = stage.GetPrimAtPath(BODY._root_path(b))
        r = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if r.IsEmpty():
            raise SystemExit(f"{VIS_MARK} object {b.name} has empty visual bounds")
        M = UsdGeom.Xformable(prim).GetLocalTransformation()
        sc = np.array([np.linalg.norm([M[i][0], M[i][1], M[i][2]]) for i in range(3)])
        out.append((np.array(r.GetMin(), float) * sc, np.array(r.GetMax(), float) * sc))
    return out


def object_corners(binds, st, centers: list) -> list:
    """Object box corners [8k, 3] per step (framing.obb_corners): centre = object root position from track_points
    (world), orientation = the trajectory state's root quaternion (wxyz), box = object_boxes."""
    import numpy as np

    from dexengine.record.render import framing as CF

    objs = [b for b in binds if b.name.startswith("object")]
    boxes = object_boxes(binds)
    quats = [np.asarray(st, float)[:, b.root_cols][:, 3:7] for b in objs]
    return [
        np.concatenate([CF.obb_corners(np.asarray(c, float)[j], quats[j][t], *boxes[j]) for j in range(len(objs))])
        for t, c in enumerate(centers)
    ]


def object_margins(binds) -> list[float]:
    """Per object binding: margin = half diagonal (object_half_diag) + OBJ_PAD_M."""
    return [round(h + OBJ_PAD_M, 4) for h in object_half_diag(binds)]


class Tally:
    """add() per frame, meta() at the end."""

    def __init__(self, obj_margin: list[float]):
        self.m = obj_margin
        self.n = {g: [0, 0, 0] for g in ("hand", "object")}  # points, inside, visible

    def add(self, pts: dict, pos, q, depth, k: dict) -> None:
        import numpy as np

        from dexengine.record.render import framing as CF

        for g, margin in (("hand", HAND_MARGIN_M), ("object", self.m)):
            P = pts[g]
            if not len(P):
                continue
            ins, vis = visible(CF.to_cam(P, pos, q), depth, k, margin)
            self.n[g][0] += len(P)
            self.n[g][1] += int(np.sum(ins))
            self.n[g][2] += int(np.sum(vis))

    def meta(self, frames: int) -> dict:
        out = {
            g: {
                "points": n[0],
                "in_frame": round(n[1] / n[0], 4) if n[0] else None,
                "visible": round(n[2] / n[0], 4) if n[0] else None,
            }
            for g, n in self.n.items()
        }
        print(
            f"{VIS_MARK} left eye {frames} frames · inside -> unoccluded · hand {out['hand']['in_frame']} -> "
            f"{out['hand']['visible']} · object {out['object']['in_frame']} -> {out['object']['visible']} "
            f"(margin hand {HAND_MARGIN_M} m · object {self.m} m)",
            flush=True,
        )
        return dict(out, frames=frames, hand_margin_m=HAND_MARGIN_M, object_margin_m=self.m)
