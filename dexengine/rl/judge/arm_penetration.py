"""Physical plausibility of judge rollouts — an auxiliary value, never the judge value: the penetration of the link
pairs the simulator does not collide.

Pairs (the classes of the collision filter, `classify_body`):
  arm_arm    right hand/arm (RH · RA) <-> left hand/arm (LH · LA)
  arm_torso  hand/arm (RH · RA · LH · LA) <-> torso/base (OTHER)
  Two links joined directly by a joint are left out (PhysX always disables that pair's collision; touching is
  normal). The original joint cut by the two-articulation split (RobotL: R root link <-> first left-arm link) is a
  joint pair too, derived by the split's owner `split_artic.cut_joint_pairs` alone (otherwise torso <-> left shoulder
  counts as ~30 mm penetration in every pose and the arm_torso time fraction sticks at 1.0). Marker `[ARM_PEN_SPLIT]`.
Geometry = the colliders the simulator uses: for every link of the spawned robot stage (env 0), the convex hull of its
collision prims (mesh points · capsule · sphere · box) in link body coordinates, up to HULL_VERTS_MAX vertices (the
PhysX hull vertex limit).
Depth = for surface samples of one hull (vertices + a SAMPLE_M grid on its faces, up to SAMPLES_MAX) inside every face
of the other hull, the distance to the nearest face; the maximum of both directions (vertices alone miss the
penetration of two crossed bars or two stacked boxes; the sampling spacing bounds the error from below). Broad phase =
only (env, pair) whose hull bounding spheres overlap are measured.
Time fraction = the fraction of measured env-steps with depth > PEN_EPS_MM for that kind. Warm-up batches and
finished envs are not counted.
Marker `[ARM_PEN] {json}`, one line at the end of a judge. Reader: judge/metrics.py (it reads this file's TAG) →
arm_pen of `<judge>_clips.json` → arm_pen of the judge record.
"""

from __future__ import annotations

import json

import numpy as np
import torch

TAG = "[ARM_PEN] "
SPLIT_TAG = "[ARM_PEN_SPLIT] "  # the joint cut by the split was added as a joint pair (prefix distinct from TAG)
PEN_EPS_MM = 0.5  # minimum depth counted as penetration, above float and hull-approximation noise
HULL_VERTS_MAX = 64  # hull vertex limit (farthest-point selection); bounds the face count
SAMPLE_M = 0.01  # sample spacing on hull faces (m)
SAMPLES_MAX = 512  # surface samples per hull (farthest-point selection)
CHUNK = 256  # (env, pair) measured at once; memory = CHUNK × samples × faces
KINDS = ("arm_arm", "arm_torso")
_RIGHT, _LEFT = ("RH", "RA"), ("LH", "LA")


def _kind(a: str, b: str) -> int | None:
    """The pair kind (index into KINDS) of two link classes, symmetric; None if neither."""
    for x, y in ((a, b), (b, a)):
        if x in _RIGHT and y in _LEFT:
            return 0
        if x in _RIGHT + _LEFT and y == "OTHER":
            return 1
    return None


def _shape_points(prim) -> np.ndarray | None:
    """Local points of one collision prim: mesh = its vertices; capsule, sphere, box = surface samples. Unknown
    type = None."""
    from pxr import UsdGeom

    d = np.array(
        [[x, y, z] for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)], float
    )
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    if prim.IsA(UsdGeom.Mesh):
        p = prim.GetAttribute("points").Get()
        return np.asarray(p, float) if p else None
    if prim.IsA(UsdGeom.Sphere):
        return d * float(prim.GetAttribute("radius").Get())
    if prim.IsA(UsdGeom.Cube):
        return (
            np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
            * float(prim.GetAttribute("size").Get())
            / 2
        )
    if prim.IsA(UsdGeom.Capsule):
        r, h = float(prim.GetAttribute("radius").Get()), float(prim.GetAttribute("height").Get())
        ax = "XYZ".index(str(prim.GetAttribute("axis").Get() or "Z"))
        off = np.zeros(3)
        off[ax] = h / 2
        return np.concatenate([d * r + off, d * r - off])
    return None


def _fps(v: np.ndarray, k: int) -> np.ndarray:
    """Farthest-point selection of k points, deterministic (first point = the farthest from the centroid)."""
    if len(v) <= k:
        return v
    pick = [int(np.argmax(np.linalg.norm(v - v.mean(0), axis=1)))]
    dist = np.linalg.norm(v - v[pick[0]], axis=1)
    while len(pick) < k:
        pick.append(int(np.argmax(dist)))
        dist = np.minimum(dist, np.linalg.norm(v - v[pick[-1]], axis=1))
    return v[pick]


def _hull(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices (V ≤ HULL_VERTS_MAX, 3), face planes (F, 4) with n·x + c ≤ 0 inside, surface samples
    (P ≤ SAMPLES_MAX, 3))."""
    from scipy.spatial import ConvexHull

    v = _fps(pts[ConvexHull(pts, qhull_options="QJ").vertices], HULL_VERTS_MAX)
    h = ConvexHull(v, qhull_options="QJ")
    s = [v]
    for a, b, c in v[h.simplices]:
        n = max(1, int(np.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a)) / SAMPLE_M)))
        ij = np.array([(i, j) for i in range(n + 1) for j in range(n + 1 - i)], float) / n
        s.append(a + ij[:, :1] * (b - a) + ij[:, 1:] * (c - a))
    return v, h.equations, _fps(np.unique(np.round(np.concatenate(s), 9), axis=0), SAMPLES_MAX)


def link_hulls(stage, robot_root: str, classify, force_left: bool = False, asset: int = 0) -> tuple[list[dict], set]:
    """(hulls [{body, asset, cls, verts, planes, samples}] in link body coordinates, set of link-name pairs joined by
    a joint). Dies on an unknown collision prim type (dropping it would show that link's penetration as 0).
    force_left = the separate articulation RobotL (everything under it is left side)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    cache, out, joints = UsdGeom.XformCache(), [], set()
    root = stage.GetPrimAtPath(robot_root)
    if not root.IsValid():
        raise RuntimeError(f"[ARM_PEN] robot prim {robot_root} is missing")
    for body in Usd.PrimRange(root):
        if body.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(body)
            ends = [t.name for r in (j.GetBody0Rel(), j.GetBody1Rel()) for t in r.GetTargets()]
            if len(ends) == 2:
                joints.add(frozenset(ends))
        if not body.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        cls = classify(body.GetName())
        if force_left:
            cls = "LH" if cls in ("RH", "LH") else "LA"
        inv = cache.GetLocalToWorldTransform(body).GetInverse()
        it = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
        for q in it:
            if q != body and q.HasAPI(UsdPhysics.RigidBodyAPI):
                it.PruneChildren()
                continue
            if not q.HasAPI(UsdPhysics.CollisionAPI):
                continue
            pts = []
            for g in Usd.PrimRange(q, Usd.TraverseInstanceProxies()):
                if not g.IsA(UsdGeom.Gprim):
                    continue
                p = _shape_points(g)
                if p is None:
                    raise RuntimeError(f"[ARM_PEN] {g.GetPath()}: unknown collision shape {g.GetTypeName()}")
                m = np.array(cache.GetLocalToWorldTransform(g) * inv, float)  # row vectors: p_body = p · M
                pts.append(np.c_[p, np.ones(len(p))] @ m)
            if pts:
                verts, planes, samples = _hull(np.concatenate(pts)[:, :3])
                out.append(
                    {
                        "body": body.GetName(),
                        "asset": asset,
                        "cls": cls,
                        "verts": verts,
                        "planes": planes,
                        "samples": samples,
                    }
                )
    return out, joints


def split_joint_pairs(stage, right_root: str, left_root: str, cut_pairs) -> set:
    """Body-name pairs of the original joint the two-articulation split cut; `cut_pairs` = `split_artic.cut_joint_pairs`
    (the split contract's one owner; it dies when it cannot derive them, since an empty set would count the cut
    neighbours as penetrating). Prints one marker line (SPLIT_TAG)."""
    pairs, why = cut_pairs(stage, right_root, left_root)
    names = sorted(tuple(sorted(p)) for p in pairs)
    print(
        SPLIT_TAG + f"{len(pairs)} original joint pairs cut by the split are excluded as joint pairs {names} — "
        f"L root {why['mount']} = the stand-in of R root {why['r_base']} (joints {why['joints']} · transform "
        f"difference {why['pose_err']:.1e}) · owner split_artic.cut_joint_pairs",
        flush=True,
    )
    return pairs


def pair_table(hulls: list[dict], joints: set) -> tuple[list[tuple[int, int, int]], int]:
    """[(hull a, hull b, kind)] and the number of pairs left out because a joint connects them."""
    pairs, skipped = [], 0
    for i, a in enumerate(hulls):
        for j in range(i + 1, len(hulls)):
            b = hulls[j]
            k = _kind(a["cls"], b["cls"])
            if k is None:
                continue
            if frozenset((a["body"], b["body"])) in joints:
                skipped += 1
                continue
            pairs.append((i, j, k))
    return pairs, skipped


def depth(va, pa, qa, pb, qb, planes_b) -> torch.Tensor:
    """Max depth (K,) [m] of hull a's samples (body coordinates va (K,P,3), pose pa (K,3), qa (K,4) wxyz) inside hull
    b (pose pb, qb, planes_b (K,F,4)): for a point inside every face, the distance to the nearest face."""
    from isaaclab.utils.math import quat_apply, quat_apply_inverse  # wxyz in this build

    v = va.shape[1]
    w = quat_apply(qa[:, None].expand(-1, v, 4), va) + pa[:, None]
    loc = quat_apply_inverse(qb[:, None].expand(-1, v, 4), w - pb[:, None])
    s = torch.einsum("kvi,kfi->kvf", loc, planes_b[..., :3]) + planes_b[:, None, :, 3]
    return (-s.amax(-1)).clamp(min=0.0).amax(-1)


def _pad(arrs: list[np.ndarray], fill_row) -> torch.Tensor:
    n = max(len(a) for a in arrs)
    return torch.tensor(
        np.stack([np.concatenate([a, np.repeat([fill_row(a)], n - len(a), 0)]) for a in arrs]), dtype=torch.float32
    )


class ArmPen:
    """Penetration collector of one judge (one run_eval_balanced call): batch(assignment, warm-up?) → sample(done)
    every step → report()."""

    def __init__(
        self, hulls: list[dict], pairs: list, skipped: int, bodies: list[str], poses, clip_names: list[str], device
    ):
        self.names, self.poses, self.skipped = clip_names, poses, skipped  # poses() → (pos (N,B,3), quat (N,B,4))
        col = {b: i for i, b in enumerate(bodies)}  # (asset index, body name) → merged column
        self.hb = torch.tensor([col[(h["asset"], h["body"])] for h in hulls], device=device)
        self.verts = _pad([h["samples"] for h in hulls], lambda a: a[0]).to(device)  # samples (and bounding spheres)
        self.planes = _pad([h["planes"] for h in hulls], lambda a: [0.0, 0.0, 0.0, -1e9]).to(device)
        self.center = torch.tensor(np.stack([h["verts"].mean(0) for h in hulls]), dtype=torch.float32, device=device)
        self.rad = (self.verts - self.center[:, None]).norm(dim=-1).amax(-1)
        p = torch.tensor(pairs, device=device).reshape(-1, 3)
        self.pa, self.pb, self.pk = p[:, 0], p[:, 1], p[:, 2]
        c = len(clip_names)
        self.mm = torch.zeros(len(KINDS), c, device=device)  # [kind, clip] max depth mm
        self.npen = torch.zeros(len(KINDS), c, device=device)  # [kind, clip] env-steps with depth > PEN_EPS_MM
        self.n = torch.zeros(c, device=device)
        self.ctx = None

    def batch(self, clip_assignment: torch.Tensor, warmup: bool) -> None:
        self.ctx = None if warmup else clip_assignment.to(self.n.device)

    def sample(self, done_mask: torch.Tensor) -> None:
        if self.ctx is None or not len(self.pa):
            return
        env = (~done_mask[: len(self.ctx)]).nonzero().squeeze(-1)
        if not len(env):
            return
        from isaaclab.utils.math import quat_apply

        pos, quat = self.poses()
        p, q = pos[env][:, self.hb], quat[env][:, self.hb]  # (M, H, 3|4)
        cw = quat_apply(q, self.center[None].expand(len(env), -1, 3)) + p
        near = (cw[:, self.pa] - cw[:, self.pb]).norm(dim=-1) < self.rad[self.pa] + self.rad[self.pb]
        pen = torch.zeros(len(env), len(self.pa), device=p.device)
        idx = near.nonzero()
        for s in range(0, len(idx), CHUNK):
            e, k = idx[s : s + CHUNK].T
            a, b = self.pa[k], self.pb[k]
            d = torch.maximum(
                depth(self.verts[a], p[e, a], q[e, a], p[e, b], q[e, b], self.planes[b]),
                depth(self.verts[b], p[e, b], q[e, b], p[e, a], q[e, a], self.planes[a]),
            )
            pen[e, k] = d
        clip = self.ctx[env]
        self.n.index_add_(0, clip, torch.ones(len(env), device=p.device))
        for k in range(len(KINDS)):
            m = self.pk == k
            v = pen[:, m].amax(-1) * 1000.0 if bool(m.any()) else torch.zeros(len(env), device=p.device)
            self.mm[k].scatter_reduce_(0, clip, v, "amax")
            self.npen[k].index_add_(0, clip, (v > PEN_EPS_MM).float())

    def report(self) -> dict:
        n = self.n.clamp(min=1)
        pairs = {k: int((self.pk == i).sum()) for i, k in enumerate(KINDS)}
        tot = max(float(self.n.sum()), 1.0)
        d = {
            "pairs": {
                k: {
                    "max_mm": round(float(self.mm[i].max()) if len(self.n) else 0.0, 2),
                    "frac": round(float(self.npen[i].sum()) / tot, 5),
                }
                for i, k in enumerate(KINDS)
            },
            "clips": {
                self.names[c]: [round(float(self.mm[i, c]), 2) for i in range(len(KINDS))]
                + [round(float(self.npen[i, c] / n[c]), 5) for i in range(len(KINDS))]
                for c in range(len(self.names))
                if float(self.n[c]) > 0
            },
            "samples": int(self.n.sum()),
            "eps_mm": PEN_EPS_MM,
            "hull_pairs": pairs,
            "adjacent_skipped": self.skipped,
        }
        print(TAG + json.dumps(d, ensure_ascii=False), flush=True)
        return d


class _Off:
    """The collector that measures nothing, run_eval_balanced's default (arm_pen=None): no marker, so the judge record
    has no arm_pen (not measured)."""

    def batch(self, *_a) -> None:
        return None

    def sample(self, *_a) -> None:
        return None

    def report(self) -> None:
        return None


OFF = _Off()


def collector_for(raw_env, clip_names: list[str] | None, num_clips: int) -> ArmPen:
    """Build the collector from the hulls of the spawned robot (env 0 stage; also RobotL for a split articulation).
    Dies without hand/arm hulls."""
    from dexengine.rl.tasks import hooks
    from dexengine.rl.tasks.scene.collision_bodies import classify_body
    from dexengine.rl.tasks.scene.split_artic import LEFT_ENTITY, LEFT_PRIM_LEAF, cut_joint_pairs

    root = raw_env.scene.env_prim_paths[0]
    assets = [raw_env.scene["robot"]] + ([raw_env.scene[LEFT_ENTITY]] if LEFT_ENTITY in raw_env.scene.keys() else [])  # noqa: SIM118  — InteractiveScene has no __contains__
    hulls, joints = link_hulls(raw_env.scene.stage, f"{root}/Robot", classify_body)
    if len(assets) == 2:
        h2, j2 = link_hulls(raw_env.scene.stage, f"{root}/{LEFT_PRIM_LEAF}", classify_body, force_left=True, asset=1)
        hulls, joints = hulls + h2, joints | j2
        if hooks.floating_active():
            # Floating hands: the two articulations are spawned separately from the start (per-side variants
            # floating_<hand>_split_{R,L}), not split from one articulation, so no original joint was cut (zero
            # pairs is the rule's answer; cut_joint_pairs requires a world-fixed root and would die).
            print(
                SPLIT_TAG + "floating hands — the two articulations are spawned per side (no original joint was "
                "cut) · 0 pairs · owner: the floating-hand plugin",
                flush=True,
            )
        else:
            joints = joints | split_joint_pairs(
                raw_env.scene.stage, f"{root}/Robot", f"{root}/{LEFT_PRIM_LEAF}", cut_joint_pairs
            )
    bodies = [(i, b) for i, a in enumerate(assets) for b in a.body_names]
    if not any(h["cls"] in _RIGHT for h in hulls) or not any(h["cls"] in _LEFT for h in hulls):
        raise RuntimeError(
            f"[ARM_PEN] hand/arm collision hulls are missing (right "
            f"{sum(h['cls'] in _RIGHT for h in hulls)} · left "
            f"{sum(h['cls'] in _LEFT for h in hulls)}) — cannot measure"
        )
    pairs, skipped = pair_table(hulls, joints)

    def poses():
        return (
            torch.cat([a.data.body_link_pos_w for a in assets], 1),
            torch.cat([a.data.body_link_quat_w for a in assets], 1),
        )

    names = list(clip_names) if clip_names else [f"clip_{i}" for i in range(num_clips)]
    return ArmPen(hulls, pairs, skipped, bodies, poses, names, raw_env.device)
