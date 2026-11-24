"""Single owner of the object pin pose (the PARK invalidity fix, diagnosis park_fix).

The four paths that **write** an object at its reference pose (reset `mdp.reset_object_to_trajectory`, SPAWN_HOLD and
the judge park `ArmResidualTrainEnv.pin_objects_to_reference`, and the PSS snap `_pss_snap_side`) and the oracle that
measures that pin (`rl.judge.park.Park.at_cut`, env window `object_pin_pose`) all call the one `pin_pose`.

Why: the data contract allows a reference pose to dig into a static support surface (rules D24 · D24′). Written as
is, PhysX depenetration kicks the object within one step, and SPAWN_HOLD and PARK rewrite the same pose every step and
repeat the same kick (diagnosis park_fix M1). The closest feasible pose = the reference pose lifted by the
penetration depth:

    lift = max(0, max_s max_{v inside the footprint and thickness of s} (top_s − z_v))      (s = support surface, v = a
    collision vertex at the reference pose)

Support surfaces = the desk collision box (bounds of `DeskCollisionTop` on the live stage; sunk in aerial scenes, so
naturally 0) + the support discs **this object touches in the physics** (`support_disc_rows`: discs stand from reset to
the end (rule D40) · top = the stamped z · owner = the statement `cfg.support_disc_owner` the physics authoring reads;
a disc owned by the other side has collision off and is no support surface). Vertices = the convex hull vertices of the
collision mesh in body coordinates (the lowest point below a plane comes from a hull vertex). The object of env i =
asset[i % n] (the MultiAssetSpawner contract, not clip_ids: the judge overwrites the clip_ids of envs outside the
batch with 0).

Not changed: the reference trajectory (the values rewards, observations and terminations read); fixing the data's desk
height (desk_z) is a data decision. Not hidden: the lift is always visible **per source**
(`LIFT_SOURCES`: desk embedding (the D24 share) · disc) in the `[PIN_FEASIBLE]` marker and the PARK line. Excess lift is
judged by the PARK oracle by measurement (free fall = invalid).
Articulated objects: only the root link (`base_link`) hull, in that link's coordinates; the same statement as rule D24
decision (3) (`dexengine.data.objects.hull.root_link_points`).
"""

from __future__ import annotations

import torch

MARK = "[PIN_FEASIBLE]"
# env chunk of the lift computation: only a (chunk, V, 3) vertex tensor at a time (V in the thousands; same values,
# split work only)
PIN_CHUNK = 512
DESK_PRIM = "DeskCollisionTop"
ARTIC_ROOT_LINK = (
    "base_link"  # env_setup.objects.ARTIC_LINK_NAMES[0]: the root link of the bake (bake_object_usd --articulated)
)


# ── Pure math (CPU-tested, runs without Isaac) ──


def quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v (...,3) by q (...,4) **wxyz** (this build's isaaclab.utils.math convention), with broadcasting."""
    shape = torch.broadcast_shapes(q.shape[:-1], v.shape[:-1])
    q, v = q.expand(*shape, 4), v.expand(*shape, 3)
    w, u = q[..., :1], q[..., 1:]
    t = 2.0 * torch.linalg.cross(u, v, dim=-1)
    return v + w * t + torch.linalg.cross(u, t, dim=-1)


#: Scene name prefix of the support object slots; writer `env_setup.supports._build_support_slots` (support_obj_<k>).
SUPPORT_OBJ_PREFIX = "support_obj_"
#: Lift sources: desk · disc (standing from reset to the end, rule D40).
LIFT_SOURCES = ("desk", "disc")
FOREIGN = (
    "foreign"  # penetration into discs owned by the other side (~own): a check, not a source; must be 0 in pin_pose
)


def support_footprints(pts: torch.Tensor, valid: torch.Tensor, desk=None, discs=None) -> tuple:
    """One definition of the support footprints (`support_lift_parts` · `support_clearance`): (desk
    (n,V)
    | None, discs (n,V,K) | None). Desk = vertices inside the box x·y · discs = vertices inside the radius of the discs
    standing in the physics (on); None without any on disc."""
    fd = fk = None
    if desk is not None:
        x0, x1, y0, y1, _top, _bot = (float(v) for v in desk)
        fd = valid & (pts[..., 0] >= x0) & (pts[..., 0] <= x1) & (pts[..., 1] >= y0) & (pts[..., 1] <= y1)
    if discs is not None and discs["on"].any():
        dx = pts[..., 0, None] - discs["x"][:, None, :]  # (n,V,K)
        dy = pts[..., 1, None] - discs["y"][:, None, :]
        fk = valid[..., None] & discs["on"][:, None, :] & (dx * dx + dy * dy <= (discs["r"][:, None, :]) ** 2)
    return fd, fk


def support_lift_parts(pts: torch.Tensor, valid: torch.Tensor, desk=None, discs=None) -> dict:
    """Reference pose vertices pts (n,V,3) (env-local) · valid (n,V) -> per source (`LIFT_SOURCES`), the minimal lift
    (n,)
    ≥ 0 that removes the support penetration.

    For one support (footprint F · slab [bottom, top]): when the z range of the vertices inside F overlaps the slab
    (lowest < top and highest > bottom), penetration = top − (lowest z inside F), otherwise 0 (floating above the slab
    or entirely below it). Objects buried deeper than the slab thickness are caught too.
    desk  = (x0, x1, y0, y1, top, bottom): the desk collision box · None = no desk
    discs = {x, y, r, top, on · own: (n,K) · h}: discs standing in the physics (on) (slab [top − h, top] · footprint =
            inside the radius). own = whether the owner statement is this object's side (required); penetration of
            on & ~own slots is counted separately as FOREIGN.
    Limit (approximation): the footprint test uses vertices, so a floor wider than the disc with its corners outside the
    disc reads as 0 penetration for that disc.
    """
    z = pts[..., 2]
    inf = torch.full_like(z, float("inf"))
    zero = torch.zeros(pts.shape[0], device=pts.device, dtype=pts.dtype)
    out = {k: zero.clone() for k in LIFT_SOURCES + (FOREIGN,)}
    f, fk = support_footprints(pts, valid, desk, discs)
    if f is not None:
        top, bot = float(desk[4]), float(desk[5])
        lo = torch.where(f, z, inf).amin(dim=1)
        hi = torch.where(f, z, -inf).amax(dim=1)
        out["desk"] = torch.where((lo < top) & (hi > bot), top - lo, torch.zeros_like(lo))
    if fk is not None:
        top = discs["top"]  # (n,K)
        zz = z[..., None].expand_as(fk)
        lo = torch.where(fk, zz, float("inf")).amin(dim=1)  # (n,K)
        hi = torch.where(fk, zz, float("-inf")).amax(dim=1)
        pen = torch.where((lo < top) & (hi > top - discs["h"]), top - lo, torch.zeros_like(lo))
        out["disc"] = pen.amax(dim=1)
        out[FOREIGN] = torch.where(discs["own"], torch.zeros_like(pen), pen).amax(dim=1)
    return out


# ── Scene geometry (live stage, read once and cached on the env) ──


def _asset_count(obj) -> int:
    """The object of env i = asset[i % n]; n = the MultiAssetSpawner asset count (a single USD = 1)."""
    acfg = getattr(getattr(obj.cfg, "spawn", None), "assets_cfg", None)
    return len(acfg) if acfg else 1


def _env_prim(env, prim_path: str, i: int) -> str:
    ns = env.scene.env_regex_ns
    for pat in (ns, "{ENV_REGEX_NS}"):
        if pat in prim_path:
            return prim_path.replace(pat, env.scene.env_prim_paths[i])
    raise RuntimeError(
        f"{MARK} object prim path {prim_path!r} has no env namespace ({ns}) — env {i}'s prim cannot be found"
    )


def _collision_meshes_body(stage, root_path: str, visual: bool = False):
    """Body-coordinate points [(V_i,3)] per collision (visual=True: visual) mesh under root_path, and whether
    articulated; the one reader."""
    import numpy as np

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"{MARK} object prim {root_path} is not on the stage")
    rng = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
    bodies = [p for p in rng if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not bodies:
        raise RuntimeError(f"{MARK} no rigid body under {root_path}")
    artic = len(bodies) > 1
    meshes = rng
    if artic:
        named = [p for p in bodies if p.GetName() == ARTIC_ROOT_LINK]
        if len(named) != 1:
            raise RuntimeError(
                f"{MARK} {root_path}: {len(bodies)} rigid bodies but {len(named)} root links "
                f"{ARTIC_ROOT_LINK!r} — the body coordinates cannot be decided"
            )
        body = named[0]
        # Only the root link pieces (rule D24 decision (3)): the spawn joint poses of other links are not the reference
        # θ
        # (no link overlap = dexengine.data.objects.hull.stage_links)
        meshes = list(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
    else:
        body = bodies[0]
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    tf = Gf.Transform(xc.GetLocalToWorldTransform(body))
    m_body = Gf.Matrix4d().SetTransform(
        tf.GetRotation(), tf.GetTranslation()
    )  # physics body frame = rotation + translation (no scale)
    inv = np.asarray(m_body.GetInverse(), dtype=float)
    pts, CA = [], UsdPhysics.CollisionAPI
    for p in meshes:
        col = p.HasAPI(CA) and CA(p).GetCollisionEnabledAttr().Get() is not False
        if (
            not p.IsA(UsdGeom.Mesh) or col == visual
        ):  # the cooking dump generator's prim selection is checked against this too
            continue
        v = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get() or [], dtype=float)
        if not len(v):
            continue
        m = (
            np.asarray(xc.GetLocalToWorldTransform(p), dtype=float) @ inv
        )  # row-vector convention: p_body = [p,1] @ M_mesh→world @ inv(M_body)
        pts.append((np.c_[v, np.ones(len(v))] @ m)[:, :3])
    if not pts:
        raise RuntimeError(f"{MARK} no {'visual' if visual else 'collision'} mesh under {root_path}")
    return pts, artic


def _hull_points_body(stage, root_path: str):
    """Body-coordinate convex hull vertices (V,3) of the points of `_collision_meshes_body` combined, and whether
    articulated."""
    import numpy as np

    pts, artic = _collision_meshes_body(stage, root_path)
    P = np.concatenate(pts)
    try:  # the lowest point below a plane comes from a hull vertex: this only reduces (the minimum stays)
        from scipy.spatial import ConvexHull

        P = P[ConvexHull(P).vertices]
    # a degenerate (planar) point set: use all points unreduced (same minimum, only more work)
    except (RuntimeError, ValueError):  # scipy QhullError is a RuntimeError
        P = np.unique(P, axis=0)
    return P, artic


def object_geometry(env, obj) -> tuple:
    """(P (A,V,3) body-coordinate vertices · valid (A,V) · A), read once per object asset (prim path) and cached on the
    env."""
    cache = env.__dict__.setdefault("_pin_geom", {})
    key = obj.cfg.prim_path
    if key in cache:
        return cache[key]
    A = _asset_count(obj)
    n_read = min(A, int(env.num_envs))
    rows, artic = [], False
    for i in range(n_read):
        P, a = _hull_points_body(env.scene.stage, _env_prim(env, key, i))
        rows.append(torch.as_tensor(P, dtype=torch.float32))
        artic = artic or a
    V = max(int(r.shape[0]) for r in rows)
    Pt = torch.zeros(n_read, V, 3)
    valid = torch.zeros(n_read, V, dtype=torch.bool)
    for i, r in enumerate(rows):
        Pt[i, : r.shape[0]] = r
        valid[i, : r.shape[0]] = True
    out = (Pt.to(env.device), valid.to(env.device), A)
    cache[key] = out
    desk = desk_box(env)
    print(
        f"{MARK} {key.rsplit('/', 1)[-1]}: assets {A} (read {n_read}) · max collision hull vertices {V} · "
        + (
            f"desk top z {desk[4]:.4f} footprint x[{desk[0]:.3f},{desk[1]:.3f}] y[{desk[2]:.3f},{desk[3]:.3f}]"
            if desk is not None
            else "no desk"
        )
        + (" · articulated: root link hull (rule D24 decision (3))" if artic else ""),
        flush=True,
    )
    return out


def desk_box(env):
    """Desk collision box (x0, x1, y0, y1, top, bottom) in env-local coordinates (env 0's live prim bounds − env 0's
    origin). None without one."""
    if "_pin_desk" in env.__dict__:
        return env._pin_desk
    from pxr import Usd, UsdGeom

    prim = env.scene.stage.GetPrimAtPath(f"{env.scene.env_prim_paths[0]}/{DESK_PRIM}")
    box = None
    if prim.IsValid():
        r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"]).ComputeWorldBound(prim).ComputeAlignedRange()
        o = [float(v) for v in env.scene.env_origins[0].tolist()]
        lo, hi = r.GetMin(), r.GetMax()
        box = (lo[0] - o[0], hi[0] - o[0], lo[1] - o[1], hi[1] - o[1], hi[2] - o[2], lo[2] - o[2])
    env._pin_desk = box
    return box


# ── Support disc table (rules D26′ · D20, from reset to the end) ──


def support_disc_state(env) -> dict | None:
    """The single reader of the support disc table: the params of the reset event term `reset_support_discs` (owner
    `env_setup.supports._build_disc_slots`). The reset (`events.reset_support_discs` -> `reset_discs`) and the pin pose
    (`pin_pose`) read the same table; whichever asks first builds it, whatever the event order. A scene without discs ->
    None. Discs stand from reset to the end of the episode (rule D40, no lifetime window). Disc prims without the event
    term, table rows other than (x, y, z, r), or an owner statement (cfg.support_disc_owner) that is missing or
    disagrees with the table fail."""
    if "_support_disc" in env.__dict__:
        return env._support_disc
    names = sorted(str(k) for k in env.scene.keys() if str(k).startswith("support_disc_"))  # noqa: SIM118  — InteractiveScene has no __contains__
    try:
        cfg = env.event_manager.get_term_cfg("reset_support_discs")
    except ValueError:
        cfg = None
    if cfg is None:
        if names:
            raise RuntimeError(f"[discs] disc prims {names} exist but there is no reset event reset_support_discs")
        env._support_disc = None
        return None
    prm = cfg.params
    tag = prm.get("tag", "AERIAL")
    tm = env.trajectory_manager
    t = torch.tensor(prm["table"], dtype=torch.float32, device=env.device)  # (C, K, 4) = x, y, z (top), r
    if t.ndim != 3 or t.shape[-1] != 4:
        raise RuntimeError(f"[{tag}][discs] disc table shape {tuple(t.shape)} — must be (clip, slot, 4: x·y·z·r)")
    if t.shape[0] != tm.num_clips:
        raise RuntimeError(
            f"[{tag}][discs] disc table clips {t.shape[0]} ≠ loader clips {tm.num_clips} — order contract broken"
        )
    if "height" not in prm or "hidden_z" not in prm:
        raise RuntimeError(
            f"[{tag}][discs] disc thickness (height) and hidden z (hidden_z) are missing from the event params"
        )
    real = t[..., 2] > -5.0
    n_real = int(real.sum().item())
    own = getattr(
        getattr(env, "cfg", None), "support_disc_owner", None
    )  # the same statement as the physics authoring (collision_authoring)
    why = (
        "is missing"
        if own is None
        else f"shape {[len(r) for r in own][:3]}… ≠ disc table {tuple(t.shape[:2])}"
        if len(own) != t.shape[0] or any(len(r) != t.shape[1] for r in own)
        else "names no owner for a real disc"
        if any(o is None and x for r, m in zip(own, real.tolist(), strict=True) for o, x in zip(r, m, strict=True))
        else ""
    )
    if why:
        raise RuntimeError(
            f"[{tag}][discs] disc owner statement cfg.support_disc_owner {why} — the physical support cannot be read"
        )
    hs = prm.get("heights")  # column heights (C, K)
    hs = torch.full_like(t[..., 0], prm["height"]) if hs is None else torch.tensor(hs).to(t).reshape(t.shape[:2])
    env._support_disc = {
        "table": t,
        "names": list(prm["disc_names"]),
        "h": hs,
        "hidden_z": float(prm["hidden_z"]),
        "tag": tag,
        "owner": own,
        "visible": prm.get("visible"),
    }
    print(
        f"[{tag}][discs] disc slots {len(prm['disc_names'])} × clips {t.shape[0]} — real discs {n_real}, "
        f"r {t[..., 3][real].min().item() if n_real else 0:.3f}~{t[..., 3][real].max().item() if n_real else 0:.3f} m "
        f"· "
        f"centre = top − {float(prm['height']) / 2 * 1000:.1f} mm · from reset to the end (rule D40)",
        flush=True,
    )
    return env._support_disc


def disc_clip(env, st: dict, env_ids: torch.Tensor) -> torch.Tensor:
    """Table rows (clips) of env_ids, clamped to the last clip by the same rule as `reference_frames` (so the global
    clip_ids of a subset load never point outside the table and die on a CUDA index assert at an unrelated line; such
    envs get their reference frames clamped by the same rule)."""
    return env.trajectory_manager.clip_ids[env_ids].clamp(max=int(st["table"].shape[0]) - 1)


def support_disc_rows(env, env_ids: torch.Tensor, side: str) -> dict | None:
    """Disc supports read by the pin pose (`pin_pose`): {x, y, r, top, on · own · h (column height): (n,K)}. A scene
    without discs -> None. on = the clip's real discs (standing from reset to the end) · own = the owner statement (the
    same cell as the physics authoring, `cfg.support_disc_owner`) is `side` or both (a shared object: one object for
    both hands)."""
    st = support_disc_state(env)
    if st is None:
        return None
    if f"own_{side}" not in st:  # per side (C, K), once
        st[f"own_{side}"] = torch.tensor([[o in (side, "both") for o in r] for r in st["owner"]]).to(st["table"].device)
    ci = disc_clip(env, st, env_ids)
    w = st["table"][ci]
    return {
        "x": w[..., 0],
        "y": w[..., 1],
        "top": w[..., 2],
        "r": w[..., 3],
        "on": w[..., 2] > -5.0,
        "own": st[f"own_{side}"][ci],
        "h": st["h"][ci],
    }


def reset_discs(env, env_ids: torch.Tensor) -> None:
    """Reset (`events.reset_support_discs`): write the disc slots of env_ids. Real discs get centre = top (stamped z) −
    thickness/2, empty slots the hidden z. Discs stand there until the end of the episode (kinematic, no lifetime).
    Before writing, the disc size is fitted to the reset clip's stamp (supports.fit_disc_geometry: render session clip
    k, read-back guard)."""
    st = support_disc_state(env)
    if st is None:
        raise RuntimeError("[discs] reset_support_discs was called but there is no disc table")
    ci = disc_clip(env, st, env_ids)
    from dexengine.rl.tasks.scene.supports import fit_disc_geometry  # disc size owner (avoids a cycle)

    fit_disc_geometry(env, st, env_ids, ci)
    w = st["table"][ci]
    pose = torch.zeros(*w.shape[:2], 7, device=w.device)
    pose[..., :3] = w[..., :3]
    z = w[..., 2]
    pose[..., 2] = torch.where(z > -5.0, z - 0.5 * st["h"][ci], torch.full_like(z, st["hidden_z"]))
    pose[..., 3] = 1.0  # scene frame as is (no rotation, no z shift)
    write_slots(env, env_ids, st["names"], pose)


def write_slots(env, env_ids: torch.Tensor, names: list, pose: torch.Tensor) -> None:
    """One writer for fixed slots (discs, support objects): pose (n, K, 7) = scene-frame xyz · wxyz (the env origin is
    added here), velocity 0."""
    for k, name in enumerate(names):
        rs = torch.zeros(len(env_ids), 13, device=env.device)
        rs[:, :7] = pose[:, k]
        rs[:, :3] += env.scene.env_origins[env_ids]
        env.scene[name].write_root_state_to_sim(rs, env_ids)


# ── Owner ──


def pin_pose(env, ids: torch.Tensor, frames: torch.Tensor, side: str, obj, ref: dict | None = None) -> tuple:
    """Pin pose of the `side` object for ids: (pos (n,3) env-local · quat (n,4) wxyz · lift (n,) m · parts {source: (n,)
    m}).
    The reference pose (`trajectory_manager.get_reference_state_for_envs(ids, frames)`) lifted by the support
    penetration. lift = the maximum over the sources in parts (`LIFT_SOURCES`); the marker and the oracle count sources
    separately. A share from other sides' discs (FOREIGN) > 0 fails."""
    if ref is None:
        ref = env.trajectory_manager.get_reference_state_for_envs(ids, frames)
    p, q = ref[f"{side}_obj_pos"], ref[f"{side}_obj_quat"]
    P, valid, A = object_geometry(env, obj)
    a = ids.long() % A  # with A > num_envs only num_envs assets are read; ids < num_envs keeps it in range
    desk, discs = desk_box(env), support_disc_rows(env, ids, side)
    if (
        discs is not None
    ):  # a disc owned by the other side never touches this object in the physics: drop it from the supports
        raw_on, discs["on"] = discs["on"], discs["on"] & discs["own"]
    parts = {k: torch.zeros(len(ids), device=p.device, dtype=p.dtype) for k in LIFT_SOURCES + (FOREIGN,)}
    for s0 in range(0, len(ids), PIN_CHUNK):  # transient memory cap for a 4096-env reset (PIN_CHUNK)
        sl = slice(s0, s0 + PIN_CHUNK)
        pts = quat_apply_wxyz(q[sl, None, :], P[a[sl]]) + p[sl, None, :]
        dk = None if discs is None else {k: v[sl] for k, v in discs.items()}
        for k, v in support_lift_parts(pts, valid[a[sl]], desk, dk).items():
            parts[k][sl] = v
    foreign = parts.pop(FOREIGN) * 1000.0
    if bool((foreign > 0).any()):
        raise RuntimeError(
            f"{MARK} {side}: foreign slot lift >0 {int((foreign > 0).sum())} max "
            f"{float(foreign.max()):.2f} mm — a foreign disc the physics never touches was used as support"
        )
    lift = torch.stack([parts[k] for k in LIFT_SOURCES]).amax(dim=0)
    pos = p.clone()
    pos[:, 2] = pos[:, 2] + lift
    said = env.__dict__.setdefault("_pin_said", set())
    tag = (obj.cfg.prim_path, side)
    if tag not in said and ids.numel() > 0:
        said.add(tag)
        mm, j = lift * 1000.0, int(torch.argmax(lift))
        top = f" · max {float(mm[j]):.1f} mm (env {int(ids[j])} clip {int(env.trajectory_manager.clip_ids[ids[j]])})"
        print(
            f"{MARK} first pin {side}: env {ids.numel()} · support penetration lift >0 {int((mm > 0).sum())} · "
            f">5 mm {int((mm > 5.0).sum())}{top if bool((mm > 0).any()) else ''} · sources {lift_sources_text(parts)} "
            f"· "
            f"foreign discs excluded (env,slot) {0 if discs is None else int((raw_on & ~discs['own']).sum())} · "
            f"foreign slot lift >0 {int((foreign > 0).sum())} max {float(foreign.max()):.2f}",
            flush=True,
        )
    return pos, q, lift, parts


def lift_sources_text(parts: dict) -> str:
    """Per-source lift line of the `[PIN_FEASIBLE] first pin` marker: `desk >0 n max x · disc …` (mm)."""
    out = []
    for k in LIFT_SOURCES:
        mm = parts[k].float() * 1000.0
        out.append(f"{k} >0 {int((mm > 0).sum())} max {float(mm.max()) if mm.numel() else 0.0:.2f}")
    return " · ".join(out)


# ---------------------------------------------------------------- support gap of the pinned pose
# The gap between the pinned pose and its supports (desk + this side's discs); read by the judge's gate D
# through env.object_support_gap.


def support_clearance(pts: torch.Tensor, valid: torch.Tensor, desk=None, discs=None) -> torch.Tensor:
    """Reference pose vertices pts (n,V,3) · valid (n,V) -> support gap (n,) m = min_s (lowest vertex z inside the
    footprint − top_s). Supports and footprints are those of `object_pin.support_lift_parts` (`support_footprints`;
    the caller drops foreign discs). A support with no vertex inside its footprint is inf. Negative = digging in, near 0
    = resting, positive = floating."""
    z = pts[..., 2]
    gap = torch.full((pts.shape[0],), float("inf"), device=pts.device, dtype=pts.dtype)
    f, fk = support_footprints(pts, valid, desk, discs)
    if f is not None:
        gap = torch.minimum(gap, torch.where(f, z, torch.full_like(z, float("inf"))).amin(dim=1) - float(desk[4]))
    if fk is not None:
        lo = torch.where(fk, z[..., None].expand_as(fk), float("inf")).amin(dim=1)  # (n,K)
        gap = torch.minimum(gap, (lo - discs["top"]).amin(dim=1))
    return gap


def support_gap(env, ids: torch.Tensor, frames: torch.Tensor, side: str, obj, ref: dict | None = None) -> torch.Tensor:
    """Support gap (n,) m of the `side` object's pin pose for ids (`pin_pose`, the pose SPAWN_HOLD writes;
    `support_clearance`). Supports = the same as pin_pose (desk + discs owned by this side).
    The 'above the support' statement of judge gate D (env window `object_support_gap`)."""
    pos, q, _lift, _parts = pin_pose(env, ids, frames, side, obj, ref)
    P, valid, A = object_geometry(env, obj)
    a = ids.long() % A
    desk, discs = desk_box(env), support_disc_rows(env, ids, side)
    if discs is not None:
        discs["on"] = discs["on"] & discs["own"]
    gap = torch.full((len(ids),), float("inf"), device=pos.device, dtype=pos.dtype)
    for s0 in range(0, len(ids), PIN_CHUNK):
        sl = slice(s0, s0 + PIN_CHUNK)
        pts = quat_apply_wxyz(q[sl, None, :], P[a[sl]]) + pos[sl, None, :]
        dk = None if discs is None else {k: v[sl] for k, v in discs.items()}  # h is (n,K) too: column heights
        gap[sl] = support_clearance(pts, valid[a[sl]], desk, dk)
    return gap
