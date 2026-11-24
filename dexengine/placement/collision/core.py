#!/usr/bin/env python3
"""Arm collision — the arm guards (the surface ruler) of judge = bake = gate = final guard.

The guards: build_arm_ctx / desk_penetration(_each)(_t) / pen_cross_guard. The ruler is the **surface of the colliders
the sim spawns** (arm_surface — shapes from sim_collider, the one generator): the desk (DESK_SURF_PEN_MM), the support
objects (support_surface, the desk's value), the two arms (ARM_OVERLAP_MAX_MM) and the torso
(torso.TORSO_OVERLAP_MAX_MM). Every consumer reads these names — change a value here only. The torch twins of the
search functions live in core_torch (the judge's GPU batch path imports it directly).

Vendor capsules: the rby1 URDF defines, per link, a capsule collider plus `coltype`/`colaffinity` (bitmask collision
groups). `<capsule>` is not a standard URDF geometry (only box/cylinder/sphere/mesh), so urdfdom gives up parsing and
pinocchio sees a robot with no colliders at all — geom.load_capsules reads the XML directly (the torso pair masks use
the vendor bitmasks: arm_3/4/5 coltype 128/256/512, affinity 30~31 → from the elbow on, checked against the torso).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from dexengine.core.errors import InputMissingError, input_missing
from dexengine.placement.collision import arm_proxy as AP
from dexengine.placement.collision import arm_surface as AS
from dexengine.placement.collision import guard_band as GB
from dexengine.placement.collision import torso as TS
from dexengine.placement.collision.geom import arm_link_transforms

# ── the arm guards (judge = bake = gate = final guard, shared) ───────

#: Desk: the deepest arm-collider surface point may sit at most this far below the desk top (inside the footprint).
DESK_SURF_PEN_MM = 2.0
#: Policy motion allowance: the judges' DP filters drop a pose whose desk penetration exceeds DESK_SURF_PEN_MM by more
#: than this (+ the proxy filter slack) — the policy moves around the reference, so a pose this close to the limit
#: cannot enter a passing path anyway (judge.gate, judge.accept).
POLICY_MOTION_ALLOWANCE_MM = 5.0
#: Two arms: their collider surfaces may overlap at most this deep (positive = allowed overlap depth).
ARM_OVERLAP_MAX_MM = 2.0
#: Desk ruler: axis samples per capsule (7 missed the deepest point). Part of the ruler identity (ruler_id) — a sidecar
#: measured with another count is another ruler.
CAPSULE_AXIS_SAMPLES = 13
#: Reason words of side_guard (`<side><word><mm>mm`) — the gate counts support failures by SUPPORT_PEN_WORD.
DESK_PEN_WORD, SUPPORT_PEN_WORD = "desk_pen", "support_pen"
#: Link name of a torso maximum in desk_penetration(_t) (no robot link has this name) — the judge compares it.
TORSO_LINK = "torso"


def _axis_ts() -> np.ndarray:
    """Capsule axis parameters in [−1, 1] — CAPSULE_AXIS_SAMPLES points (numpy and the GPU twin read this one)."""
    return np.linspace(0, 1, CAPSULE_AXIS_SAMPLES) * 2 - 1


def torso_shift_m() -> float:
    """Offset that puts the torso overlap on the desk scale inside the search scalar: torso + shift > DESK_SURF_PEN_MM
    ⇔ torso overlap > torso.TORSO_OVERLAP_MAX_MM (the two named tolerances, no third value)."""
    return (DESK_SURF_PEN_MM - TS.TORSO_OVERLAP_MAX_MM) / 1000.0


_CTX_CACHE: dict = {}
# The resident bake worker (bake.server) keeps these pure caches between clips (keys independent of the clip); every
# other global is reset to its import-time value per request.
_SERVER_KEEP = ("_CTX_CACHE",)  # (arm, hand, side, proxy) key


def build_arm_ctx(arm: str, hand: str, side: str) -> dict:
    """One set of FK context — pinocchio is used here (1 time) only; everything after is numpy
    batch composition.

    A serial arm chain is reconstructed exactly as T_k(q) = T_{k-1}(q) · A_k · R(axis_k, q_k).
    A_k is taken from oMi at the neutral pose (bent torso included) — however many fixed frames
    sit between the arm joints, they are absorbed into A_k, so this is general.
    """
    key = (arm, hand, side, AP.kind())
    if key in _CTX_CACHE:
        return _CTX_CACHE[key]
    import pinocchio as pin

    from dexengine.placement import arm_ik as AIP

    urdf, _ee, joints, base_xyz, _wd = AIP.resolve_combo(arm, hand, side)
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    q0 = pin.neutral(model)
    pin.forwardKinematics(model, data, q0)
    jids = [model.getJointId(j) for j in joints]
    B = [np.asarray(data.oMi[j].homogeneous) for j in jids]
    A = [None] + [np.linalg.solve(B[k - 1], B[k]) for k in range(1, len(jids))]
    root = ET.parse(str(urdf)).getroot()
    ax = {}
    for j in root.findall("joint"):
        a = j.find("axis")
        ax[j.get("name")] = np.array([float(v) for v in (a.get("xyz") if a is not None else "1 0 0").split()])
    axes = np.stack([ax[j] for j in joints])
    # The colliders are the sim's (arm_surface reads sim_collider — the one generator; the link ↔ joint correspondence
    # is the arm registry's ArmSpec.body_name). An arm with none cannot be guarded: it dies.
    surf = AS.build_surface(str(urdf), arm, side, list(joints))
    for h in surf["hulls"]:  # the mesh guard's band (guard_band — the hull inside its grown table capsule)
        h["band"] = GB.band(AP.arm_key(arm, h["link"]), h["v"])
    if not surf["links"]:
        raise InputMissingError(
            f"[collision] {arm}+{hand}/{side}: no arm colliders (sim_collider.bodies 0) — the "
            "arm–desk · torso guards cannot run"
        )
    # EE (hand attachment frame) — a fixed offset relative to the last joint frame (the idle arm's EE target).
    pin.updateFramePlacements(model, data)
    _fid = model.getFrameId(_ee)
    _ee_w = np.asarray(data.oMf[_fid].homogeneous)
    ee_local = np.linalg.solve(B[-1], _ee_w)  # last joint → EE fixed transform
    ctx = {
        "B0": B[0],
        "A": A,
        "axes": axes,
        "caps": surf["caps"],
        "surf": surf,
        "ee_local": ee_local,
        "base": np.asarray(base_xyz, float),
        "joints": list(joints),
        "ztop": float(AIP.DESK_HEIGHT),
        "urdf": str(urdf),
        "combo": (arm, hand, side),
        "proxy": AP.kind(),
    }
    ctx["torso"] = TS.build_torso(ctx, arm, hand, side)
    _CTX_CACHE[key] = ctx
    return ctx


def cross_min_dist_each(ctxR: dict, ctxL: dict, qR, qL, subdiv: int = 4):
    """Signed **surface** distance between the two arms over every arm body pair (T',) [m] — negative = overlap depth;
    assumes the frames are aligned (arm_surface.cross_surface_each).

    Both arms share the clip's time axis (retiming keeps a single time axis), so (qR[t], qL[t]) at
    t really are simultaneous poses. With subdiv>1 the gaps between frames are subdivided with the
    same interpolation on both arms, so the chord poses retiming would create are seen too (the
    same principle as the desk sweep check)."""
    qR = np.asarray(qR, float)
    qL = np.asarray(qL, float)
    T = min(len(qR), len(qL))
    qR, qL = qR[:T], qL[:T]
    if subdiv > 1 and T > 1:
        w = np.linspace(0.0, 1.0, subdiv, endpoint=False)[1:]

        def _mid(q):
            nj = q.shape[1]
            m = (q[:-1, None, :] * (1.0 - w)[None, :, None] + q[1:, None, :] * w[None, :, None]).reshape(-1, nj)
            return np.concatenate([q, m], 0)

        qR, qL = _mid(qR), _mid(qL)  # both grow in the same order, so the pairing is preserved
    return AS.cross_surface_each(ctxR, ctxL, arm_link_transforms(ctxR, qR), arm_link_transforms(ctxL, qL))


# ── Scene and the robot's support surface. The robot collides with the **desk top** of a table scene and with
#    nothing in an aerial scene: support discs touch objects only, never the hands, so no robot check reads them.


def _chords(q: np.ndarray, subdiv: int) -> np.ndarray:
    """Frames + the retiming chords: retiming smooths large joint jumps with straight-line interpolation in joint
    space, so the poses along each chord become real data. Each gap is split into `subdiv` parts (8: z along a chord is
    smooth; 3 parts missed the deepest point of a measured case). Order: the frames first, then the midpoints."""
    if subdiv <= 1 or len(q) <= 1:
        return q
    w = np.linspace(0.0, 1.0, subdiv, endpoint=False)[1:]
    return np.concatenate(
        [q, (q[:-1, None, :] * (1.0 - w)[None, :, None] + q[1:, None, :] * w[None, :, None]).reshape(-1, q.shape[1])], 0
    )


def _frame_of(fi: int, Tq0: int, subdiv: int, stride: int) -> int:
    """Row of the chord-augmented array → the original frame (a midpoint maps to the frame on its left)."""
    return (fi if fi < Tq0 else (fi - Tq0) // max(subdiv - 1, 1)) * stride


def desk_penetration(
    ctx: dict,
    q_all,
    z_off: float = 0.0,
    stride: int = 1,
    subdiv: int = 8,
    want_components: bool = False,
    desk_xy=None,
    torso: bool = True,
):
    """Search scalar of one arm side over the frames and chords: (worst m, frame, link).

    desk  = how deep the deepest arm-collider surface point sits below the desk top inside the footprint (m, positive
            = below; passing low beside the desk is legal). CAPSULE_AXIS_SAMPLES axis samples per capsule.
    torso = the torso filter's overlap depth (collision.torso, m, positive = overlap) — ``torso=False`` leaves it out
            (the guard measures the torso with the exact ruler instead).
    worst = max(desk, torso + torso_shift_m()) — one scalar against DESK_SURF_PEN_MM that the search reads; the link
            of a torso maximum is TORSO_LINK (the judge compares it). want_components appends
            (desk_m, torso_m | None) on their own scales."""
    q = np.asarray(q_all, float)[::stride]
    Tq0 = len(q)
    q = _chords(q, subdiv)
    Ts = arm_link_transforms(ctx, q)
    base = ctx["base"] + np.array([0.0, 0.0, z_off])
    ts = _axis_ts()[None, :, None]
    worst, desk_only = (-1e9, -1, ""), -1e9
    for lk, pen in AS.desk_pen_bodies(ctx, Ts, base, ts, desk_xy):
        m = float(pen.max())
        desk_only = max(desk_only, m)
        if m > worst[0]:
            worst = (m, _frame_of(int(np.unravel_index(int(pen.argmax()), pen.shape)[0]), Tq0, subdiv, stride), lk)
    torso_only = None
    if torso:
        tv = -TS.torso_filter_sd(ctx["torso"], Ts)
        torso_only = float(tv.max())
        if torso_only + torso_shift_m() > worst[0]:
            worst = (torso_only + torso_shift_m(), _frame_of(int(tv.argmax()), Tq0, subdiv, stride), TORSO_LINK)
    if want_components:
        return worst[0], worst[1], worst[2], desk_only, torso_only
    return worst


def _desk_each(ctx: dict, Ts, z_off: float, desk_xy) -> np.ndarray:
    """Desk component per pose (P,) from link transforms: the deepest arm-collider surface point below the desk top
    inside the footprint (m; −1e9 when no sample is inside)."""
    base = ctx["base"] + np.array([0.0, 0.0, z_off])
    ts = _axis_ts()[None, :, None]
    out = np.full(Ts.shape[0], -1e9)
    for _lk, pen in AS.desk_pen_bodies(ctx, Ts, base, ts, desk_xy):
        out = np.maximum(out, pen.max(1))
    return out


def desk_only_each(ctx: dict, q_all, z_off: float = 0.0, desk_xy=None) -> np.ndarray:
    """Per-pose desk component (T,) of the search scalar — what the IK pool clearance sidecar stores
    (reach_map.clearance: desk_rel − this). The desk only: a robot height offset dz moves the desk in the robot
    frame, so a stored desk depth shifts by dz exactly; a torso overlap moves with the robot and must not be shifted —
    folded into the same scalar it made the dz layers drop poses the judge accepts."""
    return _desk_each(ctx, arm_link_transforms(ctx, np.asarray(q_all, float)), z_off, desk_xy)


def desk_penetration_each(ctx: dict, q_all, z_off: float = 0.0, desk_xy=None) -> np.ndarray:
    """Per-pose search scalar (T,) — each pose alone, no chords: max(desk, torso + torso_shift_m()) as in
    desk_penetration. Use: per-frame candidate filtering (judge · cross-repair)."""
    q = np.asarray(q_all, float)
    Ts = arm_link_transforms(ctx, q)
    return np.maximum(_desk_each(ctx, Ts, z_off, desk_xy), -TS.torso_filter_sd(ctx["torso"], Ts) + torso_shift_m())


def ruler_sources() -> list[str]:
    """The code that sets the stored desk clearance: desk_only_each · _desk_each · _axis_ts with the capsule sample
    count's value (a name in the code carries no value), the arm surface module (bodies and desk depth), the FK
    (geom) and the proxy geometry (arm_proxy.ruler_sources — its kind, and for capsules its code and table hash)."""
    import inspect
    from pathlib import Path

    from dexengine.placement.collision import geom as CG

    return [
        inspect.getsource(desk_only_each),
        inspect.getsource(_desk_each),
        inspect.getsource(_axis_ts),
        f"CAPSULE_AXIS_SAMPLES = {CAPSULE_AXIS_SAMPLES!r}",
        Path(AS.__file__).read_text(encoding="utf-8"),
        Path(CG.__file__).read_text(encoding="utf-8"),
        *AP.ruler_sources(),
    ]


def ruler_id(sources: list[str] | None = None) -> str:
    """Identity of the desk ruler the IK pool clearance sidecar is measured with — the `ruler` half of
    reach_map.clearance.sidecar_identity (the shapes are the other half, sim_collider.measured_digest): a hash of
    the syntax of ruler_sources(), so a comment or docstring edit keeps it and a code edit changes it. The torso is
    not in it — the sidecar holds the desk component only (desk_only_each)."""
    import hashlib

    from dexengine.placement.collision.geom_cache import syntax  # the one code-identity rule (AST without docstrings)

    src = ruler_sources() if sources is None else sources
    return "desk-" + hashlib.sha256("\n".join(syntax(x) for x in src).encode()).hexdigest()[:16]


def guard_ctx(ctx: dict) -> dict:
    """The deciding ruler's context of ctx's combo/side — built inside arm_proxy.mesh_ruler() (GUARD_KIND, the original
    mesh), whatever proxy the search's ctx carries. Call it inside mesh_ruler()."""
    if not AP.guarding():
        raise RuntimeError("[COLL_PROXY] guard_ctx only inside arm_proxy.mesh_ruler()")
    return ctx if ctx["proxy"] == AP.kind() else build_arm_ctx(*ctx["combo"])


_SLACK_DIRS = 4096  # unit directions of the support-function bound below (Fibonacci sphere)


def proxy_desk_slack_m(ctx: dict) -> float:
    """ε (m) such that every body of the search ruler ``ctx`` lies inside its deciding-ruler body ⊕ ball(ε) — the
    amount a filter on ``ctx`` must be loosened by (depth threshold + ε, footprint shrunk by ε) to never reject what
    the deciding ruler (guard_ctx — the original mesh) passes ("filter ≤ judge"). Per body: a proxy capsule against
    the revision hull it was fitted to = max over directions of (capsule support − hull support) (both convex, so this
    is the one-sided Hausdorff distance); a native capsule = half the axis-sample spacing (the deciding ruler samples
    its axis at _axis_ts, the proxy ruler measures it exactly). 0 when ctx already is the deciding ruler.
    Not in ruler_sources: it reads the rulers, it is not one."""
    if ctx["proxy"] == AP.GUARD_KIND:
        return 0.0
    with AP.mesh_ruler():
        g = guard_ctx(ctx)
    if g is ctx:
        return 0.0
    hull = {(h["link"], h["k"]): np.asarray(h["v"], float) for h in g["surf"]["hulls"]}
    i = np.arange(_SLACK_DIRS) + 0.5
    ph, th = np.arccos(1.0 - 2.0 * i / _SLACK_DIRS), np.pi * (1.0 + 5.0**0.5) * i
    D = np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], axis=1)
    eps = 0.0
    for c in ctx["surf"]["caps"]:
        if not c.get("proxy"):
            eps = max(eps, float(c["L"]) / (CAPSULE_AXIS_SAMPLES - 1) / 2.0)
            continue
        v = hull.get((c["link"], c["k"]))
        if v is None:
            raise RuntimeError(
                input_missing(
                    f"[DESK_RULER] ✗ {ctx['combo']} proxy capsule {c['link']}: its original "
                    "hull is not in the deciding ruler"
                )
            )
        ax = np.asarray(c["R"], float) @ np.array([0.0, 0.0, 1.0]) * (float(c["L"]) / 2.0)
        p = np.asarray(c["p"], float)
        hc = np.maximum(D @ (p + ax), D @ (p - ax)) + float(c["r"])
        eps = max(eps, float((hc - (D @ v.T).max(axis=1)).max()))
    return eps


def side_guard(ctx: dict, q, side: str, z_off: float = 0.0, desk_xy=None, *, fingers, supports) -> tuple[bool, str]:
    """The arm guard of one side on the deciding ruler (arm_proxy.GUARD_KIND = the original mesh, never the search
    proxy): _side_guard on guard_ctx(ctx)."""
    with AP.mesh_ruler():
        return _side_guard(guard_ctx(ctx), q, side, z_off, desk_xy, fingers=fingers, supports=supports)


def _side_guard(ctx: dict, q, side: str, z_off: float = 0.0, desk_xy=None, *, fingers, supports) -> tuple[bool, str]:
    """_side_terms with the guard band's counter — marker [GUARD_BAND] (guard_band) once per side guard."""
    cnt = GB.counter(DESK_SURF_PEN_MM)
    try:
        return _side_terms(ctx, q, side, z_off, desk_xy, fingers=fingers, supports=supports, cnt=cnt)
    finally:
        GB.say(side, cnt)


def _side_terms(ctx: dict, q, side: str, z_off: float, desk_xy, *, fingers, supports, cnt) -> tuple[bool, str]:
    """One arm side: desk surface penetration ≤ DESK_SURF_PEN_MM (reason `<side>desk_pen<mm>mm`), then the arm surface
    into a support object ≤ DESK_SURF_PEN_MM (support_surface — reason `<side>support_pen<mm>mm(<object>)`), then
    arm · palm · fingers ↔ torso with the exact ruler ≤ TORSO_OVERLAP_MAX_MM (reason `<side>torso_surface<mm>mm`).
    The reason words are DESK_PEN_WORD · SUPPORT_PEN_WORD (the gate and the ledgers read them).
    fingers = that
    side's finger values (T, n) on q's time axis (hand_surface.guard_finger_q) · supports = the clip's support objects
    exactly as support_surface.scene_supports(tree) returned them (empty when it states none) — both required, a
    missing one dies, and a list from anywhere else dies in the support term (support_surface.checked)."""
    desk_m = GB.desk_max(
        ctx,
        arm_link_transforms(ctx, _chords(np.asarray(q, float), 8)),
        ctx["base"] + np.array([0.0, 0.0, z_off]),
        _axis_ts()[None, :, None],
        desk_xy,
        cnt,
    )
    if desk_m * 1000.0 > DESK_SURF_PEN_MM:
        return False, f"{side}{DESK_PEN_WORD}{desk_m * 1000.0:.1f}mm"
    from dexengine.placement.collision import support_surface as SS

    sup_m, _lk, obj = SS.arm_support_overlap(ctx, q, z_off, supports)
    if sup_m * 1000.0 > DESK_SURF_PEN_MM:
        return False, f"{side}{SUPPORT_PEN_WORD}{sup_m * 1000.0:.1f}mm({obj})"
    ok_t, why_t, _sd = TS.torso_guard(ctx, q, side, fingers, count=cnt)
    return ok_t, why_t


def term_values(
    ctx: dict[str, dict],
    q: dict[str, np.ndarray],
    z_off: float = 0.0,
    desk_xy=None,
    *,
    fingers: dict[str, np.ndarray],
    supports: list,
) -> dict[str, float | None]:
    """The maximum of every term of pen_cross_guard (mm · positive = penetration · overlap depth) — all of them, not
    stopping at the first violation. Ruler · functions · inputs are those of _side_terms · cross_guard (the deciding
    ruler — mesh_ruler · guard_ctx): desk = guard_band.desk_max (band threshold 0 — exact when positive) · support =
    support_surface.arm_support_overlap · torso = torso.torso_guard (exact when there is an overlap — the band skips
    only pairs proven > 0) · cross = cross_min_dist_each. Two sides → the larger; a scene without supports · a cross
    with one side only → None. Comparing with the thresholds is the caller's job (DESK_SURF_PEN_MM ·
    TORSO_OVERLAP_MAX_MM · ARM_OVERLAP_MAX_MM)."""
    from dexengine.placement.collision import support_surface as SS

    out: dict[str, float | None] = {"desk": None, "support": None, "torso": None, "cross": None}
    up = lambda k, v: out.__setitem__(k, v if out[k] is None else max(out[k], v))  # noqa: E731
    with AP.mesh_ruler():
        g = {s: guard_ctx(ctx[s]) for s in ("right", "left") if q.get(s) is not None and ctx.get(s) is not None}
        for side, c in g.items():
            qs = np.asarray(q[side], float)
            cnt = GB.counter(0.0)
            base = c["base"] + np.array([0.0, 0.0, z_off])
            up(
                "desk",
                1000.0
                * GB.desk_max(c, arm_link_transforms(c, _chords(qs, 8)), base, _axis_ts()[None, :, None], desk_xy, cnt),
            )
            sup_m = SS.arm_support_overlap(c, qs, z_off, supports)[0]
            if sup_m > -1e8:
                up("support", 1000.0 * sup_m)
            up("torso", -1000.0 * TS.torso_guard(c, qs, side, fingers.get(side), count=cnt)[2])
        if len(g) == 2:
            out["cross"] = -1000.0 * float(cross_min_dist_each(g["right"], g["left"], q["right"], q["left"]).min())
    return out


def cross_guard(ctxR: dict, ctxL: dict, qR, qL) -> tuple[bool, str]:
    """Two arms: signed surface distance ≥ −ARM_OVERLAP_MAX_MM (reason `cross<signed mm>mm`) on the deciding ruler."""
    with AP.mesh_ruler():
        sd = float(cross_min_dist_each(guard_ctx(ctxR), guard_ctx(ctxL), qR, qL).min()) * 1000.0
    return (True, "") if sd >= -ARM_OVERLAP_MAX_MM else (False, f"cross{sd:.1f}mm")


def pen_cross_guard(
    ctx: dict[str, dict],
    q: dict[str, np.ndarray],
    z_off: float = 0.0,
    desk_xy=None,
    *,
    fingers: dict[str, np.ndarray],
    supports: list,
) -> tuple[bool, str]:
    """**Single canonical source** of the arm hard constraint — judging (`judge._pen_cross_guard`, every adoption
    site), the bake (`final_guard`) and the gate (side_guard · cross_guard, the same two steps) — so "passes judging =
    passes the gate" holds. Order fixed (the reason strings must be deterministic for list comparison): side_guard per
    side (right, left), then cross_guard. ctx {"right": build_arm_ctx(...), "left": ...} (one side alone is fine) ·
    q {"right": (T,7), ...} (a missing side is skipped) · z_off = robot base z offset (robot_z_off of adopted.json) ·
    fingers {"right": (T, n), ...} (hand_surface.guard_finger_q — every guarded side needs its own) · supports = the
    clip's support objects (support_surface.scene_supports — both arms meet every one).
    Returns (ok, why) at the first violation."""
    for side in ("right", "left"):
        if q.get(side) is not None and ctx.get(side) is not None:
            ok, why = side_guard(
                ctx[side], q[side], side, z_off=z_off, desk_xy=desk_xy, fingers=fingers.get(side), supports=supports
            )
            if not ok:
                return False, why
    if q.get("right") is not None and q.get("left") is not None:
        return cross_guard(ctx["right"], ctx["left"], q["right"], q["left"])
    return True, ""
