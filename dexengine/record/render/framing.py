"""Ego framing rule of aerial scenes (math, numpy; tested outside Kit too): set the head joints once per clip
(fixed inside an episode) so the union of the clip trajectory's hand and object points falls inside the ego left-eye
image (calibrated K, render.cameras.zed_intrinsics). The canonical rule does not move the yaw (FRAME_YAW: pitch only).

Head joints (URDF rby1a_model_v1.2, canonical body urdf): head_0 = yaw (axis z) · head_1 = pitch (axis y, positive =
  down). The rig (render.rig.Rig) uses [0] of dexengine.core.arms head_joint_names as yaw and [-1] as pitch.
Rule (solve):
  (1) move the points into left-eye optical coordinates (ROS: x right, y down, z forward) and measure the angles
      (atan2(x, z), atan2(y, z)).
  (2) pitch only (FRAME_YAW False, canonical): the yaw stays at its start value (canonical 0); a fixed-point iteration
      pitch += vertical angle difference (positive = down) brings the vertical angle centre of the chosen point box
      to the vertical centre of the image window (the real angle range of the calibrated K, even when the principal
      point is off centre). For a box off to the side the world depression differs from the image vertical angle, so
      the image angle is fitted directly.
      Yaw and pitch (FRAME_YAW True): aim at the world azimuth and depression of the box centre direction: yaw +=
      Δazimuth, pitch += Δdepression.
      Both iterate because the camera is offset from the rotation centre; done within FRAME_TOL_DEG, and dies when
      FRAME_ITERS is not enough.
  (2') when the points span more than the field of view, only the points inside the view-wide window (span per axis)
      that holds the most points in the first pose are kept and iterated (subset; the chosen points stay fixed). With
      the yaw fixed the horizontal window is the real image range (it cannot move).
  (3) priority: first 'all' (every hand link, object centres, every step); if all are inside the image in the fitted
      pose (with a moving yaw: the angle box fits the field width) that pose. Otherwise 'core' (per-step centres of the
      active hands, object centres) with the same rule. Both record their in-image ratios (points, per kind) in meta.
  (3') grasp moment: in the chosen pose, when the in-image ratio of the 'grasp' points (grasp_points: active hand
      centres + object centres within GRASP_WIN_S around the step where the object first moves) is below
      GRASP_IN_MIN, refit to those points (rule grasp; only when the refitted pose reaches GRASP_IN_MIN); frames that
      leave the image up or down afterwards are accepted. When they leave sideways and tilting cannot bring them in,
      the pose stays (recorded: ego_view.grasp_visible false = the original exclusion trigger ORIG_OUT_RULE -> the
      orig render is skipped and aug fills rows 0-3 with placement variants, record.episodes.ORIG_FOV_RULE). The
      visibility of variants is decided separately by VIS_RULE (whole object projection, 1 s window, edge margin:
      ego_view.vis_strict).
      On a desk scene, when trajectory framing was chosen only for the grasp moment and cannot bring it in, the desk
      angle stays (render.rig.frame_head).
  A solution beyond the joint limits (URDF ±1.57 rad) is not clamped: an 'all' solution beyond the limit counts as
  'does not fit' in (3) and moves on to 'core' (reason in all_fail); a 'core' solution beyond the limit dies (in clips
  with the idle hand low beside the body, 'all' asked for 95° and killed the whole session).
Desk scenes default to the desk fill (render.cameras.head_pitch_for_desk, yaw 0); only clips whose hand or object
  in-image ratio in that pose is below DESK_FILL_MIN_IN use the trajectory rule (needs_trajectory; meta head_rule =
  desk_fill | trajectory).
"""

from __future__ import annotations

import math

import numpy as np

from dexengine.core.errors import InputMissingError

FRAME_YAW = False  # move the yaw too? False (canonical) = yaw stays at its start value (0), pitch per clip
#   (view consistency: training data needs the same head azimuth in every clip; aerial
#   scenes measured: objects in frame median 1.00 with a fixed head)
FRAME_TOL_DEG = 0.05
FRAME_ITERS = 40
FRAME_DAMP = 0.5  # when an axis error flips sign (overshoot) its step gain is multiplied by this; the camera is
#   off the yaw axis, so 1° of yaw moves the azimuth by more than 1° and a full step
#   oscillates
HEAD_LIMIT_RAD = 1.57  # URDF limit of head_0 and head_1 (rby1a_model_v1.2, both the same)
# desk scenes: trajectory framing when hand or object in-frame ratio of the desk fill is lower
DESK_FILL_MIN_IN = 0.95
GRASP_WIN_S = 0.5  # window around the grasp moment
GRASP_IN_MIN = 0.9  # grasp visible = in-frame ratio of the window points (hand, object centres) >= this
#: original exclusion trigger (original off screen = the original render is skipped, aug rows 0-3); single owner, meta
#: ego_view.grasp_visible.
#: in-frame ratio of the active hand centres and object centres in the grasp window (±win_s, grasp field) >= in_min.
#: When the original passes, row 0 = the original.
ORIG_OUT_RULE = {"name": "grasp_center_win05_f90", "win_s": GRASP_WIN_S, "in_min": GRASP_IN_MIN}
#: variant acceptance and choice criterion (strict); single owner, meta ego_view.vis_strict, the visibility of the aug
#: choice 'visible variants first' (every clip). From the grasp moment (the step where the object first moves, the
#: grasp_points rule) for win_s seconds: the share of steps where the whole object projection (object shape = the 8
#: corners of the visual-bounds OBB: the USD local bounds turned by the object pose, obb_corners) is inside the image
#: edge margin (horizontal and vertical fractions) and in front of the camera >= in_min. A still object: window = all.
VIS_RULE = {"name": "grasp_obj_obb_1s_m5_f90", "shape": "obb8", "win_s": 1.0, "margin": 0.05, "in_min": 0.9}
VIS_GROUP = "vis"  # point group name: [step, corner (8 x object), 3] · stats measures it with vis_frac
# meta note 'object mostly hidden' = in-frame ratio of the object centres over all steps < this
OBJECT_HIDDEN_MAX = 0.5


def quat_R(q) -> np.ndarray:
    """wxyz -> rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def to_cam(P: np.ndarray, pos, q_ros) -> np.ndarray:
    """World points [N, 3] -> left-eye optical coordinates (x right, y down, z forward)."""
    return (np.asarray(P, float) - np.asarray(pos, float)) @ quat_R(q_ros)


def window(k: dict) -> tuple[float, float, float, float]:
    """Angle range of the image window (horizontal lo, hi; vertical lo, hi; rad)."""
    return (
        math.atan2(-k["cx"], k["fx"]),
        math.atan2(k["width"] - k["cx"], k["fx"]),
        math.atan2(-k["cy"], k["fy"]),
        math.atan2(k["height"] - k["cy"], k["fy"]),
    )


def inside(pc: np.ndarray, k: dict) -> np.ndarray:
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = k["fx"] * pc[:, 0] / z + k["cx"], k["fy"] * pc[:, 1] / z + k["cy"]
    return (z > 0) & (u >= 0) & (u < k["width"]) & (v >= 0) & (v < k["height"])


def box(pc: np.ndarray) -> tuple[float, float, float, float]:
    ax, ay = np.arctan2(pc[:, 0], pc[:, 2]), np.arctan2(pc[:, 1], pc[:, 2])
    return float(ax.min()), float(ax.max()), float(ay.min()), float(ay.max())


def fits(pc: np.ndarray, k: dict) -> bool:
    """Do the angle box's width and height fit inside the image window (would all fit when centred)?"""
    a0, a1, b0, b1 = box(pc)
    w0, w1, h0, h1 = window(k)
    return bool((pc[:, 2] > 0).all() and a1 - a0 <= w1 - w0 and b1 - b0 <= h1 - h0)


def span(a: np.ndarray, width: float) -> tuple[float, float]:
    """Angle range of one axis: [min, max] when everything fits, else [start, end] of the narrowest 'window <= width'
    (one of the windows holding the most points: max count over the sorted points, ties to the middle), so centring
    it brings in the most points."""
    a = np.sort(a)
    if a[-1] - a[0] <= width:
        return float(a[0]), float(a[-1])
    j = np.searchsorted(a, a + width, side="right")  # count inside the window [a_i, a_i + width] = j - i
    n = j - np.arange(len(a))
    best = np.flatnonzero(n == n.max())
    i = int(best[len(best) // 2])
    return float(a[i]), float(a[i] + width)


def subset(pc: np.ndarray, k: dict) -> np.ndarray:
    """Indices of the points inside the view window (span per axis) holding the most points; chosen once in the first
    pose and kept for the whole iteration (re-choosing every step is discontinuous and the fixed point oscillates;
    first check measured)."""
    w0, w1, h0, h1 = window(k)
    ax, ay = np.arctan2(pc[:, 0], pc[:, 2]), np.arctan2(pc[:, 1], pc[:, 2])
    a0, a1 = span(ax, w1 - w0) if FRAME_YAW else (w0, w1)
    b0, b1 = span(ay, h1 - h0)
    idx = np.flatnonzero((ax >= a0) & (ax <= a1) & (ay >= b0) & (ay <= b1))
    return idx if len(idx) else np.arange(len(pc))


def center_err(pc: np.ndarray, k: dict) -> tuple[float, float]:
    """Box angle centre - window angle centre (horizontal, vertical; rad, camera angles, vertical positive = down)."""
    a0, a1, b0, b1 = box(pc)
    w0, w1, h0, h1 = window(k)
    return (a0 + a1 - w0 - w1) / 2, (b0 + b1 - h0 - h1) / 2


def target_dir(pc: np.ndarray, k: dict, R: np.ndarray) -> np.ndarray:
    """World direction (unit) the optical axis must take so the box angle centre comes to the window centre."""
    dax, day = center_err(pc, k)
    d = np.array([math.tan(dax), math.tan(day), 1.0])
    d = R @ (d / np.linalg.norm(d))
    return d


def az_el(d) -> tuple[float, float]:
    """World direction -> (azimuth, depression) deg; depression = asin(-z) (same meaning as the optical-axis pitch of
    render.rig.pose)."""
    return math.degrees(math.atan2(d[1], d[0])), math.degrees(math.asin(-float(d[2])))


class HeadLimit(SystemExit):
    """The head pose that fits this point set is beyond the joint limits; uncaught it dies as before (SystemExit).
    frame catches it only for 'all' and moves on to 'core'."""


def solve(pose_fn, pts: np.ndarray, k: dict, q0: tuple[float, float]) -> dict:
    """pose_fn((yaw, pitch) rad) -> (pos, q_ros wxyz, fwd) of the left eye in that head pose. Returns yaw, pitch and the
    iteration record."""
    yaw, pitch = q0
    hist, idx = [], None
    g_az = g_el = 1.0  # per-axis step gain (x FRAME_DAMP after an overshoot)
    for it in range(FRAME_ITERS):
        pos, q, fwd = pose_fn((yaw, pitch))
        pc = to_cam(pts, pos, q)
        if idx is None:
            idx = subset(pc, k)
        if FRAME_YAW:
            az_t, el_t = az_el(target_dir(pc[idx], k, quat_R(q)))
            az, el = az_el(fwd)
            e_az, e_el = (az_t - az + 180.0) % 360.0 - 180.0, el_t - el
        else:  # the horizontal difference is only recorded (yaw fixed)
            e_az, e_el = (math.degrees(x) for x in center_err(pc[idx], k))
        hist.append(
            {
                "iter": it,
                "yaw_deg": math.degrees(yaw),
                "pitch_deg": math.degrees(pitch),
                "err_az_deg": e_az,
                "err_el_deg": e_el,
            }
        )
        if abs(e_el) < FRAME_TOL_DEG and (abs(e_az) < FRAME_TOL_DEG or not FRAME_YAW):
            break
        if hist[:-1]:  # overshoot (sign flip) lowers that axis's gain
            prev = hist[-2]
            g_az *= FRAME_DAMP if prev["err_az_deg"] * e_az < 0 else 1.0
            g_el *= FRAME_DAMP if prev["err_el_deg"] * e_el < 0 else 1.0
        yaw, pitch = yaw + (g_az * math.radians(e_az) if FRAME_YAW else 0.0), pitch + g_el * math.radians(e_el)
        if max(abs(yaw), abs(pitch)) > HEAD_LIMIT_RAD:
            raise HeadLimit(
                f"[COLLECT_FRAME] head solution beyond the limit: yaw {math.degrees(yaw):.1f}° · "
                f"pitch {math.degrees(pitch):.1f}° (limit ±{math.degrees(HEAD_LIMIT_RAD):.1f}°)"
            )
    else:
        raise SystemExit(f"[COLLECT_FRAME] framing fixed point did not converge {hist[-3:]}")
    return {"yaw": yaw, "pitch": pitch, "iters": hist, "subset_frac": round(len(idx) / max(1, len(pts)), 4)}


def grasp_unfit(fill: dict, fr: dict | None) -> float | None:
    """Desk scenes: when the only reason for moving from the desk fill pose (fill) to trajectory framing (fr) was the
    grasp moment and fr could not bring it in, that ratio (-> keep the desk angle); otherwise None."""
    only = needs_trajectory(fill) and not needs_trajectory({g: v for g, v in fill.items() if g != "grasp"})
    if fr is None or not only or fr["rule"] == "grasp":
        return None
    print(
        f"[COLLECT_FRAME] only the grasp moment is off screen and tilting cannot bring it in (rule {fr['rule']} · "
        f"grasp {fr['after'].get('grasp')}): desk angle kept",
        flush=True,
    )
    return fr["after"].get("grasp")


def needs_trajectory(fill: dict) -> bool:
    """Desk fill pose ratios (stats) -> is trajectory framing needed? Yes when hand or object is below
    DESK_FILL_MIN_IN or the grasp moment (if present) is below GRASP_IN_MIN."""
    vals = [fill[g] for g in ("hand", "object") if fill.get(g) is not None]
    if not vals:
        raise InputMissingError(f"[COLLECT_FRAME] no hand or object points {fill}")
    return min(vals) < DESK_FILL_MIN_IN or (fill.get("grasp") is not None and fill["grasp"] < GRASP_IN_MIN)


def grasp_points(obj_steps: list, hand_steps: list, step_dt: float) -> tuple[np.ndarray, dict]:
    """Points of the grasp window: obj_steps = object centre per step [k, 3] · hand_steps = active hand centre per step
    [m, 3]. Grasp moment = the step where the object first moves (rule owner record.phases.moving_span, the same value
    as the start of the 'manipulate' phase); window = GRASP_WIN_S around it. An object that never moves (pressing and
    so on) gives the window = all steps (the object place itself must be visible). Returns (points [N, 3], info)."""
    n = len(obj_steps)
    if n == 0 or len(hand_steps) != n:
        raise SystemExit(f"[COLLECT_FRAME] grasp moment: step counts differ: object {n} · hand {len(hand_steps)}")
    from dexengine.record import phases as PH

    X = np.stack([np.asarray(o, float).reshape(-1, 3) for o in obj_steps])  # [T, k, 3]
    ms = PH.moving_span(X, None, step_dt)
    hit = [] if ms is None else [ms[0]]
    h = int(round(GRASP_WIN_S / step_dt))
    lo, hi = (max(0, int(hit[0]) - h), min(n, int(hit[0]) + h + 1)) if len(hit) else (0, n)
    pts = np.concatenate(
        [np.concatenate([np.asarray(hand_steps[t], float).reshape(-1, 3), X[t]]) for t in range(lo, hi)]
    )
    return pts, {"onset_s": round(float(hit[0]) * step_dt, 3) if len(hit) else None, "window_steps": [lo, hi]}


def obb_corners(center, quat_wxyz, lo, hi) -> np.ndarray:
    """The 8 visual-bounds OBB corners of one object (world): center [3] · quat wxyz · local bounds lo/hi [3] (USD
    local, before the pose)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    C = np.array([[(hi if (i >> a) & 1 else lo)[a] for a in range(3)] for i in range(8)])
    return C @ quat_R(quat_wxyz).T + np.asarray(center, float)


def vis_points(obj_steps: list, corner_steps: list, step_dt: float) -> tuple[np.ndarray, dict]:
    """Point group of the visibility criterion (VIS_RULE): obj_steps = object centre per step [k, 3] (sets the window)
    · corner_steps = object shape corners per step [8k, 3] (obb_corners) -> ([window steps, 8k, 3], info). Window =
    from the grasp moment (the first step of record.phases.moving_span, the grasp_points value) for win_s seconds; a
    still object gives all steps. Corners != 8 x objects dies."""
    from dexengine.record import phases as PH

    X = np.stack([np.asarray(o, float).reshape(-1, 3) for o in obj_steps])  # [T, k, 3]
    Cn = np.stack([np.asarray(c, float).reshape(-1, 3) for c in corner_steps])  # [T, 8k, 3]
    if X.shape[1] == 0 or Cn.shape != (len(X), 8 * X.shape[1], 3):
        raise SystemExit(
            f"[COLLECT_FRAME] visibility criterion: objects {X.shape[1]} · corners {Cn.shape} != "
            f"(steps {len(X)}, 8 x objects, 3)"
        )
    ms = PH.moving_span(X, None, step_dt)
    n = len(X)
    lo = 0 if ms is None else int(ms[0])
    hi = n if ms is None else min(n, lo + int(round(VIS_RULE["win_s"] / step_dt)) + 1)
    return Cn[lo:hi], {"window_steps": [lo, hi], "onset": ms is not None}


def add_grasp(groups: dict, obj_steps: list, hand_steps: list, corner_steps: list, step_dt: float) -> dict:
    """Add the grasp (grasp_points) and visibility (VIS_GROUP, vis_points) fields to the point groups -> info (window,
    onset)."""
    groups["grasp"], info = grasp_points(obj_steps, hand_steps, step_dt)
    groups[VIS_GROUP], info["vis"] = vis_points(obj_steps, corner_steps, step_dt)
    return info


def vis_frac(P: np.ndarray, pos, q, k: dict) -> float | None:
    """[step, corner, 3] -> share of steps where every object shape corner is in front of the camera and inside the
    image edge VIS_RULE.margin. The convex hull of the projected corners covers the object projection, so all corners
    inside means the whole projection is inside."""
    P = np.asarray(P, float)
    if P.ndim != 3 or P.shape[0] == 0 or P.shape[-1] != 3:
        return None
    pc = to_cam(P.reshape(-1, 3), pos, q)
    z = pc[:, 2]
    m = float(VIS_RULE["margin"])
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = k["fx"] * pc[:, 0] / z + k["cx"], k["fy"] * pc[:, 1] / z + k["cy"]
        ok = (
            (z > 0)
            & (u >= m * k["width"])
            & (u <= (1 - m) * k["width"])
            & (v >= m * k["height"])
            & (v <= (1 - m) * k["height"])
        )
    return round(float(ok.reshape(P.shape[:2]).all(axis=1).mean()), 4)


def ego_view(fill: dict, ladder: dict | None = None) -> dict:
    """Episode meta fields: in-frame ratios of the fixed head pose (stats) -> grasp_visible (original exclusion trigger
    ORIG_OUT_RULE, grasp field) · vis_strict (variant choice VIS_RULE, vis field) · 'object mostly hidden' note ·
    vis_ladder (visibility ladder, measured by check_view for later trajectories only; else None)."""
    g, o, vis = fill.get("grasp"), fill.get("object"), fill.get(VIS_GROUP)
    return {
        "grasp_in_frame": g,
        "object_in_frame": o,
        "hand_in_frame": fill.get("hand"),
        "vis_in_frame": vis,
        "grasp_visible": None if g is None else bool(g >= ORIG_OUT_RULE["in_min"]),
        "vis_strict": None if vis is None else bool(vis >= VIS_RULE["in_min"]),
        "object_hidden": None if o is None else bool(o < OBJECT_HIDDEN_MAX),
        "vis_ladder": ladder,
        "rule": {"orig_out": ORIG_OUT_RULE, "vis": VIS_RULE, "object_hidden_max": OBJECT_HIDDEN_MAX},
    }


def ros_from_opengl(q_gl) -> np.ndarray:
    """OpenGL camera pose (wxyz; columns = x right, y up, -z forward; Isaac quat_w_opengl) -> the optical convention of
    to_cam (x right, y down, z forward) = q ⊗ Rx(180°), the same transform as Rc @ diag(1, -1, -1) in
    render.rig.pose."""
    w, x, y, z = (float(v) for v in q_gl)
    return np.array([-x, w, z, -y])


def k_of(K, res) -> dict:
    """3x3 intrinsics + [W, H] -> the k of inside."""
    return {
        "fx": float(K[0][0]),
        "fy": float(K[1][1]),
        "cx": float(K[0][2]),
        "cy": float(K[1][2]),
        "width": int(res[0]),
        "height": int(res[1]),
    }


def exo_cam(cal: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Outer camera (render meta camera, fixed for the session after the dolly) -> (world position = exo.pos +
    desk.origin, optical q, k)."""
    ex = cal["exo"]
    pos = np.asarray(ex["pos"], float) + np.asarray(cal["desk"]["origin"], float)
    return pos, ros_from_opengl(ex["quat_opengl_wxyz"]), k_of(cal["K_exo"], cal["exo_res"])


def exo_view(cal: dict, groups: dict, upper) -> dict:
    """Episode meta exo_view: the same point groups and rule as ego_view, seen from the outer camera (exo_cam).
    (stats → ego_view).
    Oracle: the upper-body points the dolly put inside the image (world, Rig.view_ref P) must all be inside, else the
    camera convention is wrong (dies)."""
    pos, q, k = exo_cam(cal)
    up = inside(to_cam(upper, pos, q), k)
    if not up.all():
        raise SystemExit(
            f"[COLLECT_FRAME] exo_view oracle: upper-body points {int((~up).sum())}/{len(up)} outside the "
            "outer image: the camera convention (OpenGL, wxyz, origin) is wrong"
        )
    return ego_view(stats(groups, pos, q, k))


def stats(groups: dict, pos, q, k: dict) -> dict:
    """In-frame ratio per kind: groups = {name: points [N, 3]}; the visibility field (VIS_GROUP, [step, corner, 3]) uses
    vis_frac."""
    return {
        g: vis_frac(P, pos, q, k)
        if np.ndim(P) == 3
        else (round(float(inside(to_cam(P, pos, q), k).mean()), 4) if len(P) else None)
        for g, P in groups.items()
    }


def frame(pose_fn, groups: dict, k: dict, q0: tuple[float, float]) -> dict:
    """Priority rule: groups has 'all' and 'core' (other fields are only measured)."""
    pos, q, _f = pose_fn(q0)
    before = stats(groups, pos, q, k)
    rule, all_fail = "all", None
    try:
        sol = solve(pose_fn, groups["all"], k, q0)
        pos, q, _f = pose_fn((sol["yaw"], sol["pitch"]))
        pc = to_cam(groups["all"], pos, q)
        fit_all = fits(pc, k) if FRAME_YAW else bool(inside(pc, k).all())
    # the pose for 'all' is beyond the limit = 'all does not fit' (rule 3): core
    except HeadLimit as e:
        all_fail, fit_all = str(e), False
        print(f"{e}: fitting 'core' instead of 'all'", flush=True)
    if not fit_all:
        rule = "core"
        sol = solve(pose_fn, groups["core"], k, q0)  # core beyond the limit dies here (no clamping)
        pos, q, _f = pose_fn((sol["yaw"], sol["pitch"]))
    G = groups.get("grasp")
    if G is not None and len(G) and float(inside(to_cam(G, pos, q), k).mean()) < GRASP_IN_MIN:
        # (3') grasp off screen -> fit to it (beyond the limit: keep the pose)
        try:
            gs = solve(pose_fn, G, k, (sol["yaw"], sol["pitch"]))
            gpos, gq, _f = pose_fn((gs["yaw"], gs["pitch"]))
            if float(inside(to_cam(G, gpos, gq), k).mean()) >= GRASP_IN_MIN:  # only when it fits (no partial gain)
                rule, sol, pos, q = "grasp", gs, gpos, gq
        except HeadLimit as e:
            print(f"{e}: grasp fitting beyond the limit, previous pose kept", flush=True)
    return {
        "rule": rule,
        "yaw": sol["yaw"],
        "pitch": sol["pitch"],
        "iters": sol["iters"],
        "subset_frac": sol["subset_frac"],
        "yaw_free": FRAME_YAW,
        "before": before,
        "all_fail": all_fail,
        "after": stats(groups, pos, q, k),
    }
