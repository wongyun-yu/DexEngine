"""Inputs and prediction for the cell choice of originally off-screen clips (augment.policy.CELL_RULE_FOV); numpy,
no Isaac.

Record (view_record): the orig-round render session (render.clip_phases.orig_fov_phase) stores the calibrated ego
  left-eye pose, the intrinsics, the visibility window points (object shape vertices of render.framing.VIS_RULE; the
  window from the grasp moment; render coordinates [steps, 8 x objects, 3]) and the rotation centre q (frame-0 object
  xy mean in render coordinates; the render-coordinate twin of the placement Src.object_xy_frame0) in
  render_clip.json orig_fov.view.
Prediction (predict): a variant (ψ deg, d_scene m) moves the window points rigidly (same form as augment.scene.Tf:
  x' = R(ψ)(x - q) + q + d, z unchanged) -> (visible fraction vis_frac, distance px of the window mean projection
  from the image centre). Render coordinates = scene coordinates + a translation (canary prediction oracle: 9
  candidates, max |measured - predicted| 0.039, judged 9/9). The step where objects start to move is invariant under
  the rigid transform, so the window is the same.
The ranking is only a prediction: acceptance and row choice use the render session's measured vis_strict
(record.episodes.choose_rows).
"""

from __future__ import annotations

import math

import numpy as np

from dexengine.core.errors import InputMissingError
from dexengine.record.render import framing as CF

VIEW_KEYS = ("cam_pos", "cam_quat_ros_wxyz", "K", "vis_pts", "q_xy")
K_KEYS = ("fx", "fy", "cx", "cy", "width", "height")
MARK = "[COLLECT_FOV]"


def view_record(cal: dict, T: dict, clip: str) -> dict:
    """Calibration result cal (Rig.calibrate) · trajectory context T (render.episode.load_traj: groups, tf, st) ->
    record."""
    from dexengine.record.render import framing as CF

    st = np.asarray(T["st"], float)
    objs = [e for e in T["tf"].entities if str(e["name"]).startswith("object")]
    if not objs:
        raise InputMissingError(f"{MARK} {clip}: the trajectory has no object entity: no rotation centre")
    q = np.mean([st[0, e["lo"] : e["lo"] + 2] for e in objs], axis=0)
    eye, kz = cal["ego_left_pose"], cal["zed"]
    return {
        "cam_pos": [float(v) for v in eye["pos"]],
        "cam_quat_ros_wxyz": [float(v) for v in eye["quat_ros_wxyz"]],
        "K": {k: float(kz[k]) for k in K_KEYS},
        "q_xy": [float(q[0]), float(q[1])],
        "vis_pts": np.round(np.asarray(T["groups"][CF.VIS_GROUP], float), 5).tolist(),
    }


def check(view: dict) -> dict:
    """Record shape check: a missing field or a point cloud not shaped [steps, vertices, 3] dies (the old sphere shape
    [.., 4] dies too)."""
    miss = [k for k in VIEW_KEYS if k not in (view or {})] + [k for k in K_KEYS if k not in (view or {}).get("K", {})]
    S4 = np.asarray((view or {}).get("vis_pts", []), float)
    if miss or S4.ndim != 3 or S4.shape[-1] != 3 or S4.shape[0] == 0:
        raise SystemExit(
            f"{MARK} off-screen cell choice record is broken: missing fields {miss} · point shape {S4.shape}"
        )
    return view


def transform(S4, q_xy, psi_deg: float, d_xy) -> np.ndarray:
    """Window points [steps, vertices, 3] -> variant window points (xy rigid transform, z unchanged; the object pose
    turns with it)."""
    c, s = math.cos(math.radians(psi_deg)), math.sin(math.radians(psi_deg))
    R = np.array([[c, -s], [s, c]])
    q = np.asarray(q_xy, float)
    Y = np.array(S4, float)
    Y[..., :2] = (Y[..., :2] - q) @ R.T + q + np.asarray(d_xy, float)
    return Y


def predict(view: dict, psi_deg: float, d_xy) -> tuple[float, float]:
    """Prediction for one variant -> (visible fraction 0..1, distance px of the window mean projection from the image
    centre; inf behind the camera)."""
    from dexengine.record.render import framing as CF

    Y = transform(view["vis_pts"], view["q_xy"], psi_deg, d_xy)
    K, pos, q = view["K"], view["cam_pos"], view["cam_quat_ros_wxyz"]
    v = CF.vis_frac(Y, pos, q, K)
    pc = CF.to_cam(Y[..., :3].reshape(-1, 3), pos, q)
    if np.any(pc[:, 2] <= 1e-6):
        return float(v or 0.0), math.inf
    u = K["fx"] * pc[:, 0] / pc[:, 2] + K["cx"]
    w = K["fy"] * pc[:, 1] / pc[:, 2] + K["cy"]
    return float(v or 0.0), float(math.hypot(u.mean() - K["width"] / 2, w.mean() - K["height"] / 2))


ORIG_FOV_OWNER_MARK = "[ORIG_FOV_OWNER]"
FOVOUT_MARK = "[AUG_FOVOUT_CANON]"
FOVOUT_CANON_RULE = "fovout_canon_drop_variants"


def orig_fov_decide(measured, owner) -> bool:
    """aug-round original visibility decision (input to the row-0 choice): the owner's value when there is one, else
    the session's measured value (old path). The owner carries off-screen (False) only; any other value dies. The
    measured value must be a bool too."""
    if not isinstance(measured, bool):
        raise ValueError(f"original grasp-moment visibility is not a bool {measured!r}")
    if owner is None:
        return measured
    if owner is not False:
        raise ValueError(f"orig_fov_owner takes False (off-screen) only: {owner!r}")
    return False


def choice_expect(c: str, tc: dict, vis: dict) -> list[str]:
    """The original's row of the trajectory choice check (record.driver_check.choice_problems) under an owner record
    (traj_choice.orig_fov carries owner): the original candidate's grasp_visible there is the owner's judgment (input
    of choose_rows), not a measurement. The judgment is checked on its own against orig_fov_decide(measured, owner),
    and the original's expected value in vis {name: (grasp_visible, vis_strict)} becomes its measurement
    (orig_fov.measured, the original candidate's vis_strict; in place), so the render's ego_view is compared
    measurement to measurement. orig_fov.measured is the original's own measured value (render.clip_phases.
    choose_phase, whatever variants were probed). No owner = [] (the old path)."""
    of = tc.get("orig_fov") or {}
    if "owner" not in of:
        return []
    o = next((x for x in tc.get("cands", []) if x.get("role") == "orig"), None)
    if o is None:
        return [f"{c}: owner record without an original candidate"]
    try:
        dec = orig_fov_decide(of.get("measured"), of["owner"])
    except ValueError as ex:
        return [f"{c}: the owner judgment cannot be recomputed: {ex}"]
    bad = [] if o["grasp_visible"] is dec else [f"{c}: original judgment {o['grasp_visible']} != owner rule {dec}"]
    vis[o["name"]] = (of["measured"], o.get("vis_strict"))
    return bad


def owner_set(a) -> frozenset[str]:
    """--orig-fov-out of the record driver (comma separated, a.orig_fov_out) -> set of clip base names. The first call
    checks aug round and ⊆ --clips (else dies) and prints the on marker once (cached on a). Missing or empty = empty
    set (old path)."""
    from dexengine.core import paths as P

    got = getattr(a, "_fov_owner", None)
    if got is not None:
        return got
    out = frozenset(P.clip_name(c.strip()) for c in (getattr(a, "orig_fov_out", "") or "").split(",") if c.strip())
    if out:
        if a.round != "aug":
            raise SystemExit(f"{ORIG_FOV_OWNER_MARK} --orig-fov-out is for the aug round only: round {a.round}")
        if miss := sorted(out - {P.clip_name(c) for c in a.clips.split(",") if c}):
            raise InputMissingError(
                f"{ORIG_FOV_OWNER_MARK} clips {miss[:4]} (total {len(miss)}) of --orig-fov-out are not in --clips"
            )
        print(
            f"{ORIG_FOV_OWNER_MARK} on · owners off-screen {len(out)} {sorted(out)[:4]}: aug row 0 uses this value · "
            f"passing variants roll out even when the canonical cell drops ({FOVOUT_CANON_RULE})",
            flush=True,
        )
    a._fov_owner = out
    return out


def gate(a, m: dict, c: dict, clip: str, cp: str) -> tuple[str | None, bool]:
    """Rollout gate of augment.session one_clip: canon_stop (owner list owner_set); when the rule applies,
    c.canon_fovout + a marker line."""
    from dexengine.core import paths as P

    stop, fo = canon_stop(m, cp, P.clip_name(clip), owner_set(a))
    if fo:
        c["canon_fovout"] = cp
        print(f"{FOVOUT_MARK} {clip}: {cp} -> passing variants {c['pre']} roll out · post gate --no-ref", flush=True)
    return stop, fo


def canon_stop(m: dict, cp: str, nm: str, fov: frozenset[str]) -> tuple[str | None, bool]:
    """Generation record m · canonical cell problem cp (augment.stage.canon_problem) -> (reason to stop without
    rollout | None, rule applied). Off-screen (nm ∈ fov) ∧ cp ∧ passing variants >= 1 does not stop. Otherwise the
    old rule (canonical cell dropped or 0 passing -> stop)."""
    if cp and nm in fov and any(v["status"] == "ok" for v in m["variants"] if v["k"] != 0):
        return None, True
    if cp or not any(v["status"] == "ok" for v in m["variants"]):
        return cp or "", False
    return None, False


#: visibility ladder for filling rows 0-3 of originally off-screen clips (single owner; strictest first).
#: Point cloud = VIS_GROUP (VIS_RULE window, object OBB vertices). Step form all = every vertex inside the image
#: (inside the edge margin, in front of the camera) · center_half = vertex mean (centre) inside the image ∧ at least
#: half the vertices inside (a proxy of 50 % projected area). Pass = step fraction of the window >= in_min. L0 =
#: render.framing.VIS_RULE as it is (= vis_strict). Bottom (VIS_LADDER_BOTTOM) = top by visibility score (in-image
#: fraction of all window vertices, margin 0); always present.
VIS_LADDER = (
    {"name": "L0", "step": "all", "margin": CF.VIS_RULE["margin"], "in_min": CF.VIS_RULE["in_min"]},
    {"name": "L1", "step": "all", "margin": 0.0, "in_min": CF.VIS_RULE["in_min"]},
    {"name": "L2", "step": "all", "margin": 0.0, "in_min": 0.75},
    {"name": "L3", "step": "center_half", "margin": 0.0, "in_min": 0.75},
)
VIS_LADDER_BOTTOM = "L4"


def vis_ladder(P: np.ndarray, pos, q, k: dict) -> dict | None:
    """[steps, vertices, 3] -> visibility ladder (VIS_LADDER) {pass: {level: bool}, frac: {level: step fraction},
    score}; None when not a point cloud. L0 is the same computation as render.framing.vis_frac (same margin, same
    rounding), so it must equal vis_strict (choose_rows checks)."""
    if P is None:
        return None
    P = np.asarray(P, float)
    if P.ndim != 3 or P.shape[0] == 0 or P.shape[-1] != 3:
        return None
    pc = CF.to_cam(P.reshape(-1, 3), pos, q)
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = k["fx"] * pc[:, 0] / z + k["cx"], k["fy"] * pc[:, 1] / z + k["cy"]
        cc = CF.to_cam(P.mean(axis=1), pos, q)
        cu, cv = k["fx"] * cc[:, 0] / cc[:, 2] + k["cx"], k["fy"] * cc[:, 1] / cc[:, 2] + k["cy"]

    def ins(m: float) -> np.ndarray:
        return (
            (z > 0)
            & (u >= m * k["width"])
            & (u <= (1 - m) * k["width"])
            & (v >= m * k["height"])
            & (v <= (1 - m) * k["height"])
        ).reshape(P.shape[:2])

    cin = (cc[:, 2] > 0) & (cu >= 0) & (cu <= k["width"]) & (cv >= 0) & (cv <= k["height"])
    fr, ok = {}, {}
    for lv in VIS_LADDER:
        step = ins(float(lv["margin"])).all(axis=1) if lv["step"] == "all" else (cin & (ins(0.0).mean(axis=1) >= 0.5))
        fr[lv["name"]] = round(float(step.mean()), 4)
        ok[lv["name"]] = bool(fr[lv["name"]] >= float(lv["in_min"]))
    return {"pass": ok, "frac": fr, "score": round(float(ins(0.0).mean()), 4)}


def view_measures(groups: dict, pos, q, k: dict) -> dict:
    """Measurement of the later-trajectory camera check (render.clip_session.check_view): fill (in-frame ratio per
    kind, render.framing.stats) + ladder (visibility ladder; None without visibility points)."""
    return {"fill": CF.stats(groups, pos, q, k), "ladder": vis_ladder(groups.get(CF.VIS_GROUP), pos, q, k)}
