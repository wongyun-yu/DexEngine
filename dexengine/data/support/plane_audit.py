"""Support-plane audit of one clip: object vertices, rest intervals and the keypoint audit (driven by
support/plane.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dexengine.data.support import desk_rule as DR
from dexengine.data.support.rest import REST_MIN_S, TIP_FREE, V_REST, Z_STAB

FOOT_MARGIN, HAND_TOL, DT, SMOOTH, STEP = 0.10, 0.02, 0.02, 15, 5


def obj_verts(mesh_dir: Path, n=2000):
    p = mesh_dir / "visual.obj"
    if not p.exists():
        return None
    v = np.array(
        [list(map(float, ln.split()[1:4])) for ln in p.read_text(errors="replace").split("\n") if ln.startswith("v ")],
        dtype=float,
    )
    return v[np.linspace(0, len(v) - 1, min(n, len(v))).astype(int)] if len(v) else None


def qmat(q):
    q = q / max(np.linalg.norm(q), 1e-9)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def intervals(mask, min_len):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def audit_clip(mano_dir: Path, ti: dict, obj_root: Path, ds: str = "", h2o_untilted: bool = False):
    """Support audit of one clip. `h2o_untilted`: the H2O first pass without the desk-plane rotation — it only
    gives the rest intervals the desk plane's pivot reads (data/support/h2o.py; READERS h2o derive order)."""
    kp = np.load(mano_dir / "0/trajectory_keypoints.npz")
    kp = {k: kp[k] for k in kp.files}
    tilt = None
    if (
        ds == "h2o" and not h2o_untilted
        # D10-e'': desk plane rotation (h2o_frame.tilt_transform), the same M as base_adapter. No plane → exception →
        # cls=error (excluded)
    ):
        from dexengine.data.readers import h2o_frame as HF

        tilt = HF.tilt_transform(mano_dir.name)
        kp = HF.apply_rigid_keypoints(kp, tilt["M"])
    rec = _audit_keypoints(kp, ti, obj_root)
    if tilt is not None:
        rec["h2o_tilt"] = {
            "tilt_deg": round(tilt["tilt_deg"], 2),
            "desk_z_plane": round(tilt["desk_z"], 4),
            "pivot_source": tilt["pivot_source"],
            "version": HF.TILT_VERSION,
        }
    return rec


def _hand_tips(kp: dict) -> tuple[list, list]:
    """Fingertip positions and fingertip + wrist z of each present hand."""
    tips, hands_z = [], []
    for side in ("right", "left"):
        f = kp.get(f"qpos_finger_{side}")
        w = kp.get(f"qpos_wrist_{side}")
        if f is None or w is None or np.abs(w[:, :3]).sum() == 0:
            continue
        tips.append(f[..., :3])
        hands_z.append(np.concatenate([f[..., 2], w[:, 2:3]], axis=1))
    return tips, hands_z


def _free_samples(o, v, tips, i: int, j: int) -> tuple[list, list]:
    """Bottom z and xy of the object on the samples (every STEP frames) of rest interval [i, j) where every fingertip
    is at least TIP_FREE away from the object surface.

    A rest interval mixes in the moments the hand approaches or releases; samples away for at least REST_MIN_S are
    "placed", and the bottom z is measured on those samples.
    """
    zs = []
    xys = []
    ks = list(range(i, j, STEP))
    for k in ks:
        R = qmat(o[k, 3:7])
        vw = v @ R.T + o[k, :3]
        d = np.min(np.linalg.norm(tips[k][:, None, :] - vw[None, :, :], axis=2))
        if d >= TIP_FREE:
            zs.append(float(vw[:, 2].min()))
            xys.append(vw[:, :2])
    return zs, xys


def _add_placed_rests(rec: dict, rests_z: list, rests_xy: list, side: str, o, v, tips) -> None:
    """Placed intervals of one side's object: still (smoothed speed < V_REST) for REST_MIN_S with the hand away and the
    bottom z stable within Z_STAB; each goes into rec["rest_intervals"], its bottom z / xy into rests_z / rests_xy."""
    ok = np.isfinite(o).all(1)
    pos = o[:, :3]
    vel = np.linalg.norm(np.gradient(pos, DT, axis=0), axis=1)
    kern = np.ones(SMOOTH) / SMOOTH
    vel = np.convolve(vel, kern, mode="same")
    still = ok & (vel < V_REST)
    n_min = int(REST_MIN_S / DT)
    for i, j in intervals(still, n_min):
        zs, xys = _free_samples(o, v, tips, i, j)
        free = len(zs) * STEP * DT >= REST_MIN_S and (max(zs) - min(zs)) <= Z_STAB
        if free:
            rec["rest_intervals"].append(
                {"side": side, "start": int(i), "end": int(j), "bottom_z": round(float(np.median(zs)), 4)}
            )
            rests_z.append(np.median(zs))
            rests_xy.append(np.concatenate(xys))


def _set_support(rec: dict, rests_z: list, rests_xy: list, tips) -> None:
    """Support z (median placed bottom), footprint (placed xy ± FOOT_MARGIN), conflict frames and depth, class."""
    tz = float(np.median(rests_z))
    rec["table_z"] = round(tz, 4)
    xy = np.concatenate(rests_xy)
    lo = xy.min(0) - FOOT_MARGIN
    hi = xy.max(0) + FOOT_MARGIN
    rec["footprint"] = [round(float(x), 3) for x in (*lo, *hi)]
    tipxy = tips[..., :2]
    inside = (tipxy[..., 0] > lo[0]) & (tipxy[..., 0] < hi[0]) & (tipxy[..., 1] > lo[1]) & (tipxy[..., 1] < hi[1])
    below = tips[..., 2] < tz - HAND_TOL
    conf = int(np.any(inside & below, axis=1).sum())
    rec["conflict_frames"] = conf
    if conf:  # conflict depth: depth of fingertips below the support inside the footprint (median · max)
        depth = (tz - tips[..., 2])[inside & below]
        rec["conflict_depth_p50"] = round(float(np.median(depth)), 3)
        rec["conflict_depth_max"] = round(float(depth.max()), 3)
    rec["cls"] = "table_conflict" if conf > 0 else "table"
    rec["rest_at_start"] = bool(any(r["start"] == 0 for r in rec["rest_intervals"]))


def _hand_rest_estimate(kp: dict):
    """Desk height cue without a placed interval: a hand lingering slowly (<3 cm/s) for ≥0.3 s within 1.5 cm of its
    own lowest point is taken to rest on the desk (None when no hand does). It cannot be verified (no source offset),
    so it stays as desk_z_est + confidence=low only; a delivered ground_offset_z wins."""
    est = None
    for side in ("right", "left"):
        w = kp.get(f"qpos_wrist_{side}")
        f = kp.get(f"qpos_finger_{side}")
        if w is None or f is None or np.abs(w[:, :3]).sum() == 0:
            continue
        z = np.minimum(w[:, 2], f[..., 2].min(1))
        sp = np.convolve(np.linalg.norm(np.gradient(w[:, :3], DT, axis=0), axis=1), np.ones(10) / 10, mode="same")
        near = (z < z.min() + 0.015) & (sp < 0.03)
        run = 0
        for v in near:
            run = run + 1 if v else 0
            if run >= 15:
                est = float(z.min()) if est is None else min(est, float(z.min()))
                break
    return est


def _object_lows(kp: dict, ti: dict, obj_root: Path) -> tuple[list, list]:
    """Lowest object point on ~30 sampled frames (obj_zmin, copied into trees) and over all frames (obj_zmin_all,
    read by the desk height rule; INTAKE only), per side."""
    ozs = []
    lows = []
    for side in ("right", "left"):
        md = ti.get(f"{side}_object_mesh_dir")
        o = kp.get(f"qpos_obj_{side}")
        if not md or o is None or o.shape[0] < 2:
            continue
        v = obj_verts(obj_root / Path(md).name)
        if v is None:
            continue
        ozs.extend(
            float((v @ qmat(o[k, 3:7]).T + o[k, :3])[:, 2].min())
            for k in range(0, o.shape[0], max(1, o.shape[0] // 30))
            if np.isfinite(o[k]).all()
        )
        lows.append(DR.low_all_frames(v, o))
    return ozs, lows


def _audit_keypoints(kp: dict, ti: dict, obj_root: Path):
    """Support verdict of one clip from keypoints: placed intervals, support z, footprint, conflicts and low points."""
    tips, hands_z = _hand_tips(kp)
    T = kp["timestamps"].shape[0] if "timestamps" in kp else (tips[0].shape[0] if tips else 0)
    if not tips:
        return {"cls": "no_hand", "T": int(T)}
    tips = np.concatenate(tips, axis=1)
    hands_z = np.concatenate(hands_z, axis=1)
    rec = {
        "T": int(T),
        "objects": [],
        "rest_intervals": [],
        "table_z": None,
        "footprint": None,
        "conflict_frames": 0,
        "cls": "none",
    }
    rests_z, rests_xy = [], []
    for side in ("right", "left"):
        md = ti.get(f"{side}_object_mesh_dir")
        o = kp.get(f"qpos_obj_{side}")
        if not md or o is None or o.shape[0] < 5:
            continue
        v = obj_verts(obj_root / Path(md).name)
        if v is None:
            continue
        _add_placed_rests(rec, rests_z, rests_xy, side, o, v, tips)
        rec["objects"].append(side)
    if rests_z:
        _set_support(rec, rests_z, rests_xy, tips)
    rec["hand_zmin"] = round(float(np.nanmin(hands_z)), 4)
    est = _hand_rest_estimate(kp)
    rec["desk_z_est_handrest"] = round(est, 4) if est is not None else None
    ozs, lows = _object_lows(kp, ti, obj_root)
    rec["obj_zmin"] = round(min(ozs), 4) if ozs else None
    lows = [x for x in lows if x is not None]
    rec["obj_zmin_all"] = round(min(lows), 4) if lows else None
    return rec
