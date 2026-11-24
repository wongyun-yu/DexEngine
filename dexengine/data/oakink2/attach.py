#!/usr/bin/env python3
"""OakInk2 attached parts (fixed, lever): attach pose and the rulers of the lever rule.

No thresholds here: the lever band is oakink2.part_kinds.LEVER_TIP_M · LEVER_DRIVE_SHARE (read by lever_in_band), and
the
one-body thresholds (oakink2.artic_fit.ROT_TOL_DEG · POS_TOL_M) are passed by the callers (pairing part_clusters,
converter). Input = the part's pose sequence in the attachment representative's body frame (positions (T,3), rotations
Rotation; oakink2.artic_fit.rel_pose).

  is_lever(o)                       whether the spec lists the part as a lever
  part_points(o)                    part mesh vertices (body frame), the surface the tip displacement reads
  surface_shift(pts, pa, Ra, pb, Rb) max distance surface points move between two poses of the same body
  lever_rest(Rrel)                  the lever's rest pose (frame) and the per-frame angle away from it
  lever_motion(prel, Rrel, pts, …)  tip displacement · share of the jointed part's motion with the lever turned
  lever_in_band(tip_m, …)           whether the lever is inside the fixed-attachment band; the one check pairing and the
      converter FK oracle read
  lever_fixed(prel, Rrel, o, …)     verdict for one lever (record · reason when it cannot be fixed)
  attach_pose(prel, Rrel, …)        fixed attach pose = clip mean (for a lever, frames in the rest pose only)
  fk_errors(m, q_new, chk, pts)     body FK of the merged model ↔ source pose sequence (position · rotation · surface
      shift); the converter oracle's measurement
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import paths as P  # noqa: E402
from dexengine.data.oakink2 import part_kinds as PK  # noqa: E402 — owner of part kinds
from dexengine.data.oakink2 import parts as OP  # noqa: E402


def is_lever(o: dict) -> bool:
    """Whether the spec lists the part as a lever (a hinged lever on the attachment representative)."""
    return PK.kind_of(OP.rid_of(o["obj_id"]))[0] == "lever"


def part_points(o: dict) -> np.ndarray:
    """Mesh vertices of a part object (body frame, (N,3)). The max surface displacement between two rigid poses occurs
    at
    a vertex. A missing mesh fails (no fallback to the origin alone)."""
    f = P.main_root() / P.data_layout.object_path("oakink2", o["mesh_dir"]) / "visual.obj"
    v = [ln.split()[1:4] for ln in f.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.startswith("v ")]
    if not v:
        raise ValueError(f"[ARTIC_ATTACH] {f}: no vertices — cannot measure the lever tip displacement")
    return np.asarray(v, float)


def surface_shift(pts: np.ndarray, pa, Ra, pb, Rb) -> np.ndarray:
    """Max distance (m) surface points move between two poses of the same body (positions (…,3) · rotation matrices
    (…,3,3)), per frame."""
    dR, dp = np.asarray(Ra) - np.asarray(Rb), np.asarray(pa) - np.asarray(pb)
    return np.linalg.norm(np.einsum("...ij,nj->...ni", dR, pts) + dp[..., None, :], axis=-1).max(-1)


def lever_rest(Rrel) -> tuple[int, np.ndarray]:
    """The lever's rest pose = the frame whose relative rotation is closest to the clip median (a lever is pressed and
    returns, so most of the clip is at rest) · the per-frame angle away from it (°). (rest frame, angles (T,))."""
    v = (Rrel * Rrel[0].inv()).as_rotvec()
    rest = int(np.linalg.norm(v - np.median(v, 0), axis=1).argmin())
    return rest, np.degrees((Rrel * Rrel[rest].inv()).magnitude())


def lever_motion(prel, Rrel, pts: np.ndarray, alpha: np.ndarray | None, away_deg: float) -> dict:
    """Measurements read by the lever rule.
    tip_m        tip displacement: the farthest distance the lever surface (vertices `pts`) travels from the rest pose
    drive_share  share of the jointed part's motion (joint values `alpha`) during which the lever is turned more than
                 `away_deg` from rest; counted per interval of constant lever state from that interval's joint-value
                 range (summing frame differences would accumulate noise per frame). 0 without a jointed part (`alpha`
                 None) or when it did not move."""
    rest, dev = lever_rest(Rrel)
    Rm = Rrel.as_matrix()
    out = {"tip_m": float(surface_shift(pts, prel, Rm, prel[rest], Rm[rest]).max()), "drive_share": 0.0}
    if alpha is not None:
        away = dev > away_deg
        cut = [0, *(np.flatnonzero(np.diff(away.astype(np.int8))) + 1).tolist(), len(away) - 1]
        span = [
            (bool(away[a]), float(np.ptp(alpha[a : b + 1]))) for a, b in zip(cut[:-1], cut[1:], strict=True) if b > a
        ]
        total = sum(v for _f, v in span)
        if total > 0:
            out["drive_share"] = sum(v for f, v in span if f) / total
    return out


def lever_in_band(tip_m: float, drive_share: float = 0.0, slack_m: float = 0.0) -> bool:
    """Whether the lever is inside the fixed-attachment band: tip displacement ≤ LEVER_TIP_M (+ `slack_m`) and the share
    of pushing the jointed part < LEVER_DRIVE_SHARE (values owned by oakink2.part_kinds). Read by pairing (lever_fixed ←
    oakink2.artic_fit.part_clusters) and the converter FK oracle (oakink2.artic_convert.convert_tree: lever surface
    shift
    in the merged model; the spec asset's attach pose may differ from this clip's rest pose by up to the one-body
    translation threshold, which is passed as `slack_m`)."""
    return tip_m <= PK.LEVER_TIP_M + slack_m and drive_share < PK.LEVER_DRIVE_SHARE


def lever_fixed(prel, Rrel, o: dict, alpha: np.ndarray | None, away_deg: float) -> tuple[dict, str | None]:
    """Verdict for one lever: (record {obj_id · tip_m · drive_share}, the reason it cannot be fixed in its rest pose, or
    None). `alpha` = joint values of the jointed part the lever can push (None if none). The reason text reaches the
    exclusion ledger and stays as written."""
    lm = lever_motion(prel, Rrel, part_points(o), alpha, away_deg)
    rec = {"obj_id": o["obj_id"], "tip_m": round(lm["tip_m"], 4), "drive_share": round(lm["drive_share"], 3)}
    if lever_in_band(lm["tip_m"], lm["drive_share"]):
        return rec, None
    return rec, (
        f"cannot be fixed (tip displacement {lm['tip_m'] * 1e3:.1f} mm · limit {PK.LEVER_TIP_M * 1e3:.0f} · "
        "share of the "
        f"joint-part motion with the lever turned {lm['drive_share']:.2f} · limit {PK.LEVER_DRIVE_SHARE}) — two joints "
        f"are outside the converter's one-joint contract (not a data defect)"
    )


def attach_pose(prel, Rrel, lever: bool, away_deg: float):
    """Fixed attach pose (attachment representative body frame (p, R)) = clip mean. Taking one frame of a relative pose
    that wobbles even within one body would grow the FK oracle's max error by the whole wobble. A lever is fixed in its
    rest pose, so only frames at rest are averaged (pressed intervals do not pull the mean; a still lever uses every
    frame
    and gives the same value)."""
    if lever:
        keep = lever_rest(Rrel)[1] <= away_deg
        prel, Rrel = prel[keep], Rrel[keep]
    return prel.mean(0), Rrel.mean()


def fk_errors(m, q_new: np.ndarray, chk: list, pts: dict | None = None) -> dict | None:
    """FK oracle measurement: per body of the merged model `m`, max (position m, rotation °, surface shift m) of the
    world
    pose vs the source pose sequence, {body: [dp, dr, ds]}. `chk` = [(body name, source 7 columns xyz + wxyz)] · `pts` =
    {body: surface points}; ds is measured only for those bodies (levers, else 0). None when a body is missing (with -1,
    d.xpos[-1] would silently read the last body). Up to 200 evenly spaced frames."""
    import mujoco  # noqa: PLC0415

    bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b, _qo in chk]
    if min(bids) < 0:
        return None
    d = mujoco.MjData(m)
    err = {b: [0.0, 0.0, 0.0] for b, _qo in chk}
    for t in np.linspace(0, len(q_new) - 1, min(len(q_new), 200)).astype(int):
        d.qpos[:] = q_new[t]
        mujoco.mj_forward(m, d)
        for (b, qo), bid in zip(chk, bids, strict=True):
            Rf, Ro = Rot.from_matrix(d.xmat[bid].reshape(3, 3)), Rot.from_quat(qo[t, [4, 5, 6, 3]])
            err[b][0] = max(err[b][0], float(np.linalg.norm(d.xpos[bid] - qo[t, :3])))
            err[b][1] = max(err[b][1], float(np.degrees((Rf.inv() * Ro).magnitude())))
            if pts and b in pts:
                ds = surface_shift(pts[b], d.xpos[bid], Rf.as_matrix(), qo[t, :3], Ro.as_matrix())
                err[b][2] = max(err[b][2], float(ds))
    return err
