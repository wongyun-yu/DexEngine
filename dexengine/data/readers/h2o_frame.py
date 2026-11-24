#!/usr/bin/env python3
"""h2o source pose archive IO; **single owner** of the source↔delivered frame relation.

Settled facts (measured on 184 clips):
  · `cam4/{hand_pose,obj_pose_rt}` of the official pose archive are **coordinates in that camera's frame k** (cam4,
    egocentric), not world. `cam_pose` is cam→world (frame 0 ≈ identity).
  · Delivered mano/retarget trajectories X' are in a **fixed world frame**:  X'_k = G · C_k · X_cam_k  (C_k =
    cam_pose_k, G = fixed rigid transform per take); residual 1.5 mm (p95 1.8).
    Misreading "source = world" and fitting one rigid transform leaves residual p50 22.6 mm (max 106); that is head
        motion, not a delivery defect.
    Deciding oracle: velocity of objects with the hand ≥ 10 cm away (at rest): delivered frame 0.3–0.7 cm/s (still) vs
        camera-frame hypothesis 1.1–5.2 cm/s.
  · So there is **nothing to correct** in the delivered frame. The remaining issue is the tilt of the world frame itself
    (cam4 initial pose ≈ world, desk normal ≠ +z): `dexengine/data/support/h2o.py` (D10-e'').
Use of G: moves the depth-measured world plane into the delivered (capture) frame:  n' = G.R n_w,  c' = G c_w.
Inputs: data/h2o/raw/dl/pose<S>/<subject>/<scene>/<take>/cam4/{cam_pose,hand_pose,obj_pose_rt}/*.txt (official, 30 fps)
        mano layer <clip>/0/trajectory_keypoints.npz (delivered, 50 Hz; wrist 7 · fingertips 5×7 · object 7, quaternion
            wxyz); source frame k=3m and delivered frame j=5m coincide exactly.
Fit result: INTAKE/h2o_frame.json (per clip G(R,t), residual, head motion), `--fit`.
Exception messages raised while fitting or loading land in INTAKE/h2o_frame.json, h2o_table_plane.json, the support
audit or the exclusion ledger, so they stay as written.
Usage: $RETARGET_PY dexengine/data/readers/h2o_frame.py --fit [--clip <clip>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.core import paths as P  # noqa: E402
from dexengine.core.paths import (
    main_root as _data_root,  # noqa: E402 — owner of the data root (env DEXENGINE_MAIN_ROOT)
)

RAWPOSE = _data_root() / "data/h2o/raw/dl"
FIT_VERSION = "2026-09-15b"  # b: source = camera frame, delivered = G·C_k·source = fixed world
RAW_DT, DEL_DT = 1.0 / 30.0, 0.02
TIP_IDX = (
    4,
    8,
    12,
    16,
    20,
    # fingertips of the 21 MANO joints (thumb, index, middle, ring, little); same order as the delivered fingertips
    # (permutation (0,1,2,3,4) measured)
)
_FIT_CACHE: dict | None = None


def fit_path() -> Path:
    return P.layer_path("intake", "h2o") / "h2o_frame.json"


def take_key(clip: str) -> str:
    """Key of the per-take tables (fit, desk plane) = source take: tree name → clip (paths.clip_name) → placement
    variant → source clip (paths.variant_source) → segment's source take (paths.parent_clip). Unknown suffixes remain
    and fail the table lookup."""
    return P.parent_clip(P.variant_source(P.clip_name(clip)))


def take_of(clip: str) -> tuple[str, str, str]:
    """'subject1_h1__take0_cam4[_segNN][_pvNN]' → ('subject1', 'h1', '0')."""
    base = take_key(clip)
    subj, rest = base.split("_", 1)
    scene, tk = rest.split("__take")
    take = tk.split("_cam")[0]
    return subj, scene, take


def pose_dir(clip: str) -> Path:
    subj, scene, take = take_of(clip)
    d = RAWPOSE / f"pose{subj[-1]}" / subj / scene / take / "cam4"
    if not d.is_dir():
        raise FileNotFoundError(
            f"h2o raw pose missing: {d} (extract subject{subj[-1]}_pose_v1_1.tar.gz into data/h2o/raw/dl/pose"
            f"{subj[-1]})"
        )
    return d


def read_rows(d: Path) -> list[np.ndarray]:
    return [np.array(list(map(float, f.read_text().split()))) for f in sorted(d.glob("*.txt"))]


def raw_hands(pd: Path):
    """(K, 2, 21, 3) camera-frame hand joints (left, right) + (K, 2) validity flags."""
    rows = read_rows(pd / "hand_pose")
    H = np.full((len(rows), 2, 21, 3), np.nan)
    F = np.zeros((len(rows), 2), bool)
    for k, r in enumerate(rows):
        if r.shape[0] != 128:
            raise ValueError(f"hand_pose width {r.shape[0]} != 128: {pd}")
        F[k] = (r[0] > 0.5, r[64] > 0.5)
        H[k, 0] = r[1:64].reshape(21, 3)
        H[k, 1] = r[65:128].reshape(21, 3)
    return H, F


def raw_obj(pd: Path):
    """(K, 4, 4) camera-frame object pose, class id."""
    rows = read_rows(pd / "obj_pose_rt")
    return np.stack([r[1:17].reshape(4, 4) for r in rows]), (int(rows[0][0]) if rows else -1)


def cam_pose(pd: Path) -> np.ndarray:
    """(K, 4, 4) cam→world."""
    C = np.stack(read_rows(pd / "cam_pose")).reshape(-1, 4, 4)
    if not np.allclose(C[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"cam_pose last row is not [0 0 0 1]: {pd}")
    return C


def kabsch(A: np.ndarray, B: np.ndarray):
    """B ≈ R A + t (rotation only, no reflection). Returns R, t, rms (m), Umeyama scale diagnostic (should be 1)."""
    ca, cb = A.mean(0), B.mean(0)
    Hm = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(Hm)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    res = B - (A @ R.T + t)
    rms = float(np.sqrt((res**2).sum(1).mean()))
    return R, t, rms, float((S * np.diag(D)).sum() / ((A - ca) ** 2).sum())


def _delivered_obj(kp) -> np.ndarray | None:
    for s in ("right", "left"):
        o = kp.get(f"qpos_obj_{s}")
        if o is not None and np.abs(o[:, :3]).sum() > 0:
            return o
    return None


def fit_G(clip: str, kp) -> dict:
    """Fit G of delivered X'_k = G · C_k · X_cam_k from wrist, fingertip and object correspondences at exactly matching
    frames (source k=3m ↔ delivered j=5m). Also records, as an oracle, the residual of the 'source = world' misreading
    (rms_if_raw_were_world_mm); its size is the size of the head motion."""
    pd = pose_dir(clip)
    H, F = raw_hands(pd)
    M, _ = raw_obj(pd)
    C = cam_pose(pd)
    K = min(len(H), len(M), len(C))
    T = int(kp["qpos_wrist_right"].shape[0])
    ms = np.arange(0, min((K - 1) // 3, (T - 1) // 5) + 1)
    ks, js = 3 * ms, 5 * ms
    A, A0, B = [], [], []
    for si, s in enumerate(("left", "right")):
        w = kp[f"qpos_wrist_{s}"][:, :3]
        f = kp[f"qpos_finger_{s}"][:, :, :3]
        ok = F[ks, si] & (np.abs(w[js]).sum(1) > 0)
        kk = ks[ok]
        for q, ti in ((None, 0),) + tuple(enumerate(TIP_IDX)):
            a = H[kk, si, ti]
            A0.append(a)
            A.append(np.einsum("nij,nj->ni", C[kk, :3, :3], a) + C[kk, :3, 3])  # camera → world
            B.append(w[js[ok]] if q is None else f[js[ok], q])
    do = _delivered_obj(kp)
    if do is not None:
        A0.append(M[ks, :3, 3])
        A.append(np.einsum("nij,nj->ni", C[ks, :3, :3], M[ks, :3, 3]) + C[ks, :3, 3])
        B.append(do[js, :3])
    A = np.concatenate(A)
    A0 = np.concatenate(A0)
    B = np.concatenate(B)
    R, t, rms, scale = kabsch(A, B)
    _, _, rms0, _ = kabsch(A0, B)
    trav = np.linalg.norm(C[:K, :3, 3] - C[0, :3, 3], axis=1)
    rot = Rot.from_matrix(C[:K, :3, :3]) * Rot.from_matrix(C[0, :3, :3]).inv()
    return {
        "R": R.tolist(),
        "t": t.tolist(),
        "rms_mm": round(rms * 1000, 2),
        "rms_if_raw_were_world_mm": round(rms0 * 1000, 2),
        "scale": round(scale, 4),
        "n_pairs": int(len(A)),
        "n_raw_frames": int(K),
        "T_delivered": T,
        "cam_travel_cm_max": round(float(trav.max()) * 100, 2),
        "cam_rot_deg_max": round(float(np.degrees(rot.magnitude()).max()), 2),
        "version": FIT_VERSION,
    }


def load_fits() -> dict:
    global _FIT_CACHE
    if _FIT_CACHE is None:
        p = fit_path()
        if not p.is_file():
            raise FileNotFoundError(
                f"h2o frame fit missing: {p} — run `dexengine/data/readers/h2o_frame.py --fit` first"
            )
        _FIT_CACHE = json.loads(Path(p).read_text())
    return _FIT_CACHE


def G_of(clip: str) -> np.ndarray:
    """(4,4) world → delivered (capture) frame."""
    rec = load_fits()["clips"].get(take_key(clip))
    if rec is None:
        raise KeyError(f"clip not in the h2o frame fit: {clip}")
    if rec.get("version") != FIT_VERSION:
        raise ValueError(f"h2o frame fit version mismatch {rec.get('version')} != {FIT_VERSION}: {clip} — rerun --fit")
    G = np.eye(4)
    G[:3, :3] = np.array(rec["R"])
    G[:3, 3] = np.array(rec["t"])
    return G


# D10-e'': per-take desk plane rotation; rotates the capture frame so the desk normal → +z, and desk_z = plane height.
# dexengine/data/support/h2o.py writes the planes into INTAKE/h2o_table_plane.json (source depth oracle). Consumers:
# dexengine/data/support/plane.py
# (mano keypoints) · base_adapter.py (kin qpos + mano). Both use the same M, so desk height, the hand's lowest point and
# rest runs are defined in one frame. A missing plane or failed quality **fails** (no fallback).
TILT_VERSION = "2026-09-15a"
_PLANE_CACHE: dict | None = None


def plane_path() -> Path:
    return P.layer_path("intake", "h2o") / "h2o_table_plane.json"


def load_planes() -> dict:
    global _PLANE_CACHE
    if _PLANE_CACHE is None:
        p = plane_path()
        if not p.is_file():
            raise FileNotFoundError(
                f"h2o desk plane missing: {p} — run `dexengine/data/support/h2o.py` first (needs raw depth streaming)"
            )
        _PLANE_CACHE = json.loads(Path(p).read_text())
    return _PLANE_CACHE


def rot_to_z(n: np.ndarray) -> np.ndarray:
    """Minimal rotation sending unit normal n to +z (Rodrigues)."""
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c = float(np.dot(n, z))
    if s < 1e-12:
        return np.eye(3) if c > 0 else Rot.from_euler("x", 180, degrees=True).as_matrix()
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K * s + K @ K * (1 - c)


def tilt_transform(clip: str) -> dict:
    """D10-e'' rigid transform of a clip (segment and placement-variant suffixes ignored, via take_key).
    Returns {M(4,4), tilt_deg, desk_z, pivot, pivot_source, plane}."""
    base = take_key(clip)
    rec = load_planes()["clips"].get(base)
    if rec is None:
        raise KeyError(f"clip not in the h2o desk plane: {base} (depth not extracted?)")
    if rec.get("version") != TILT_VERSION:
        raise ValueError(
            f"h2o desk plane version mismatch {rec.get('version')} != {TILT_VERSION}: {base} — rerun "
            "dexengine/data/support/h2o.py"
        )
    if not rec.get("quality_ok"):
        raise ValueError(f"h2o desk plane below quality ({rec.get('quality_why')}): {base}")
    n = np.array(rec["n_capture"], float)
    p0 = np.array(rec["pivot_capture"], float)
    R = rot_to_z(n)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p0 - R @ p0
    return {
        "M": M,
        "tilt_deg": float(rec["tilt_deg"]),
        "desk_z": float(p0[2]),
        "pivot": p0.tolist(),
        "pivot_source": rec.get("pivot_source"),
        "plane": rec,
    }


def apply_points(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Transform (..., 3) points by the constant M(4,4)."""
    return X @ M[:3, :3].T + M[:3, 3]


def apply_pose7(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Transform a (T, 7) = [pos(3), quat wxyz(4)] pose sequence by the constant M (zero quaternions stay)."""
    Y = X.copy()
    Y[:, :3] = apply_points(X[:, :3], M)
    ok = np.linalg.norm(X[:, 3:7], axis=1) > 1e-6
    if ok.any():
        r = Rot.from_matrix(M[:3, :3]) * Rot.from_quat(X[ok][:, [4, 5, 6, 3]])
        Y[np.where(ok)[0][:, None], [[3, 4, 5, 6]]] = r.as_quat()[:, [3, 0, 1, 2]]
    return Y


def apply_rigid_keypoints(kp: dict, M: np.ndarray) -> dict:
    """Apply the constant M to a mano keypoint dict (qpos_wrist_*, qpos_finger_*, qpos_obj_*). *_wrist_frame keys
    (wrist-relative) and zero columns (absent hand or object) stay."""
    out = dict(kp)
    for s in ("right", "left"):
        w = kp.get(f"qpos_wrist_{s}")
        if w is not None and np.abs(w[:, :3]).sum() > 0:
            out[f"qpos_wrist_{s}"] = apply_pose7(w, M)
        f = kp.get(f"qpos_finger_{s}")
        if f is not None and np.abs(f[..., :3]).sum() > 0:
            f2 = f.copy()
            f2[..., :3] = apply_points(f[..., :3], M)
            out[f"qpos_finger_{s}"] = f2
        o = kp.get(f"qpos_obj_{s}")
        if o is not None and np.abs(o[:, :3]).sum() > 0:
            out[f"qpos_obj_{s}"] = apply_pose7(o, M)
    return out


def tilt_marker(clip: str, info: dict, who: str) -> str:
    return (
        f"[H2O_TILT] {who} clip={clip} tilt_deg={info['tilt_deg']:.1f} desk_z={info['desk_z']:+.4f} pivot="
        f"{np.round(info['pivot'], 3).tolist()} "
        f"pivot_source={info['pivot_source']} inlier_frac={info['plane'].get('inlier_frac')}"
    )


def _fit_into(out: dict, mano_root, c: str) -> None:
    """Fit one clip into ``out["clips"]``, or record why it could not be fitted in ``out["errors"]``."""
    try:
        out["clips"][c] = fit_G(c, np.load(mano_root / c / "0/trajectory_keypoints.npz"))
    except Exception as e:  # noqa: BLE001  a clip that cannot be fitted is recorded in errors; the batch continues
        out["errors"][c] = f"{type(e).__name__}: {str(e)[:160]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--clip", default="")
    a = ap.parse_args()
    if not a.fit:
        ap.print_help()
        return 2
    mano_root = P.layer_path("mano", "h2o", emb="bimanual")
    # names from outside go through the single entry
    clips = (
        P.read_clip_list([a.clip], dataset="h2o", quiet=True)
        if a.clip
        else sorted(p.name for p in mano_root.iterdir() if (p / "0/trajectory_keypoints.npz").is_file())
    )
    out = {
        "tool": "dexengine/data/readers/h2o_frame.py",
        "version": FIT_VERSION,
        "model": "X'_k = G · C_k · X_cam_k (delivered = fixed world; raw per-camera pose = camera frame)",
        "clips": {},
        "errors": {},
    }
    for c in clips:
        _fit_into(out, mano_root, c)
    p = P.clips_record(fit_path(), bool(a.clip))  # a single --clip run does not overwrite the dataset-wide fit record
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    ok = list(out["clips"].values())
    if ok:
        r = np.array([x["rms_mm"] for x in ok])
        r0 = np.array([x["rms_if_raw_were_world_mm"] for x in ok])
        tr = np.array([x["cam_travel_cm_max"] for x in ok])
        print(
            f"[h2o_frame] fit {len(ok)}/{len(clips)} (errors {len(out['errors'])}) · rms_mm p50={np.median(r):.2f} "
            f"p95={np.percentile(r, 95):.2f} max={r.max():.2f} · "
            f"'source=world' misreading rms_mm p50={np.median(r0):.1f} max={r0.max():.1f} · head travel cm "
            f"max p50={np.median(tr):.1f} max={tr.max():.1f} → {p}"
        )
    for c, e in list(out["errors"].items())[:5]:
        print("  ERR", c, e)
    return 0 if not out["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
