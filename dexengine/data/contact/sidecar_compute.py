"""Contact sidecar computation: per-side fingertip/region contact against every object of a base tree
(compute_side) and the per-object stacking (stack_objects); driven by contact/sidecar.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dexengine.data.contact import common as CC
from dexengine.data.contact import (
    fingerprint as SF,  # owner of input fingerprints, code version and palm coefficients
)

VERSION = SF.CODE_VERSION  # SF owns the version; consumers (bake, status) compare it without heavy imports


FINGERS = np.array(CC.FINGERS)


REGIONS = np.array(["fingertip"])


def compute_side(
    ti: dict, kin: np.ndarray, mano, side: str, obj: dict, surf: Path, _alpha: float | None = None
) -> dict:
    from dexengine.data.contact import geom as CG

    T = len(kin)
    sg = ti["segment"]
    off = int(sg["frame_map"].get("offset", 0))
    a = int(sg["start"])
    psi = float((ti.get("frame") or {}).get("yaw_deg", 0.0))
    M = Rot.from_euler("z", psi, degrees=True).as_matrix()  # base = Rz(yaw)·capture; MANO gets the same rotation
    Tm = len(mano[f"qpos_finger_{side}"])
    idx = np.clip(np.arange(T) + a + off, 0, Tm - 1)
    tips = mano[f"qpos_finger_{side}"][idx, :, :3].astype(np.float64) @ M.T
    wrist = mano[f"qpos_wrist_{side}"][idx, :3].astype(np.float64) @ M.T
    qs = obj["qpos_start"]
    o7 = kin[:, qs : qs + 7]
    art = obj.get("articulation")
    hinge = kin[:, art["qpos_index"]] if art else None
    oside = obj["name"].split("_")[0]
    mism = None
    if f"qpos_obj_{oside}" in mano:
        mism = float(np.median(np.linalg.norm(mano[f"qpos_obj_{oside}"][idx, :3] @ M.T - o7[:, :3], axis=1)) * 1000)
    dist, wit_w, n_pts = CG.tip_distances(tips, o7, surf, art, hinge)
    # palm reference point: the wrist keypoint by default (bytes equal to old outputs); all-objects uses the measured α
    # proxy.
    _p = wrist if _alpha is None else wrist + _alpha * (tips[:, :, :3].mean(1) - wrist)
    palm_d, palm_w, _ = CG.point_distances(_p, o7, surf, art, hinge)
    dist = dist.astype(np.float32)[:, :, None]
    contact_bin = dist[..., None] < CC.THRESHOLDS_M[None, None, None, :]
    contact = CG.contact_from_dist(dist[:, :, 0])
    payload = {
        "frames": np.arange(T, dtype=np.int32),
        "fingers": FINGERS,
        "regions": REGIONS,
        "dist": dist,
        "penetration": np.zeros_like(contact),
        "effective_dist": dist,
        "witness_hand": tips.astype(np.float32)[:, :, None, :],
        "witness_obj": wit_w.astype(np.float32)[:, :, None, :],
        "palm_dist": palm_d.astype(np.float32),
        "palm_penetration": np.zeros(T, bool),
        "palm_effective_dist": palm_d.astype(np.float32),
        "palm_witness_hand": wrist.astype(np.float32),
        "palm_witness_obj": palm_w.astype(np.float32),
        "contact_bin": contact_bin,
        "thresholds_m": CC.THRESHOLDS_M,
        "contact": contact,
        "contact_threshold_m": np.float32(CG.KEYPOINT_THRESHOLD_M),
        "smooth_max_gap": np.int32(CC.SMOOTH_MAX_GAP),
        "smooth_min_on": np.int32(CC.SMOOTH_MIN_ON),
        "embodiment": "mano_keypoints",
        "side": side,
        "tool_mesh_id": obj["obj_id"],
        "n_obj_surface_pts": np.int64(n_pts),
        "frame": "base(yaw)",
        "yaw_deg": np.float32(psi),
        "units": "meters",
        "fps": np.float32(ti["segment"]["fps"]),
        "penetration_available": False,
        "palm_source": ("mano_wrist_keypoint" if _alpha is None else f"mano_palm_proxy(alpha={_alpha})"),
        "palm_alpha": np.float32(np.nan if _alpha is None else _alpha),
        "articulated": bool(art),
        "mano_obj_mismatch_mm": np.float32(np.nan if mism is None else mism),
        "source": "dexengine/data/contact/sidecar.py",
        "version": VERSION,
        "parent_clip": ti["parent_clip"],
        "segment_start": np.int32(a),
        "frame_offset": np.int32(off),
    }
    return payload


def stack_objects(pls: list, objs: list) -> dict:
    """Sidecar with an object axis (D-MANO).

    Why separate: `contact_mano_base` is attached only to the **one object** with `role == "main"`, and D23 decides that
    role from the case; using that sidecar as the case input would be circular. Measuring every object **before** roles
    are decided breaks the loop and makes cases and roles independent of the robot hand.
    """
    return {
        "frames": pls[0]["frames"],
        "fingers": FINGERS,
        "regions": REGIONS,
        "obj_names": np.array([str(o["name"]) for o in objs]),
        "obj_ids": np.array([str(o["obj_id"]) for o in objs]),
        # the role at build time is **only recorded**; a reader depending on it would bring the cycle back.
        "obj_role_at_build": np.array([str(o.get("role", "")) for o in objs]),
        "dist": np.stack([p["dist"][:, :, 0] for p in pls], 1).astype(np.float32),  # (T, n_obj, 5 fingers)
        "palm_dist": np.stack([p["palm_dist"] for p in pls], 1).astype(np.float32),  # (T, n_obj)
        # Co-motion checks (the case asks "did hand and object move together") need the nearest point on the object
        # side.
        # Without it, case classification cannot run that check on the MANO source and silently gives another answer.
        "witness_obj": np.stack([p["witness_obj"][:, :, 0, :] for p in pls], 1).astype(np.float32),  # (T, n_obj, 5, 3)
        "contact": np.stack(
            [np.squeeze(p["contact"], -1) if p["contact"].ndim == 3 else p["contact"] for p in pls], 1
        ),  # (T, n_obj, 5 fingers)
        "contact_bin": np.stack([p["contact_bin"][:, :, 0, :] for p in pls], 1),  # (T, n_obj, 5, 6 thresholds)
        "thresholds_m": CC.THRESHOLDS_M,
        "contact_threshold_m": pls[0]["contact_threshold_m"],
        "smooth_max_gap": np.int32(CC.SMOOTH_MAX_GAP),
        "smooth_min_on": np.int32(CC.SMOOTH_MIN_ON),
        "embodiment": "mano_keypoints",
        "side": pls[0]["side"],
        "palm_source": pls[0]["palm_source"],
        "palm_alpha": pls[0]["palm_alpha"],
        "penetration_available": False,
        "frame": pls[0]["frame"],
        "yaw_deg": pls[0]["yaw_deg"],
        "units": "meters",
        "fps": pls[0]["fps"],
        "articulated": np.array([bool(p["articulated"]) for p in pls]),
        "mano_obj_mismatch_mm": np.array([float(p["mano_obj_mismatch_mm"]) for p in pls], dtype=np.float32),
        "source": "dexengine/data/contact/sidecar.py --all-objects",
        "version": VERSION,
        "parent_clip": pls[0]["parent_clip"],
        "segment_start": pls[0]["segment_start"],
        "frame_offset": pls[0]["frame_offset"],
    }
