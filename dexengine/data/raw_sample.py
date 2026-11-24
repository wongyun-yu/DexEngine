#!/usr/bin/env python3
"""Sampling rule from source tracked pose sequences onto our time grid; single owner (dependencies numpy and scipy
only).

Rule: all objects of a scene are sampled on **the same time grid** with **the same interpolation**: position = linear
interpolation · rotation = spherical linear interpolation (slerp) · times outside the source range take the end value.
The delivered objects are sampled that way (`drift_deg` measures it).
Why: loading each object's pose at a different time (one the interpolated value at that time, another the next source
frame) opens gaps between parts on frames where the whole object turns fast; a hinge exact on one axis in the source
leaves the hinge for a few frames in our data, and supports and extra objects are loaded at a different instant than the
body.
The version `RAW_SAMPLE` is stamped into the supplement files (the sample field of INTAKE/raw_objects/*.npz) and the
trees (objects[].supplement.sample). Readers (base_layer.hand_dir.supplements_for · oakink2.artic_fit.part_clusters)
fail with a reason when the version differs; old samples are rewritten with `oakink2/raw_objects.py --retransform` and
base is rebuilt. Marker = `[RAW_TIME] same_grid n=…`.
The stamp_reason / yaw_reason texts end in the base ledger and the exclusion ledger, so they stay as written.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

#: Version of the sampling rule: linear position · slerp rotation · one time grid for every object of the scene
RAW_SAMPLE = "same_grid-1"
MARK = "[RAW_TIME]"
#: Max angle (deg) the delivered object's rotation may differ from the rotation we sample from the same source. With the
#: same grid and the same interpolation it is 0 (measured: max 0.000° over 2,708 axis-hand base trees).
DRIFT_MAX_DEG = 0.5


def sample(T4: np.ndarray, t_raw: np.ndarray, t_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample a source 4x4 pose sequence (times t_raw, ascending) at times t_q → (position (T,3), rotation matrices
    (T,3,3))."""
    T4 = np.asarray(T4, float)
    t_raw = np.asarray(t_raw, float)
    if len(t_raw) < 2 or len(T4) != len(t_raw) or np.any(np.diff(t_raw) <= 0):
        raise ValueError(
            f"{MARK} cannot sample the source pose sequence — frames {len(T4)} · times {len(t_raw)} (at least "
            "two, ascending)"
        )
    tq = np.clip(np.asarray(t_q, float), t_raw[0], t_raw[-1])
    pos = np.stack([np.interp(tq, t_raw, T4[:, k, 3]) for k in range(3)], 1)
    return pos, Slerp(t_raw, Rot.from_matrix(T4[:, :3, :3]))(tq).as_matrix()


def drift_deg(R_ours: np.ndarray, R_ref: np.ndarray) -> np.ndarray:
    """Per-frame difference (deg) of two rotation sequences of one object (ours · reference). When both are poses at the
    same times the difference is a constant (mesh frame offset); returns the angle left after removing that constant
    (robust mean)."""
    d = Rot.from_matrix(np.asarray(R_ours)).inv() * Rot.from_matrix(np.asarray(R_ref))
    m = d.mean()
    for _ in range(3):
        a = np.degrees((d * m.inv()).magnitude())
        m = d.mean(weights=np.where(a < DRIFT_MAX_DEG, 1.0, DRIFT_MAX_DEG / np.maximum(a, 1e-9)))
    return np.degrees((d * m.inv()).magnitude())


def stamp_reason(stamp, what: str) -> str:
    """Whether a supplement's sampling version is the current rule: "" if so, else the reason to fail."""
    s = None if stamp is None else str(stamp)
    if s == RAW_SAMPLE:
        return ""
    return (
        f"{MARK} {what}: sample version {s!r} ≠ {RAW_SAMPLE!r} — pose from a different time than the body "
        "(old sample = rotation from the next raw frame). "
        "Rewrite with oakink2_raw_objects.py --retransform and rebuild base"
    )


def mark(n: int, where: str) -> str:
    """Marker saying n supplements sampled on the same grid were read; a run without this line did not check the
    version."""
    return f"{MARK} same_grid n={n} sample={RAW_SAMPLE} {where}"


# Scene rotation (yaw) repair of supplements; off by default.
#: Switch (value '1' = on · '0' or empty = off · anything else fails). Off = unchanged behaviour (supplement file
#: content and reads).
#: On = when writing a supplement, read the scene rotation from its owner (base_layer.hand_dir.yaw_for, reference hand
#: direction
#: dexengine.data.align.REF_DEG) instead of the tree record (fail when they differ), write the YAW_KEY field into the
#: file, and base rejects supplements without that field or with a rotation different from its own. On expiry, on
#: becomes the default and the switch is removed.
YAW_FIX_ENV = "OAKINK2_SUPP_YAW_FIX"
YAW_MARK = "[SUPP_YAW]"
YAW_KEY = "yaw_deg"
#: Tolerance (°) for treating the supplement's rotation as equal to the tree's; trees write rotation to four decimals
#: (base_adapter frame.yaw_deg).
YAW_TOL_DEG = 1e-3
_MARKED: set = set()


def yaw_fix_on() -> bool:
    """Read the switch; unknown values fail. When on, prints the marker once (a run without the line = an off run)."""
    v = os.environ.get(YAW_FIX_ENV, "")
    if v not in ("", "0", "1"):
        raise SystemExit(f"{YAW_MARK} {YAW_FIX_ENV}={v!r} — only '1' (on) · '0' · empty (off) are accepted")
    return v == "1"


def yaw_mark(where: str, yaw: float) -> None:
    """Marker of an on run, once per location (where)."""
    if where not in _MARKED:
        _MARKED.add(where)
        print(f"{YAW_MARK} on {where} yaw={yaw:.4f}° ({YAW_FIX_ENV}=1)", flush=True)


def yaw_reason(stamp, want: float, what: str) -> str:
    """Supplement file rotation field (stamp) ↔ rotation of the tree loading that supplement (want): '' when equal, else
    the reason to fail."""
    if stamp is None:
        return (
            f"{YAW_MARK} {what}: no rotation field ({YAW_KEY}) — an old supplement whose reference hand "
            "direction cannot be checked · "
            "rewrite with oakink2_raw_objects.py --retransform"
        )
    if abs(float(stamp) - float(want)) > YAW_TOL_DEG:
        return (
            f"{YAW_MARK} {what}: supplement rotation {float(stamp):.4f}° ≠ tree {float(want):.4f}° — the "
            "supplement sits at a different "
            "rotation than the delivered object · rewrite with --retransform"
        )
    return ""
