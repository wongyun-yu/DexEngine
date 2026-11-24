"""Geometry helpers of the E5 offline judge: quaternion → rotation, angles, hold windows, speeds (50 Hz rows)."""

from __future__ import annotations

import numpy as np

HZ = 50.0  # judge row rate (Hz): rollout states, the kinematic reference and reference.npz are all on this clock


def quat_R(q: np.ndarray) -> np.ndarray:
    """wxyz (…,4) → (…,3,3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def rot_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Rotation angle (degrees) between two rotation-matrix stacks."""
    tr = np.einsum("...ij,...ij->...", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def tilt_deg(Rs: np.ndarray) -> np.ndarray:
    """Angle (degrees) between world up and the body axis that pointed world-up at frame 0."""
    up0 = Rs[0].T @ np.array([0, 0, 1.0])
    return np.degrees(np.arccos(np.clip(np.einsum("tij,j->ti", Rs, up0)[:, 2], -1, 1)))


def longest(mask: np.ndarray, hz: float = HZ) -> float:
    """Longest run of True rows, in seconds."""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best / hz


def first_hold(mask: np.ndarray, hold_s: float, hz: float = HZ) -> int | None:
    """First row at which a run of True rows has lasted hold_s (success happens at that row); None if never.
    hold_s ≤ 0 gives the first True row."""
    need = max(1, int(round(hold_s * hz)))
    cur = 0
    for i, m in enumerate(mask):
        cur = cur + 1 if m else 0
        if cur >= need:
            return i
    return None


def speeds(p: np.ndarray, Rs: np.ndarray, hz: float = HZ) -> tuple[np.ndarray, np.ndarray]:
    """Per-row linear (m/s) and angular (rad/s) speed; row 0 is 0."""
    lin = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1) * hz]
    ang = np.r_[0.0, np.radians(rot_deg(Rs[1:], Rs[:-1])) * hz]
    return lin, ang
