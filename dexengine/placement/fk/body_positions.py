#!/usr/bin/env python3
"""Pre-compute body world positions of a tree from its qpos trajectory using MuJoCo FK.

Retargeted trajectories store only joint angles (qpos); the observations need body world positions. This runs MuJoCo
forward kinematics on each frame and writes <sequence_dir>/0/body_positions.npz.

Usage:
    python -m dexengine.placement.fk.body_positions <sequence_dir> [-q]
"""

import argparse
import os
import sys
import time

import mujoco
import numpy as np

# Prefixes to exclude from hand body discovery (IK auxiliary bodies)
EXCLUDE_PREFIXES = ("ref_", "trace_", "track_", "mocap_")

# Site names to extract (fingertips and palm)
RIGHT_SITE_NAMES = [
    "right_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_ring_tip",
    "right_pinky_tip",
    "right_palm",
]
LEFT_SITE_NAMES = [
    "left_thumb_tip",
    "left_index_tip",
    "left_middle_tip",
    "left_ring_tip",
    "left_pinky_tip",
    "left_palm",
]

FPS = 50.0


def _under_object_body(model: mujoco.MjModel, i: int) -> bool:
    """True when body i or one of its ancestors has a name ending in `_object` (part of an object)."""
    while i > 0:
        if model.body(i).name.endswith("_object"):
            return True
        i = int(model.body_parentid[i])
    return False


def discover_hand_bodies(model: mujoco.MjModel, prefix: str) -> list[tuple[int, str]]:
    """Discover hand body indices and names by prefix, excluding IK auxiliaries.

    Args:
        model: MuJoCo model.
        prefix: "right_" or "left_".

    Returns:
        List of (body_index, body_name) sorted by index.
    """
    bodies = []
    for i in range(model.nbody):
        name = model.body(i).name
        if not name.startswith(prefix):
            continue
        if any(name.startswith(ep) for ep in EXCLUDE_PREFIXES):
            continue
        # Also skip object bodies (e.g. "right_object", "left_object") and their descendants: a part body of an
        # articulated object such as `right_object_top` has the `right_` prefix but is not a hand body.
        if _under_object_body(model, i):
            continue
        bodies.append((i, name))
    return sorted(bodies, key=lambda x: x[0])


def discover_sites(model: mujoco.MjModel, site_names: list[str]) -> list[tuple[int, str]]:
    """Find site indices for requested site names. Skip missing sites gracefully.

    Returns:
        List of (site_index, site_name) for sites that exist in the model.
    """
    # Build lookup of all site names
    all_sites = {}
    for i in range(model.nsite):
        all_sites[model.site(i).name] = i

    found = []
    for name in site_names:
        if name in all_sites:
            found.append((all_sites[name], name))
        else:
            print(f"  [WARN] Site '{name}' not found in model, skipping.")
    return found


def compute_velocities(positions: np.ndarray, fps: float) -> np.ndarray:
    """Compute velocities via finite differences.

    Args:
        positions: (T, N, 3) position array.
        fps: Sampling rate in Hz.

    Returns:
        (T, N, 3) velocity array.
    """
    T = positions.shape[0]
    vel = np.zeros_like(positions)
    if T < 2:
        return vel
    vel[:-1] = (positions[1:] - positions[:-1]) * fps
    vel[-1] = vel[-2]
    return vel


def compute_angular_velocities(quats: np.ndarray, fps: float) -> np.ndarray:
    """Compute angular velocities from quaternion time series via finite differences.

    Uses the relative rotation between consecutive frames: dq = q(t+1) * conj(q(t)),
    then converts to axis-angle and scales by fps to get world-frame angular velocity.

    Args:
        quats: (T, N, 4) quaternion array in wxyz format (MuJoCo convention).
        fps: Sampling rate in Hz.

    Returns:
        (T, N, 3) angular velocity array in world frame.
    """
    T, N, _ = quats.shape
    ang_vel = np.zeros((T, N, 3), dtype=np.float32)
    if T < 2:
        return ang_vel

    q0 = quats[:-1]  # (T-1, N, 4) wxyz
    q1 = quats[1:]  # (T-1, N, 4) wxyz

    # conj(q0): negate xyz, keep w
    q0_conj = q0.copy()
    q0_conj[..., 1:] *= -1

    # dq = q1 * conj(q0) — Hamilton product in wxyz format
    w0, x0, y0, z0 = q0_conj[..., 0], q0_conj[..., 1], q0_conj[..., 2], q0_conj[..., 3]
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]

    dq_w = w1 * w0 - x1 * x0 - y1 * y0 - z1 * z0
    dq_x = w1 * x0 + x1 * w0 + y1 * z0 - z1 * y0
    dq_y = w1 * y0 - x1 * z0 + y1 * w0 + z1 * x0
    dq_z = w1 * z0 + x1 * y0 - y1 * x0 + z1 * w0

    # Ensure shortest path (w >= 0)
    sign = np.sign(dq_w)
    sign[sign == 0] = 1.0
    dq_w *= sign
    dq_x *= sign
    dq_y *= sign
    dq_z *= sign

    # axis-angle: angle = 2*arccos(w), axis = xyz/sin(angle/2)
    dq_w_clamped = np.clip(dq_w, -1.0, 1.0)
    half_angle = np.arccos(dq_w_clamped)  # (T-1, N)
    angle = 2.0 * half_angle

    sin_half = np.sin(half_angle)
    safe_sin = np.where(sin_half > 1e-8, sin_half, 1.0)

    axis_x = dq_x / safe_sin
    axis_y = dq_y / safe_sin
    axis_z = dq_z / safe_sin

    # For near-zero rotations, angular velocity ≈ 0
    mask = (sin_half > 1e-8).astype(np.float32)

    ang_vel[:-1, :, 0] = axis_x * angle * fps * mask
    ang_vel[:-1, :, 1] = axis_y * angle * fps * mask
    ang_vel[:-1, :, 2] = axis_z * angle * fps * mask
    ang_vel[-1] = ang_vel[-2]  # repeat last frame

    return ang_vel


_SITE_NAMES = {"right": RIGHT_SITE_NAMES, "left": LEFT_SITE_NAMES}


def _trajectory_file(traj_dir: str) -> str | None:
    """The kinematic trajectory (ground truth from MANO retargeting), else the physics roll-out (may drift)."""
    traj_kin = os.path.join(traj_dir, "trajectory_kinematic.npz")
    traj_ik = os.path.join(traj_dir, "trajectory_ikrollout.npz")
    if os.path.isfile(traj_kin):
        return traj_kin
    if os.path.isfile(traj_ik):
        print("  WARNING: No trajectory_kinematic.npz, falling back to trajectory_ikrollout.npz")
        return traj_ik
    print(f"[ERROR] No trajectory file found in {traj_dir}")
    return None


def _load(sequence_dir: str):
    """(trajectory folder, model, data, qpos of every frame) of a sequence, or None after printing why."""
    seq_name = os.path.basename(sequence_dir)
    scene_xml = os.path.join(sequence_dir, "scene.xml")
    traj_dir = os.path.join(sequence_dir, "0")
    traj_file = _trajectory_file(traj_dir)
    if traj_file is None:
        return None
    if not os.path.isfile(scene_xml):
        print(f"[ERROR] scene.xml not found: {scene_xml}")
        return None
    print(f"Processing: {seq_name}")
    try:
        model = mujoco.MjModel.from_xml_path(scene_xml)
    except Exception as e:  # noqa: BLE001 — a scene that does not load fails this sequence only
        print(f"[ERROR] Failed to load scene.xml: {e}")
        return None
    data = mujoco.MjData(model)
    qpos_all = np.load(traj_file)["qpos"]
    T, nq = qpos_all.shape
    print(f"  Trajectory: {T} frames, nq={nq} (model nq={model.nq})")
    if nq != model.nq:
        print(f"[ERROR] qpos dimension mismatch: traj has {nq}, model expects {model.nq}")
        return None
    return traj_dir, model, data, qpos_all


def _discover(model: mujoco.MjModel, verbose: bool) -> tuple[dict, dict]:
    """Hand bodies and fingertip/palm sites per side: {side: [(index, name), ...]} each."""
    bodies = {side: discover_hand_bodies(model, f"{side}_") for side in ("right", "left")}
    if verbose:
        for side, label in (("right", "Right"), ("left", "Left")):
            print(f"  {label} hand bodies ({len(bodies[side])}):")
            for idx, name in bodies[side]:
                print(f"    [{idx}] {name}")
    sites = {side: discover_sites(model, _SITE_NAMES[side]) for side in ("right", "left")}
    if verbose:
        for side, label in (("right", "Right"), ("left", "Left")):
            print(f"  {label} sites ({len(sites[side])}): {[s[1] for s in sites[side]]}")
    return bodies, sites


def _forward_kinematics(model, data, qpos_all, bodies: dict, sites: dict, verbose: bool) -> dict:
    """Body positions, body quaternions (wxyz) and site positions of every frame, per side (float32)."""
    T = len(qpos_all)
    out = {}
    for side in ("right", "left"):
        out[side] = {
            "pos": np.zeros((T, len(bodies[side]), 3), dtype=np.float32),
            "quat": np.zeros((T, len(bodies[side]), 4), dtype=np.float32),
            "site": np.zeros((T, len(sites[side]), 3), dtype=np.float32),
        }
    t0 = time.time()
    for t in range(T):
        data.qpos[:] = qpos_all[t]
        mujoco.mj_forward(model, data)
        for side in ("right", "left"):
            for j, (bidx, _name) in enumerate(bodies[side]):
                out[side]["pos"][t, j] = data.xpos[bidx]
                out[side]["quat"][t, j] = data.xquat[bidx]
        for side in ("right", "left"):
            for j, (sidx, _name) in enumerate(sites[side]):
                out[side]["site"][t, j] = data.site_xpos[sidx]
        if verbose and (t + 1) % 50 == 0:
            elapsed = time.time() - t0
            fps_actual = (t + 1) / elapsed
            print(f"  Frame {t + 1}/{T} ({fps_actual:.0f} frames/sec)")
    elapsed = time.time() - t0
    print(f"  FK complete: {T} frames in {elapsed:.2f}s ({T / elapsed:.0f} frames/sec)")
    for side in ("right", "left"):
        out[side]["vel"] = compute_velocities(out[side]["pos"], FPS)
    for side in ("right", "left"):
        out[side]["ang_vel"] = compute_angular_velocities(out[side]["quat"], FPS)
    return out


def _arrays(fk: dict, bodies: dict, sites: dict) -> dict:
    """The body_positions.npz members, in their file order."""
    names = {side: np.array([b[1] for b in bodies[side]]) for side in ("right", "left")}
    tips = {side: np.array([s[1] for s in sites[side]]) for side in ("right", "left")}
    arr = {}
    for side in ("right", "left"):
        arr[f"{side}_hand_body_pos"] = fk[side]["pos"]
        arr[f"{side}_hand_body_names"] = names[side]
        arr[f"{side}_hand_body_quat"] = fk[side]["quat"]
    for side in ("right", "left"):
        arr[f"{side}_fingertip_pos"] = fk[side]["site"]
        arr[f"{side}_fingertip_names"] = tips[side]
    for side in ("right", "left"):
        arr[f"{side}_hand_body_vel"] = fk[side]["vel"]
    for side in ("right", "left"):
        arr[f"{side}_hand_body_ang_vel"] = fk[side]["ang_vel"]
    arr["fps"] = np.float32(FPS)
    return arr


def _print_verification(arr: dict) -> None:
    """--verbose: array shapes and frame-0 samples of the right hand."""
    print("\n  === Verification ===")
    print("  Output arrays:")
    for key in (
        "right_hand_body_pos",
        "left_hand_body_pos",
        "right_hand_body_quat",
        "left_hand_body_quat",
        "right_fingertip_pos",
        "left_fingertip_pos",
        "right_hand_body_vel",
        "left_hand_body_vel",
        "right_hand_body_ang_vel",
        "left_hand_body_ang_vel",
    ):
        print(f"    {key + ':':<25}{arr[key].shape} {arr[key].dtype}")
    names = list(arr["right_hand_body_names"])
    pos, vel = arr["right_hand_body_pos"], arr["right_hand_body_vel"]
    for title, values, word in (("\n  Frame 0 samples:", pos, "pos"), ("\n  Frame 0 velocities:", vel, "vel")):
        print(title)
        if pos.shape[1] > 0:
            print(f"    right_hand_base {word}: {values[0, 0]}")
        if "right_thumb_distal" in names:
            print(f"    right_thumb_distal {word}: {values[0, names.index('right_thumb_distal')]}")
    print("\n  Frame 0 fingertip sites:")
    for j, name in enumerate(arr["right_fingertip_names"]):
        print(f"    {name}: {arr['right_fingertip_pos'][0, j]}")


def process_sequence(sequence_dir: str, verbose: bool = True) -> bool:
    """Process a single sequence directory.

    Args:
        sequence_dir: Path to the sequence directory containing scene.xml and 0/.
        verbose: Print detailed progress information.

    Returns:
        True if successful, False otherwise.
    """
    loaded = _load(os.path.abspath(sequence_dir))
    if loaded is None:
        return False
    traj_dir, model, data, qpos_all = loaded
    bodies, sites = _discover(model, verbose)
    fk = _forward_kinematics(model, data, qpos_all, bodies, sites, verbose)
    arr = _arrays(fk, bodies, sites)
    output_path = os.path.join(traj_dir, "body_positions.npz")
    np.savez(output_path, **arr)
    print(f"  Saved: {output_path}")
    if verbose:
        _print_verification(arr)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute body world positions from a tree's qpos trajectory using MuJoCo FK."
    )
    parser.add_argument(
        "sequence_dir",
        help="Path to a sequence directory (contains scene.xml and 0/). Required — there is no default tree.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress detailed per-frame output.",
    )
    args = parser.parse_args()

    seq_dir = os.path.abspath(args.sequence_dir)
    if not os.path.isdir(seq_dir):
        print(f"[ERROR] Sequence directory not found: {seq_dir}")
        sys.exit(1)

    if not process_sequence(seq_dir, verbose=not args.quiet):
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
