"""Custom event functions for Taco training reset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from dexengine.rl.tasks import env_flags

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from dexengine.rl.tasks.mdp.events.reset_arm import reset_arm_to_trajectory  # noqa: F401

# Root orientation: y_180_x_m90 — matches the replay pipeline (Rule 13).
# This compensates for the MJCF body quaternion lost during USD conversion.
# With this root, PhysX body_link_pos_w directly equals MuJoCo FK positions
# (no pre-rotation needed on reference body positions).
_ROOT_QUAT_WXYZ = [0.0, 0.0, 0.7071068, -0.7071068]


def reset_hands_to_trajectory(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    right_hand_cfg: SceneEntityCfg = SceneEntityCfg("right_hand"),  # noqa: B008 — Isaac Lab term signature: the default names the scene entity
    left_hand_cfg: SceneEntityCfg = SceneEntityCfg("left_hand"),  # noqa: B008 — Isaac Lab term signature: the default names the scene entity
) -> None:
    """Reset hands to trajectory reference pose at RSI start frame.

    Sets joint positions, joint velocities (zero), root pose (pinned at env origin
    with correct rotation), and joint position targets.
    """
    # Get reference state at RSI start frame for the reset envs
    start_frames = env.trajectory_manager.start_frames[env_ids]
    ref = env.trajectory_manager.get_reference_state_for_envs(env_ids, start_frames)

    right_hand = env.scene[right_hand_cfg.name]
    left_hand = env.scene[left_hand_cfg.name]

    # Set joint states (position + zero velocity)
    rh_qpos = ref["right_hand_qpos"]  # (N, 18) or (N, 12) PhysX order
    lh_qpos = ref["left_hand_qpos"]  # (N, 18) or (N, 12) PhysX order
    zero_vel_r = torch.zeros_like(rh_qpos)
    zero_vel_l = torch.zeros_like(lh_qpos)

    # For full robot (finger-only mode), specify joint_ids to write only hand joints.
    # For floating hand (18 joints), joint_ids=None writes all joints.
    traj_mgr = env.trajectory_manager
    if hasattr(traj_mgr, "_fingers_only") and traj_mgr._fingers_only:
        rh_joint_ids = traj_mgr._right_hand_ids.tolist()
        lh_joint_ids = traj_mgr._left_hand_ids.tolist()
    else:
        rh_joint_ids = None
        lh_joint_ids = None

    # LEFT-UNI: DEXENGINE_RIGHT_PARK=1 — park the RIGHT hand fingers at
    # their DEFAULT pose instead of the trajectory reference (mirror of how the uni
    # right-only scene leaves the left hand at default). Full-robot mode only.
    if env_flags.get("DEXENGINE_RIGHT_PARK") == "1" and rh_joint_ids is not None:
        rh_qpos = right_hand.data.default_joint_pos[env_ids][:, rh_joint_ids]

    right_hand.write_joint_state_to_sim(rh_qpos, zero_vel_r, rh_joint_ids, env_ids)
    left_hand.write_joint_state_to_sim(lh_qpos, zero_vel_l, lh_joint_ids, env_ids)

    # Set joint position targets so PD controller matches
    right_hand.set_joint_position_target(rh_qpos, joint_ids=rh_joint_ids, env_ids=env_ids)
    left_hand.set_joint_position_target(lh_qpos, joint_ids=lh_joint_ids, env_ids=env_ids)

    # Pin root bodies at env origins with correct rotation
    # Use rotated root if trajectory manager has +X forward rotation enabled
    if hasattr(env, "trajectory_manager") and env.trajectory_manager.root_quat_rotated is not None:
        root_quat = env.trajectory_manager.root_quat_rotated.unsqueeze(0)
        # Add desk height Z offset to env origins
        origins = env.scene.env_origins[env_ids].clone()
        origins[:, 2] += env.trajectory_manager._desk_height
    elif hasattr(env, "trajectory_manager") and hasattr(env.trajectory_manager, "_base_root_quat"):
        # Non-rotated case: use base root quat (identity for URDF, y_180_x_m90 for MJCF)
        root_quat = env.trajectory_manager._base_root_quat.unsqueeze(0)
        origins = env.scene.env_origins[env_ids]
    else:
        root_quat = torch.tensor([_ROOT_QUAT_WXYZ], device=env.device, dtype=torch.float32)
        origins = env.scene.env_origins[env_ids]
    root_pose = torch.cat(
        [
            origins,
            root_quat.expand(len(env_ids), -1),
        ],
        dim=-1,
    )  # (N, 7)

    # Pin root bodies — skip for full robot (finger-only mode) where right_hand
    # is aliased to the full robot articulation and pinning would teleport it.
    if not (hasattr(traj_mgr, "_fingers_only") and traj_mgr._fingers_only):
        zero_root_vel = torch.zeros(len(env_ids), 6, device=env.device)
        right_hand.write_root_pose_to_sim(root_pose, env_ids)
        right_hand.write_root_velocity_to_sim(zero_root_vel, env_ids)
        left_hand.write_root_pose_to_sim(root_pose, env_ids)
        left_hand.write_root_velocity_to_sim(zero_root_vel, env_ids)


def randomize_initial_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    right_hand_cfg: SceneEntityCfg = SceneEntityCfg("right_hand"),  # noqa: B008 — Isaac Lab term signature: the default names the scene entity
    randomize_fraction: float = 0.0,
    finger_open_deg: float = 0.0,
    wrist_z_max_cm: float = 0.0,
) -> None:
    """Randomly perturb the initial hand pose for robustness.

    Applied AFTER reset_hands_to_trajectory. Per-episode sampling: each env
    is randomized with probability ``randomize_fraction`` (like RSI).

    Two independent perturbation channels:
    - Fingers: open by up to ``finger_open_deg`` degrees (opening-only)
    - Wrist Z: lift by up to ``wrist_z_max_cm`` cm (upward-only, +Z)

    Args:
        randomize_fraction: Probability [0, 1] that an episode gets randomized.
            0.0 = disabled (no-op). 0.5 = 50% of episodes randomized.
        finger_open_deg: Maximum additional finger opening in degrees.
            Each joint is independently sampled uniformly from
            [0, finger_open_deg] degrees. Default 0 = disabled.
        wrist_z_max_cm: Maximum upward wrist displacement in centimeters.
            Sampled uniformly from [0, wrist_z_max_cm] cm, applied to
            the wrist pos_z joint (+Z only). Default 0 = disabled.
    """
    has_fingers = finger_open_deg > 0.0
    has_wrist_z = wrist_z_max_cm > 0.0
    if randomize_fraction <= 0.0 or (not has_fingers and not has_wrist_z):
        return None  # no-op when disabled

    right_hand = env.scene[right_hand_cfg.name]
    n = len(env_ids)
    device = env.device

    return _randomize_episode_subset(
        env,
        env_ids,
        randomize_fraction,
        finger_open_deg,
        wrist_z_max_cm,
        device,
        has_fingers,
        has_wrist_z,
        n,
        right_hand,
    )


def _randomize_episode_subset(
    env, env_ids, randomize_fraction, finger_open_deg, wrist_z_max_cm, device, has_fingers, has_wrist_z, n, right_hand
):
    """Per-episode coin flip: only randomize a fraction of episodes."""
    mask = torch.rand(n, device=device) < randomize_fraction
    if not mask.any():
        return  # no episodes selected this reset

    randomized_env_ids = env_ids[mask]
    n_rand = randomized_env_ids.shape[0]

    # Read current joint positions (set by reset_hands_to_trajectory)
    joint_pos = right_hand.data.joint_pos[randomized_env_ids].clone()

    # Determine finger joint indices and wrist joint index
    traj_mgr = env.trajectory_manager
    if hasattr(traj_mgr, "_fingers_only") and traj_mgr._fingers_only:
        finger_ids = traj_mgr._right_hand_ids
        wrist_z_idx = None  # full robot — wrist controlled by arm, not virtual joints
    else:
        n_joints = joint_pos.shape[1]
        finger_ids = list(range(6, n_joints))
        wrist_z_idx = 2  # joint index 2 = pos_z (prismatic, meters)

    # --- Finger randomization (opening-only) ---
    if has_fingers and finger_ids is not None and len(finger_ids) > 0:
        import math

        max_opening_rad = finger_open_deg * math.pi / 180.0
        fids = torch.tensor(finger_ids, device=device) if not isinstance(finger_ids, torch.Tensor) else finger_ids

        lower = right_hand.data.joint_pos_limits[0, fids, 0]
        current_finger_pos = joint_pos[:, fids]
        headroom = (current_finger_pos - lower).clamp(min=0.0)

        raw_opening = torch.rand(n_rand, len(fids), device=device) * max_opening_rad
        opening = torch.min(raw_opening, headroom)
        joint_pos[:, fids] -= opening

    # --- Wrist Z randomization (upward-only, +Z) ---
    if has_wrist_z and wrist_z_idx is not None:
        max_z_m = wrist_z_max_cm * 0.01  # cm → meters
        z_displacement = torch.rand(n_rand, device=device) * max_z_m  # [0, max_z_m]
        joint_pos[:, wrist_z_idx] += z_displacement

    # Write perturbed state only for randomized envs
    zero_vel = torch.zeros_like(joint_pos)
    if hasattr(traj_mgr, "_fingers_only") and traj_mgr._fingers_only:
        rh_ids = traj_mgr._right_hand_ids.tolist()
        right_hand.write_joint_state_to_sim(joint_pos[:, rh_ids], zero_vel[:, rh_ids], rh_ids, randomized_env_ids)
        right_hand.set_joint_position_target(joint_pos[:, rh_ids], joint_ids=rh_ids, env_ids=randomized_env_ids)
    else:
        right_hand.write_joint_state_to_sim(joint_pos, zero_vel, env_ids=randomized_env_ids)
        right_hand.set_joint_position_target(joint_pos, env_ids=randomized_env_ids)


def reset_object_to_trajectory(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_right"),  # noqa: B008 — Isaac Lab term signature: the default names the scene entity
    side: str = "right",
) -> None:
    """Reset object to trajectory reference pose at RSI start frame.

    Teleports the object to the primary object position from the trajectory.

    Args:
        side: Which hand's object to use ("right" or "left").
    """
    start_frames = env.trajectory_manager.start_frames[env_ids]
    ref = env.trajectory_manager.get_reference_state_for_envs(env_ids, start_frames)

    obj = env.scene[object_cfg.name]

    # Object pose: position (relative to env origin) + quaternion — one owner of the pinned pose (a reference
    # pose that digs into a static support is lifted by the penetration; the same function as SPAWN_HOLD, the judge
    # park and the PSS snap)
    from dexengine.rl.tasks.scene.object_pin import pin_pose as _pin_pose

    _pp, obj_quat, _, _ = _pin_pose(env, env_ids, start_frames, side, obj, ref)
    obj_pos = _pp + env.scene.env_origins[env_ids]

    # Combine into root state: (N, 13) = pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
    root_state = torch.zeros(len(env_ids), 13, device=env.device)
    root_state[:, :3] = obj_pos
    root_state[:, 3:7] = obj_quat
    # velocities stay zero
    # DEXENGINE_OBJ_REF_VEL_INIT=1: an object placed at **zero velocity** on a middle frame immediately disagrees with
    # a reference that is "being carried" (the poison that killed RSI mid-starts; the DEC snap takes the same path).
    # The initial velocity comes from the reference finite difference. Default OFF = byte-identical.
    if env_flags.get("DEXENGINE_OBJ_REF_VEL_INIT") == "1":
        _tm_v = env.trajectory_manager
        # The same index guard as above: no out-of-range index when a subset is loaded.
        _ci_v = _tm_v.clip_ids[env_ids].clamp(max=int(_tm_v.clip_lengths.numel()) - 1)
        _f2 = torch.minimum(start_frames + 1, _tm_v.clip_lengths[_ci_v] - 1)
        _ref2 = _tm_v.get_reference_state_for_envs(env_ids, _f2)
        _dtv = 0.02
        _dp = (_ref2[f"{side}_obj_pos"] - ref[f"{side}_obj_pos"]) / _dtv
        root_state[:, 7:10] = _dp.clamp(min=-2.0, max=2.0)
        _q1 = ref[f"{side}_obj_quat"]
        _q2 = _ref2[f"{side}_obj_quat"]
        # small-angle approximation: omega ≈ 2 * vec(q2 ⊗ q1^-1) / dt  (flip the sign when w<0)
        _q1c = torch.cat([_q1[:, :1], -_q1[:, 1:]], dim=1)
        _w = _q2[:, 0] * _q1c[:, 0] - (_q2[:, 1:] * (-_q1c[:, 1:])).sum(-1)
        _xyz = _q1c[:, :1] * _q2[:, 1:] + _q2[:, :1] * _q1c[:, 1:] + torch.cross(_q2[:, 1:], _q1c[:, 1:], dim=1)
        _sgn = torch.where(_w < 0, -torch.ones_like(_w), torch.ones_like(_w)).unsqueeze(1)
        root_state[:, 10:13] = (2.0 * _sgn * _xyz / _dtv).clamp(min=-6.0, max=6.0)
        if not getattr(env, "_objvel_logged", False):
            env._objvel_logged = True
            print(
                f"[reset_object] REF_VEL_INIT on ({side}): mean |v| "
                f"{root_state[:, 7:10].norm(dim=1).mean().item():.3f} m/s"
            )

    obj.write_root_state_to_sim(root_state, env_ids)

    # Articulated objects get their **joint values** back to the reference too: resetting the pose only would start the
    # lid at the last episode's angle, off the reference. Rigid objects have no joints (0 written). Owner of the joint
    # write = spawn_hold.pin_joints (the same function as SPAWN_HOLD).
    from dexengine.rl.tasks.scene.spawn_hold import pin_joints as _pin_joints

    _nj = _pin_joints(env, env_ids, side, obj, ref)
    if _nj and not getattr(env, "_artic_reset_logged", False):
        env._artic_reset_logged = True
        _jp = ref[f"{side}_obj_arti"]
        print(
            f"[reset_object_to_trajectory] articulated {side}: joint values restored ON "
            f"(n_joints={_nj}, mean {_jp.mean().item():.4f} rad)",
            flush=True,
        )


def reset_support_discs(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    disc_names: list,
    table: list,
    tag: str = "AERIAL",
    height: float | None = None,
    hidden_z: float | None = None,
    heights: list | None = None,
    visible: bool | None = None,
) -> None:
    """Aerial scene (rule D20) and desk support (rule D26′, tag="DESK_SUPPORT"): write the per-clip support disc slot
    poses with **that env's clip** values. The desk support's value owner is the tree's
    `task_info[qpos_layout.DESK_SUPPORT_KEY].discs` (bake owner `coords.compute_desk_support`; same coordinate
    convention as the aerial discs below).

    `table[c][k] = [x, y, z, r]` (c = loader clip order, k = slot). Owner of the values: the tree's
    `task_info.aerial_support.discs` (bake owner `coords.aerial_support`). **The disc xy is already in the scene (= env
    world) frame**: coords applied the capture->scene rotation RZ_CAP2SCENE (Rz(−90°)), so it is not rotated again here.
    Oracle (arctic s01-box_grab_01 f=47): the floor centre of bottom.obj at the tree pose, (+0.022, −0.136), rotated by
    Rz(−90°) is (−0.136, −0.022) = disc (−0.137, −0.019), r 0.213=0.211. z is already a world value
    (bottom_z_capture + z_off) and is the **top face** (the supported object's floor; the same meaning as the render's
    robot_motion_viz), so the cylinder centre is z − height/2 (a centre at z would sink objects by
    half the 5 mm thickness). Empty slots go to the hidden z. A disc stands from reset to the episode end (rule D40;
    the data rules read the same physics; no lifetime window). Discs are kinematic, so one write stays;
    they are written at every reset because the env->clip assignment is fixed per reset. `object_pin.support_disc_state`
    reads the table once from the event term params (disc_names · table · height · hidden_z · heights are those params;
    heights[c][k] = column height per slot: the stamped h for a pedestal, height for a disc; visible = the session's
    disc visibility statement).
    """
    from dexengine.rl.tasks.scene import object_pin as _op  # owner of the disc table and its write

    _op.reset_discs(env, env_ids)


def reset_support_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    names: list,
    poses: list,
    assets: list | None = None,
    tag: str = "SUPPORT",
) -> None:
    """Support-object slots (rule D42): written at the scene pose of the env clip's first frame (empty slots hidden);
    kinematic, no lifetime. Owner of the table and the write: `scene.supports.reset_supports` (names · poses · assets ·
    tag are the event params it reads)."""
    from dexengine.rl.tasks.scene import supports as _sr

    _sr.reset_supports(env, env_ids)


def reset_extra_objects(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Extra object slots: written at the env's start-frame reference pose (empty slots hidden); owner
    `scene.extra_slots.reset_extra_slots`."""
    from dexengine.rl.tasks.scene import extra_slots as _xe

    _xe.reset_extra_slots(env, env_ids)
