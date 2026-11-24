"""Arm reset onto the reference: the IK cache pose at each env's start frame, parking and the left arm."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg

from dexengine.rl.tasks import env_flags

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from dexengine.rl.tasks.mdp.events.reset_cache import _envz_apply_root_z, _load_arm_ik_cache, reset_cache_all

#: Returned by a split-out step that did not return early (its caller then goes on).
_CONTINUE = object()


def reset_arm_to_trajectory(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008 — Isaac Lab term signature: the default names the scene entity
    arm_joint_names: list[str] | None = None,
    ee_body_name: str | None = None,
    ik_lambda_val: float = 0.005,
    left_arm_joint_expr: str | None = None,
) -> None:
    """Reset arm joints to place wrist at trajectory frame-0 pose.

    If pre-computed arm IK poses are available,
    directly teleports the arm joints to the converged positions. Otherwise falls
    back to a single DifferentialIK step from the current configuration.

    The pre-computed path is preferred because it provides ~100-step converged
    joint positions (sub-centimeter wrist error), while the single IK step
    fallback gives ~5cm initial error that needs PD convergence during stepping.

    This should be called AFTER reset_hands_to_trajectory (which sets fingers).

    Args:
        env: The environment.
        env_ids: Environments to reset.
        robot_cfg: Robot scene entity config.
        arm_joint_names: Arm joint name patterns. None = the active arm's right arm.
        ee_body_name: End-effector body name for IK. None = the active arm's right EE.
        ik_lambda_val: DLS regularization for IK (used only in fallback path).
        left_arm_joint_expr: LEFT arm joint regex. None = the active arm's. The
            cfg factories always pass all three from the arm spec (explicit
            dataflow); a None is resolved from the active selection, which dies
            when DEXENGINE_ARM is unset or unregistered — never a silent default.
    """
    if arm_joint_names is None or ee_body_name is None:
        # Missing values are filled from the **active arm**. The cfg factory always passes params, but a new wiring or
        # diagnostic call without them would silently drift onto another arm's joint names. The legacy arm fallback
        # (`or _ARM`) and that arm's default EE name are gone: without an active selection resolve_robot_selection
        # fails.
        from dexengine.robots.selection import resolve_robot_selection as _rrs

        _arm_now = _rrs()[0]
        if arm_joint_names is None:
            arm_joint_names = [_arm_now.joint_expr("right")]
        if ee_body_name is None:
            ee_body_name = _arm_now.ee_body_name("right")

    robot = env.scene[robot_cfg.name]

    return _pin_head_joints(env, env_ids, arm_joint_names, ee_body_name, ik_lambda_val, left_arm_joint_expr, robot)


def _pin_head_joints(env, env_ids, arm_joint_names, ee_body_name, ik_lambda_val, left_arm_joint_expr, robot):
    """Pin the head (camera) joints with a PD target Symptom: on the episode's first frame the head is pitched 30° down
    (initial joint_pos) and right after rises to 0°, so the camera looks straight ahead. Cause: only the initial
    state (joint_pos) was 30°; nobody wrote the **actuator position target**. The ImplicitActuator target buffer
    starts at 0, so the PD pulls the head up to 0°. Arm and hand get set_joint_position_target at every reset, but
    the policy does not drive the head, so it had no such call. Fix: at every reset, head joint target =
    default_joint_pos (= the arm spec's head_drive.pitch_deg). The joint ids are looked up once and cached on the env
    (the names come from arm_spec).
    """
    _head_ids = getattr(env, "_head_jids_t", None)
    if _head_ids is None:
        from dexengine.robots.selection import resolve_robot_selection as _rrs_head

        _arm_h = _rrs_head()[0]
        _names = list(getattr(_arm_h, "head_joint_names", ()) or ())
        _head_ids = robot.find_joints(_names)[0] if _names else []
        env._head_jids_t = _head_ids
        if _head_ids:
            _deg = [round(float(robot.data.default_joint_pos[0, j]) * 180.0 / 3.141592653589793, 1) for j in _head_ids]
            print(f"[reset_arm_to_trajectory] head joints PD-pinned {_names} -> {_deg}deg", flush=True)
    if _head_ids:
        _hd = robot.data.default_joint_pos[env_ids][:, _head_ids]
        robot.set_joint_position_target(_hd, joint_ids=_head_ids, env_ids=env_ids)

    traj_mgr = env.trajectory_manager

    # ── ENVZ is applied by _envz_apply_root_z(), called **after** the caches are loaded (left above / right below), at
    # the two exits (just before the parked return, just before the right teleport). Called before loading, the first
    # reset was skipped silently, so pose_render (one reset) never had it.

    # --- Reset / pin the LEFT arm ---
    # Right-only training: the left arm is in no obs/reward, so pin its PD target to default
    # (cosmetic; unchanged byte-for-byte). BIMANUAL: the left arm IS policy-driven + tracked,
    # so reset it to the LEFT arm-IK cache (precompute/.../arm_ik_poses_left/, generated by
    # the arm IK stage, left side, all frames) at the per-env start_frame — mirroring the
    # right-arm reset below — so the left wrist starts on the left reference.
    # GATED split-articulation: the LEFT arm is a SEPARATE `robot_left` articulation, so ALL
    # left-arm joint lookups + writes below target it (via the `left_hand` alias). Unset ->
    # left_robot is robot (byte-identical).
    from dexengine.rl.tasks.scene import split_artic as _split_artic

    _split_ev = _split_artic.enabled()
    left_robot = env.scene["left_hand"] if (_split_ev and "left_hand" in env.scene.keys()) else robot  # noqa: SIM118  — InteractiveScene has no __contains__
    if not hasattr(env, "_left_arm_jids_cache"):
        # Same as the right fallback: the active arm's left joint expressions (a zero match here would read as 'no left
        # arm' and skip the left reset silently).
        from dexengine.robots.selection import resolve_robot_selection as _rrs_l

        left_jids, _ = left_robot.find_joints(
            [left_arm_joint_expr if left_arm_joint_expr is not None else _rrs_l()[0].joint_expr("left")]
        )
        env._left_arm_jids_cache = left_jids
        # Canonical side-suffixed name: these ids pair with _arm_ik_cache_all_left the
        # same way the RSI snap path already pairs them (tasks/env.py: cache_all_left + _left_arm_jids_t).
        # A LEFT-uni reset returns early at the parked-right-arm branch below, so the right-arm id
        # publication is never reached — without this the ARM_FF velocity feedforward finds no ids.
        env._arm_ik_joint_ids_left = left_jids
    left_jids = env._left_arm_jids_cache
    _bimanual = "left_hand_joint_pos" in env.action_manager._terms
    return _reset_arm_right_park_and_ik(
        env, env_ids, arm_joint_names, ee_body_name, ik_lambda_val, _bimanual, left_jids, left_robot, robot, traj_mgr
    )


def _reset_arm_right_park_and_ik(
    env, env_ids, arm_joint_names, ee_body_name, ik_lambda_val, _bimanual, left_jids, left_robot, robot, traj_mgr
):
    """Right-park branch and the arm-IK cache / frame-0 IK reset (statements unchanged; its returns are the
    caller's)."""
    if left_jids and _bimanual:
        if not hasattr(env, "_arm_ik_cache_left"):
            env._arm_ik_cache_left = _load_arm_ik_cache(env, side="left")
            if env._arm_ik_cache_left is not None:
                env._left_arm_jids_t = torch.tensor(left_jids, device=env.device, dtype=torch.long)
        lc = getattr(env, "_arm_ik_cache_left", None)
        if lc is not None:
            clip_ids = traj_mgr.clip_ids[env_ids]
            cache_all_l, len_l = reset_cache_all(env, "left")
            if cache_all_l is not None:
                sf = traj_mgr.start_frames[env_ids].clamp(min=0)
                sf = torch.minimum(sf, len_l[clip_ids] - 1)
                left_jpos = cache_all_l[clip_ids, sf]  # (N,7) frame-indexed (RSI)
            else:
                left_jpos = lc[clip_ids]  # (N,7) frame-0
            left_robot.write_joint_state_to_sim(
                left_jpos, torch.zeros_like(left_jpos), joint_ids=env._left_arm_jids_t, env_ids=env_ids
            )
            left_robot.set_joint_position_target(left_jpos, joint_ids=env._left_arm_jids_t, env_ids=env_ids)
        else:
            left_defaults = left_robot.data.default_joint_pos[env_ids][:, left_jids]
            left_robot.set_joint_position_target(left_defaults, joint_ids=left_jids, env_ids=env_ids)
            if not getattr(env, "_left_cache_warned", False):
                env._left_cache_warned = True
                print(
                    "[reset_arm_to_trajectory] WARNING: bimanual but NO left arm-IK cache "
                    "(arm_ik_poses_left/) — left arm pinned to default. Run "
                    "the arm IK stage (scripts/stage2_ground.sh) for the left side, all frames."
                )
    elif left_jids:
        left_defaults = left_robot.data.default_joint_pos[env_ids][:, left_jids]
        left_robot.set_joint_position_target(left_defaults, joint_ids=left_jids, env_ids=env_ids)

    _ret = _park_right_arm(env, env_ids, arm_joint_names, robot, traj_mgr)
    if _ret is not _CONTINUE:
        return _ret

    # --- Fallback: single IK step ---
    N = int(env_ids.numel())  # number of envs being reset (IK controller batch + jacobian select)

    _reset_arm_ik_from_wrist(env, env_ids, arm_joint_names, ee_body_name, ik_lambda_val, N, robot, traj_mgr)
    return None


def _park_right_arm(env, env_ids, arm_joint_names, robot, traj_mgr):
    """LEFT-UNI: DEXENGINE_RIGHT_PARK=1 — park the RIGHT arm (joint-state write + PD pin) at its default pose. The LEFT
    branch above already ran (left arm gets its own RSI reset). Unset -> byte-identical. PARK POSE: holds
    default_joint_pos, exactly like the LEFT arm parks in right-only training. The robot default is SYMMETRIC (both
    arms 0 -> straight-down hang, clear of the desk; scene/robot._joint_pos_map) — a right-only elbow-up override
    once made the parked right arm reach the desk back-edge (cosmetic desk-touch in L renders). No env gate: right
    parks the same way left does.
    """
    if env_flags.get("DEXENGINE_RIGHT_PARK") == "1":
        if not hasattr(env, "_right_park_jids"):
            env._right_park_jids, _ = robot.find_joints(arm_joint_names)
            print(
                "[reset_arm_to_trajectory] RIGHT-PARK gate ACTIVE: right arm held at "
                "default pose (symmetric with the left arm's park; both hang straight down)"
            )
        rj = env._right_park_jids
        r_def = robot.data.default_joint_pos[env_ids][:, rj]
        robot.write_joint_state_to_sim(r_def, torch.zeros_like(r_def), joint_ids=rj, env_ids=env_ids)
        robot.set_joint_position_target(r_def, joint_ids=rj, env_ids=env_ids)
        # left-uni exit: apply root z from the dz of the left cache (loaded above), then return.
        _envz_apply_root_z(env, robot, traj_mgr, env_ids)
        return None

    # --- Try pre-computed arm IK poses (cached on first call) ---
    if not hasattr(env, "_arm_ik_cache_right"):
        env._arm_ik_cache_right = _load_arm_ik_cache(env)
        # Also cache arm joint IDs for the pre-computed path.
        # NAMING: `_arm_ik_joint_ids_right` has always held the ACTIVE arm's ids —
        # on a LEFT-uni build `arm_joint_names` is the LEFT arm, so the "right" name is a legacy
        # misnomer. Publish the canonical side-suffixed name too so consumers (e.g. the ARM_FF
        # velocity feedforward) can resolve by side instead of guessing.
        if env._arm_ik_cache_right is not None:
            _ik_ids, _ = robot.find_joints(arm_joint_names)
            env._arm_ik_joint_ids_right = _ik_ids
            _act_side = "left" if any("left" in n for n in arm_joint_names) else "right"
            setattr(env, f"_arm_ik_joint_ids_{_act_side}", _ik_ids)

    # right-uni and bimanual exit: right after loading the right cache, in the same reset as the joint teleport.
    _envz_apply_root_z(env, robot, traj_mgr, env_ids)

    if env._arm_ik_cache_right is not None:
        clip_ids = traj_mgr.clip_ids[env_ids]
        # RSI: if the all-frame cache is available, reset the arm to the IK config at each
        # env's start_frame (so a mid-trajectory RSI reset lands the wrist on the reference
        # at that frame, consistent with the fingers/object which reset to the same frame).
        # Else (no all-frame data) use the frame-0 config — legacy behavior, byte-identical.
        cache_all, len_r = reset_cache_all(env, "right")
        if cache_all is not None:
            sf = traj_mgr.start_frames[env_ids].clamp(min=0)
            sf = torch.minimum(sf, len_r[clip_ids] - 1)
            arm_jpos = cache_all[clip_ids, sf]  # (N, 7) frame-indexed
        else:
            arm_jpos = env._arm_ik_cache_right[clip_ids]  # (N, 7) frame-0
        arm_jids = env._arm_ik_joint_ids_right
        robot.write_joint_state_to_sim(
            arm_jpos,
            torch.zeros_like(arm_jpos),
            joint_ids=arm_jids,
            env_ids=env_ids,
        )
        robot.set_joint_position_target(
            arm_jpos,
            joint_ids=arm_jids,
            env_ids=env_ids,
        )
        return None
    return _CONTINUE


def _reset_arm_ik_from_wrist(env, env_ids, arm_joint_names, ee_body_name, ik_lambda_val, N, robot, traj_mgr):
    """Frame-0 wrist pose from the trajectory and the IK reset of the arm (statements unchanged)."""
    arm_joint_ids, ee_pos_b, ee_quat_b, ik_command, ik_controller, jacobian = _frame0_wrist_pose_world(
        env, env_ids, arm_joint_names, ee_body_name, ik_lambda_val, N, robot, traj_mgr
    )
    if jacobian.shape[0] > N:
        jacobian = jacobian[env_ids]

    # Current arm joint positions as starting point
    arm_joint_pos = robot.data.joint_pos[env_ids][:, arm_joint_ids].clone()

    # Single IK step: compute joint position delta toward target
    ik_controller.set_command(ik_command, ee_pos_b, ee_quat_b)
    arm_joint_pos = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, arm_joint_pos)

    # Teleport arm joints and set PD target
    robot.write_joint_state_to_sim(
        arm_joint_pos,
        torch.zeros_like(arm_joint_pos),
        joint_ids=arm_joint_ids,
        env_ids=env_ids,
    )
    robot.set_joint_position_target(
        arm_joint_pos,
        joint_ids=arm_joint_ids,
        env_ids=env_ids,
    )


def _frame0_wrist_pose_world(env, env_ids, arm_joint_names, ee_body_name, ik_lambda_val, N, robot, traj_mgr):
    """Get trajectory frame-0 wrist pose in world frame."""
    try:
        # get_wrist_pos/quat_w expects (num_envs,) shape, uses internal clip_ids.
        # We call with frame_ids=None (uses start_frames) then select env_ids.
        all_wrist_pos = traj_mgr.get_wrist_pos_w(None)  # (num_envs, 3)
        all_wrist_quat = traj_mgr.get_wrist_quat_w(None)  # (num_envs, 4)
        wrist_pos_w = all_wrist_pos[env_ids] + env.scene.env_origins[env_ids]
        wrist_quat_w = all_wrist_quat[env_ids]
    except RuntimeError as e:
        # wrist_poses_dir_right not configured AND arm_ik_poses unavailable.
        # Silently skipping here leaves the arm at its default rest pose, which is
        # far from the trajectory reference and causes immediate early termination
        # (fingers appear fixed at the rest-pose global coordinate while the
        # reference advances to the grasp position).  Raise loudly so the run owner
        # knows to set wrist_poses_dir_right in the training config.
        raise RuntimeError(
            "[reset_arm_to_trajectory] Cannot reset arm: no pre-computed arm IK poses "
            "found in arm_ik_poses_dir_right AND no wrist_poses_dir_right is configured as fallback. "
            "Set trajectory.wrist_poses_dir_right in the env config to enable the single-IK "
            "fallback, or regenerate the arm IK poses in preprocessing. "
            f"Original error: {e}"
        ) from e

    # Convert wrist target -> link7 target via the SIDE-AWARE mount, funneled through
    # mdp.wrist_kinematics. The old hardcoded Rz(+90) inverse was SIDE-BLIND: a
    # left-arm reset via this single-IK fallback would place link7 180deg wrong (same class
    # as the original left-wrist bug). Right is byte-identical. (Primary reset uses the
    # precomputed side-aware arm_ik_poses cache; this fallback only fires if it is absent.)
    from dexengine.rl.tasks.mdp.wrist_kinematics import mount_tensors, side_from_body_name, virtual_wrist_to_link7

    _, _r_inv, _t_off = mount_tensors(robot, side_from_body_name(ee_body_name), env.device)
    link7_pos_w, link7_quat_w = virtual_wrist_to_link7(wrist_pos_w, wrist_quat_w, _r_inv, _t_off)

    # Convert link7 world pose to robot root frame
    root_pos_w = robot.data.root_pos_w[env_ids]
    root_quat_w = robot.data.root_quat_w[env_ids]
    link7_pos_b, link7_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, link7_pos_w, link7_quat_w)

    # Resolve arm joint IDs
    arm_joint_ids, _ = robot.find_joints(arm_joint_names)

    # Resolve EE body index
    ee_body_ids, _ = robot.find_bodies(ee_body_name)
    ee_body_idx = ee_body_ids[0]

    # Jacobian body index (fixed-base robots have one fewer row)
    if robot.is_fixed_base:
        jacobi_body_idx = ee_body_idx - 1
    else:
        jacobi_body_idx = ee_body_idx

    # Create IK controller for single-step solve
    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": ik_lambda_val},
    )
    ik_controller = DifferentialIKController(ik_cfg, num_envs=N, device=env.device)

    # IK target: link7 pose in robot root frame
    ik_command = torch.cat([link7_pos_b, link7_quat_b], dim=-1)  # (N, 7)

    # Current EE pose in root frame (from last sim step / default pose)
    ee_pos_w = robot.data.body_pos_w[env_ids, ee_body_idx]
    ee_quat_w = robot.data.body_quat_w[env_ids, ee_body_idx]
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

    # Get Jacobian for arm joints only
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, arm_joint_ids]
    return arm_joint_ids, ee_pos_b, ee_quat_b, ik_command, ik_controller, jacobian
