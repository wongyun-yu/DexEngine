"""Floating hand: the armless hand body of the paper's floating rows (Tab. 2) and the floating RL ablation arms.

Our RL method (bundle training · the same rewards · PARK judge · the same success rule) runs on the hand alone: no
arm, no IK, no arm joint PD. Each hand is one free-root articulation (fix_base False — the repository's bake rule
"floating hand = False"); the canonical training already uses two articulations (right ``robot`` · left ``robot_left``,
``DEXENGINE_BI_SPLIT_ARTIC=1``), so the downstream wiring is unchanged. The whole hand has gravity disabled
(FlashCHORD puts gravcomp 1.0 on every hand body).

Wrist drives (``DEXENGINE_FH_WRIST``):
  wrench (default)  an external wrench (force 3 + torque 3) on the hand root body: an SE(3) spring-damper toward the
                    reference wrist pose + the policy residual (wrench.py). A 3+3 virtual-joint chain was not the
                    default because the reference wrist's Euler chain comes within 10° of the gimbal (±90°) on 5
                    right / 9 left clips of the 64 (world-fixed anchor · XYZ order).
  joint             six virtual wrist joints from a per-episode anchor, driven by PhysX implicit joint drives
                    (wrist_joints.py, anchor rule anchor.py).

This module is the import surface other packages use (frozen): it re-exports the names of identity.py, wrench.py,
wrist_joints.py and anchor.py and owns the reset event. It loads without isaaclab and without torch.

Env flags read inside the Isaac process (a process boundary, so they stay env vars): DEXENGINE_ARM=floating (owner
dexengine.core.arms) · DEXENGINE_FH_WRIST · DEXENGINE_FH_ANCHOR. RL switches (dexengine.rl.tasks.switches, the
trainer's switches file): fh_nan_reset (DEXENGINE_FH_NAN_RESET) · fh_palm_point (DEXENGINE_FH_PALM_POINT).
The RL task reaches this module through dexengine.rl.tasks.hooks (plugin.py registers it).
Marker ``[FLOATING_HAND]``: a run whose log lacks it is not a floating-hand run.
"""

from __future__ import annotations

from experiments.ablations.floating_hand.anchor import (  # noqa: F401
    anchor_check,
    anchor_minmid,
    anchor_reset_stamp,
    anchor_rule,
    anchor_start_angles,
    anchor_table,
    center_branch,
    vj_lin_limit,
)
from experiments.ablations.floating_hand.identity import (  # noqa: F401
    ARM_GAIN_ENVS,
    DATA_WORLD_ARM,
    FLOATING_ARM,
    MARK,
    SUPPORTED_HANDS,
    active,
    data_world_arm,
    describe,
    is_floating,
    require_no_arm_gains,
    require_supported,
    root_body,
    source_urdf,
)
from experiments.ablations.floating_hand.wrench import (  # noqa: F401
    DAMP_INERTIA_RULE,
    MASS_RULE,
    WRIST_GAINS,
    LoadObserver,
    damping_inertia_w,
    wrench_to_link_frame,
    wrist_wrench,
)
from experiments.ablations.floating_hand.wrist_joints import (  # noqa: F401
    NAN_RESET_ENV,
    PALM_POINT_ENV,
    VJ_ARMATURE_RULE,
    VJ_AXES,
    VJ_GIMBAL_WARN,
    VJ_LIN_LIMIT_M,
    VJ_LINK_INERTIA,
    VJ_LINK_MASS,
    VJ_ROT_MID_LIMIT,
    VJ_ROT_OUTER_LIMIT,
    WRIST_DRIVE_ENV,
    joint_drive,
    mode_canon_path,
    nan_reset,
    palm_point_mode,
    vj_actuator_gains,
    vj_armature,
    vj_armature_mark,
    vj_canon_path,
    vj_joint_names,
    vj_link_names,
    vj_targets,
)

#: head of the once-per-run reset line; the run checks of the E1-E3 jobs read it from the logs
RESET_LINE = f"{MARK} reset applied:"

_anchor_reset_stamp = anchor_reset_stamp  # the name before the split


def reset_floating_hands_to_trajectory(env, env_ids) -> None:
    """Reset event: per hand, ROOT POSE = the reference wrist (the env's start frame) · root velocity 0.

    The arm body's counterpart writes the arm joints from the arm-IK pool to put the wrist at frame 0. The floating
    hand's wrist is its root, so there is no IK — it reads the same reference source as the outer action term and the
    observations (``trajectory_manager.get_wrist_pos_w/quat_w``). The finger joints are written by the parent event
    (hand trajectory reset), as on the arm body. ENVZ (robot base height) does not apply: there is no base.
    Joint drive: the articulation root is the world-fixed anchor, put at the start reference wrist (or at the clip's
    minmid anchor pose); the virtual joints start at 0 (minmid: rotation start angles = YXZ(anchor⁻¹ ⊗ start hand)).
    """
    import torch  # noqa: PLC0415

    from dexengine.rl.tasks.scene import split_artic  # noqa: PLC0415 — split_artic imports this module

    tm = env.trajectory_manager
    left = split_artic.left_articulation(env)
    if left is None:
        raise RuntimeError(
            f"{MARK} reset: the left articulation ({split_artic.LEFT_ENTITY}) is not in the scene — "
            "the split wiring is off."
        )
    n = len(env_ids)
    zero6 = torch.zeros(n, 6, device=env.device)
    joint = joint_drive()
    anchor = anchor_rule() if joint else ""
    if anchor and getattr(env, "_fh_anchor_tab", None) is None:  # rule minmid — the clip table once (gates · markers)
        env._fh_anchor_tab = anchor_table(tm, env.device)
    for side, art in (("right", env.scene["robot"]), ("left", left)):
        pos = tm.get_wrist_pos_w(None, side=side)[env_ids] + env.scene.env_origins[env_ids]
        quat = tm.get_wrist_quat_w(None, side=side)[env_ids]
        root_quat = env._fh_anchor_tab[side][tm.clip_ids[env_ids]] if anchor else quat
        art.write_root_pose_to_sim(torch.cat([pos, root_quat], dim=-1), env_ids)
        if not joint:  # the joint drive's root (the anchor) is world-fixed — virtual joint velocity 0 takes its place
            art.write_root_velocity_to_sim(zero6, env_ids)
        if joint:
            ids, names = art.find_joints(vj_joint_names(side), preserve_order=True)
            if list(names) != vj_joint_names(side):
                raise RuntimeError(
                    f"{MARK} reset ({WRIST_DRIVE_ENV}=joint): {side} virtual joints {names} ≠ {vj_joint_names(side)}"
                )
            z = torch.zeros(n, len(ids), device=env.device)
            q0 = z
            if anchor:
                # the outer branch follows the unwrapped reference at the start frame (RSI included · the same absolute
                # frame as get_wrist_*(None))
                cid, fr = tm.clip_ids[env_ids], tm.start_frames[env_ids].long()
                u_tab = env._fh_anchor_tab["u"][side]
                fr = fr.clamp(0, u_tab.shape[1] - 1)
                e0 = anchor_start_angles(root_quat, quat, u_tab[cid, fr])
                q0 = torch.cat([z[:, :3], e0], dim=-1)
                anchor_reset_stamp(env, side, tm, cid, fr, e0)
            art.write_joint_state_to_sim(q0, z, joint_ids=ids, env_ids=env_ids)
            art.set_joint_position_target(q0, joint_ids=ids, env_ids=env_ids)
    if not getattr(env, "_floating_reset_logged", False):
        env._floating_reset_logged = True
        # a run check reads this line from the logs (RESET_LINE, then anything)
        print(
            f"{RESET_LINE} both hand roots = reference wrist (start frame) · velocity 0 — first batch {n} envs"
            + (
                (
                    " · path C: anchor = rule minmid (per clip · side) · virtual rotation start angle = "
                    "YXZ(anchor⁻¹ ⊗ start hand)"
                    if anchor
                    else " · path C: anchor = reference wrist · 6 virtual joints = 0"
                )
                if joint
                else ""
            ),
            flush=True,
        )
