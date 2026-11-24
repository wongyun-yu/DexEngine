"""Shared base of the reference-tracking rewards: per-side body, joint and fingertip lookups."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from dexengine.rl.tasks.mdp.naming import (
    TIP_FINGERS,
    resolve_independent_joints,
    resolve_link7_to_wrist_offset,
    resolve_palm_body_id,
    resolve_wrist_joints,
)

# MuJoCo FK body index mapping (per hand, 13 bodies):
# [0] hand_base  (= wrist/palm)
# [1] thumb_proximal_base  (level_1)
# [2] thumb_proximal        (level_2)
# [3] thumb_intermediate    (level_2)
# [4] thumb_distal          (nearest to thumb_tip site)
# [5] index_proximal        (level_1)
# [6] index_intermediate    (level_2, nearest to index_tip site)
# [7] middle_proximal       (level_1)
# [8] middle_intermediate   (level_2, nearest to middle_tip site)
# [9] ring_proximal         (level_1)
# [10] ring_intermediate    (level_2, nearest to ring_tip site)
# [11] pinky_proximal       (level_1)
# [12] pinky_intermediate   (level_2, nearest to pinky_tip site)

# The constants above describe the legacy 13-body RH56F1 FK layout. For other
# layouts (SharpA: 23 bodies) the trajectory manager exposes name-resolved
# indices via ``fk_finger_indices`` and the reward base class prefers those.

# Body indices for level groups
_PALM_IDX = 0
_LEVEL_1_INDICES = [1, 5, 7, 9, 11]  # proximal joints
_LEVEL_2_INDICES = [2, 3, 6, 8, 10, 12]  # intermediate joints
# Fingertip body indices (nearest body to each MuJoCo site)
_THUMB_TIP_IDX = 4  # thumb_distal
_INDEX_TIP_IDX = 6  # index_intermediate
_MIDDLE_TIP_IDX = 8  # middle_intermediate
_RING_TIP_IDX = 10  # ring_intermediate
_PINKY_TIP_IDX = 12  # pinky_intermediate


class _BimanualRewardBase(ManagerTermBase):
    """Base class caching body/joint indices for both hands."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._side = cfg.params.get("side", "both")
        self._traj_mgr = env.trajectory_manager

        # Cache right hand body IDs (Isaac Lab indices in MuJoCo FK order)
        self._rh_body_ids = torch.tensor(self._traj_mgr.right_body_ids_isaac, device=env.device)
        _fk = self._cache_left_hand_body_ids(env)
        if _fk is not None:
            self._fk_tip_idx = dict(_fk["tips"])
            self._fk_level1 = list(_fk["level1"])
            self._fk_level2 = list(_fk["level2"])
        else:
            self._fk_tip_idx = {
                "thumb": _THUMB_TIP_IDX,
                "index": _INDEX_TIP_IDX,
                "middle": _MIDDLE_TIP_IDX,
                "ring": _RING_TIP_IDX,
                "pinky": _PINKY_TIP_IDX,
            }
            self._fk_level1 = list(_LEVEL_1_INDICES)
            self._fk_level2 = list(_LEVEL_2_INDICES)

    def _cache_left_hand_body_ids(self, env):
        """Cache left hand body IDs."""
        self._lh_body_ids = torch.tensor(self._traj_mgr.left_body_ids_isaac, device=env.device)

        # Cache palm body index (wrist_rot_z = end of wrist chain)
        # Uses fallback for placeholder left hand (right-hand USD with right_* names)
        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]
        self._rh_palm_body_id = resolve_palm_body_id(rh, "right")
        self._lh_palm_body_id = resolve_palm_body_id(lh, "left")

        # Wrist rotation joint IDs in PhysX order (for rotation tracking)
        self._rh_wrist_rot_ids = self._find_joint_ids(rh, resolve_wrist_joints(rh, "right", rot_only=True))
        self._lh_wrist_rot_ids = self._find_joint_ids(lh, resolve_wrist_joints(lh, "left", rot_only=True))

        # Full-robot (link7) detection + the fixed link7->virtual-wrist transform.
        # For the full robot the palm body resolves to the arm end-effector
        # link (link7 on the legacy arm), whose body frame differs from the mocap wrist/palm
        # reference by a FIXED Rz(-90deg) + 5.55cm-along-Z transform. Applying it
        # makes eef_pos/eef_rot frame-consistent with the reference / IK target
        # (verified: collapses a spurious 5.8cm/88deg offset to ~0.5cm/5deg).
        # Constants MUST match dexengine.rl.tasks.mdp.actions.arm_ik (_r_offset_fwd_wxyz,
        # _link7_to_wrist_trans). For the floating hand the palm body already is
        # the wrist, so no transform is applied (is_link7 is False).
        from dexengine.rl.tasks.mdp.naming import is_arm_ee_body as _is_ee

        self._rh_is_link7 = _is_ee(rh.data.body_names[self._rh_palm_body_id])
        self._lh_is_link7 = _is_ee(lh.data.body_names[self._lh_palm_body_id])
        # SIDE-AWARE mounting rotation (right Rz-90 / left Rz+90; the mirror is proven
        # from the URDF). The prior single Rz(-90) gave the LEFT
        # virtual-wrist orientation a 180deg error in RewardEefRot. MUST match the action
        # term's _r_offset_fwd_wxyz.
        from dexengine.rl.tasks.mdp.naming import resolve_ee_to_wrist_rot as _rot_pair  # side-aware pair

        self._rh_wrist_r_offset = torch.tensor(list(_rot_pair(rh, "right")), device=env.device)
        self._lh_wrist_r_offset = torch.tensor(list(_rot_pair(lh, "left")), device=env.device)
        # Back-compat alias (right-hand value) for any external readers.
        self._wrist_r_offset = self._rh_wrist_r_offset
        # Translation is hand-dependent (RH56F1 0.0555, SharpA 0.0655) and the
        # two hands can differ on a mixed robot; resolve per side.
        self._rh_wrist_t_offset = torch.tensor(list(resolve_link7_to_wrist_offset(rh, "right")), device=env.device)
        self._lh_wrist_t_offset = torch.tensor(list(resolve_link7_to_wrist_offset(lh, "left")), device=env.device)
        # Back-compat alias (right-hand value) for any external readers.
        self._wrist_t_offset = self._rh_wrist_t_offset

        # Independent joint IDs in PhysX order (for velocity reward, indexing joint_pos/joint_vel)
        self._rh_indep_ids = self._find_joint_ids(rh, resolve_independent_joints(rh, "right"))
        self._lh_indep_ids = self._find_joint_ids(lh, resolve_independent_joints(lh, "left"))

        # Trajectory-space indices for indexing ref qpos (same as PhysX for floating hand,
        # but remapped for full robot where traj is 12-dim finger-only).
        traj_mgr = env.trajectory_manager
        if hasattr(traj_mgr, "_fingers_only") and traj_mgr._fingers_only:
            # Per-side widths: on a mixed robot (right SharpA 22 / left RH56F1 12) the
            # two perms differ — using the right width for both indexes past the left
            # qpos array (CUDA device-side assert, found at first SharpA smoke).
            n_traj_r = len(traj_mgr._right_perm)
            _lp = getattr(traj_mgr, "_left_perm", None)
            n_traj_l = len(_lp) if _lp is not None else n_traj_r
            self._rh_traj_indep_ids = torch.arange(n_traj_r, device=env.device)
            self._lh_traj_indep_ids = torch.arange(n_traj_l, device=env.device)
        else:
            self._rh_traj_indep_ids = self._rh_indep_ids
            self._lh_traj_indep_ids = self._lh_indep_ids

        self._fps = 50.0  # trajectory frame rate

        # FK finger indices: legacy 13-body RH56F1 layout keeps the module
        # constants; other layouts use the manager's name-resolved indices.
        _fk = getattr(traj_mgr, "fk_finger_indices", None)
        return _fk

    @staticmethod
    def _find_joint_ids(asset, joint_names: list[str]) -> torch.Tensor | None:
        """Find PhysX joint indices for given joint names. Returns None if empty."""
        if not joint_names:
            return None
        ids = [asset.data.joint_names.index(n) for n in joint_names]
        return torch.tensor(ids, device=asset.device)

    def _combine(self, rh_reward: torch.Tensor, lh_reward: torch.Tensor) -> torch.Tensor:
        """Combine per-hand rewards based on side configuration."""
        if self._side == "right":
            return rh_reward
        elif self._side == "left":
            return lh_reward
        return 0.5 * (rh_reward + lh_reward)

    def _virtual_wrist_pose(self, asset, palm_body_id, is_link7, origins, t_offset=None, r_offset=None):
        """Sim wrist pose in the mocap-reference frame: (env-local pos, world quat).

        Full robot: palm body is link7 -> apply the fixed link7->wrist transform so
        the result is frame-consistent with the reference. Floating hand: palm body
        already is the wrist (no transform). ``t_offset`` / ``r_offset`` select the
        per-side translation / mounting rotation (both default to the right-hand
        value for back-compat; the LEFT arm's rotation is the Rz(+90) mirror).
        """
        from isaaclab.utils.math import quat_apply, quat_mul

        pos = asset.data.body_link_pos_w[:, palm_body_id, :] - origins
        quat = asset.data.body_link_quat_w[:, palm_body_id, :]
        if is_link7:
            n = pos.shape[0]
            t_off = t_offset if t_offset is not None else self._wrist_t_offset
            r_off = r_offset if r_offset is not None else self._wrist_r_offset
            pos = pos + quat_apply(quat, t_off.expand(n, -1))
            quat = quat_mul(quat, r_off.expand(n, -1))
        return pos, quat

    def _fingertip_reward(
        self,
        env: ManagerBasedEnv,
        finger: str,
        k: float,
        hinge_threshold_m: float = 0.0,
        contact_gated_hinge: bool = False,
        hinge_threshold_in_contact: float = 0.04,
        contact_force_threshold: float = 5.0,
    ) -> torch.Tensor:
        """Compute fingertip position tracking reward for one finger.

        Args:
            env: The environment instance.
            finger: One of TIP_FINGERS (thumb .. pinky); its MuJoCo FK body index is ``self._fk_tip_idx[finger]``.
            k: Exponential kernel sharpness.
            hinge_threshold_m: Soft hinge threshold in metres (uniform mode).
                When > 0 and contact_gated_hinge=False, errors within this
                distance yield full reward.
            contact_gated_hinge: If True (H10 reward-form variant), use
                `hinge_threshold_in_contact` per-env when the fingertip is in
                contact with the object (||F|| > contact_force_threshold N),
                otherwise use `hinge_threshold_m` for that env. State-dependent
                hinge.
            hinge_threshold_in_contact: Threshold applied when in contact.
            contact_force_threshold: Force magnitude threshold (N) to declare contact.

        Returns:
            Reward tensor of shape (num_envs,).
        """
        body_idx = self._fk_tip_idx[finger]
        ref_body = self._traj_mgr.get_reference_body_pos(env.episode_length_buf)
        origins = env.scene.env_origins

        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]

        # Right hand
        rh_cur = rh.data.body_link_pos_w[:, self._rh_body_ids[body_idx], :] - origins
        rh_ref = ref_body["right_body_pos"][:, body_idx, :]
        rh_dist = torch.norm(rh_ref - rh_cur, dim=-1)

        # Left hand
        lh_cur = lh.data.body_link_pos_w[:, self._lh_body_ids[body_idx], :] - origins
        lh_ref = ref_body["left_body_pos"][:, body_idx, :]
        lh_dist = torch.norm(lh_ref - lh_cur, dim=-1)

        if contact_gated_hinge and hinge_threshold_m > 0.0:
            # H10 reward-form variant: per-env threshold depending on contact state.
            # In-contact fingertips → wider hinge (license to deviate to grasp);
            # free fingertips → tighter hinge (must follow mocap).
            rh_contact = self._fingertip_contact(env, "right", finger, contact_force_threshold)
            lh_contact = self._fingertip_contact(env, "left", finger, contact_force_threshold)
            rh_thresh = torch.where(rh_contact, hinge_threshold_in_contact, hinge_threshold_m)
            lh_thresh = torch.where(lh_contact, hinge_threshold_in_contact, hinge_threshold_m)
            rh_excess = torch.clamp(rh_dist - rh_thresh, min=0.0)
            lh_excess = torch.clamp(lh_dist - lh_thresh, min=0.0)
            return self._combine(torch.exp(-k * rh_excess), torch.exp(-k * lh_excess))

        if hinge_threshold_m > 0.0:
            # Uniform hinge (H2)
            rh_excess = torch.clamp(rh_dist - hinge_threshold_m, min=0.0)
            lh_excess = torch.clamp(lh_dist - hinge_threshold_m, min=0.0)
            return self._combine(torch.exp(-k * rh_excess), torch.exp(-k * lh_excess))

        return self._combine(torch.exp(-k * rh_dist), torch.exp(-k * lh_dist))

    def _fingertip_contact(self, env: ManagerBasedEnv, side: str, finger: str, force_threshold: float) -> torch.Tensor:
        """Per-env bool: the `finger` tip of the `side` hand has a net contact force above `force_threshold` N
        (contact-gated hinge). The hand's contact sensor must exist and carry every fingertip body: the hinge was asked
        for, so a missing sensor or tip dies instead of reading no contact."""
        sensor = env.scene[f"{side}_hand_contact"]
        cache_key = f"_contact_tip_ids_{side}"
        if not hasattr(self, cache_key):
            from dexengine.rl.tasks.mdp.naming import resolve_tip_body_names  # noqa: PLC0415

            tip_names = resolve_tip_body_names(env.scene[f"{side}_hand"], side)  # TIP_FINGERS order
            missing = [n for n in tip_names if n not in sensor.body_names]
            if missing:
                raise ValueError(f"{side}_hand_contact has no body {missing} (sensor bodies {sensor.body_names})")
            setattr(self, cache_key, [sensor.body_names.index(n) for n in tip_names])
        tip_id = getattr(self, cache_key)[TIP_FINGERS.index(finger)]
        force_xyz = sensor.data.net_forces_w[:, tip_id, :]
        return torch.norm(force_xyz, dim=-1) > force_threshold

    def _body_group_reward(self, env: ManagerBasedEnv, body_indices: list[int], k: float) -> torch.Tensor:
        """Compute mean position tracking reward over a group of bodies.

        Args:
            env: The environment instance.
            body_indices: List of MuJoCo FK body indices.
            k: Exponential kernel sharpness.

        Returns:
            Reward tensor of shape (num_envs,).
        """
        ref_body = self._traj_mgr.get_reference_body_pos(env.episode_length_buf)
        origins = env.scene.env_origins

        rh = env.scene["right_hand"]
        lh = env.scene["left_hand"]

        # Right hand: average L2 distance across body group
        rh_isaac_ids = self._rh_body_ids[body_indices]
        rh_cur = rh.data.body_link_pos_w[:, rh_isaac_ids, :] - origins.unsqueeze(1)
        rh_ref = ref_body["right_body_pos"][:, body_indices, :]
        rh_dist = torch.norm(rh_ref - rh_cur, dim=-1).mean(dim=-1)

        # Left hand
        lh_isaac_ids = self._lh_body_ids[body_indices]
        lh_cur = lh.data.body_link_pos_w[:, lh_isaac_ids, :] - origins.unsqueeze(1)
        lh_ref = ref_body["left_body_pos"][:, body_indices, :]
        lh_dist = torch.norm(lh_ref - lh_cur, dim=-1).mean(dim=-1)

        return self._combine(torch.exp(-k * rh_dist), torch.exp(-k * lh_dist))
