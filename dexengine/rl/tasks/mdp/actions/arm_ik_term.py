"""The arm-IK fingers action term: wrist pose deltas through differential IK, absolute finger targets."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm

from dexengine.rl.tasks import env_flags
from dexengine.rl.tasks.mdp.naming import resolve_ee_to_wrist_rot, resolve_link7_to_wrist_offset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from typing import TYPE_CHECKING as _TC

from dexengine.rl.tasks.mdp.actions.arm_ik_kinematics import ArmKinematicsMixin
from dexengine.rl.tasks.mdp.actions.arm_ik_process import ArmIKProcessMixin

if _TC:
    from dexengine.rl.tasks.mdp.actions.arm_ik import ArmIKFingersPositionActionCfg


logger = logging.getLogger("dexengine.rl.tasks.mdp.actions.arm_ik")  # the name it had before the split


def check_ema_weight(name: str, w) -> float:
    """An EMA weight on the new value must lie in (0, 1]. 1.0 turns the filter off; 0 is not "off" — it keeps the
    previous value forever, so a filter meant to be off would hold the wrist target at the reset pose for the whole run.
    Any value outside (0, 1] dies here, when the action term is built."""
    w = float(w)
    if not 0.0 < w <= 1.0:
        raise ValueError(
            f"{name}={w} is outside (0, 1]: 1.0 turns the filter off, 0 freezes the target at its first value"
        )
    return w


class ArmIKFingersPositionAction(ArmKinematicsMixin, ArmIKProcessMixin, ActionTerm):
    """Arm IK from persistent wrist target + policy delta, plus finger PD from RL policy.

    Action dim = 12 (6 wrist deltas + 6 finger positions in [-1, 1]).
    The arm wrist target is computed as prev_wrist_target + policy_delta,
    where prev_wrist_target is initialized from actual link7 state on reset
    and updated purely by policy deltas (no per-step state reading).

    Pipeline each step:

    1. ``process_actions``: receive 12-dim actions from RL policy.
       - EMA-smooth actions.
       - Compute wrist_qpos_target = prev_target + scaled_delta (no state-chasing).
       - Optionally apply IK target EMA for additional smoothing.
       - Update persistent prev_wrist_target.
       - Convert wrist world pose -> link7 world target (apply fixed offset).
       - Transform link7 target to robot root frame.
       - Feed absolute pose command to DifferentialIK controller.

    2. ``apply_actions``: compute and write joint targets.
       - Run DifferentialIK to get 7 arm joint position targets.
       - Scale 6 finger actions [-1,1] -> joint limits.
       - Write arm + finger position targets to articulation.
    """

    cfg: ArmIKFingersPositionActionCfg
    """The configuration of the action term."""

    #: Arm joint count this term expects; the floating hand (``experiments.ablations.floating_hand.action``) overrides
    #: it with 0. One owner:
    # the class.
    EXPECTED_ARM_JOINTS = 7

    def __init__(self, cfg: ArmIKFingersPositionActionCfg, env: ManagerBasedEnv) -> None:
        # Initialize the base action term (sets self._asset, self.device, etc.)
        super().__init__(cfg, env)

        self._resolve_arm_joints(cfg)
        # Source indices: which independent joint drives each coupled joint
        # Legacy: thumb_3 ← thumb_2 (idx 1), thumb_4 ← thumb_2 (idx 1),
        # index_2 ← index_1 (idx 2), middle_2 ← middle_1 (idx 3),
        # ring_2 ← ring_1 (idx 4), little_2 ← little_1 (idx 5)
        if cfg.coupling_source_idx is None:
            self._coupling_source_idx = [1, 1, 2, 3, 4, 5]
        else:
            self._coupling_source_idx = list(cfg.coupling_source_idx)

        body_names = self._resolve_ee_body(cfg, env)
        self._prev_wrist_quat_local[:, 0] = 1.0  # identity quaternion (wxyz)
        self._ik_target_ema = check_ema_weight("ik_target_ema", cfg.ik_target_ema)

        # -- Action buffers (6 wrist delta + n finger) --
        _A = self._action_dim
        self._raw_actions = torch.zeros(self.num_envs, _A, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, _A, device=self.device)
        self._prev_actions = torch.zeros(self.num_envs, _A, device=self.device)
        self._ema = check_ema_weight("action_ema", cfg.action_ema)
        self._wrist_pos_scale = cfg.wrist_pos_scale
        self._wrist_rot_scale = cfg.wrist_rot_scale

        # -- Temporal ensembling (EgoVLA-style exp-weighted action history) --
        self._wrist_smooth_weight = cfg.wrist_smooth_weight
        self._finger_smooth_weight = cfg.finger_smooth_weight
        self._use_temporal_ensembling = cfg.wrist_smooth_weight >= 0 or cfg.finger_smooth_weight >= 0
        self._init_smoothing_and_reference(cfg, env, _A, body_names)

    def _resolve_ee_body(self, cfg, env):
        """Resolve EE body."""
        body_ids, body_names = self._asset.find_bodies(cfg.ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one match for EE body '{cfg.ee_body_name}', got {len(body_ids)}: {body_names}")
        self._ee_body_idx = body_ids[0]

        # Jacobian body index: fixed-base robots have one fewer row in Jacobian
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
        else:
            self._jacobi_body_idx = self._ee_body_idx

        # -- Create DifferentialIK controller --
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": cfg.ik_lambda_val},
        )
        self._ik_controller = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # -- IK command buffer (7: pos_xyz + quat_wxyz in root frame) --
        self._ik_commands = torch.zeros(self.num_envs, 7, device=self.device)

        # -- link7 -> hand_base mounting offset constants (GPU tensors, computed once) --
        # Translation (0,0,0.0555 RH56F1 / 0.0655 SharpA) is the same on both arms.
        # ROTATION is SIDE-AWARE: the right arm mounts the hand at Rz(-90deg), the
        # mirrored LEFT arm at Rz(+90deg) (measured
        # against the bimanual URDF: right rot_err 0deg, left 180deg vs the old single
        # Rz(-90) constant). The old code hardcoded the right's Rz(-90) for BOTH hands,
        # flipping the LEFT wrist orientation IK target 180deg -> the left hand tracked a
        # wrong-axis pose in eval/replay. fwd: hand_base_quat = link7_quat (x) fwd;
        # inv (wrist->link7) = conjugate(fwd).
        # The rotation is pair-aware like the translation: the rby1 mount adds roll 180° (sharpa) / ±170.5° (wuji) on
        # top of the first arm's constant. Legacy hands keep that constant.
        _r_fwd = list(resolve_ee_to_wrist_rot(self._asset, self._side))
        self._r_offset_fwd_wxyz = torch.tensor(_r_fwd, device=self.device)
        self._r_offset_inv_wxyz = torch.tensor([_r_fwd[0], -_r_fwd[1], -_r_fwd[2], -_r_fwd[3]], device=self.device)
        # The translation fallback is **asset-based and pair-aware** like the rotation above. The cfg used to set this
        # value only in the hand_spec branch, so rby1 + a legacy hand (hand_spec=None) fell back to the legacy arm
        # literal +0.0555 where MOUNT_REGISTRY says -0.1321: an 18.8 cm error applied back and forth by the IK while
        # reset, obs and rewards were pair-consistent (the worst silent defect: only learning fails). (The factory
        # always passes the cfg value now; the spec is required.)
        _t_off = (
            cfg.link7_to_wrist_offset
            if cfg.link7_to_wrist_offset is not None
            else resolve_link7_to_wrist_offset(self._asset, self._side)
        )
        self._t_offset = torch.tensor(list(_t_off), device=self.device)

        # -- Floating hand root pose (for computing wrist world pose from trajectory qpos) --
        # In training, the floating hand root is at env_origins + (0,0,desk_height)
        # with orientation Rz(-90°). We use this to convert trajectory qpos to world pose.
        traj_mgr = env.trajectory_manager
        if hasattr(traj_mgr, "root_quat_rotated") and traj_mgr.root_quat_rotated is not None:
            self._floating_root_quat = traj_mgr.root_quat_rotated.clone()
        else:
            self._floating_root_quat = torch.tensor([0.7071068, 0.0, 0.0, -0.7071068], device=self.device)
        self._floating_root_z_offset = getattr(traj_mgr, "_desk_height", 0.0)

        # link7 -> virtual wrist translation in link7 frame (same value as _t_offset).
        # (_r_offset_fwd_wxyz / _r_offset_inv_wxyz are set above, side-aware.)
        self._link7_to_wrist_trans = self._t_offset.clone()

        # -- Arm IK target buffer (for stable IK: apply delta to previous target, not actual) --
        self._prev_arm_targets = torch.zeros(self.num_envs, self._num_arm_joints, device=self.device)

        # -- Persistent wrist target for IK target EMA (pos + quat, no euler) --
        self._prev_wrist_pos_local = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_wrist_quat_local = torch.zeros(self.num_envs, 4, device=self.device)
        return body_names

    def _resolve_arm_joints(self, cfg):
        """Resolve arm joints."""
        self._arm_joint_ids, self._arm_joint_names = self._asset.find_joints(cfg.arm_joint_names)
        self._num_arm_joints = len(self._arm_joint_ids)
        if self._num_arm_joints != self.EXPECTED_ARM_JOINTS:
            raise ValueError(
                f"Expected {self.EXPECTED_ARM_JOINTS} arm joints, got {self._num_arm_joints}: {self._arm_joint_names}"
            )

        # -- Resolve finger joints (independent; count from cfg) --
        # preserve_order=True ensures the resolved IDs match the config order:
        # [thumb_1, thumb_2, index_1, middle_1, ring_1, little_1] for RH56F1,
        # the 22 vendor-order joints for SharpA.
        # Without this, find_joints returns PhysX order (alphabetical) which
        # mismatches the policy's action order.
        self._finger_joint_ids, self._finger_joint_names = self._asset.find_joints(
            cfg.finger_joint_names, preserve_order=True
        )
        self._num_finger_joints = len(self._finger_joint_ids)
        if self._num_finger_joints != len(cfg.finger_joint_names):
            raise ValueError(
                f"Expected {len(cfg.finger_joint_names)} finger joints, "
                f"got {self._num_finger_joints}: {self._finger_joint_names}"
            )
        # Policy action dimension: 6 wrist deltas + n independent finger targets.
        self._action_dim = 6 + self._num_finger_joints

        # -- Resolve coupled finger joints (dependent) for explicit coupling --
        # Full robot USD may not have PhysxMimicJointAPI, so we handle coupling here.
        # cfg.coupled_joint_names=None → legacy RH56F1 auto-list (drift invariant);
        # []   → no coupling (fully-actuated hand, e.g. SharpA).
        side = "right" if "right" in cfg.finger_joint_names[0] else "left"
        self._side = side  # used below for the side-aware link7->hand_base mounting rotation
        if cfg.coupled_joint_names is None:
            coupled_names = [
                f"{side}_hand_thumb_3_joint",
                f"{side}_hand_thumb_4_joint",
                f"{side}_hand_index_2_joint",
                f"{side}_hand_middle_2_joint",
                f"{side}_hand_ring_2_joint",
                f"{side}_hand_little_2_joint",
            ]
        else:
            coupled_names = list(cfg.coupled_joint_names)
        if coupled_names:
            try:
                self._coupled_joint_ids, _ = self._asset.find_joints(coupled_names, preserve_order=True)
                self._has_coupling = True
            except Exception:  # noqa: BLE001 — a hand without coupled joints runs uncoupled
                self._coupled_joint_ids = []
                self._has_coupling = False
        else:
            self._coupled_joint_ids = []
            self._has_coupling = False
        self._num_coupled_joints = len(coupled_names)

        # Coupling expansion: independent → coupled physical joints.
        # Legacy RH56F1 (cfg None): thumb_3 = 1.2953 * thumb_2,
        # thumb_4 = 0.8962 * thumb_3, {index,middle,ring,little}_2 = 1.1545 * {}_1
        if cfg.coupling_ratios is None:
            _ratios = [
                1.2953,  # thumb_3 = ratio * thumb_2
                0.8962 * 1.2953,  # thumb_4 = ratio * thumb_2 (chained)
                1.1545,  # index_2 = ratio * index_1
                1.1545,  # middle_2 = ratio * middle_1
                1.1545,  # ring_2 = ratio * ring_1
                1.1545,  # little_2 = ratio * little_1
            ]
        else:
            _ratios = list(cfg.coupling_ratios)
        self._coupling_ratios = torch.tensor(_ratios, device=self.device)

    def _init_smoothing_and_reference(self, cfg, env, _A, body_names):
        """Action smoothing history and weights, the trajectory reference mode and the finger limit scaling."""
        if self._use_temporal_ensembling:
            self._smooth_history_len = cfg.smooth_history_len
            # Per-env action history: (history_len, num_envs, action_dim)
            # Using a tensor ring buffer instead of deque for GPU efficiency
            self._action_history = torch.zeros(cfg.smooth_history_len, self.num_envs, _A, device=self.device)
            self._history_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            # Pre-compute weights for wrist and finger dims
            indices = torch.arange(cfg.smooth_history_len, device=self.device, dtype=torch.float32)
            w_wrist = torch.exp(-max(cfg.wrist_smooth_weight, 0.0) * indices)  # (H,)
            w_finger = torch.exp(-max(cfg.finger_smooth_weight, 0.0) * indices)  # (H,)
            # Combine into (H, action_dim) weight matrix
            self._smooth_weights = torch.zeros(cfg.smooth_history_len, _A, device=self.device)
            self._smooth_weights[:, :6] = w_wrist.unsqueeze(1).expand(-1, 6)
            self._smooth_weights[:, 6:] = w_finger.unsqueeze(1).expand(-1, self._num_finger_joints)

        # -- Trajectory reference mode --
        self._use_traj_ref = cfg.use_trajectory_reference
        if self._use_traj_ref:
            self._traj_mgr = env.trajectory_manager
            print("[ArmIKFingersPositionAction] Using trajectory reference for wrist target")

        self._resolve_finger_joint_limits(cfg, body_names)
        if self._park:
            print(f"[ArmIKFingersPositionAction:{self._side}] RIGHT-PARK ACTIVE (default-pose PD hold)")

    def _resolve_finger_joint_limits(self, cfg, body_names):
        """Finger joint limits for ManipTrans-style scaling scale(a, lo, hi) = 0.5 * (a + 1) * (hi - lo) + lo."""
        joint_pos_limits = self._asset.data.soft_joint_pos_limits[0]  # (num_joints, 2)
        self._finger_lower = joint_pos_limits[self._finger_joint_ids, 0]  # (6,)
        self._finger_upper = joint_pos_limits[self._finger_joint_ids, 1]  # (6,)
        # ── Finger ROM floor (wuji hyperextension experiment) ──────────────────────
        # Why: the wuji vendor URDF lets the flexion joints (joint1/3/4) hyperextend −27°..−9° (scene vs URDF 0/20
        # mismatches, so the conversion is innocent; it is the spec itself). Human PIP/DIP stop near 0°, so as a direct
        # test of "fingertip alignment alone gives unnatural poses", only the lower bound of the action mapping is
        # floored (no USD rebuild). The residual REFERENCE-BASE normalisation uses the same _finger_lower/_range, so
        # the base clamps too (where the reference asks for poses outside the ROM it is deliberately not followed;
        # that is the experiment). The exclude regex skips the abduction joints (±21°, the same as a human hand).
        _romf = getattr(cfg, "finger_rom_floor", None)
        if _romf is not None:
            import re as _re

            _ex = getattr(cfg, "finger_rom_floor_exclude", None)
            if not _ex:
                # The old default was wuji joint names (finger[2-5]_joint2): on another hand it matched nothing and
                # silently cut the abduction joints' negative bound too. The exclude set is owned by
                # HandSpec.abduction_joint_regex (the action contract fills it).
                raise ValueError(
                    "finger_rom_floor is on but the abduction joints to skip (finger_rom_floor_exclude) are "
                    "empty — declare HandSpec.abduction_joint_regex (the action contract fills it)"
                )
            _mask = torch.tensor(
                [not _re.fullmatch(_ex, n) for n in self._finger_joint_names],
                device=self._finger_lower.device,
                dtype=torch.bool,
            )
            _floored = torch.where(_mask, torch.clamp(self._finger_lower, min=float(_romf)), self._finger_lower)
            _n = int((_floored > self._finger_lower + 1e-9).sum().item())
            self._finger_lower = _floored
            print(
                f"[ArmIKFingersPositionAction][W1b] finger ROM floor {_romf} rad applied — "
                f"{_n}/{self._num_finger_joints} joint lower bounds raised (exclude regex {_ex})"
            )
        self._finger_range = self._finger_upper - self._finger_lower  # (6,)

        # -- Debug output --
        logger.info(f"[ArmIKFingersPositionAction] Arm joints ({self._num_arm_joints}): {self._arm_joint_names}")
        logger.info(
            f"[ArmIKFingersPositionAction] Finger joints ({self._num_finger_joints}): {self._finger_joint_names}"
        )
        logger.info(f"[ArmIKFingersPositionAction] EE body: '{body_names[0]}' (idx={self._ee_body_idx})")
        logger.info(f"[ArmIKFingersPositionAction] IK lambda: {cfg.ik_lambda_val}, EMA: {cfg.action_ema}")
        traj_ref_str = "trajectory_ref" if self._use_traj_ref else "current_state"
        ensemble_str = (
            (f"wrist={cfg.wrist_smooth_weight}, finger={cfg.finger_smooth_weight}, history={cfg.smooth_history_len}")
            if self._use_temporal_ensembling
            else "off"
        )
        print(
            f"[ArmIKFingersPositionAction] action_dim={self._action_dim} "
            f"(6 wrist + {self._num_finger_joints} fingers), "
            f"arm={self._num_arm_joints} joints (IK-driven), "
            f"EE='{body_names[0]}', lambda={cfg.ik_lambda_val}, "
            f"target={traj_ref_str}, ensemble={ensemble_str}, "
            f"action_ema={cfg.action_ema}, ik_target_ema={cfg.ik_target_ema}"
        )
        print(f"[ArmIKFingersPositionAction] Finger limits lower: {self._finger_lower}")
        print(f"[ArmIKFingersPositionAction] Finger limits upper: {self._finger_upper}")

        # LEFT-UNI: DEXENGINE_RIGHT_PARK=1 — the RIGHT arm+fingers hold their
        # DEFAULT pose all episode (policy + GT reference both ignored), mirroring the uni
        # scene's parked left arm (PD target pin at default). Reset-side parking lives in
        # mdp/events (reset_arm_to_trajectory / reset_hands_to_trajectory). Left side and
        # unset env -> byte-identical.
        self._park = self._side == "right" and env_flags.get("DEXENGINE_RIGHT_PARK") == "1"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        """Policy action dimension: 6 wrist delta + n finger positions.

        12 for RH56F1 (6 fingers), 28 for SharpA (22 fingers). Per-hand layout:
          action[0:3] = wrist position delta (scaled by wrist_pos_scale)
          action[3:6] = wrist rotation delta (scaled by wrist_rot_scale)
          action[6:]  = finger positions in [-1, 1]
        """
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
