"""Trajectory manager for Taco bimanual training.

Wraps TacoTrajectoryLoader and provides reference trajectory data
to observation, reward, and event MDP terms. Also loads pre-computed
body positions from MuJoCo FK for ManipTrans-style observations.
"""

from __future__ import annotations

import os

import torch

from isaaclab.assets import Articulation

from dexengine.rl.tasks.mdp.trajectory.bodies import BodyPositionMixin
from dexengine.rl.tasks.mdp.trajectory.bodies_load import BodyLoadMixin
from dexengine.rl.tasks.mdp.trajectory.clock import ClipClockMixin

# Import the existing trajectory loader (Rule 1: no edits to replay code)
# The loader lives in loader.py (the old taco_replay task is gone).
from dexengine.rl.tasks.mdp.trajectory.loader import TacoTrajectoryLoader

# SharpA Wave: 28 qpos per hand (6 wrist + 22 fully-actuated finger joints).
# Joint names are identical between the rig MJCF (dataset) and the merged
# full-robot URDF, so dataset order == vendor order. Suffix table lives in the
# naming seam.
from dexengine.rl.tasks.mdp.trajectory.manager_cfg import (
    DATASET_LEFT_JOINT_ORDER,
    DATASET_RIGHT_JOINT_ORDER,
    ROOT_QUAT_WXYZ,
    SHARPA_LEFT_FINGER_ORDER,
    SHARPA_LEFT_JOINT_ORDER,
    SHARPA_RIGHT_FINGER_ORDER,
    SHARPA_RIGHT_JOINT_ORDER,
    TrajectoryManagerCfg,
    ref_table_requested,
    virtual_wrist_order,
)
from dexengine.rl.tasks.mdp.trajectory.reference import ReferenceStateMixin
from dexengine.rl.tasks.mdp.trajectory.wrist import WristPoseMixin


class TrajectoryManager(WristPoseMixin, BodyPositionMixin, BodyLoadMixin, ClipClockMixin, ReferenceStateMixin):
    """Provides reference trajectory data for training MDP terms.

    This wraps TacoTrajectoryLoader and adds:
    - Per-env clip assignment and tracking
    - Dataset->PhysX joint permutation
    - Pre-computed body position loading for ManipTrans-style observations
    - Convenient accessors for MDP functions
    """

    def __init__(
        self,
        cfg: TrajectoryManagerCfg,
        device: str | torch.device,
        num_envs: int,
        right_hand: Articulation,
        left_hand: Articulation,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = num_envs

        self._load_trajectories(cfg, num_envs, right_hand, left_hand)
        self.schedule_progress = (
            1.0  # linear threshold schedule: 0=start(loose), 1=end(final/tight); default 1.0 = final
        )

        # Set up coordinate rotation (+Y forward → +X forward)
        self._rotate = cfg.rotate_to_x_forward
        self._desk_height = cfg.desk_height
        self._init_root_rotation_and_corrections(cfg)

    def _load_trajectories(self, cfg, num_envs, right_hand, left_hand):
        """Load trajectories using the existing loader (Rule 1: reuse, not rewrite)."""
        self._loader = TacoTrajectoryLoader(
            data_dir=cfg.data_dir,
            device=self.device,
            subject_filter=cfg.subject_filter,
            scene_filter=cfg.scene_filter,
            sequence_filter=cfg.sequence_filter,
            hand_qpos_dim=getattr(cfg, "hand_qpos_dim", 18),
            world_z_offset=float(
                cfg.desk_height
            ),  # rule D20: the loader pre-applies the aerial trees' per-clip z shift as the difference to this constant
            expect_idle_side=str(getattr(cfg, "idle_side", "") or ""),
            n_extra_slots=int(getattr(cfg, "n_extra_slots", 0) or 0),
            extra_slot_cols=getattr(cfg, "extra_slot_cols", None),
            extra_slot_static=getattr(cfg, "extra_slot_static", None),
        )
        self._hand_qpos_dim = int(getattr(cfg, "hand_qpos_dim", 18))
        # The dataset order and coupling of the finger joints are vendor facts of that hand, so the spec brings them.
        # Taken once here, whatever the mode.
        self._dataset_finger_names = tuple(getattr(cfg, "dataset_finger_joint_names", ()) or ())
        self._dataset_coupled_names = tuple(getattr(cfg, "dataset_coupled_finger_names", ()) or ())
        self._fk_body_aliases = tuple(getattr(cfg, "fk_body_aliases", ()) or ())
        self._fk_tip_body_names = tuple(getattr(cfg, "fk_tip_body_names", ()) or ())
        self._fk_palm_isaac_right = str(getattr(cfg, "fk_palm_isaac_body_right", "") or "")
        self._fk_palm_isaac_left = str(getattr(cfg, "fk_palm_isaac_body_left", "") or "")
        if self._dataset_finger_names:
            _expect = 6 + len(self._dataset_finger_names)
            if _expect != self._hand_qpos_dim:
                raise ValueError(
                    f"hand_qpos_dim={self._hand_qpos_dim} but the spec gives "
                    f"{len(self._dataset_finger_names)} fingers (6+{len(self._dataset_finger_names)}"
                    f"={_expect}). Both are the same fact of the same hand; a mismatch shifts every data "
                    "slice."
                )

        # Build trajectory->PhysX permutation from live joint names (Rule 4)
        # _build_permutation returns (perm, physx_hand_ids) — gracefully
        # skips non-hand joints (arm, head) when full robot is passed.
        # Spec order first (virtual wrist + fingers, then fingers-only for the full
        # robot where virtual wrist joints don't exist); legacy name probes otherwise.
        self._fingers_only = False
        if self._dataset_finger_names:
            _spec_full = virtual_wrist_order("right") + list(self._dataset_finger_names)
            _candidates = [(_spec_full, "spec"), (list(self._dataset_finger_names), "spec_fingers")]
        else:
            _candidates = [
                (DATASET_RIGHT_JOINT_ORDER, "dataset"),
                (SHARPA_RIGHT_JOINT_ORDER, "sharpa"),
                (SHARPA_RIGHT_FINGER_ORDER, "sharpa_fingers"),
            ]
        for order, convention in _candidates:
            try:
                self._right_perm, self._right_hand_ids = self._build_permutation(right_hand.data.joint_names, order)
                self._joint_convention = convention
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot match joints. PhysX has: {right_hand.data.joint_names}")

        if self._joint_convention.endswith("_fingers"):
            self._fingers_only = True

        self._match_left_hand_bodies(cfg, num_envs, right_hand, left_hand)
        # RSI band (fraction of clip length): RSI start frames are sampled from
        # [rsi_lo_frac, rsi_hi_frac] * (clip_len-1). Defaults (0,1) = legacy uniform-over-clip.
        # A reverse curriculum (RsiSchedule) anneals the band toward 0 over training.
        self.rsi_lo_frac = 0.0
        self.rsi_hi_frac = 1.0

    def _match_left_hand_bodies(self, cfg, num_envs, right_hand, left_hand):
        """Left hand: try matching order, fallback to right."""
        _left_fingers = [n.replace("right_", "left_", 1) for n in self._dataset_finger_names]
        left_orders = {
            "spec": virtual_wrist_order("left") + _left_fingers,
            "spec_fingers": _left_fingers,
            "dataset": DATASET_LEFT_JOINT_ORDER,
            "sharpa": SHARPA_LEFT_JOINT_ORDER,
            "sharpa_fingers": SHARPA_LEFT_FINGER_ORDER,
        }
        try:
            self._left_perm, self._left_hand_ids = self._build_permutation(
                left_hand.data.joint_names, left_orders[self._joint_convention]
            )
        except ValueError:
            # Mixed-hand robot (right and left hands differ): retry the
            # left hand against the other conventions before the placeholder
            # right-perm fallback. The dataset left-hand qpos belongs to the
            # RIGHT hand's embodiment in that case, so the left reference is
            # meaningless — zero it (open pose) instead of writing garbage.
            self._left_perm = None
            for _order in (DATASET_LEFT_JOINT_ORDER, SHARPA_LEFT_JOINT_ORDER, SHARPA_LEFT_FINGER_ORDER):
                try:
                    self._left_perm, self._left_hand_ids = self._build_permutation(left_hand.data.joint_names, _order)
                    self._left_ref_zero = True
                    print(
                        "[TrajectoryManager] Left hand embodiment differs from "
                        "dataset — left reference qpos zeroed (open pose)."
                    )
                    break
                except ValueError:
                    continue
            if self._left_perm is None:
                self._left_perm = self._right_perm
                self._left_hand_ids = self._right_hand_ids

        print(f"[TrajectoryManager] Joint convention: {self._joint_convention}")
        print(f"[TrajectoryManager] Right perm: {self._right_perm.tolist()}")
        print(f"[TrajectoryManager] Left perm: {self._left_perm.tolist()}")

        # Load body positions and build body name permutations (Rule 4: runtime)
        self._load_body_positions(cfg.data_dir, right_hand.data.body_names, left_hand.data.body_names)

        # Left-hand reference validity (False = zero the left ref qpos; set
        # above when the left hand embodiment differs from the dataset hand).
        if not hasattr(self, "_left_ref_zero"):
            self._left_ref_zero = False

        # Per-env state
        self.clip_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.clip_lengths = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._static_clips = bool(getattr(cfg, "static_clip_assignment", False))
        if self._static_clips:
            self.clip_ids = torch.arange(num_envs, device=self.device) % self._loader.num_clips
            self.clip_lengths = self._loader.get_clip_lengths(self.clip_ids)
            print(
                f"[TrajectoryManager] STATIC clip assignment: env_i -> clip i % "
                f"{self._loader.num_clips} (multi-object; no per-reset resampling)"
            )

        # RSI (Reference State Initialization) state
        self.start_frames = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # PER-SIDE CLOCK (reset_scheme=decoupled): when enabled each hand
        # tracks its OWN clip segment — side i's absolute frame is
        # floor((elen - _side_birth[:,i]) / slowdown) + _side_start[:,i]. Default OFF:
        # every mapping below reduces to the shared start_frames timeline, byte-identical.
        self.per_side_clock = False
        self._side_start: torch.Tensor | None = None  # (N,2) long [right,left] RSI starts
        self._side_birth: torch.Tensor | None = None  # (N,2) long elen at segment birth
        self.rsi_fraction = 0.0  # Probability of starting from random frame (0=always frame 0)

    def _init_root_rotation_and_corrections(self, cfg):
        """Root rotation (hand USD source, Rz(-90)), desk z offset and the initial pose corrections."""
        self._urdf_hand = cfg.urdf_hand

        self._base_root_quaternion(cfg)
        if cfg.wrist_poses_dir_right is not None:
            self._wrist_poses, self._wrist_quats = self._load_wrist_poses(cfg.wrist_poses_dir_right, cfg.traj_offset_z)
            # Bimanual: load the LEFT-hand wrist reference table. Explicit dir if given,
            # else the sibling "<dir>_left" by convention (matches arm_ik_poses_left/
            # contact_*_left). Loaded only if present; the left table is queried solely
            # by get_wrist_*(side='left') from the left ResidualArmIKAction, so it stays
            # inert in single-hand mode (no left action term => never read).
            _left_dir = cfg.wrist_poses_dir_left
            if _left_dir is None and cfg.wrist_poses_dir_right.rstrip("/").endswith("_right"):
                _cand = cfg.wrist_poses_dir_right.rstrip("/")[: -len("_right")] + "_left"
                if os.path.isdir(_cand):
                    _left_dir = _cand
            if _left_dir is not None and os.path.isdir(_left_dir):
                self._left_wrist_poses, self._left_wrist_quats = self._load_wrist_poses(_left_dir, cfg.traj_offset_z)
                print(
                    f"[TrajectoryManager] LEFT wrist poses loaded from {_left_dir}: "
                    f"{tuple(self._left_wrist_poses.shape)}"
                )

        # Phase 1 contact-aware lag compensation:
        # Slow-down factor for runtime trajectory time-warp.
        self._slowdown_factor = float(getattr(cfg, "slowdown_factor", 1.0))
        if self._slowdown_factor != 1.0:
            print(f"[TrajectoryManager] Runtime slow-down enabled: factor={self._slowdown_factor:.3f}")
            assert self._slowdown_factor > 0, "slowdown_factor must be > 0"

        # Stats
        print(f"[TrajectoryManager] Loaded {self._loader.num_clips} clips, max_length={self._loader.max_length}")
        if self._wrist_poses is not None:
            print(f"[TrajectoryManager] Wrist poses loaded: {self._wrist_poses.shape}")
        if self._urdf_hand:
            print("[TrajectoryManager] URDF hand: analytical dataset->URDF-hand wrist transform enabled")
        if ref_table_requested():
            self._build_ref_table()

    def _base_root_quaternion(self, cfg):
        """Base root quaternion depends on hand USD source: - MJCF-converted (dataset): y_180_x_m90 cancels baked
        converter rotation - URDF-converted: identity (no baked rotation).
        """
        if cfg.urdf_hand:
            self._base_root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            print("[TrajectoryManager] URDF hand: base root quat = identity (no baked rotation)")
        else:
            self._base_root_quat = torch.tensor(ROOT_QUAT_WXYZ.tolist(), device=self.device, dtype=torch.float32)

        if self._rotate:
            from isaaclab.utils.math import quat_mul as _quat_mul

            # Rz(-90°) quaternion in wxyz
            rz_m90_wxyz = torch.tensor([0.7071068, 0.0, 0.0, -0.7071068], device=self.device)
            # Rotated root orientation: Rz(-90°) * base_root_quat
            self.root_quat_rotated = _quat_mul(rz_m90_wxyz.unsqueeze(0), self._base_root_quat.unsqueeze(0)).squeeze(
                0
            )  # (4,) wxyz
            # Rz(-90°) rotation matrix for positions: (x,y,z) → (y,-x,z)
            self._rz_mat = torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=self.device,
                dtype=torch.float32,
            )
            self._rz_quat = rz_m90_wxyz
            self._z_offset = torch.tensor([0.0, 0.0, self._desk_height], device=self.device)
            print(
                f"[TrajectoryManager] Rotate to +X forward: root_quat={self.root_quat_rotated.tolist()}, "
                f"desk_height={self._desk_height}m"
            )
        else:
            self.root_quat_rotated = None

        # Load initial pose corrections (from penetration optimization).
        # initial_pose_corrections.json was an artifact of
        # the legacy hand only (deleted with its arm) — no registered hand can
        # use it. Asking for it dies instead of being silently switched off; before, every
        # other hand printed a warning and ran without it.
        if cfg.pose_corrections:
            raise ValueError(
                f"[TrajectoryManager] pose_corrections=True for hand_name={cfg.hand_name!r}: the "
                "corrections file was a legacy-hand artifact and no registered hand supports it."
            )
        self._use_pose_corrections = False
        if self._use_pose_corrections:
            self._pose_corrections = self._load_pose_corrections(cfg.data_dir)
        else:
            self._pose_corrections = None

        # Load pre-dumped wrist poses for the arm IK (optional)
        self._wrist_poses = None
        self._wrist_quats = None
        self._left_wrist_poses = None
        self._left_wrist_quats = None

    def rsi_adapt_report(self) -> str:
        """Kept for rl/train/runner.py, which prints it after each eval; the adaptive RSI sampler was removed."""
        return "rsi_adaptive=off"

    # ------------------------------------------------------------------
    # Reference-state precomputed table (DEXENGINE_REF_TABLE, canonical default on; 0 = the old cache path).
    # The reference state is a pure function of (clip, absolute frame): the loader table (qpos_all) is static, pose
    # corrections fail when on (always off), and the wrist transform, joint permutation, Rz(−90°) and desk height are
    # init constants. So init computes every (clip, frame) once **with the existing functions** into a table, and a run
    # gets the same value with one gather (index_select). Index = clip × T + clamp(frame, 0, T−1), the loader's rule.
    # The table is used only after a full comparison with the old path (REF_TABLE_ATOL). Off = the old path as is.
    # Difference when on: without a cache the side stash (_last_*) refreshes on every call (the old path skipped it on
    # cache hits; its only readers are the non-fullrobot observation fallback and the boot smoke, never canonical
    # rby1). Values differ by 1 float32 ulp (shadow comparison over 11,262 training rollout calls, max|Δ| 1.19e-7, a
    # kernel difference between pieces with other row counts and orders). The get_reference_body_pos cache is outside.
    # ------------------------------------------------------------------
    _ref_table: dict | None = None

    # ------------------------------------------------------------------
    # Per-step caching for get_reference_body_pos / get_reference_state
    # These are called many times per step (by 6+ reward terms and
    # observation/termination terms). Caching avoids redundant tensor
    # indexing on large pre-loaded arrays.
    # ------------------------------------------------------------------
    _cache_step: int = -1
    _cache_body_pos: dict | None = None
    _cache_state: dict | None = None
    _cache_state_key: int = -1
    _cache_state_p1: dict | None = None
    _cache_state_p1_key: int = -1
    # Per cache slot, the real input that made it (:meth:`_ref_input_sig`): an equal sum key with other inputs is not a
    # hit
    _cache_body_sig: torch.Tensor | None = None
    _cache_state_sig: torch.Tensor | None = None
    _cache_state_p1_sig: torch.Tensor | None = None
    _ref_cache_collisions: int = 0

    @property
    def num_clips(self) -> int:
        return self._loader.num_clips

    @property
    def max_length(self) -> int:
        return self._loader.max_length
