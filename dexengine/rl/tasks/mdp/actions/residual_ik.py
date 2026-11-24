"""Residual arm IK action for reference-base object manipulation.

Single-stage pipeline: the base action is the KINEMATIC REFERENCE (wrist = reference pose
via use_trajectory_reference; fingers = reference qpos), computed by
``compute_reference_base_action()``. The SKRL PPO policy outputs 12-dim residual
corrections on top of that reference base. (The legacy frozen-base-model path is
gone.)

Base and residual are processed separately through proper rotation composition:

  1. Base action (kinematic reference) → scale → euler_to_quat → delta
  2. Residual action (PPO) → scale * rs → euler_to_quat → delta
  3. Combine on current wrist pose:
     - Position: current + base_delta + residual_delta  (linear)
     - Rotation: current ⊗ base_quat ⊗ residual_quat  (composition)
  4. Fingers: (base + residual*rs) → ManipTrans scaling  (affine, order-safe)

Action dimension is 12 (residual only - what SKRL PPO sees).
The base action is likewise 12-dim (same layout as the residual).
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dexengine.rl.tasks.mdp.actions.residual_ik_term import ResidualArmIKAction


@configclass
class ResidualArmIKActionCfg(ActionTermCfg):
    """Configuration for ResidualArmIKAction.

    12-dim residual action combined with the kinematic reference base.
    The combined action is fed to an internal ArmIKFingersPositionAction
    which handles IK-driven arm control and finger position control.

    Arm IK config and wrist scales are passed through to the inner action term.
    """

    class_type: type[ActionTerm] = ResidualArmIKAction

    # Arm IK configuration (passed to inner ArmIKFingersPositionAction)
    asset_name: str = "robot"
    """Name of the robot articulation in the scene."""

    arm_joint_names: list[str] = MISSING
    """Regex/glob patterns matching the 7 arm joints.

    Example: ``["right_arm_[0-6]"]``
    """

    ee_body_name: str = MISSING
    """Name of the end-effector body for IK (link7).

    Example: ``"link_right_arm_6"``
    """

    finger_joint_names: list[str] = MISSING
    """Explicit list of the independent finger joint names (defines the finger
    action dimension; action_dim = 6 + len(finger_joint_names)).

    RH56F1 example (6)::

        [
            "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
            "right_hand_index_1_joint", "right_hand_middle_1_joint",
            "right_hand_ring_1_joint", "right_hand_little_1_joint",
        ]
    """

    coupled_joint_names: list[str] | None = None
    """Coupled (mimic) joint names — passed to the inner ArmIKFingersPositionAction.
    None (default) = legacy RH56F1 auto-list; [] = no coupling (e.g. SharpA)."""

    coupling_ratios: list[float] | None = None
    """Coupling ratios — passed to inner term. None = legacy RH56F1 ratios."""

    coupling_source_idx: list[int] | None = None
    """Driving-joint indices for coupled joints — passed to inner term.
    None = legacy RH56F1 mapping."""

    link7_to_wrist_offset: tuple[float, float, float] | None = None
    """link7 -> virtual-wrist translation — passed to inner term.
    None (default) = RH56F1 (0, 0, 0.0555); SharpA = (0, 0, 0.0655)."""

    ik_lambda_val: float = 0.005
    """Damping coefficient for DLS inverse kinematics."""

    wrist_pos_scale: float = 0.1
    """Scaling factor for wrist position delta (meters per unit action).

    Passed through to the inner ArmIKFingersPositionAction.
    """

    wrist_rot_scale: float = 0.5
    """Scaling factor for wrist rotation delta (radians per unit action).

    Passed through to the inner ArmIKFingersPositionAction.
    """

    # Residual configuration
    finger_rom_floor: float | None = None
    """Finger ROM floor (rad) passed through to the inner term. None (default) = unchanged."""

    finger_rom_floor_exclude: str | None = None
    """Fullmatch regex of the joints the ROM floor skips (None = none skipped; the hand recipe sets it)."""

    residual_scale: float = 0.3
    """Default residual scale used when wrist_residual_scale or
    finger_residual_scale are left as None.

    combined = base + residual * residual_scale.
    Default 0.3 for position control (same as ResidualWristPositionAction).
    """

    wrist_residual_scale: float | None = None
    """Per-component residual scale for the wrist dims (0:6).

    If None, falls back to ``residual_scale``. Set this to tune wrist
    residual magnitude independently of the finger residual.
    """

    finger_residual_scale: float | None = None
    """Per-component residual scale for the finger dims (6 to 11).

    If None, falls back to ``residual_scale``. Set this to tune finger
    residual magnitude independently of the wrist residual.
    """

    wrist_pos_residual_max_m: float | None = None
    """Hard upper bound (meters) on the wrist POSITION residual offset from the
    reference. When set (NOT None), the wrist position residual uses the SAME
    convention as the fingers — clamp(raw, -1, 1) * bound — instead of
    ``wrist_pos_scale * wrist_residual_scale``. The clamp makes ``bound`` a true
    per-axis cap; the residual_l2 penalty (on the raw action) is the soft
    restoring pull toward 0 (= the reference). e.g. 0.05 -> +/-5 cm."""

    wrist_rot_residual_max_deg: float | None = None
    """Hard upper bound (degrees) on the wrist ROTATION residual (geodesic
    angle) from the reference. When set, the wrist rotation residual is
    clamp(raw, -1, 1) * radians(bound), applied via Rodrigues axis-angle (3-DoF,
    not Euler). e.g. 60 -> +/-60 deg. None = legacy wrist_rot_scale * scale."""

    finger_residual_scale_end: float | None = None
    """Six residual axes: when not None, the finger residual scale goes finger_residual_scale (start) -> this value,
    linearly with **the same progress as the gate curriculum** (trajectory_manager.schedule_progress: 0 -> 1 over the
    first half of training, fixed at 1.0 for frozen/standalone evaluation), so evaluation automatically becomes the
    'final training condition'. None (default) = no ramp, byte-identical."""

    finger_effort_anneal_start: float | None = None
    """Effort curriculum: when not None, the finger cap goes down linearly from the build cap (--finger_effort) to
    finger_effort_hw (the hardware cap) while schedule_progress runs from this value to 1.0. Evaluation (progress fixed
    at 1.0) always has the hardware cap. None (default) = off."""

    finger_effort_hw: object = None
    """Hardware finger cap (a scalar N·m, or a joint regex -> N·m dict). The HandSpec owns the value
    (finger_effort_applied); the action contract (mdp/actions/contract.apply_action_contract) fills it.
    Unused when finger_effort_anneal_start is None."""

    finger_effort_build: float | None = None
    """Start cap of the effort curriculum = the declared build cap (--finger_effort, N·m). Filled by the action
    contract, never read back from PhysX."""

    finger_effort_build_applied: bool | None = None
    """Whether this path's builder loaded the build cap into PhysX (training True, eval = DEXENGINE_EVAL_TRAIN_PARITY).
    Filled by the action contract; init compares the initial PhysX cap with (loaded -> build cap, not loaded ->
    hardware cap)."""

    wrist_pos_residual_max_m_end: float | None = None
    """When not None, the wrist position cap (m) goes wrist_pos_residual_max_m (start) -> this value with
    schedule_progress. ⚠ The cap path owns the wrist authority: once a cap is set wrist_residual_scale is ignored,
    so a wrist ramp must go on this cap axis."""

    wrist_rot_residual_max_deg_end: float | None = None
    """When not None, the wrist rotation cap (deg) goes start -> this value with schedule_progress."""

    action_ema: float = 0.4
    """Exponential moving average factor for action smoothing.

    Applied to the raw residual only (residual_ik_process._smooth_residual), before
    it is added to the frozen base. The inner action term's EMA is forced to 1.0
    (no smoothing) to avoid double-EMA.
    """

    # Passthrough to inner ArmIKFingersPositionAction
    use_trajectory_reference: bool = False
    """If True, inner action term uses trajectory reference + delta instead of
    current state + delta. Must match the arm-IK training config."""

    wrist_ref_lookahead_frames: int = 0
    """WRIST LOOKAHEAD: the wrist residual base targets ref(t+k)
    instead of ref(t) — phase-lead feed-forward for the arm-chain PD/IK tracking
    lag. Fingers and the ReferenceBaseAction obs stay on the current frame.
    ACTION-CONTRACT parameter: eval/render must pass the same value as training.
    0 (default) = byte-identical legacy behavior."""

    # ------------------------------------------------------------------
    # Bimanual Option 2: which hand this residual action term controls.
    # 'right' (default) keeps the legacy right-only path byte-identical;
    # the left residual term passes side='left' so compute_reference_base_action
    # reads ref['left_hand_qpos'] + traj_mgr._left_hand_ids.
    # ------------------------------------------------------------------
    side: str = "right"
    """Which hand ('right' | 'left') this residual term controls. Selects the
    reference finger-qpos key and trajectory-manager hand-id list used by
    ``compute_reference_base_action``. Default 'right' = byte-identical."""

    object_key: str | None = None
    """LEFT-UNI: scene key of THIS hand's live object. None (default)
    keeps the side-derived mapping (right→'object', left→'object_left'). A left-uni
    build passes 'object' because its LEFT object spawns under the single scene key."""

    # ------------------------------------------------------------------
    # Reference-base mode
    # ------------------------------------------------------------------
    reference_base: bool = False
    """If True, the residual policy's BASE action is the kinematic reference
    (wrist = reference wrist via ``use_trajectory_reference``, fingers =
    reference finger qpos). The
    residual still trains on top with the identical composition/clamps.

    Requires ``use_trajectory_reference=True`` (asserted in __init__): the
    base wrist delta is zeroed so the wrist target reduces to the pure
    reference pose, and the finger base is set to the reference finger qpos
    mapped into normalized [-1, 1] action space. Default False = byte-identical
    to the frozen-base behavior."""

    wrist_smooth_weight: float = -1.0
    """Temporal ensembling weight for wrist dims. Passed to inner action term.
    -1.0 = disabled."""

    finger_smooth_weight: float = -1.0
    """Temporal ensembling weight for finger dims. Passed to inner action term.
    -1.0 = disabled."""

    smooth_history_len: int = 8
    """Temporal ensembling history length. Passed to inner action term."""

    ik_target_ema: float = 1.0
    """IK target EMA. Passed to inner action term. 1.0 = disabled."""

    inner_action_ema: float = 1.0
    """Action EMA on the inner action term (applied to the combined base+residual action).
    Must match the arm-IK training action_ema for consistent dynamics.
    1.0 = disabled (default, backward compatible)."""

    # ------------------------------------------------------------------
    # Pre-contact finger override
    # ------------------------------------------------------------------
    pre_contact_finger_open: bool = False
    """Force fingers to the open pose (``pre_contact_open_action``) until the
    palm approaches the object within ``pre_contact_release_palm_dist``.

    Motivation: some retargeted TACO trajectories place fingers inside the
    object at frame 0. The frozen base policy drives fingers to that
    penetrating pose within a couple of sim steps, producing the "thumb
    strikes object" failure regardless of reset-time randomization.

    The override is applied to the final combined finger action, overriding
    both the base and residual components. It latches off the first time the
    palm gets close enough and stays off until the next reset.
    """

    pre_contact_release_palm_dist: float = 0.07
    """Palm-to-object center distance (meters) below which the pre-contact
    finger override releases. Use ~0.05-0.10m.

    Too small (e.g., 0.02m): release happens after fingers should already
    be closing → still late.
    Too large (e.g., 0.15m): release happens far from object → base policy
    closes fingers early, may still hit.
    """

    pre_contact_open_action: float = -1.0
    """Override value in the normalized [-1, 1] finger action space sent to
    all 6 independent finger joints during the pre-contact phase. -1.0 maps
    to the lower joint limit (fully extended) for Inspire RH56F1.

    If the default produces an unexpected thumb pose (e.g., thumb tucked
    rather than spread), reduce magnitude (e.g., -0.5) or change sign.
    """
