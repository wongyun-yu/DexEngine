"""The residual arm-IK action term: a residual policy output composed onto the kinematic reference."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers.action_manager import ActionTerm

from dexengine.rl.tasks import env_flags, hooks
from dexengine.rl.tasks.mdp.actions.arm_ik import (
    ArmIKFingersPositionAction,
    ArmIKFingersPositionActionCfg,
    check_ema_weight,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from dexengine.rl.tasks.mdp.actions.residual_ik import ResidualArmIKActionCfg
from dexengine.rl.tasks.mdp.actions.residual_ik_feedforward import ArmFeedforwardMixin
from dexengine.rl.tasks.mdp.actions.residual_ik_process import ResidualProcessMixin
from dexengine.rl.tasks.mdp.actions.residual_ik_state import ResidualStateMixin
from dexengine.rl.tasks.mdp.actions.residual_ik_target import ResidualTargetMixin


class ResidualArmIKAction(
    ArmFeedforwardMixin, ResidualProcessMixin, ResidualTargetMixin, ResidualStateMixin, ActionTerm
):
    """Residual action term for arm-in-loop training: reference base + learned residual.

    Base and residual are processed separately through proper rotation
    composition, then combined on the current (or trajectory reference) wrist
    pose. The inner ArmIKFingersPositionAction handles only IK computation
    and joint target writes (its process_actions is bypassed).

    Pipeline each step:
      1. Kinematic reference → 12-dim base action (deterministic)
      2. PPO → 12-dim residual (sampled with exploration noise)
      3. Base wrist: scale → euler_to_quat → position delta + rotation quat
      4. Residual wrist: scale * rs → euler_to_quat → position delta + rotation quat
      5. Combine on current wrist pose:
         - pos: current + base_delta + residual_delta
         - rot: current ⊗ base_quat ⊗ residual_quat
      6. Fingers: (base + residual*rs) → ManipTrans scaling (affine)
      7. Convert wrist target → link7 → robot root frame → IK command
      8. Inner apply_actions: DifferentialIK + finger PD + coupling
    """

    cfg: ResidualArmIKActionCfg

    def __init__(self, cfg: ResidualArmIKActionCfg, env: ManagerBasedEnv) -> None:
        _A, _fh_rb = self._init_base_action_term(cfg, env)
        if self._zero_residual:
            print("[ResidualArmIKAction] DEXENGINE_ZERO_RESIDUAL=1: learned residual forced to 0 (PD reference base).")
        # DEXENGINE_ACTION_ABS=1 (RL ablation c5 · default OFF = byte-identical): FULL-MOTION action. The
        # policy outputs the absolute wrist pose target and the absolute finger target instead of a correction on the
        # reference; the reference stays in the observation and in every reward.
        #   wrist pos = anchor + clamp(a,±1)·half      one box per side over every loaded clip's reference wrist
        #                                              (floating-root frame) widened by ABS_MARGIN_M — clip-independent
        #   wrist rot = anchor_q ⊗ exp(clamp(a,±1)·π)  rotation vector, the whole SO(3)
        #   fingers   = clamp(a,±1)                    normalized joint target, no reference base
        # The penalties that read the action (residual norm · wrist hinge via raw_actions; residual_action_rate via
        # implied_cur/implied_prev) read the IMPLIED residual: the residual-mode action that gives the same target,
        # (target − reference target)/cap, extended linearly past the box so they keep their restoring pull. Every
        # reward is therefore the same function of the commanded target as in residual mode.
        self._action_abs = env_flags.get("DEXENGINE_ACTION_ABS") == "1"
        self._init_feedforward_and_residual_mode(cfg, _A, _fh_rb)

    def _init_base_action_term(self, cfg, env):
        """Initialize the base ActionTerm (sets self._asset, self.device, etc.)."""
        super().__init__(cfg, env)

        _fh_rb = self._init_hand_side(cfg, env)
        self._arm_q_target = None
        self._arm_dims_logged = False
        self._arm_step = 0
        _A = self._nw + self._n_fingers
        self._action_dim_value = _A
        self._raw_actions = torch.zeros(self.num_envs, _A, device=self.device)
        self._smoothed_residual_actions = torch.zeros(self.num_envs, _A, device=self.device)
        self._combined_actions = torch.zeros(self.num_envs, _A, device=self.device)
        self._pre_clamp_combined = torch.zeros(self.num_envs, _A, device=self.device)

        # -- Step 3: Base action buffer (zeros; the single-stage base is the kinematic
        # reference computed in process_actions, not a model output) --
        self._base_action_buffer = torch.zeros(self.num_envs, _A, device=self.device)

        # -- Step 4: Trajectory reference mode (for wrist base pose) --
        self._use_traj_ref = cfg.use_trajectory_reference
        if self._use_traj_ref:
            self._traj_mgr = env.trajectory_manager

        # WRIST LOOKAHEAD: the WRIST base targets ref(t+k) instead of
        # ref(t) to feed-forward the arm-chain PD/IK tracking lag (phase-lead).
        # Fingers/obs base stay on the current frame — their lag is ~0, and shifting
        # them would force early closure (the rpan failure mode). k is in TRAJECTORY
        # frames; _abs_frames' per-clip clamp makes the last k frames all target the
        # final frame. 0 (default) = byte-identical.
        self._wrist_lookahead = int(getattr(cfg, "wrist_ref_lookahead_frames", 0) or 0)
        if self._wrist_lookahead > 0:
            print(
                f"[ResidualArmIKAction] WRIST LOOKAHEAD ENABLED: wrist base = "
                f"ref(t+{self._wrist_lookahead} frames); fingers/obs base stay at ref(t)"
            )
        # Judge park mask (num_envs,) bool, owned by dexengine.rl.judge.park (handed over with set_parked). Parked envs
        # get wrist lead 0 (wrist reference = ref(clock 0) = frame 0) and zero arm feed-forward reference derivatives.
        # None = no parking (training, byte-identical).
        self._parked: torch.Tensor | None = None
        # the mask process_actions last used (dexengine.rl.judge.park's proof of use)
        self.parked_seen: torch.Tensor | None = None

        # -- reference-base mode --
        # When enabled, the residual policy's BASE action is the kinematic
        # reference (wrist + finger qpos).
        # The residual composition/clamps downstream are unchanged. Requires
        # use_trajectory_reference=True so the wrist target reduces to the pure
        # reference pose once the base wrist delta is zeroed in process_actions.
        self._reference_base = bool(getattr(cfg, "reference_base", False))
        self._refbase_finger_in_hand = None  # lazy: finger-joint positions within right_hand_qpos (PhysX order)
        if self._reference_base:
            assert self._use_traj_ref, (
                "reference_base=True requires use_trajectory_reference=True so "
                "the wrist base reduces to the pure reference pose. Set "
                "use_trajectory_reference=True (or pass --reference_base, which "
                "forces it on)."
            )
            print(
                "[ResidualArmIKAction] REFERENCE-BASE mode ENABLED: "
                "base = reference (wrist + fingers), residual trains on top. "
                "Base wrist delta zeroed; finger base = reference finger qpos "
                "mapped to normalized [-1, 1]."
            )

        # -- Zero-residual eval hook (env-var gated; EVAL/RENDER ONLY) --
        # DEXENGINE_ZERO_RESIDUAL=1 forces the learned residual to 0 every step, so the
        # physical target reduces to the (reference) base under normal physics/PD. This
        # is the "PD-controlled reference base" render/audit: same mechanism as the
        # reference-base mode but with no learned correction. No env var => no effect.
        self._zero_residual = env_flags.get("DEXENGINE_ZERO_RESIDUAL") == "1"
        return _A, _fh_rb

    def _init_hand_side(self, cfg, env):
        """Bimanual Option 2: which hand this term controls. Drives the reference finger-qpos key + traj-manager hand-id
        list in compute_reference_base_action. Default 'right' = byte-identical.
        """
        self._side_rb = getattr(cfg, "side", "right")
        # Live-object scene key for THIS hand (right→"object", left→"object_left").
        # Used by the optional release-latch / phase-C readiness / scripted-grip paths
        # so a LEFT term reads the LEFT object's live pose, not the right's. Default
        # "object" keeps the right path byte-identical.
        # LEFT-UNI: cfg.object_key overrides the side-derived default —
        # a left-uni scene spawns the LEFT object under the single "object" key, so
        # side='left' + object_key='object' must not resolve to the absent 'object_left'.
        _cfg_obj_key = getattr(cfg, "object_key", None)
        if _cfg_obj_key:
            self._obj_scene_key = _cfg_obj_key
        else:
            self._obj_scene_key = "object_left" if self._side_rb == "left" else "object"

        # -- Step 1: Create internal arm-IK action term --
        # Build an ArmIKFingersPositionActionCfg from our cfg fields,
        # then instantiate it. This handles IK arm control, finger limits,
        # coupling, etc.
        # Inner action_ema=1.0 (no smoothing) to avoid double-EMA:
        # the outer term smooths the learned residual before composition.
        inner_cfg = ArmIKFingersPositionActionCfg(
            asset_name=cfg.asset_name,
            arm_joint_names=cfg.arm_joint_names,
            ee_body_name=cfg.ee_body_name,
            finger_joint_names=cfg.finger_joint_names,
            coupled_joint_names=cfg.coupled_joint_names,
            coupling_ratios=cfg.coupling_ratios,
            coupling_source_idx=cfg.coupling_source_idx,
            link7_to_wrist_offset=cfg.link7_to_wrist_offset,
            ik_lambda_val=cfg.ik_lambda_val,
            wrist_pos_scale=cfg.wrist_pos_scale,
            wrist_rot_scale=cfg.wrist_rot_scale,
            action_ema=cfg.inner_action_ema,  # inner term EMA (1.0: the residual EMA is applied once, outside)
            # Passthrough: trajectory reference and temporal ensembling settings
            use_trajectory_reference=cfg.use_trajectory_reference,
            wrist_smooth_weight=cfg.wrist_smooth_weight,
            finger_smooth_weight=cfg.finger_smooth_weight,
            smooth_history_len=cfg.smooth_history_len,
            ik_target_ema=cfg.ik_target_ema,
            # Finger ROM floor pass-through (the inner term uses it for the bound and the normalisation)
            finger_rom_floor=getattr(cfg, "finger_rom_floor", None),
            finger_rom_floor_exclude=getattr(cfg, "finger_rom_floor_exclude", None),
        )
        # Floating hand (DEXENGINE_ARM=floating): only the inner term becomes the wrist wrench term; this outer term,
        # which builds the wrist target, is the same code as on the arm body. The one answer is floating_hand.active().
        # The arm body never takes this branch.
        _fh_rb = hooks.floating_hand()
        if hooks.floating_active():
            self._inner = hooks.get(hooks.FLOATING_HAND_ACTION)(inner_cfg, env)
        else:
            self._inner = ArmIKFingersPositionAction(inner_cfg, env)

        # -- Step 2: Action buffers (6 wrist + n finger) --
        self._n_fingers = len(cfg.finger_joint_names)
        # DEXENGINE_ARM_JOINT_RESIDUAL=1 (RL ablation j · default OFF = byte-identical): ARM-JOINT
        # action, the OATH-style alternative to the wrist-space residual. The leading action block is one residual per
        # arm
        # joint instead of the 6 wrist-pose dims: arm target = the reference arm joints of the frame (the all-frame arm
        # IK
        # cache, the same table the reset and the feedforward read, at the wrist look-ahead frame) + clamp(a,±1)·cap,
        # written as the arm joint position target (no IK). Fingers are unchanged. The policy also observes the arm
        # joints
        # (arm_joint_obs). nw = the leading block's width; legacy wrist/finger dims of the penalties map onto it
        # (legacy_dims).
        self._arm_joint_res = env_flags.get("DEXENGINE_ARM_JOINT_RESIDUAL") == "1"
        self._nw = len(self._inner._arm_joint_ids) if self._arm_joint_res else 6
        self._arm_res_cap = float(env_flags.get("DEXENGINE_ARM_JOINT_RESIDUAL_RAD"))
        return _fh_rb

    def _init_feedforward_and_residual_mode(self, cfg, _A, _fh_rb):
        """Arm velocity / gravity feedforward setup and the residual-mode checks (statements unchanged)."""
        if self._arm_joint_res:
            # Split articulation is fine: each side's term owns its side's asset, and env._arm_ik_joint_ids_<side> are
            # found on that same asset (mdp/events — the arm feedforward relies on the same pairing); _arm_ref_q checks
            # the pairing against this term's own arm joint ids on the first call.
            if self._action_abs or not (self._arm_res_cap > 0):
                raise RuntimeError(
                    f"[ARM_JOINT_RESIDUAL] needs DEXENGINE_ACTION_ABS off and a positive "
                    f"DEXENGINE_ARM_JOINT_RESIDUAL_RAD (got abs={self._action_abs} cap={self._arm_res_cap})"
                )
            print(
                f"[ARM_JOINT_RESIDUAL] on side={self._side_rb} arm_dims={self._nw} fingers={self._n_fingers} "
                f"action_dim={self._nw + self._n_fingers} cap={self._arm_res_cap} rad "
                f"joints={list(self._inner._arm_joint_names)}",
                flush=True,
            )
        self._abs_box = None  # built below when the switch is on (trajectory tables + root frame exist at init)
        self._abs_logged = False
        self._implied_cur = torch.zeros(self.num_envs, _A, device=self.device)
        self._implied_prev = torch.zeros(self.num_envs, _A, device=self.device)
        if self._action_abs:
            bad = [
                n
                for n, b in (
                    ("use_trajectory_reference", not self._use_traj_ref),
                    ("reference_base", not self._reference_base),
                    ("wrist_pos_residual_max_m", cfg.wrist_pos_residual_max_m is None),
                    ("wrist_rot_residual_max_deg", cfg.wrist_rot_residual_max_deg is None),
                )
                if b
            ]
            if bad:
                raise RuntimeError(
                    f"[ACTION_ABS] DEXENGINE_ACTION_ABS=1 needs {bad} (the implied residual is defined "
                    "against the reference base and the wrist caps)"
                )
            self._abs_box_build(
                math_utils.quat_conjugate(self._inner._floating_root_quat.unsqueeze(0))[0],
                float(self._inner._floating_root_z_offset),
            )

        self._init_arm_feedforward(cfg, _fh_rb)

    def _init_arm_feedforward(self, cfg, _fh_rb):
        """Arm velocity + gravity FEEDFORWARD (env-var gated) DEXENGINE_ARM_FF=1 adds, every apply_actions, on top of
        the arm IK position PD: (a) gravity compensation effort = get_gravity_compensation_forces() on ALL arm joints
        (cancels the no-gravity-comp droop), and (b) velocity target = reference arm-IK joint velocity of THIS TERM'S
        SIDE (finite-diff of env._arm_ik_cache_all_<side>), so the Kd term stops braking the intended ramp motion.
        Without it the position-only PD must hold a velocity-proportional error e_ss = (Kd/Kp)*qdot to move at all
        (measured: |e| = 1.0-1.26x that). Parity-relevant controller change (you'd add the same FF on real). Wiring:
        (b) once read the hardcoded `_arm_ik_cache_all_right`, but the all-frame cache is stored SIDE- SUFFIXED (sfx
        = f"_{side}"). On a LEFT-uni run the right-named attr was None, so the velocity FF was SILENTLY skipped
        (gravity comp still applied). Now side-aware, and it LOUDLY reports when the FF cannot engage instead of no-
        oping in silence. Default 1 (ON); the uni/bi launchers always pass it explicitly. DEXENGINE_ARM_FF=0
        reproduces the old (OFF) behaviour and old checkpoints.
        """
        self._arm_ff = env_flags.get("DEXENGINE_ARM_FF") == "1"
        if hooks.floating_active():
            # The arm feed-forward (arm joint gravity compensation + reference velocity) only means something for arm
            # joints; the floating hand has no arm and the whole hand is weightless.
            if env_flags.get("DEXENGINE_ARM_FF", None) not in (None, "", "0", "1"):
                raise ValueError(f"DEXENGINE_ARM_FF={env_flags.get('DEXENGINE_ARM_FF', None)!r} — 0|1 only")
            print(
                f"{_fh_rb.MARK} {self._side_rb}: arm feed-forward off (no arm joints; DEXENGINE_ARM_FF="
                f"{env_flags.get('DEXENGINE_ARM_FF', '(unset)')} has no consumer on this body)",
                flush=True,
            )
            self._arm_ff = False
        self._ff_arm_all_ids = None  # all 14 arm joint ids (gravity comp)
        self._ff_vel_ids = None  # THIS side's 7 arm ids, ordered to match the IK cache columns
        self._ff_warned = False  # one-shot: FF enabled but prerequisites missing
        self._ff_logged = False  # one-shot: FF engaged (prints cache shape + |qd_ref|)
        self._ff_grav_sign = 1.0
        # Feedforward CHANNEL: 'vel' (joint velocity target — measured INEFFECTIVE:
        # lag ratio 1.00 with vs 1.03 without), 'torque' (tau_ff = Kd*qd_ref through the EFFORT
        # channel that gravity compensation already proves works), or 'both'.
        # DEXENGINE_ARM_FF_VEL_SCALE exaggerates qd_ref to probe whether a channel is live at all.
        # Default "lead" ("vel" measured ineffective: lag ratio 1.00 vs 1.03). The uni/bi launchers pass it
        # explicitly; DEXENGINE_ARM_FF_MODE=vel reproduces old checkpoints.
        self._ff_mode = env_flags.get("DEXENGINE_ARM_FF_MODE").strip().lower()
        self._ff_vel_scale = float(env_flags.get("DEXENGINE_ARM_FF_VEL_SCALE"))
        self._ff_damping = None  # cached PhysX joint damping for the FF ids (torque mode)
        self._ff_kdkp = None  # cached Kd/Kp per FF joint (lead mode)
        # Central-difference half-window (frames) for the reference derivatives. The 1-frame
        # difference is retarget noise (see apply); 5 frames removed the spikes in measurement.
        self._ff_smooth = 13
        self._ff_minv = None  # cached M_diag/Kp (lead2 inertial term)
        self._ff_acc_scale = 1.0
        self._ff_idff_probed = False  # one-shot: report which inverse-dynamics APIs exist
        self._ff_massfn = None  # callable -> generalized mass matrices, if available

        self._ff_vel_pos_in_all = None  # index of each FF arm id within _ff_arm_all_ids
        self._ff_ids_src = "?"
        self._init_pre_contact_and_limits(cfg)

    def _init_pre_contact_and_limits(self, cfg):
        """Pre-contact finger override and the later init steps (statements unchanged)."""
        if self._arm_ff:
            print(
                f"[ResidualArmIKAction] DEXENGINE_ARM_FF=1: arm feedforward ENABLED "
                f"(mode={self._ff_mode}, vel_scale={self._ff_vel_scale}, gravity comp always on)."
            )

        _eff_finger_rs, _eff_wrist_rs, ensemble_str, traj_ref_str = self._init_pre_contact_override(cfg)
        if cfg.wrist_pos_residual_max_m is not None:
            _rot = f"/±{cfg.wrist_rot_residual_max_deg:.0f}°" if cfg.wrist_rot_residual_max_deg is not None else ""
            _wrist_auth = (
                f"CAP ±{cfg.wrist_pos_residual_max_m * 100:.0f}cm{_rot} "
                f"(wrist_residual_scale={_eff_wrist_rs} IGNORED when cap set)"
            )
            if cfg.wrist_pos_residual_max_m_end is not None or cfg.wrist_rot_residual_max_deg_end is not None:
                _pe = cfg.wrist_pos_residual_max_m_end or cfg.wrist_pos_residual_max_m
                _re = cfg.wrist_rot_residual_max_deg_end or cfg.wrist_rot_residual_max_deg
                _wrist_auth += f" RAMP→±{_pe * 100:.0f}cm/±{_re:.0f}°@sched"
        else:
            _wrist_auth = f"scale {_eff_wrist_rs}"
        print(
            f"[ResidualArmIKAction] residual_scale={cfg.residual_scale} "
            f"(wrist={_wrist_auth}, finger={_eff_finger_rs}) "
            f"target={traj_ref_str} ensemble={ensemble_str} "
            f"ik_lambda={cfg.ik_lambda_val} ema={cfg.action_ema}"
        )

        # Effort curriculum: when on, the finger cap goes down from the build cap (--finger_effort) to the hardware
        # cap (HandSpec) with schedule_progress. Evaluation (progress fixed at 1.0) always has the hardware cap.
        # None (default) = off.
        self._eff_anneal = None
        if cfg.finger_effort_anneal_start is not None:
            self._init_effort_anneal()

    def _init_pre_contact_override(self, cfg):
        """Step 4b: Pre-contact finger override Mitigates initial trajectory interpenetration: some TACO trajectories
        position fingers already inside the object at frame 0. Under residual scale 0.3 the frozen base closes
        fingers back to that penetrating pose within a couple of sim steps, producing the "thumb strikes object"
        failure regardless of reset-time randomization. When enabled, fingers are forced to
        ``pre_contact_open_action`` (default -1.0 → lower joint limit → extended) until the palm gets within
        ``pre_contact_release_palm_dist`` of the object, then the override releases permanently for the rest of the
        episode.
        """
        self._pre_contact_enabled = cfg.pre_contact_finger_open
        if self._pre_contact_enabled:
            self._pre_contact_release_dist = cfg.pre_contact_release_palm_dist
            self._pre_contact_open_action = cfg.pre_contact_open_action
            self._pre_contact_active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            # Reuse inner's ee body id (link7) as the palm proxy. This sits
            # ~2-3cm behind the actual palm center, which is fine
            # at the 5-10cm release-distance scale.
            print(
                f"[ResidualArmIKAction] Pre-contact finger override ENABLED: "
                f"release_dist={self._pre_contact_release_dist:.3f}m, "
                f"open_action={self._pre_contact_open_action}, "
                f"palm_body_idx={self._inner._ee_body_idx}"
            )

        # -- Sanity asserts: enforce the only configurations the bypassing
        # process_actions actually supports. The outer process_actions
        # bypasses inner.process_actions entirely, so any inner-only feature
        # (EMA on inner, temporal ensembling) is silently ignored. We assert
        # the safe defaults to prevent silent misconfiguration.
        assert cfg.inner_action_ema == 1.0, (
            "ResidualArmIKAction bypasses inner.process_actions, so "
            "inner_action_ema cannot be applied. Set inner_action_ema=1.0 "
            f"(currently {cfg.inner_action_ema})."
        )
        check_ema_weight("action_ema (outer residual)", cfg.action_ema)  # 0 would freeze the residual at zero
        assert cfg.wrist_smooth_weight < 0, (
            "ResidualArmIKAction bypasses inner.process_actions, so "
            "wrist temporal ensembling cannot be applied. Set "
            f"wrist_smooth_weight=-1.0 (currently {cfg.wrist_smooth_weight})."
        )
        assert cfg.finger_smooth_weight < 0, (
            "ResidualArmIKAction bypasses inner.process_actions, so "
            "finger temporal ensembling cannot be applied. Set "
            f"finger_smooth_weight=-1.0 (currently {cfg.finger_smooth_weight})."
        )

        traj_ref_str = "trajectory_ref" if cfg.use_trajectory_reference else "current_state"
        ensemble_str = (
            (f"wrist={cfg.wrist_smooth_weight}, finger={cfg.finger_smooth_weight}, history={cfg.smooth_history_len}")
            if (cfg.wrist_smooth_weight >= 0 or cfg.finger_smooth_weight >= 0)
            else "off"
        )
        # Resolve effective wrist/finger residual scales for the startup banner.
        # ⚠ IMPORTANT (do not be misled): when a hard cap is set (wrist_pos_residual_max_m /
        # wrist_rot_residual_max_deg), the wrist residual takes the CAP path in apply_actions
        # (clamp(raw,±1)·cap) and IGNORES wrist_residual_scale ENTIRELY. So a printed
        # wrist_residual_scale=0.0 does NOT mean "wrist frozen / follows reference" — the ±cap
        # (e.g. ±5cm / ±60°) IS the wrist's residual authority. The banner below prints the
        # EFFECTIVE authority (cap when set, else the legacy scale) to avoid that confusion.
        _eff_wrist_rs = cfg.wrist_residual_scale if cfg.wrist_residual_scale is not None else cfg.residual_scale
        _eff_finger_rs = cfg.finger_residual_scale if cfg.finger_residual_scale is not None else cfg.residual_scale
        if cfg.finger_residual_scale_end is not None:
            _eff_finger_rs = f"{_eff_finger_rs}→{cfg.finger_residual_scale_end}@sched"
        return _eff_finger_rs, _eff_wrist_rs, ensemble_str, traj_ref_str

    @property
    def action_dim(self) -> int:
        """Policy action dimension: residual corrections (6 wrist, or one per arm joint under
        DEXENGINE_ARM_JOINT_RESIDUAL, + n finger)."""
        return getattr(self, "_nw", 6) + len(self.cfg.finger_joint_names)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    # DEXENGINE_ACTION_ABS — what residual_action_rate reads instead of the flat policy action (see __init__).
    @property
    def action_abs(self) -> bool:
        return self._action_abs

    @property
    def implied_cur(self) -> torch.Tensor:
        return self._implied_cur

    @property
    def implied_prev(self) -> torch.Tensor:
        return self._implied_prev

    ABS_MARGIN_M = 0.07  # box margin = the recipe's wrist position cap, so reference ± one cap is reachable

    @property
    def processed_actions(self) -> torch.Tensor:
        """The combined (base + residual) actions fed to the inner action term."""
        return self._combined_actions

    def set_parked(self, mask: torch.Tensor | None) -> None:
        """Judge park env mask (num_envs,) bool; only dexengine.rl.judge.park calls it (None = no parking). Reset does
        not clear it."""
        self._parked = None if mask is None else mask.to(device=self.device, dtype=torch.bool)
