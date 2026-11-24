"""Pilot flags, second half: power-reward strip, finger and wrist hinges, the contact pilot (source, reward mode,
termination grace), witness-object contact rewards and the contact-conditioned release of finger imitation.

Pure argparse (no Isaac imports); flag names, types, defaults and actions are the launcher's argv contract
(tests/rl/launch_cases/*.case). dexengine.rl.args calls this right after pilot.add_pilot_args."""

from __future__ import annotations


def add_pilot_contact_args(parser) -> None:
    """Add the second half of the pilot flags to ``parser`` (in launch-contract order)."""
    _add_finger_hinge_args(parser)
    _add_contact_hinge_args(parser)
    _add_contact_match_args(parser)
    _add_grasp_target_args(parser)
    _add_imit_release_args(parser)


def _add_finger_hinge_args(parser) -> None:
    """Power reward strip, held-out data and the finger/wrist hinge thresholds."""
    parser.add_argument(
        "--strip_power_reward",
        type=int,
        default=1,
        choices=[0, 1],
        help="Drop RewardPower (exp(-10*P) saturates to zero "
        "on the full robot -- 0.000 logged contribution, zero gradient). Dead term, so "
        "parity expected. Applied in both train_cfg_overrides and eval overrides.",
    )
    parser.add_argument(
        "--heldout_data_dir",
        type=str,
        default=None,
        help=(
            "Paper 1 B1a Round 2: path to held-out clip directory "
            "(e.g. a held-out split of the combo tree root). "
            "When set, training saves a CSV at <exp_dir>/heldout_eval_manifest.csv "
            "listing the best_agent.pt path and held-out dir for a separate "
            "post-training eval pass (eval/heldout/completion_rate). "
            "Does not run held-out eval inline."
        ),
    )
    parser.add_argument(
        "--enable_finger_hinge",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Paper 1 B1a H2: soft hinge on fingertip tracking rewards. "
            "0=disabled (default, standard exp(-k*error)). "
            "1=enabled: set hinge_threshold_m=0.02 (2 cm) on thumb_tip, index_tip, "
            "middle_tip, ring_tip, pinky_tip rewards. Errors within 2 cm yield full "
            "reward (1.0); exponential decay applies only to the excess beyond 2 cm."
        ),
    )
    parser.add_argument(
        "--finger_hinge_threshold_m",
        type=float,
        default=None,
        help="Override: custom fingertip-hinge threshold in metres (default None = use 0.02). Requires "
        "--enable_finger_hinge 1.",
    )
    parser.add_argument(
        "--wrist_hinge_threshold_m",
        type=float,
        default=None,
        help="Override: custom wrist-hinge threshold in metres (default None = use 0.04). Requires "
        "--enable_wrist_hinge 1.",
    )


def _add_contact_hinge_args(parser) -> None:
    """Contact pilot: EEF/fingertip hinges, contact source and contact reward weight."""
    parser.add_argument(
        "--eef_pos_hinge_cm",
        type=float,
        default=None,
        help=(
            "eef_pos hinge threshold in CENTIMETRES (default None = disabled, behavior unchanged). "
            "When set, the wrist-position reward gives zero gradient for errors within this threshold "
            "(eef_pos error < eef_pos_hinge_cm cm); exponential decay applies only to the excess beyond. "
            "Converted to metres and passed as hinge_threshold_m to RewardEefPos. "
            "Independent of --enable_wrist_hinge / --wrist_hinge_threshold_m."
        ),
    )
    parser.add_argument(
        "--eef_rot_hinge_deg",
        type=float,
        default=None,
        help=(
            "eef_rot hinge threshold in DEGREES (default None = disabled, behavior unchanged). "
            "When set, the wrist-rotation reward gives zero gradient for geodesic errors within this "
            "threshold; exponential decay applies only to the excess beyond. Applies only on the "
            "full-robot (link7) geodesic path; floating-hand path is unchanged. "
            "Converted to radians and passed as hinge_threshold_rad to RewardEefRot."
        ),
    )
    parser.add_argument(
        "--fingertip_hinge_cm",
        type=float,
        default=2.0,
        help=(
            "R1 fingertip-reward soft-hinge threshold in CENTIMETRES. Default 2.0 (hinge ON: "
            "fingertip errors within 2 cm yield full reward, exp decay only beyond). Set 0 for "
            "hinge OFF (pure exp(-k*error)). Single source of truth for the R1 hinge/non-hinge A/B; "
            "applied unconditionally in train_cfg_overrides to all 5 fingertip rewards UNLESS a legacy "
            "hinge flag (--enable_finger_hinge / --enable_hinge) is set (those take precedence)."
        ),
    )
    # "Contact pilot" flags (per-finger contact-MATCH reward + hinge); all off by default
    # (env_setup.contact_pilot.apply_contact_pilot_overrides).
    parser.add_argument(
        "--contact_source",
        type=str,
        default="mano",
        choices=["mano", "robot"],
        help="Contact-pilot: which fingertip-contact annotation to use as the reference. "
        "Loads data/<...>/<seq>/0/contact_<source>.npz field `contact` (T,5,2) bool, "
        "fingertip region [:, :, 0]. Default 'mano'.",
    )
    parser.add_argument(
        "--contact_reward_weight",
        type=float,
        default=0.5,
        help="Contact-pilot: per-finger weight on RewardContactMatch (binary additive, "
        "max raw value 5 → scaled by this weight). Default 0.5. Only used when "
        "--enable_contact_match_reward 1.",
    )


def _add_contact_match_args(parser) -> None:
    """Contact-MATCH reward mode/softness and the contact termination."""
    parser.add_argument(
        "--enable_contact_match_reward",
        type=int,
        default=0,
        choices=[0, 1],
        help="Contact-pilot master gate: 0 (default) = OFF, no behavior change. "
        "1 = add a per-finger contact reward (form set by --contact_reward_mode) + 5 "
        "near-zero-weight (~1e-9) contact diagnostic terms logged via "
        "Episode_Reward/diag_*.",
    )
    parser.add_argument(
        "--contact_reward_mode",
        type=str,
        default="binary",
        choices=["binary", "soft", "penalty"],
        help="Contact-pilot (Batch 2) contact-reward formulation. "
        "'binary' (default) = RewardContactMatch sum(ref AND robot) in [0,5] (inert "
        "at zero contact); 'soft' = RewardGraspTargetSoft sum over ref-contact fingers of "
        "exp(-k*d_target) in [0,5] (smooth gradient toward the witness_obj GT contact "
        "point, k=--contact_softness_k); "
        "'penalty' = RewardContactPenalty -sum(ref AND NOT robot) in [-5,0]. "
        "A looser binary gate is 'binary' with a larger --contact_distance_threshold.",
    )
    parser.add_argument(
        "--contact_softness_k",
        type=float,
        default=50.0,
        help="Contact-pilot: falloff k (1/m) for --contact_reward_mode soft, reward "
        "exp(-k*d_live). Default 50 → d=0.5cm:0.78, 1cm:0.61, 2cm:0.37, 3cm:0.22.",
    )
    parser.add_argument(
        "--enable_contact_termination",
        type=int,
        default=0,
        choices=[0, 1],
        help="Contact-pilot: 0 (default) = OFF. 1 = add a hand-level contact-mismatch "
        "TERMINATION (NEW per-finger annotation + object-filtered contact force). "
        "Terminates when (ref expects any fingertip contact) AND (robot has no object "
        "contact force) is sustained >= --contact_term_grace_steps.",
    )
    parser.add_argument(
        "--contact_term_force_threshold",
        type=float,
        default=0.05,
        help="Contact-pilot termination: object-filtered fingertip force threshold (N) "
        "above which the robot counts as 'in contact'. Default 0.05.",
    )
    parser.add_argument(
        "--contact_term_grace_steps",
        type=int,
        default=50,
        help="Contact-pilot termination: consecutive mismatch steps before termination "
        "fires. 50 = ~1s at 50Hz policy. Used by --enable_contact_termination.",
    )


def _add_grasp_target_args(parser) -> None:
    """Level-1 keep, reward diagnostics, GT-target grasp/approach rewards and fingertip rebalance."""
    parser.add_argument(
        "--keep_level1",
        type=int,
        default=0,
        choices=[0, 1],
        help="Contact-pilot: 0 (default) = zero level_1 reward weight (along with level_2). "
        "1 = keep level_1. Only takes effect when a contact-pilot flag is active "
        "(see --enable_hinge / --enable_contact_match_reward).",
    )
    parser.add_argument(
        "--enable_reward_diagnostics",
        type=int,
        default=0,
        choices=[0, 1],
        help="0 (default) = strip weight-0 diagnostic-only reward terms (diag_*) so they are NOT "
        "registered/computed during training (efficiency; ~32 fewer reward-manager terms on TACO "
        "bimanual). 1 = keep them for W&B logging / debugging. Behavior-neutral for shaping "
        "(diagnostics are zero-gradient).",
    )
    # -------- witness_obj GT-target contact rewards + fingertip rebalance --------
    parser.add_argument(
        "--enable_grasp_target",
        type=int,
        default=0,
        choices=[0, 1],
        help="INFORMATIONAL ONLY — soft mode ALWAYS uses the witness_obj "
        "GT-target distance (RewardGraspTargetSoft); the legacy nearest-surface "
        "RewardContactSoft no longer exists. "
        "The GT target is the per-finger object-surface contact point, object-relative.",
    )
    parser.add_argument(
        "--enable_approach_reward",
        type=int,
        default=0,
        choices=[0, 1],
        help="Exp6B/7B: 0 (default)=OFF. 1 = add a DENSE (not ref-gated) approach reward "
        "(RewardApproachSoft) on the witness_obj GT-target distance, active every frame "
        "for the will-contact fingers. Requires --enable_contact_match_reward 1.",
    )
    parser.add_argument(
        "--contact_approach_weight",
        type=float,
        default=1.5,
        help="Exp6B/7B: weight of the dense GT-target approach reward. Default 1.5.",
    )
    parser.add_argument(
        "--contact_approach_k",
        type=float,
        default=12.0,
        help="Exp6B/7B: falloff k (1/m) for the dense approach reward exp(-k*d_target). "
        "Default 12 → d=3cm:0.70, 5cm:0.55, 8cm:0.38, 12cm:0.24.",
    )
    parser.add_argument(
        "--fingertip_weight_scale",
        type=float,
        default=1.0,
        help="multiply the 5 fingertip-tracking reward weights by this. 0.5 down-weights "
        "the fingertip imitation prior so contact/object dominate the contact phase. "
        "Default 1.0 (no change).",
    )


def _add_imit_release_args(parser) -> None:
    """Contact-conditioned release of finger imitation."""
    parser.add_argument(
        "--finger_pose_weight",
        type=float,
        default=None,
        help="Finger joint-angle (pose) tracking reward weight (RewardFingerPose). Pins "
        "high-DoF finger pose to the faithful retarget so fingers don't over-bend to hit tip "
        "positions. None/0 = off (RH56F1 default). SharpA hand-tracking: ~0.6.",
    )
    parser.add_argument(
        "--finger_pose_k",
        type=float,
        default=None,
        help="Exponential kernel sharpness for --finger_pose_weight (default cfg 4.0; reward = "
        "exp(-k * mean|q - q_ref|) over independent finger joints).",
    )
    # Contact-conditioned release of finger imitation (Dexplore-style); applied by
    # env_setup.objtrack.apply_finger_imit_release.
    parser.add_argument(
        "--imit_release_min",
        type=float,
        default=None,
        help="Finger-imitation release (Dexplore-style): the minimum multiplier v in [0,1) applied to the finger "
        "position/pose imitation rewards (5 fingertips · level_1/2 · finger_pose) only on frames where the "
        "reference hand touches the object (label contact_<source>_<side>). The multiplier goes linearly "
        "1 -> v while schedule_progress goes --imit_release_start -> 1.0 (frozen eval = 1.0 -> v); the "
        "approach phase stays at 1. Pass with --imit_release_start. Default None = off.",
    )
    parser.add_argument(
        "--imit_release_start",
        type=float,
        default=None,
        help="Start schedule_progress p0 in [0,1) of the finger-imitation release ramp. Pass with "
        "--imit_release_min. Default None = off.",
    )
