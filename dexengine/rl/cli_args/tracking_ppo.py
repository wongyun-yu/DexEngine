"""Tracking flags, second half: wrist residual deadzone and action rates, finger penalties, the wrist L2 hinge, actor
weight decay, pure tracking, PPO hyperparameter overrides, gravity ramp, perturbations and the initial-pose render.

Pure argparse (no Isaac imports); flag names, types, defaults and actions are the launcher's argv contract
(tests/rl/launch_cases/*.case). dexengine.rl.args calls this right after tracking.add_tracking_args."""

from __future__ import annotations

from dexengine.rl.cli_args.choices import FINGER_PENALTY_SPAN_CHOICES


def add_tracking_ppo_args(parser) -> None:
    """Add the second half of the tracking flags to ``parser`` (in launch-contract order)."""
    _add_finger_penalty_args(parser)
    _add_wrist_hinge_and_reset_args(parser)
    _add_ppo_override_args(parser)
    _add_physics_ramp_args(parser)
    _add_pose_render_args(parser)


def _add_finger_penalty_args(parser) -> None:
    """Residual L2 deadzones, wrist look-ahead, action-rate weights and the finger penalty span."""
    parser.add_argument(
        "--wrist_residual_l2_deadzone",
        type=float,
        default=None,
        help="Deadzone for the wrist residual_l2 (6-d). Default falls back to --residual_l2_deadzone.",
    )
    parser.add_argument(
        "--finger_residual_l2_deadzone",
        type=float,
        default=None,
        help="BASE deadzone for the finger residual_l2 when split, in the old 6-finger-dim norm "
        "units; the term uses base x dz_scale of --finger_penalty_span (derived_norm: "
        "sqrt(n/6), legacy: 1). Default falls back to --residual_l2_deadzone.",
    )
    parser.add_argument(
        "--wrist_ref_lookahead_frames",
        type=int,
        default=5,
        help="Canonical 5. WRIST LOOKAHEAD: wrist residual base targets ref(t+k) "
        "trajectory frames ahead (phase-lead vs arm PD/IK lag). Fingers/obs "
        "base stay at ref(t). ACTION-CONTRACT param — eval/render must match "
        "training. Last k frames clamp to the final frame. 0 = off.",
    )
    parser.add_argument(
        "--wrist_action_rate_weight",
        type=float,
        default=0.30,
        help="Anti-trembling: weight for -sum((a_t - a_{t-1})^2) on the WRIST "
        "residual dims (0:6) only — fingers untouched (grasp-quality guard). "
        "Canonical 0.30. 0 = off.",
    )
    parser.add_argument(
        "--finger_action_rate_weight",
        type=float,
        default=0.10,
        help="Anti-chatter rate penalty on the FINGER residual (span = --finger_penalty_span). "
        "Canonical 0.10. 0 = off. Measured finger "
        "target chatter 0.371/step on hard clips vs 0.092 on winners.",
    )
    # Action span, weight scale and deadzone scale of the two finger penalties (split residual L2, finger action
    # rate); owner core.hands.finger_penalty_plan. None (default) = the training path resolves the canon
    # (core.hands.CANON_FINGER_PENALTY_SPAN); the penalty marker tells explicit from default.
    parser.add_argument(
        "--finger_penalty_span",
        type=str,
        default=None,
        choices=FINGER_PENALTY_SPAN_CHOICES,
        help="Span + weight/deadzone scales of the finger residual-L2 (split) and finger "
        "action-rate penalties, per side. Default (unset) = the canon "
        "core.hands.CANON_FINGER_PENALTY_SPAN. derived_norm = every finger dim of that side's "
        "residual action term (6, 6+n), weight x core.hands.FINGER_PENALTY_REF_N/n and the "
        "residual-L2 norm deadzone x sqrt(n/REF_N) (total penalty kept at the old span's level). "
        "legacy = core.hands.LEGACY_FINGER_PENALTY_DIMS, weight/deadzone x1 - explicit choice "
        "only (reproducing/resuming runs of that older span). Retired names "
        "(core.hands.RETIRED_FINGER_PENALTY_SPANS: derived, derived_scaled) die. Train-only "
        "reward: the eval path marks it and consumes nothing. "
        "Owner: core.hands.finger_penalty_plan.",
    )


def _add_wrist_hinge_and_reset_args(parser) -> None:
    """Wrist residual-size hinge, reset scheme and actor weight decay."""
    # The "hinge" of these four flags is the penalty-free zone of the residual-size penalty, NOT a tracking hinge
    # (that is --enable_hinge).
    parser.add_argument(
        "--wrist_l2_hinge",
        type=int,
        default=0,
        choices=[0, 1],
        help="Wrist residual-L2 FREE ZONE (NOT a tracking hinge): replace "
        "the wrist residual_l2 norm-deadzone with a PER-DIM hinge "
        "-sum relu(|r_i|-dz)^2, dz ramped 0 -> wrist_l2_hinge_free_frac by the "
        "WristResidualFreeZoneRamp curriculum. Bounds how large the residual "
        "ACTION may grow untaxed; says nothing about tracking error. Requires "
        "the wrist split (--wrist_residual_l2_weight). 0 = off.",
    )
    parser.add_argument(
        "--wrist_l2_hinge_free_frac",
        type=float,
        default=0.7,
        help="Residual free zone, final width per dim, as a fraction of the hard cap "
        "(clamp(raw,±1)*cap mapping => raw units). 0.7 = 70%% of cap untaxed; "
        "with caps 7cm/120deg that is 4.9cm / 84deg of residual action.",
    )
    parser.add_argument(
        "--wrist_l2_hinge_start_frac",
        type=float,
        default=0.10,
        help="Residual free-zone ramp START as a fraction of total training steps. "
        "Before this: dz=0 (per-dim full-L2 guide phase).",
    )
    parser.add_argument(
        "--wrist_l2_hinge_end_frac",
        type=float,
        default=0.25,
        help="Residual free-zone ramp END as a fraction of total training steps. After "
        "this: dz = wrist_l2_hinge_free_frac (held; eval keeps this value).",
    )
    parser.add_argument(
        "--reset_scheme",
        type=str,
        default="or",
        choices=["or", "park"],
        help="Bimanual episode reset scheme. 'or' (default): either side's "
        "gate/contact termination ends the episode for BOTH hands. 'park' "
        "(under validation): the failed side publishes done_side (its GAE is "
        "cut), then PARKS — kinematic-follows its reference every step (single "
        "clock kept), reward 0 for that side — while the other side keeps "
        "learning to the clip end; the episode ends when both sides are parked. "
        "Requires --per_side_advantage 1. The bimanual value is the recipe's bi column "
        "(configs/rl/recipe.yaml).",
    )
    parser.add_argument(
        "--actor_weight_decay",
        type=float,
        default=0.0,
        help="AdamW-style weight decay applied to the PPO optimizer "
        "(actor+critic). Directly opposes weight-norm inflation. Try 1e-4-1e-3.",
    )


def _add_ppo_override_args(parser) -> None:
    """Early stop, fingertip force weight, pure tracking and the PPO hyperparameter overrides."""
    parser.add_argument(
        "--early_stop_return_frac",
        type=float,
        default=0.0,
        help="early-stop if frozen-eval episode return falls below "
        "this fraction of its running max for --early_stop_patience evals "
        "(0 = off). Try 0.5.",
    )
    parser.add_argument(
        "--early_stop_patience", type=int, default=3, help="consecutive sub-threshold evals before early stop."
    )
    parser.add_argument(
        "--fingertip_force_weight",
        type=float,
        default=None,
        help="Override the (deprecated) RewardFingertipForce weight. Exp-1 sets 0 "
        "to disable the stale force-only contact reward for a fair baseline "
        "(it has no pre-contact gradient). None = leave cfg default.",
    )
    parser.add_argument(
        "--pure_tracking",
        action="store_true",
        default=False,
        help="Diagnostic: zero out object-related reward weights, keep only tracking rewards.",
    )
    # -------- PPO hyperparameter overrides --------
    # All default None = use YAML value. Each flag overrides the corresponding YAML key.
    parser.add_argument(
        "--value_loss_scale", type=float, default=None, help="Override PPO value_loss_scale. ManipTrans uses 4.0."
    )
    parser.add_argument("--entropy_loss_scale", type=float, default=None, help="Override PPO entropy_loss_scale.")
    parser.add_argument(
        "--mini_batches",
        type=int,
        default=None,
        help="Override PPO mini_batches (yaml default 4). Unlocks token-arch training on a "
        "On a 32 GB GPU, 8 halves the attention activation memory per PPO update (avoids OOM).",
    )
    parser.add_argument("--learning_rate", type=float, default=None, help="Override PPO (constant) learning_rate.")
    parser.add_argument(
        "--kl_threshold", type=float, default=None, help="Override PPO kl_threshold for KL early stop (0 = disabled)."
    )
    parser.add_argument("--ratio_clip", type=float, default=None, help="Override PPO ratio_clip.")
    parser.add_argument("--grad_norm_clip", type=float, default=None, help="Override PPO grad_norm_clip.")
    parser.add_argument("--initial_log_std", type=float, default=None, help="Override policy initial_log_std.")
    parser.add_argument("--max_log_std", type=float, default=None, help="Override policy max_log_std (upper clip).")
    parser.add_argument("--min_log_std", type=float, default=None, help="Override policy min_log_std (lower clip).")
    parser.add_argument(
        "--freeze_curriculum",
        action="store_true",
        default=False,
        help="Disable threshold tightening curriculum (scale=1.0 throughout).",
    )


def _add_physics_ramp_args(parser) -> None:
    """Gravity ramp, friction decay, object velocity perturbation and tightening step scale."""
    parser.add_argument(
        "--disable_gravity_ramp",
        action="store_true",
        default=False,
        help="Start at full gravity (start_fraction=1.0, ramp_steps=0).",
    )
    parser.add_argument(
        "--gravity_ramp_end_iter",
        type=int,
        default=None,
        help="Iteration at which gravity reaches 100%% (linear ramp from 0%% at iter 0). "
        "Sets ramp_steps = gravity_ramp_end_iter * rollouts. Overrides "
        "--gravity_ramp_fraction. Mutually exclusive with --disable_gravity_ramp.",
    )
    parser.add_argument(
        "--gravity_ramp_fraction",
        type=float,
        default=0.10,
        help="gravity_ramp + friction_decay ramp duration as a FRACTION of total "
        "training timesteps (= max_iterations * rollouts), resolved at runtime to "
        "ramp_steps = int(total_timesteps * fraction). Default 0.10 (first 10%%) — "
        "reproduces the historical 6400 steps at the canonical 2000 iters x 32 rollouts "
        "and scales automatically when total timesteps change. --gravity_ramp_end_iter "
        "(absolute) and --disable_gravity_ramp still take precedence.",
    )
    parser.add_argument(
        "--disable_friction_decay",
        action="store_true",
        default=False,
        help="Start at final friction (initial_multiplier=final_multiplier).",
    )
    parser.add_argument(
        "--perturb_obj_vel",
        type=float,
        default=0.0,
        help="TRAIN robustify: additive random velocity kicks on the RIGHT object "
        "(native mdp.push_by_setting_velocity, interval 1-2 s). Max |linear| in m/s per axis. "
        "0 (default) = no event injected. Basin-widening lever for marginal grasp/rotation "
        "policies (uni-robustify -> attach pipeline).",
    )
    parser.add_argument(
        "--perturb_obj_angvel",
        type=float,
        default=0.0,
        help="TRAIN robustify: max |angular| velocity kick in rad/s per axis "
        "(same interval event as --perturb_obj_vel).",
    )
    parser.add_argument(
        "--tighten_steps_scale",
        type=float,
        default=None,
        help="Override the linear-schedule anneal_steps fraction-of-total-timesteps factor (default 0.5).",
    )


def _add_pose_render_args(parser) -> None:
    """Initial-pose still render (policy-free diagnostic)."""
    # --- Initial-pose still render ---
    # A policy-free diagnostic, so no --checkpoint: the reset sends the arm to the arm-IK frame-0 solution, the
    # hand to the reference qpos and the object to the reference pose. Needs --render_per_clip_videos --num_envs 1
    # (front_camera).
    parser.add_argument(
        "--pose_render_out",
        type=str,
        default="",
        help="(Eval only) Render the reference frame-0 initial pose as PNGs into this directory "
        "and exit. The 5 MANO fingertips are overlaid as green spheres. "
        "No checkpoint needed. Empty string (default) = no-op.",
    )
    parser.add_argument(
        "--pose_render_views",
        type=str,
        default="iso,front,side,top",
        help="View list of --pose_render_out (iso,front,side,back,top).",
    )
    parser.add_argument(
        "--pose_render_zoom",
        type=float,
        default=2.6,
        help="Camera distance of --pose_render_out = robot bounding-box radius x this value.",
    )
    parser.add_argument(
        "--pose_render_frame",
        type=int,
        default=0,
        help="Reference frame rendered by --pose_render_out (default 0 = initial pose).",
    )
    parser.add_argument(
        "--pose_render_seq",
        type=str,
        default="",
        help="Sequence rendered by --pose_render_out: picks the clip when --sequence_dir loads "
        "several. Empty = the clip assigned to env 0.",
    )
