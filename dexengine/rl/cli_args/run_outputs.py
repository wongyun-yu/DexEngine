"""Run flags, second half: pose randomization, the record-session seam, arm IK smoothing and gains, frozen eval and
the reference-trajectory switch. The removed video flags die with a pointer to the record render
(dexengine.rl.render_removed).

Pure argparse (no Isaac imports); flag names, types, defaults and actions are the launcher's argv contract
(tests/rl/launch_cases/*.case). dexengine.rl.args calls this right after run.add_run_args."""

from __future__ import annotations

from dexengine.rl import render_removed as RR
from dexengine.rl.cli_args.choices import _ArmGainGiven


def add_run_output_args(parser) -> None:
    """Add the second half of the run flags to ``parser`` (in launch-contract order)."""
    _add_reset_and_session_args(parser)
    _add_usd_args(parser)
    _add_action_args(parser)
    RR.add_removed_args(parser)
    _add_arm_gain_and_mass_args(parser)
    _add_residual_scale_args(parser)


def _add_reset_and_session_args(parser) -> None:
    """Reset randomization fraction, wrist limits and the record-session flags."""
    parser.add_argument(
        "--randomize_fraction",
        type=float,
        default=0.5,
        help="Fraction of episodes with randomized initial pose (0-1). Default: 0.5",
    )
    parser.add_argument(
        "--finger_open_deg", type=float, default=15.0, help="Max finger opening in degrees per joint. Default: 15.0"
    )
    parser.add_argument(
        "--wrist_z_max_cm", type=float, default=1.5, help="Max upward wrist displacement in cm (+Z only). Default: 1.5"
    )
    parser.add_argument(
        "--record_session",
        action="store_true",
        default=False,
        help="Hand the built env to the render function a record session attached (dexengine.rl.session_render; "
        "the launcher's dump modes set it, scripts/stage4_record.py replays those dumps). Without a record session "
        "the process dies: the RL trainer draws no frames. Needs --num_envs 1.",
    )
    parser.add_argument(
        "--kinematic_replay",
        action="store_true",
        default=False,
        help="Eval task, record session only: no policy and no checkpoint load — the record render writes the "
        "recorded states itself (dexengine.record.render.episode).",
    )
    parser.add_argument(
        "--videos_out_dir",
        type=str,
        default="videos/per_clip",
        help="Output folder handed to the record session's render function (relative = under the run tree).",
    )


def _add_usd_args(parser) -> None:
    """Robot USD variant."""
    parser.add_argument(
        "--usd",
        type=str,
        default=None,
        metavar="VARIANT",
        help="Robot USD variant to spawn — one choice for train, eval and render alike. "
        "Three equivalent forms: '_<variant>' / '<variant>' / '<arm>_<hand>_<variant>'. "
        "The asset path is data/usd_cache/robots/full_robot_gen/<arm>_<hand>_<variant>/. "
        "Unset = the env value, else the canonical plate (owner dexengine.core.arms.plate_variant — one plate for "
        "every process). An empty string names the body generation whose suffix is empty. Sets "
        "DEXENGINE_COMBO_USD_VARIANT internally (this flag wins over that env).",
    )


def _add_action_args(parser) -> None:
    """Wrist position scale and action EMA."""
    parser.add_argument(
        "--wrist_pos_scale",
        type=float,
        default=None,
        help="Override wrist position scale (inner action term). Default: use config value.",
    )
    parser.add_argument(
        "--action_ema",
        type=float,
        default=None,
        help="Override action EMA on inner action term. Default: use config value.",
    )


def _add_arm_gain_and_mass_args(parser) -> None:
    """IK target EMA, arm PD gains (global and per joint) and object mass overrides."""
    parser.add_argument(
        "--ik_target_ema",
        type=float,
        default=0.40,
        help="Low-pass on the IK JOINT TARGET (1.0 = none). Canonical 0.40. "
        "The policy steps at 50 Hz, so each action is a STEP change of the "
        "target; with a stiff arm that step becomes a torque spike and reads "
        "as stutter. Values ~0.3-0.6 smooth the staircase; the velocity lead "
        "(ARM_FF_MODE=lead) compensates the small extra lag it adds.",
    )
    parser.add_argument(
        "--arm_kp",
        action=_ArmGainGiven,
        type=float,
        default=900.0,
        help="Arm PD stiffness. Canonical 900.",
    )
    parser.add_argument(
        "--arm_kd",
        action=_ArmGainGiven,
        type=float,
        default=90.0,
        help="Arm PD damping. Canonical 90.",
    )
    parser.add_argument(
        "--arm_kp_per_joint",
        action=_ArmGainGiven,
        type=str,
        default=None,
        help="PER-JOINT arm stiffness: 7 comma-separated values (joint1..joint7, applied to "
        "BOTH arms via regex). Overrides --arm_kp. Real-robot driver parity: "
        "'240,240,240,240,24,31,25'. Applied on train AND eval paths.",
    )
    parser.add_argument(
        "--arm_kd_per_joint",
        action=_ArmGainGiven,
        type=str,
        default=None,
        help="PER-JOINT arm damping: 7 comma-separated values (joint1..joint7, both arms). "
        "Overrides --arm_kd. REAL-robot parity: '3,3,3,3,0.7,0.6,0.5'.",
    )
    parser.add_argument(
        "--object_mass_right",
        type=float,
        default=None,
        help="Override the RIGHT object total mass (kg), replacing density-based mass. "
        "e.g. 0.2 for a 200 g knife. Applied train+eval; physics-DR mass-scale "
        "(train only) randomizes around this nominal.",
    )
    parser.add_argument(
        "--object_mass_left",
        type=float,
        default=None,
        help="Bimanual: override the LEFT object total mass (kg). If unset, the left "
        "object keeps its own (USD/density) mass — --object_mass does NOT apply to "
        "the left. Set this explicitly to control the left object's mass.",
    )


def _add_residual_scale_args(parser) -> None:
    """Hand friction, frozen eval switch, residual scales and the reference base."""
    parser.add_argument(
        "--hand_friction",
        type=float,
        default=None,
        help="Set the robot/hand collision static+dynamic friction (runtime "
        "set_material_properties). Default: USD-baked value. e.g. 2.0 (> rubber).",
    )
    parser.add_argument(
        "--no-frozen-eval", action="store_true", default=False, help="Disable frozen evaluation during training."
    )
    parser.add_argument(
        "--residual_scale",
        type=float,
        default=None,
        help="Override residual scale (0.0 for base-only diagnostic). Default: use config value (0.3). "
        "Acts as the fallback when --wrist_residual_scale / --finger_residual_scale are not set.",
    )
    parser.add_argument(
        "--wrist_residual_scale",
        type=float,
        default=None,
        help="Per-component residual scale for wrist dims (0:6). Falls back to --residual_scale if not set.",
    )
    parser.add_argument(
        "--finger_residual_scale",
        type=float,
        default=None,
        help="Per-component residual scale for finger dims (everything after the 6 wrist dims: "
        "6:12 for RH56F1, 6:28 for SharpA). Falls back to --residual_scale if not set. "
        "With reference-base ON (recipe default A1=1) the finger "
        "base IS the reference qpos, so this is the allowed deviation from the human prior "
        "in normalized half-range units (campaign incumbent 0.3).",
    )
    parser.add_argument(
        "--use_trajectory_ref",
        action="store_true",
        default=False,
        help="Force trajectory-reference mode in action term (diagnostic).",
    )
    parser.add_argument(
        "--reference_base",
        type=int,
        default=0,
        help="Base action = kinematic reference (wrist+fingers); residual trains on top.",
    )
