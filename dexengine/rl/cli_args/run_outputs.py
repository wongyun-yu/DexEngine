"""Run flags, second half: pose randomization, video and render outputs, arm IK smoothing and gains, frozen eval and
the reference-trajectory switch.

Pure argparse (no Isaac imports); flag names, types, defaults and actions are the launcher's argv contract
(tests/rl/launch_cases/*.case). dexengine.rl.args calls this right after run.add_run_args."""

from __future__ import annotations

from dexengine.rl.cli_args.choices import _ArmGainGiven


def add_run_output_args(parser) -> None:
    """Add the second half of the run flags to ``parser`` (in launch-contract order)."""
    _add_video_args(parser)
    _add_video_selection_args(parser)
    _add_video_rank_and_action_args(parser)
    _add_arm_gain_and_mass_args(parser)
    _add_residual_scale_args(parser)


def _add_video_args(parser) -> None:
    """Reset randomization fraction, wrist limits and the video / per-clip render flags."""
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
    parser.add_argument("--video", action="store_true", default=False, help="Record debug video.")
    parser.add_argument(
        "--video_dir",
        type=str,
        default="videos/",
        help="Video output directory (relative = under the run tree, not the cwd).",
    )
    parser.add_argument(
        "--render_per_clip_videos",
        action="store_true",
        default=False,
        help="Eval-only: render one success + one failure video per clip. Forces --num_envs 1. Off by default.",
    )
    parser.add_argument(
        "--render_markers",
        type=int,
        default=1,
        choices=[0, 1],
        help="Replay render: draw the RGB wrist/object TARGET+CURRENT frame markers (the 'arrows'). "
        "1=on (default), 0=off (clean render, no frames). Active-hand only: right-uni hides left "
        "frames, left-uni hides parked-right frames, dual draws both.",
    )
    parser.add_argument(
        "--kinematic_replay",
        action="store_true",
        default=False,
        help="Render-only: KINEMATIC replay of the REFERENCE per frame (no policy/physics) — teleport "
        "the robot to the all-frame arm-IK + reference finger qpos and the object to its reference "
        "pose, then render front+ego. Produces the photorealistic reference trajectory (Isaac front "
        "view + per-frame PNGs) for RSI frame selection. Requires --render_per_clip_videos --num_envs 1.",
    )
    parser.add_argument(
        "--videos_max_attempts",
        type=int,
        default=100,
        help="Max rollouts per clip when --render_per_clip_videos is set.",
    )
    parser.add_argument(
        "--videos_out_dir",
        type=str,
        default="videos/per_clip",
        help="Output dir for per-clip videos + manifest (relative = under the run tree, not the cwd). "
        "Default: videos/per_clip",
    )


def _add_video_selection_args(parser) -> None:
    """Video fps, episode/attempt budgets, best-success selection and per-clip eval."""
    parser.add_argument(
        "--videos_fps",
        type=int,
        default=50,
        help="Output video FPS. Sim runs at 50 Hz, so the default 50 is REAL TIME. Pass 25 for "
        "the old 0.5x slow-motion replay (that used to be the default; the side-by-side "
        "composer now probes the file's fps instead of assuming 25, so either works).",
    )
    parser.add_argument(
        "--videos_max_episode_steps", type=int, default=600, help="Cap per-episode steps during video rendering."
    )
    parser.add_argument(
        "--videos_warmup_attempts",
        type=int,
        default=5,
        help="Per-clip warmup rollouts to discard before counting (sim-settling). Default 5.",
    )
    # 1 (default) writes all four views for a success or a failure (best / failure_best); 0 is the old
    # first-success + first-failure path, which writes only the ego/front views.
    parser.add_argument(
        "--videos_select_best_success",
        type=int,
        default=1,
        choices=[0, 1],
        help="When 1, run all evaluated rollouts per clip and save only the success "
        "with highest mean obj_pos reward (drops failure videos entirely). "
        "When 0 (default), legacy behavior: save first success + first failure. "
        "Bit-identical to pre-flag behavior when 0.",
    )
    parser.add_argument(
        "--videos_rollouts_per_clip",
        type=int,
        default=None,
        help="Fixed number of evaluated rollouts per clip when --videos_select_best_success=1. "
        "Default None → falls back to --videos_max_attempts.",
    )
    parser.add_argument(
        "--videos_perclip_eval",
        type=str,
        default=None,
        help="JSON {short_seq_id: completion_rate 0..1} from the eval pass. best-success "
        "renders adapt each clip's retry budget: rate==0 -> one batch (fast "
        "failure_best); rate>0 -> cap ceil(4.6/rate) rollouts (~99%% success coverage).",
    )
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


def _add_video_rank_and_action_args(parser) -> None:
    """Video clip filter, retry and ranking; wrist position scale and action EMA."""
    parser.add_argument(
        "--videos_only_clips",
        type=str,
        default=None,
        help="Comma-separated substrings; render ONLY clips whose name contains one of them "
        "(others are skipped, not counted as failures). Used by the 2-clip pair render "
        "(target + pad) so the pad clip is never rolled out.",
    )
    parser.add_argument(
        "--videos_allfail_retry_batches",
        type=int,
        default=4,
        help="Under --videos_select_best_success=1: if ALL rollouts in the initial batch "
        "(--videos_rollouts_per_clip) fail, keep running additional batches of the same "
        "size until at least one success is found, up to this many EXTRA batches "
        "(total cap = rollouts_per_clip * (1 + this)). A run whose eval completion "
        "rate is >0 should yield a success video, not failure_best. "
        "0 = legacy single-batch behavior.",
    )
    parser.add_argument(
        "--videos_rank_metric",
        type=str,
        default="obj_pos_mean",
        choices=[
            "obj_pos_mean",
            "obj_pos_cum",
            "obj_pos_obj_rot_mean",
            "fingertip_quality_mean",
            "fingertip_quality_cum",
        ],
        help="Which scalar to rank successful rollouts by under --videos_select_best_success=1. "
        "Envs without an obj_pos reward (hand-only) should use fingertip_quality_*; task envs use obj_pos_mean by "
        "default. "
        "Auto-fallback applies if rank_metric=obj_pos_* but obj_pos reward absent.",
    )
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
