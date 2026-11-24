"""Pilot flags, first half: physics DR, the B1a components and contact-aware lag compensation (Phase 1) (the rest:
pilot_contact).

Pure argparse (no Isaac imports); flag names, types, defaults and actions are the launcher's argv contract
(tests/rl/launch_cases/*.case)."""

from __future__ import annotations


def add_pilot_args(parser) -> None:
    """Add the pilot flags to ``parser`` (in launch-contract order)."""
    _add_physics_dr_and_aux_args(parser)
    _add_policy_shape_args(parser)
    _add_hinge_variant_args(parser)
    _add_lag_compensation_args(parser)
    _add_object_mode_args(parser)


def _add_physics_dr_and_aux_args(parser) -> None:
    """Physics DR, additional components and the auxiliary force loss."""
    # -------- Paper 1 B1a pilot flags --------
    # Each flag independently activates one component; together they reproduce P0.1→P0.4
    # and the four ablation runs of the pilot.
    parser.add_argument(
        "--enable_physics_dr",
        type=str,
        default="none",
        choices=["none", "train", "train_narrow", "near_ood", "far_ood"],
        help=(
            "Paper 1 B1a P0.2: per-episode physics DR range to activate. "
            "'none'=disabled (default, P0.1 / R1 baseline). "
            "'train'=training range [mass:0.7-1.3, friction:0.5-1.5, restitution:0.5-1.5, COM:±2cm] (P0.2 / R2-R4 "
            "train). "
            "'near_ood'=near-OOD eval ranges. "
            "'far_ood'=far-OOD eval ranges."
        ),
    )
    parser.add_argument(
        "--enable_additional_components",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Paper 1 B1a P0.3: enable grasp-stability contact additional components. "
            "0=disabled (default). 1=set RewardGraspStability weight to 0.2."
        ),
    )
    parser.add_argument(
        "--aux_force_loss_scale",
        type=float,
        default=0.0,
        help=(
            "Paper 1 B1a R2 (Round 2): aux contact-force prediction loss weight. "
            "0.0=disabled (default, matches P0.1-P0.4 baseline). "
            "0.1=enabled (recommended). Adds LAMBDA*MSE(pred_force, next_frame_force) to PPO loss."
        ),
    )
    parser.add_argument(
        "--aux_force_warmup_iter",
        type=int,
        default=0,
        help=(
            "Paper 1 B1a R2.5 (H7 two-stage training): iters to wait before activating aux loss. "
            "0=enable from iter 0 (default). >0=hinge alone for N iters, then enable aux. "
            "Tests whether aux+hinge interference is a timing issue (encoder must settle into "
            "hinge-relevant features before aux supervision is applied)."
        ),
    )
    parser.add_argument(
        "--aux_decouple_encoder",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Paper 1 B1a R2.5 (H3 encoder decoupling): if 1, aux head uses a separate aux_encoder. "
            "Aux loss does NOT backpropagate into the main encoder. Policy receives aux_encoder "
            "features as detached additional input (concatenation). Tests whether shared-encoder "
            "gradient interference was the failure mode of naive aux+hinge."
        ),
    )


def _add_policy_shape_args(parser) -> None:
    """Policy size/architecture, per-side advantage and human-pose reward removal."""
    parser.add_argument(
        "--policy_size",
        type=str,
        # Canonical: tiny (with the token arch).
        default="tiny",
        choices=["tiny", "tiny2x", "tiny3x", "small", "base", "large"],
        help="Size preset for --policy_arch token (d_model/layers): tiny(64/2)~0.12M, "
        "tiny2x(96/2)~0.25M, tiny3x(112/2)~0.34M, "
        "small(128/4)~0.9M, base(256/6)~4.8M, large(384/8)~14M. MUST match between "
        "train and eval (state_dict shapes fail loud on mismatch).",
    )
    parser.add_argument(
        "--policy_arch",
        type=str,
        # Canonical: the token transformer; the legacy MLP is --policy_arch shared.
        default="token",
        choices=["shared", "perhand", "structured", "small", "token"],
        help=(
            "Bimanual actor architecture. 'token' (default) = see below. 'shared' = single MLP over the "
            "full obs -> full action (also the only valid arch for single-hand). "
            "'perhand' = two-tower decoupled actor: right obs slice -> right tower -> right action[12], "
            "left obs slice -> left tower -> left action[12], concatenated [right,left]; no shared trunk "
            "or gradient coupling. Critic stays a single shared net on the full obs. Requires --bimanual 1. "
            "Motivation: the shared MLP coupled the two hands' exploration and let one hand's residual "
            "actor-gain cascade drag the whole policy down."
        ),
    )
    parser.add_argument(
        "--per_side_advantage",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "PER-SIDE ADVANTAGE (factored credit) for bimanual PPO. 0 (default) = off. "
            "1 (requires --bimanual 1 + token/small "
            "two-tower arch): the env publishes per-step reward_right/reward_left "
            "(side-tagged reward terms; shared/global terms split 50/50 so R+L == total), "
            "the critic gets TWO heads [V_right, V_left], GAE runs per side, and each "
            "tower's log-prob (mu dims [0:12]=right, [12:24]=left) is graded ONLY by its "
            "own side's advantage. KL for early-stop/adaptive-LR = sum of per-side KLs. "
            "Checkpoint format in this mode differs (2-dim value head + 2-dim value "
            "preprocessor); eval of such a checkpoint must pass the same flag."
        ),
    )
    parser.add_argument(
        "--disable_human_pose_rewards",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Paper 1 B1a R2.5 (G limit-case reward): if 1, zero out all human-pose imitation rewards "
            "(eef_pos, eef_rot, all 5 fingertips, level_1, level_2). Keeps object rewards, contact "
            "rewards, and motion penalties. Tests whether imitation framework is required at all."
        ),
    )


def _add_hinge_variant_args(parser) -> None:
    """Human-pose weight scale, finger/contact-gated/wrist hinge variants and action-aware aux."""
    parser.add_argument(
        "--human_pose_weight_scale",
        type=float,
        default=1.0,
        help=(
            "Paper 1 B1a R2.5 (H10): scale all human-pose imitation reward weights by this factor. "
            "1.0=default (no change). 0.5=halve all (eef_pos, eef_rot, fingertips, level_1/2)."
        ),
    )
    parser.add_argument(
        "--enable_finger_hinge_asymmetric",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Paper 1 B1a R2.5 (H6): if 1, apply per-finger asymmetric hinge — thumb/index/middle "
            "threshold 3cm (conflict-heavy fingers), ring/pinky 2cm. Requires --enable_finger_hinge 1."
        ),
    )
    parser.add_argument(
        "--enable_contact_gated_hinge",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Paper 1 B1a R2.5 (H10 reward-form variant): if 1, hinge threshold becomes state-dependent. "
            "Fingertips in contact (||F||>5N) use the in-contact threshold (4cm by default); free "
            "fingertips use --enable_finger_hinge baseline (2cm). Tests 'contact moments are exactly "
            "when the policy needs license to deviate from mocap'. Requires --enable_finger_hinge 1."
        ),
    )
    parser.add_argument(
        "--enable_wrist_hinge",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Paper A2: if 1, apply hinge to eef_pos (wrist position) reward with threshold=0.04m (4cm). "
            "Tests whether unrelieved wrist tracking pressure was constraining the hinge mechanism. "
            "Note: eef_rot is a no-op constant in full-robot mode (IK handles wrist orientation), "
            "so no hinge is applied to eef_rot. Default 0 = disabled."
        ),
    )
    parser.add_argument(
        "--aux_action_aware",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "Paper A1: if 1, aux head receives current action concatenated with encoder features. "
            "Aux head input dim becomes 256+act_dim=268. Requires --aux_force_loss_scale > 0. "
            "Default 0 preserves bit-exact behavior of all pre-A1 baselines."
        ),
    )


def _add_lag_compensation_args(parser) -> None:
    """Contact-aware lag compensation (slowdown, look-ahead obs, contact thresholds) and input paths."""
    # Contact-aware lag compensation Phase 1 flags (all default 0 / 1.0 = off).
    parser.add_argument(
        "--slowdown_factor",
        type=float,
        default=1.0,
        help="Phase 1 H1(a): runtime trajectory slow-down factor. 1.0 = no slow-down "
        "(default). 1.5 = stretch trajectory in time by 1.5x via linear interp / "
        "Slerp. Episode length auto-scales.",
    )
    parser.add_argument(
        "--enable_lookahead_obs",
        type=int,
        default=0,
        choices=[0, 1],
        help="Phase 1 H1(b): add lookahead wrist target obs (delta to ref at t+k). "
        "Adds 7*K dims per side (default 14 per side, K=2 at t+10 and t+30).",
    )
    parser.add_argument(
        "--lookahead_frames",
        type=str,
        default="10,30",
        help="Comma-separated lookahead frame offsets for --enable_lookahead_obs. "
        "Default '10,30' = 7*2 = 14 dims per side.",
    )
    parser.add_argument(
        "--contact_force_threshold",
        type=float,
        default=0.5,
        help="PhysX fingertip force threshold (N) for robot 'in contact'. Used by the "
        "contact-match reward distance/force gate.",
    )
    parser.add_argument(
        "--contact_distance_threshold",
        type=float,
        default=0.01,
        help="Tip-to-object-surface distance threshold (m) for robot 'in contact'. "
        "Combined with force threshold via AND to reject desk/table false positives. "
        "Default 0.01 = 1cm. Used by the contact-match reward.",
    )
    parser.add_argument(
        "--object_mesh_dir_right",
        type=str,
        default=None,
        help="RIGHT object mesh directory (containing surface_points.npz); the active-object mesh "
        "for right-side runs. --side left runs use --object_mesh_dir_left. "
        "With --multi_object 1: comma-list aligned with the --sequence order.",
    )
    parser.add_argument(
        "--sequence_dir",
        type=str,
        default=None,
        help="FOLDER-based multi-sequence loading: use EVERY sequence directory under this "
        "folder (per-seq layout = data/<ds>/trajectories: <seq>/0/trajectory_kinematic.npz "
        "+ <seq>/task_info.json). Per-seq objects resolve from task_info.json "
        "({side}_object_mesh_dir); >1 distinct object auto-enables --multi_object. "
        "Placement = whatever _shift variant dirs the folder contains (placement search "
        "is separate and NOT finalized). Mutually exclusive "
        "with --sequence and the --object_usd_*/--object_mesh_dir_* flags; uni only.",
    )


def _add_object_mode_args(parser) -> None:
    """Multi-object and shared-object scene modes."""
    parser.add_argument(
        "--multi_object",
        type=int,
        default=0,
        choices=[0, 1],
        help="Multi-object multi-clip training: the --sequence comma-list clips use DIFFERENT "
        "objects. The active side's --object_usd_*/--object_mesh_dir_* become comma-lists aligned with the "
        "--sequence order (normalized to loader order at startup). One mesh per env via a "
        "deterministic MultiAssetSpawner (env_i -> clip/asset i %% n; static clip assignment, "
        "no per-reset clip resampling). Supports right-uni and "
        "--side left (single object entity); NOT supported with --bimanual/--dual_eval. "
        "0 (default) = the single-object path.",
    )
    parser.add_argument(
        "--shared_object",
        type=int,
        default=0,
        choices=[0, 1],
        help="Bimanual SHARED object: both hands manipulate ONE physical object, so the scene "
        "spawns ONE prim (object_right) and every left-side object term/sensor/DR re-points to it "
        "(env_setup._apply_shared_object). --sequence_dir sets this itself from each tree's "
        "task_info objects[] (role=main side=both, or identical main qpos block) and REFUSES a mix of "
        "shared/non-shared clips in one run. 0 (default) = two prims.",
    )
