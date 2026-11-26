"""Run flags, first half: task, robot, data, run control, logging, resume and objects (the rest: run_outputs).

Pure argparse (no Isaac imports); flag names, types, defaults and actions are the launcher's argv contract
(tests/rl/launch_cases/*.case)."""

from __future__ import annotations

from dexengine.rl.cli_args.choices import ARM_CHOICES, HAND_CHOICES, TASK_CHOICES


def add_run_args(parser) -> None:
    """Add the run flags to ``parser`` (in launch-contract order)."""
    _add_task_and_eval_args(parser)
    _add_data_and_logging_args(parser)
    _add_wandb_eval_and_resume_args(parser)
    _add_bimanual_scene_args(parser)
    _add_side_and_randomize_args(parser)


def _add_task_and_eval_args(parser) -> None:
    """Task, robot, env count, iteration budget, in-training eval and the checkpoint flag."""
    parser.add_argument("--task", type=str, required=True, choices=TASK_CHOICES, help="Pipeline mode.")
    # --arm/--hand have no default: the launcher always passes both.
    parser.add_argument(
        "--arm",
        type=str,
        required=True,
        choices=ARM_CHOICES,
        help="Arm platform; choices are DERIVED from ARM_REGISTRY (dexengine.core.arms). "
        "Required (no default): the launcher states it. A registry "
        "entry whose spec is None is a stub and fail-fasts with the porting checklist.",
    )
    parser.add_argument(
        "--hand",
        type=str,
        required=True,
        choices=HAND_CHOICES,
        help="Hand end-effector; choices are DERIVED from HAND_REGISTRY "
        "(dexengine.core.hands). A registry entry "
        "whose spec is None is a stub and fail-fasts with the porting checklist. "
        "Port a hand by filling a HandSpec — SHARPA_WAVE_RIGHT is the worked example.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=None, help="Override num envs. Default: 4096 (train), 1 (eval/replay)."
    )
    parser.add_argument("--max_iterations", type=int, default=5000, help="PPO iterations (train only).")
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=50,
        help="Frozen-eval/checkpoint interval in iterations (train only). Canonical 50 "
        "(a longer interval lets best_agent miss the best point).",
    )
    parser.add_argument(
        "--eval_evals_per_clip",
        type=int,
        default=4,
        help="Frozen eval: episodes per clip. Total = num_clips * this. Default: 4",
    )
    parser.add_argument(
        "--eval_threshold_progress",
        type=str,
        default="1.0",
        help="Threshold-schedule progress forced at eval time: float in [0,1] or the "
        "string 'final' (= 1.0). Applied at BOTH the in-training frozen-eval "
        "forcing site and the standalone-eval forcing site. Default 1.0 = current "
        "behavior (final/tight thresholds). E.g. 0.245 evaluates at the effective "
        "thresholds of a run whose anneal stopped at 24.5%% progress.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Policy checkpoint to evaluate (required for --task eval)."
    )


def _add_data_and_logging_args(parser) -> None:
    """Dual eval checkpoints, dataset selection, experiment name, seed, distributed and wandb project."""
    parser.add_argument(
        "--dual_eval",
        type=int,
        default=0,
        choices=[0, 1],
        help="Two-policy dual eval/render: run the RIGHT and LEFT uni policies "
        "independently in one bimanual scene (obs 414 -> 207/207, actions "
        "12|12 -> 24; no merged checkpoint). Requires --bimanual 1 plus "
        "--dual_ckpt_right/--dual_ckpt_left.",
    )
    parser.add_argument(
        "--dual_ckpt_right", type=str, default=None, help="RIGHT-uni policy checkpoint for --dual_eval 1."
    )
    parser.add_argument(
        "--dual_ckpt_left", type=str, default=None, help="LEFT-uni policy checkpoint for --dual_eval 1."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Trajectory data directory. Either this OR --dataset must be set. "
        "If both, --data_dir wins (lets you override the registry default).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset id (dataset registry, dexengine.core). Resolves train_data_dir + "
        "heldout_data_dir + object_usd + object_mesh_dir from the registry. "
        "Either this OR --data_dir must be set; explicit --data_dir / "
        "--heldout_data_dir / --object_usd / --object_mesh_dir override the "
        "registry values.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "heldout"],
        help="When --dataset is set, picks which split's data_dir to use. "
        "Train→registry.train_data_dir; Heldout→registry.heldout_data_dir. "
        "Ignored when --data_dir is explicitly passed.",
    )
    parser.add_argument("--sequence", type=str, default=None, help="Sequence filter for trajectory manager.")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name. Default: auto-generated.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        default=False,
        help="Multi-GPU PPO via torch.distributed (train only). Launch with the Isaac Lab "
        "standard pattern: isaaclab.sh -p -m torch.distributed.run --nnodes=1 "
        "--nproc_per_node=<N> dexengine/rl/cli.py ... --distributed. AppLauncher "
        "binds each rank to cuda:LOCAL_RANK; --num_envs is PER-GPU; the effective "
        "seed becomes seed+rank; only rank 0 writes checkpoints/W&B/tee-log.",
    )
    parser.add_argument("--wandb", action="store_true", default=False, help="Enable W&B logging (train only).")
    parser.add_argument("--wandb_project", type=str, default="dexengine-train", help="W&B project name.")


def _add_wandb_eval_and_resume_args(parser) -> None:
    """Wandb group/tags, eval steps and resume flags."""
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="W&B group name (e.g. 'lag_compensation_ablation') for run aggregation. None = ungrouped.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default=None,
        help="Comma-separated W&B tags (e.g. 'uni,taco,seed_42'). None = no tags.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=10000,
        help="[DEPRECATED] Timesteps for standalone eval mode. "
        "Standalone eval now uses episode-based loop (run_eval_balanced) "
        "and this argument is ignored. Kept for backward CLI compatibility.",
    )
    parser.add_argument(
        "--eval_warmup_batches",
        type=int,
        default=1,
        help="Number of warmup batches to discard from standalone eval (eliminates cold-start bias). Default 1.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint to load into the agent BEFORE training starts "
        "(diagnostic: run the trained policy through the training code path).",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume training from this checkpoint (loads PPO agent weights)."
    )
    parser.add_argument(
        "--resume_iteration",
        type=int,
        default=0,
        help="With --resume_from, CONTINUE training at this iteration: trainer.initial_timestep and "
        "env.common_step_counter are set to iteration x rollouts, so every iteration-based "
        "schedule (threshold_tightening anneal, ramps, frozen-eval period, total MI) is kept. "
        "0 (default) = start at iteration 0 (weights only).",
    )


def _add_bimanual_scene_args(parser) -> None:
    """Curriculum offset/horizon, MANO fingertip viz and the per-side object USDs (bimanual routing)."""
    parser.add_argument(
        "--curriculum_step_offset",
        type=int,
        default=None,
        help="Seed env.common_step_counter with N so every step-based schedule "
        "(threshold tightening, obj-follow ramp, level1/2 ramp-down, RSI band, "
        "gravity/friction DR, collision curriculum) CONTINUES instead of "
        "restarting. Default None = auto: with --resume/--resume_from, derive "
        "N from the checkpoint (agent_<N>.pt -> N; best_agent.pt -> MI from the "
        "run's TRAIN_DONE sentinel x 32 rollouts); without resume, 0. "
        "Pass 0 to force a "
        "fresh schedule on top of resumed weights.",
    )
    parser.add_argument(
        "--curriculum_horizon_iterations",
        type=int,
        default=None,
        help="Denominator (iterations) of the curriculum schedules. Default None = --max_iterations. "
        "Every schedule computed as a fraction of training (object-follow ramp 75 %%, gate "
        "tightening 50 %% — the force curriculum and the imitation release use this progress — "
        "RSI, gravity, friction, wrist residual, VOC, collision curriculum) uses this length. "
        "For an extended resume (e.g. MI 3200 from a 1600 checkpoint) pass the original run "
        "length (1600): the schedules then stay at their final values instead of rewinding. "
        "1 <= value <= --max_iterations.",
    )
    parser.add_argument(
        "--object_usd_right",
        type=str,
        default=None,
        help="RIGHT object USD path (the active object for right-side runs and the "
        "right partner in bimanual). --side left runs use --object_usd_left. "
        "With --multi_object 1: comma-list aligned with --sequence.",
    )
    # Bimanual: separate right and left residual paths trained jointly (summed reward); routes the task to the
    # bimanual gym id and adds the left object USD / mesh.
    parser.add_argument(
        "--bimanual",
        type=int,
        default=0,
        choices=[0, 1],
        help="Bimanual Option 2: 0 (default)=right-only. "
        "1 = SEPARATE right+left residual paths trained jointly in one env "
        "(summed reward). Routes train/eval to the "
        "Bimanual gym id; requires --object_usd_left + --object_mesh_dir_left.",
    )
    parser.add_argument(
        "--object_usd_left",
        type=str,
        default=None,
        help="LEFT object USD path (Bimanual Option 2). Required when --bimanual 1.",
    )


def _add_side_and_randomize_args(parser) -> None:
    """Left object meshes, uni side selection and finger/wrist randomization."""
    parser.add_argument(
        "--object_mesh_dir_left",
        type=str,
        default=None,
        help="LEFT object mesh directory (surface_points.npz) for the left contact rewards (Bimanual Option 2).",
    )
    # Uni side selection (symmetric interface). Each side is the exact mirror of the other: single 12-dim policy
    # on that hand, the side's object in the single 'object' scene slot (--object_usd_<side>), obs 207,
    # self-collision OFF, the OTHER arm/hand parked at default PD.
    parser.add_argument(
        "--side",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Which hand the uni policy controls. 'right' (default) = right-uni "
        "build (RightUni gym id); 'left' = the exact LEFT mirror (LeftUni gym "
        "id, 12-dim left_hand_joint_pos, right arm parked at default PD, "
        "DEXENGINE_RIGHT_PARK set internally). Pass the active object via "
        "--object_usd_<side>/--object_mesh_dir_<side>. Ignored-with-error "
        "under --bimanual (two-policy scenes are side-neutral).",
    )
    parser.add_argument(
        "--randomize_finger",
        action="store_true",
        default=False,
        help="Enable finger opening randomization at initial pose.",
    )
    parser.add_argument(
        "--randomize_wrist",
        action="store_true",
        default=False,
        help="Enable wrist Z (upward) randomization at initial pose.",
    )
