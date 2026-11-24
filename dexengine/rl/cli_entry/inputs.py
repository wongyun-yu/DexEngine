"""PRE-KIT: validate and complete the parsed trainer arguments (side, dataset, active object, task requirements).

Every function takes the parser (errors go through ``parser.error``, rc 2) and the parsed namespace, which it
completes in place. Standard library only (plus the dataset registry, imported lazily).
"""

from __future__ import annotations

import os


def check_side(parser, args_cli) -> None:
    """Derive ``args_cli.left_uni`` from --side and check the --bimanual / --dual_eval combinations."""
    # --side {right,left} is the symmetric uni interface; left_uni (read by the env-cfg wiring) is derived from it.
    args_cli.left_uni = 1 if args_cli.side == "left" else 0
    if getattr(args_cli, "bimanual", 0) and args_cli.side != "right":
        parser.error("--side is a UNI-build switch; --bimanual scenes are side-neutral (drop --side).")

    # --dual_eval: two-policy dual eval needs the bimanual scene and both uni checkpoints.
    if getattr(args_cli, "dual_eval", 0):
        if not getattr(args_cli, "bimanual", 0):
            parser.error("--dual_eval 1 requires --bimanual 1 (two-hand scene).")
        if not (args_cli.dual_ckpt_right and args_cli.dual_ckpt_left):
            parser.error("--dual_eval 1 requires --dual_ckpt_right and --dual_ckpt_left.")


def check_left_uni(parser, args_cli) -> None:
    """--side left: no right-side object or gate flags; parks the right arm by setting DEXENGINE_RIGHT_PARK=1.

    Right-park = joint-state write + default PD pin every reset (mdp events); setting it here (not by the caller)
    makes train and standalone eval of a left-uni run identical.
    """
    if getattr(args_cli, "left_uni", 0):
        if getattr(args_cli, "bimanual", 0):
            parser.error("--side left is mutually exclusive with --bimanual")
        for _lu_flag in (
            "object_usd_right",
            "object_mesh_dir_right",
            "object_mass_right",
            "obj_pos_thresh_const_left",
            "obj_rot_thresh_const_left",
        ):
            if getattr(args_cli, _lu_flag, None):
                parser.error(
                    f"--side left forbids --{_lu_flag} (a left-uni build has NO right object/gate: "
                    f"pass --object_usd_left/--object_mesh_dir_left and the unsuffixed thresh flags)."
                )
        from dexengine.rl.tasks import switches  # stdlib only

        switches.override(right_park=True)
        print(
            "[side=left] DEXENGINE_RIGHT_PARK=1 set internally (right arm+fingers "
            "park at default PD every reset; events.py RIGHT-PARK gate)."
        )


def resolve_eval_threshold(parser, args_cli) -> None:
    """--eval_threshold_progress: a float in [0, 1], or 'final' (= 1.0)."""
    if str(args_cli.eval_threshold_progress).strip().lower() == "final":
        args_cli.eval_threshold_progress = 1.0
    else:
        try:
            args_cli.eval_threshold_progress = float(args_cli.eval_threshold_progress)
        except ValueError:
            parser.error(
                f"--eval_threshold_progress must be a float in [0,1] or 'final'; "
                f"got '{args_cli.eval_threshold_progress}'."
            )
        if not (0.0 <= args_cli.eval_threshold_progress <= 1.0):
            parser.error(f"--eval_threshold_progress must be in [0,1]; got {args_cli.eval_threshold_progress}.")


def resolve_dataset(parser, args_cli) -> None:
    """Fill the data paths from the dataset registry when --dataset is given.

    Either --data_dir or --dataset must be set. The registry fills train/heldout data dirs and the right object
    USD / mesh dir only where the explicit flags left them unset.
    """
    if args_cli.dataset is None and args_cli.data_dir is None:
        parser.error("Either --dataset or --data_dir must be set.")
    if args_cli.dataset is not None:
        try:
            from dexengine.core import dataset_dirs as _ds_mod

            _spec = _ds_mod.resolve(args_cli.dataset)
        except Exception as e:  # noqa: BLE001 — any resolve failure becomes a CLI usage error
            parser.error(f"--dataset='{args_cli.dataset}' failed to resolve: {e}")
        # Pick split for the primary data_dir
        _split_paths = {"train": _spec.train_data_dir, "heldout": _spec.heldout_data_dir}
        # With --sequence_dir (a tree view) the sequence_dir resolution sets data_dir; the split path is not used.
        if args_cli.data_dir is None and getattr(args_cli, "sequence_dir", None):
            print(
                f"[dataset] data_dir ← --sequence_dir (the split='{args_cli.split}' path is not used by a "
                f"tree-view run)",
                flush=True,
            )
        elif args_cli.data_dir is None:
            _resolved_dd = _split_paths[args_cli.split]
            if _resolved_dd is None:
                parser.error(
                    f"--dataset='{args_cli.dataset}' has no path for split='{args_cli.split}'. "
                    f"Available splits: {[k for k, v in _split_paths.items() if v is not None]}"
                )
            args_cli.data_dir = _resolved_dd
            print(f"[dataset] --data_dir resolved from registry: {args_cli.data_dir}")
        if args_cli.heldout_data_dir is None and _spec.heldout_data_dir is not None:
            args_cli.heldout_data_dir = _spec.heldout_data_dir
            print(f"[dataset] --heldout_data_dir resolved from registry: {args_cli.heldout_data_dir}")
        if (
            getattr(args_cli, "object_usd_right", None) is None
            and _spec.object_usd is not None
            and not getattr(args_cli, "left_uni", 0)
            and not getattr(args_cli, "sequence_dir", None)
        ):
            args_cli.object_usd_right = _spec.object_usd
            print(f"[dataset] --object_usd_right resolved from registry: {args_cli.object_usd_right}")
        if (
            getattr(args_cli, "object_mesh_dir_right", None) is None
            and _spec.object_mesh_dir is not None
            and not getattr(args_cli, "left_uni", 0)
            and not getattr(args_cli, "sequence_dir", None)
        ):
            args_cli.object_mesh_dir_right = _spec.object_mesh_dir
            print(f"[dataset] --object_mesh_dir_right resolved from registry: {args_cli.object_mesh_dir_right}")
        print(f"[dataset] '{args_cli.dataset}': {_spec.notes}")


def resolve_active_object(parser, args_cli) -> tuple[str, str]:
    """Set the side-neutral active-object triple and normalize --multi_object lists; returns (side, scene key).

    The CLI takes side-explicit --object_{usd,mesh_dir,mass}_{right,left}. The single 'object' scene slot reads
    args_cli.object_usd / object_mesh_dir / object_mass: right-side and bimanual runs use the right flags (the
    bimanual left partner keeps *_left); --side left uses the left flags.
    """
    _active_side = "left" if getattr(args_cli, "left_uni", 0) else "right"
    _ACTIVE_OBJ_KEY = f"object_{_active_side}"  # the single active-object scene entity
    args_cli.object_usd = getattr(args_cli, f"object_usd_{_active_side}")
    args_cli.object_mesh_dir = getattr(args_cli, f"object_mesh_dir_{_active_side}")
    args_cli.object_mass = getattr(args_cli, f"object_mass_{_active_side}", None)

    # --multi_object: clips use different objects. The comma-lists (user order, aligned with --sequence) are
    # reordered to loader order (sorted sequence names, as the loader sorts its glob) so env_i -> clip i%n ->
    # asset i%n line up, and each token is made absolute. The bimanual left lists get the same order.
    if getattr(args_cli, "multi_object", 0):
        if getattr(args_cli, "dual_eval", 0):
            parser.error("--multi_object 1 is not supported with --dual_eval")
        _mo_bi = bool(getattr(args_cli, "bimanual", 0))
        _mo_seqs = [t.strip() for t in (args_cli.sequence or "").split(",") if t.strip()]
        _mo_usds = [t.strip() for t in (args_cli.object_usd or "").split(",") if t.strip()]  # active side
        _mo_dirs = [t.strip() for t in (args_cli.object_mesh_dir or "").split(",") if t.strip()]
        if len(_mo_seqs) < 1:
            parser.error("--multi_object 1 requires at least one --sequence")
        if len(_mo_seqs) == 1:
            print("[multi_object] single clip under the multi-clip env contract (asset list of 1)")
        if not (len(_mo_seqs) == len(_mo_usds) == len(_mo_dirs)):
            parser.error(
                f"--multi_object 1: comma-list lengths must match --sequence "
                f"(seqs={len(_mo_seqs)} usds={len(_mo_usds)} mesh_dirs={len(_mo_dirs)})"
            )
        _mo_order = sorted(range(len(_mo_seqs)), key=lambda i: _mo_seqs[i])
        args_cli.sequence = ",".join(_mo_seqs[i] for i in _mo_order)
        args_cli.object_usd = ",".join(os.path.abspath(_mo_usds[i]) for i in _mo_order)
        args_cli.object_mesh_dir = ",".join(os.path.abspath(_mo_dirs[i]) for i in _mo_order)
        setattr(args_cli, f"object_usd_{_active_side}", args_cli.object_usd)
        setattr(args_cli, f"object_mesh_dir_{_active_side}", args_cli.object_mesh_dir)
        if _mo_bi:
            _mo_lu = [t.strip() for t in (args_cli.object_usd_left or "").split(",") if t.strip()]
            _mo_ld = [t.strip() for t in (args_cli.object_mesh_dir_left or "").split(",") if t.strip()]
            if len(_mo_lu) == 1 and len(_mo_seqs) > 1:  # one left object shared by every clip
                _mo_lu, _mo_ld = _mo_lu * len(_mo_seqs), _mo_ld * len(_mo_seqs)
            if not (len(_mo_lu) == len(_mo_ld) == len(_mo_seqs)):
                parser.error(
                    f"--multi_object 1 + --bimanual: LEFT comma-list lengths must match "
                    f"--sequence (seqs={len(_mo_seqs)} usds={len(_mo_lu)} mesh_dirs={len(_mo_ld)})"
                )
            args_cli.object_usd_left = ",".join(os.path.abspath(_mo_lu[i]) for i in _mo_order)
            args_cli.object_mesh_dir_left = ",".join(os.path.abspath(_mo_ld[i]) for i in _mo_order)
        print(
            f"[multi_object] {len(_mo_seqs)} clips (loader order): "
            + "; ".join(
                f"{i}:{_mo_seqs[j]} obj={os.path.basename(os.path.dirname(os.path.abspath(_mo_usds[j])))}"
                for i, j in enumerate(_mo_order)
            )
        )
    return _active_side, _ACTIVE_OBJ_KEY


def check_task(parser, args_cli) -> None:
    """Eval needs --checkpoint (except the policy-free pose render and kinematic replay); --distributed is
    train-only."""
    if (
        args_cli.task == "eval"
        and args_cli.checkpoint is None
        and not getattr(args_cli, "pose_render_out", "")
        and not getattr(args_cli, "kinematic_replay", False)
    ):
        # --kinematic_replay teleports to the reference every frame and --pose_render_out shows the reset pose:
        # neither runs a policy, and a new (arm, hand) pair has no checkpoint yet when it is first inspected.
        parser.error("--checkpoint is required for eval tasks.")

    # --distributed (multi-GPU PPO): eval/render paths are single-GPU (balanced-eval bookkeeping and video
    # writers are not rank-aware).
    if getattr(args_cli, "distributed", False) and args_cli.task != "train":
        parser.error("--distributed supports --task train only.")
