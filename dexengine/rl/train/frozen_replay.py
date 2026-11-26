"""Fresh-process judge entry (``DEXENGINE_FROZEN_REPLAY=1``): one frozen eval of the loaded checkpoint, then exit.

The ``[FROZEN_REPLAY]`` and ``[TERMDIAG]`` lines are parsed by the judge sidecar; their text must not change.
"""

import os


def frozen_replay(*, _OT_VAR, _ot_recorder_class, frozen_eval_snapshot, raw_env, wrapped, args_cli):
    """Fresh-process judge (``DEXENGINE_FROZEN_REPLAY=1``): one frozen eval of the loaded checkpoint,
    then close the env. ``runner.run()`` is never called, so the weights stay as loaded."""
    # With DEXENGINE_FROZEN_RENDER=1 the frozen eval hands its env to a record session instead (same frozen settings).
    print("[FROZEN_REPLAY] ── actual replay call (frozen_eval_snapshot, not a copy) ──", flush=True)
    _fr_iter = int(os.environ.get("DEXENGINE_FROZEN_REPLAY_ITER", "0") or 0)
    _fr_epc_req = int(os.environ.get("DEXENGINE_FROZEN_REPLAY_EVALS_PER_CLIP", str(args_cli.eval_evals_per_clip)))
    print(
        f"[FROZEN_REPLAY] resume_from={args_cli.resume_from} iteration={_fr_iter} "
        f"eval_evals_per_clip(requested)={_fr_epc_req} num_envs={wrapped.num_envs}",
        flush=True,
    )
    # Trajectory record (off by default; DEXENGINE_TRAJ_RECORD=<output .npz>): a read-only recorder attached to the
    # judge loop (dexengine.rl.eval.traj_record).
    _tr_rec = None
    _tr_out = os.environ.get("DEXENGINE_TRAJ_RECORD", "")
    if _tr_out:
        from dexengine.rl.eval.traj_meta import run_meta as _run_meta
        from dexengine.rl.eval.traj_record import TrajRecorder as _TrajRec

        if os.environ.get(_OT_VAR, ""):  # object-pose trace (passed the launch check): wrap the recorder
            _TrajRec = _ot_recorder_class(_TrajRec, os.environ[_OT_VAR])
        _tr_rec = _TrajRec(
            raw_env,
            list(raw_env.trajectory_manager._loader.sequence_names),
            _tr_out,
            meta=_run_meta(args_cli, iteration=_fr_iter, evals_per_clip=_fr_epc_req),
        )
    frozen_eval_snapshot(_fr_iter, evals_per_clip=_fr_epc_req, traj_recorder=_tr_rec)
    _fr_rate = getattr(frozen_eval_snapshot, "_last_completion", float("nan"))
    _fr_agg = getattr(frozen_eval_snapshot, "_last_aggregate", {}) or {}
    _fr_sides = getattr(frozen_eval_snapshot, "_last_side_rates", None)
    print(
        f"[FROZEN_REPLAY] DONE headline_completion={_fr_rate:.4f} "
        f"aggregate_completion_rate={float(_fr_agg.get('completion_rate', float('nan'))):.4f} "
        f"mean_episode_length={float(_fr_agg.get('mean_episode_length', float('nan'))):.2f} "
        f"side_rates={_fr_sides}",
        flush=True,
    )
    # With the object-tracking termination off (dexengine.rl.judge.terms): print the TERMDIAG remainder (steps after
    # its last periodic line) in the same form, so the log holds per-term termination counts over the whole judge
    # (a consumer checks object_tracking* = 0). Nothing is printed when the term is on.
    from dexengine.rl.judge.terms import objtrack_term as _ot_term

    if _ot_term()[0] == 0 and getattr(raw_env, "_termdiag_counts", None) is not None:
        _tdm = raw_env.termination_manager
        _tdv = raw_env._termdiag_counts.tolist()
        _tdc = int(raw_env.common_step_counter)
        print(
            f"[TERMDIAG] step {_tdc} (last {_tdc - raw_env._termdiag_last} steps · rest at the judge end): "
            + " ".join(f"{n}={v}" for n, v in zip(_tdm._term_names, _tdv, strict=True))
            + f" total={sum(_tdv)}",
            flush=True,
        )
    if _tr_rec is not None:
        _tr_rec.finalize(_fr_agg)
    print("[FROZEN_REPLAY] EXIT (runner.run() is not called — the weights are --resume_from as loaded)", flush=True)
    wrapped.close()
