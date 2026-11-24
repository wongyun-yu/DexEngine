"""Best-checkpoint selection after a frozen eval: headline completion, best_agent.pt, best_return_agent.pt, early
stop."""

import os

import torch


def update_best_checkpoints(
    *,
    _dist,
    _dist_rank,
    _side_rates,
    _voc_flush_on,
    agent,
    aggregate,
    args_cli,
    best_frozen_completion,
    best_frozen_return,
    evals_per_clip,
    exp_dir,
    frozen_eval_snapshot,
    iteration,
    num_clips,
    raw_env,
    saved_snap_only,
):
    """Headline completion, best_agent.pt / best_return_agent.pt and the return-collapse early stop."""
    # 5. Update best_agent.pt based on frozen eval completion rate.
    # SKRL's default best_agent.pt logic uses rolling-100 training reward
    # (a shaped sum), which is NOT the metric we care about. Override it
    # with the frozen-eval completion rate.
    # `eval/completion_rate` uses the same function (run_eval_balanced) and physics as the standalone eval, with the
    # frozen eval's sample size (eps_per_clip of the [FROZEN EVAL iter N] line). The pick below is a greedy argmax
    # over evals, so it carries a small upward bias; the final verdict comes from the judge, not from this number.
    # DECOUPLED: the classic completion_rate saturates (time_out-only episodes),
    # so the best metric becomes min(right_rate, left_rate) — both hands must work.
    import copy as _copy
    import math as _math

    _restore_snap_state(
        _dist,
        _dist_rank,
        _side_rates,
        _voc_flush_on,
        agent,
        aggregate,
        args_cli,
        best_frozen_completion,
        best_frozen_return,
        evals_per_clip,
        exp_dir,
        frozen_eval_snapshot,
        iteration,
        num_clips,
        raw_env,
        saved_snap_only,
        _copy,
        _math,
    )


def _restore_snap_state(
    _dist,
    _dist_rank,
    _side_rates,
    _voc_flush_on,
    agent,
    aggregate,
    args_cli,
    best_frozen_completion,
    best_frozen_return,
    evals_per_clip,
    exp_dir,
    frozen_eval_snapshot,
    iteration,
    num_clips,
    raw_env,
    saved_snap_only,
    _copy,
    _math,
):
    """When snap-only was turned off for this eval (saved_snap_only), the gates terminate, no `fail` masks accumulate
    and seg_completion = comp/(comp+fail) would read 100%; min(right, left) would then pin best_agent at 100% and
    pick checkpoints by rolling return alone. So the per-side rates are not used then (the headline rate is).
    """
    if saved_snap_only:
        _side_rates = None
    if _side_rates is not None and not any(_math.isnan(x) for x in _side_rates):
        current_completion = float(min(_side_rates))
    else:
        _raw = aggregate.get("completion_rate", None)
        current_completion = float(_raw) if (_raw is not None and not _math.isnan(float(_raw))) else -1.0
    # Stash this call's headline numbers on the function object so a caller outside the training loop (the
    # DEXENGINE_FROZEN_REPLAY judge) can read what this call produced; frozen_eval_snapshot returns None.
    frozen_eval_snapshot._last_completion = current_completion
    frozen_eval_snapshot._last_aggregate = aggregate
    frozen_eval_snapshot._last_side_rates = list(_side_rates) if _side_rates is not None else None
    frozen_eval_snapshot._last_evals_per_clip = evals_per_clip
    frozen_eval_snapshot._last_num_clips = num_clips
    _es = _select_best_checkpoint(
        _dist_rank,
        _voc_flush_on,
        agent,
        best_frozen_completion,
        best_frozen_return,
        exp_dir,
        frozen_eval_snapshot,
        iteration,
        raw_env,
        _copy,
        current_completion,
    )
    _tr = list(getattr(agent, "_track_rewards", []))
    if _tr:
        current_return = float(sum(_tr) / len(_tr))
        if current_return > _es._best_return and _dist_rank == 0:
            _es._best_return = current_return
            _ret_modules = {}
            for _name, _module in agent.checkpoint_modules.items():
                _ret_modules[_name] = _copy.deepcopy(agent._get_internal_value(_module))
            _ckpt_dir = os.path.join(exp_dir, "checkpoints")
            os.makedirs(_ckpt_dir, exist_ok=True)
            torch.save(_ret_modules, os.path.join(_ckpt_dir, "best_return_agent.pt"))
            print(
                f"    [BEST RETURN] episode return {current_return:.3f} — saved best_return_agent.pt @ iter {iteration}"
            )
        _early_stop_detector_off(_dist, args_cli, _es, current_return)


def _early_stop_detector_off(_dist, args_cli, _es, current_return):
    """Early-stop detector (off unless --early_stop_return_frac > 0). DISTRIBUTED: disabled — the stop decision is per-
    rank (local return); one rank stopping while others keep issuing gradient all-reduces would deadlock NCCL. (A
    rank-synced stop needs an all_reduce'd decision; not needed for current runs, so fail safe = off.).
    """
    _es_frac = float(getattr(args_cli, "early_stop_return_frac", 0.0))
    if _es_frac > 0.0 and _dist:
        print(
            "    [EARLY-STOP] DISABLED under --distributed (per-rank stop "
            "would desync the gradient all-reduce; see comment)"
        )
        _es_frac = 0.0
    if _es_frac > 0.0:
        _es._es_max = max(_es._es_max, current_return)
        if _es._es_max > 0 and current_return < _es_frac * _es._es_max:
            _es._es_below += 1
            print(
                f"    [EARLY-STOP watch] return {current_return:.3f} < "
                f"{_es_frac:.2f}×max({_es._es_max:.3f}) "
                f"[{_es._es_below}/{getattr(args_cli, 'early_stop_patience', 3)}]"
            )
        else:
            _es._es_below = 0
        if _es._es_below >= int(getattr(args_cli, "early_stop_patience", 3)):
            print(
                f"    [EARLY-STOP] return collapsed below {_es_frac:.2f}×max for "
                f"{_es._es_below} evals — stopping (best_return_agent.pt kept)."
            )
            _es._stop = True  # honored in patched_post after env restore


def _select_best_checkpoint(
    _dist_rank,
    _voc_flush_on,
    agent,
    best_frozen_completion,
    best_frozen_return,
    exp_dir,
    frozen_eval_snapshot,
    iteration,
    raw_env,
    _copy,
    current_completion,
):
    """Rolling-100 training return (SKRL's _track_rewards deque) as the tie-break: with a strict '>' alone,
    best_agent.pt would freeze at the first eval that hit 100% completion.
    """
    _tr_b1 = list(getattr(agent, "_track_rewards", []))
    current_rolling_return = float(sum(_tr_b1) / len(_tr_b1)) if _tr_b1 else -float("inf")
    _improved = current_completion > best_frozen_completion[0]
    _tie_break = (
        current_completion == best_frozen_completion[0]
        and current_completion >= 0.0
        and current_rolling_return > best_frozen_return[0]
    )
    # VOC v2.1 flush hygiene: while assist is still active (flush pending),
    # this eval ran WITH assist — suppress the best_agent.pt update.
    _voc_ctl = getattr(raw_env, "_voc", None)
    _voc_suppress = _voc_flush_on and _voc_ctl is not None and not getattr(_voc_ctl, "flush_complete", True)
    if _voc_suppress and (_improved or _tie_break):
        print(
            f"    [BEST AGENT] SUPPRESSED (VOC flush pending — assist active, "
            f"kp_pos={_voc_ctl.kp_pos:.3f}): completion "
            f"{current_completion * 100:.1f}% not eligible for best_agent.pt"
        )
    return _distributed_rank_writes_best(
        _dist_rank,
        agent,
        best_frozen_completion,
        best_frozen_return,
        exp_dir,
        frozen_eval_snapshot,
        iteration,
        _copy,
        current_completion,
        _improved,
        _tie_break,
        _voc_suppress,
        current_rolling_return,
    )


def _distributed_rank_writes_best(
    _dist_rank,
    agent,
    best_frozen_completion,
    best_frozen_return,
    exp_dir,
    frozen_eval_snapshot,
    iteration,
    _copy,
    current_completion,
    _improved,
    _tie_break,
    _voc_suppress,
    current_rolling_return,
):
    """DISTRIBUTED: only rank 0 writes best_agent.pt. Weights are identical across ranks (broadcast init + gradient all-
    reduce), so rank 0's copy is the policy; its own frozen-eval completion is the decision metric (other ranks'
    evals differ only by env seed). A judge (DEXENGINE_FROZEN_REPLAY=1 — the fresh-process judge, its eval worker, a
    trajectory record) trains nothing: no best pick to keep, so it writes no best_agent.pt in its run folder
    (finalize would only delete it). The numbers above are already taken — this only skips the file.
    """
    _replay = os.environ.get("DEXENGINE_FROZEN_REPLAY", "0") == "1"
    if _replay and (_improved or _tie_break) and not getattr(frozen_eval_snapshot, "_said_no_best", False):
        frozen_eval_snapshot._said_no_best = True
        print("    [BEST AGENT] judge (replay) — best_agent.pt not written", flush=True)
    if (_improved or _tie_break) and not _voc_suppress and _dist_rank == 0 and not _replay:
        prev = best_frozen_completion[0]
        prev_return = best_frozen_return[0]
        best_frozen_completion[0] = current_completion
        best_frozen_return[0] = current_rolling_return
        best_modules = {}
        for _name, _module in agent.checkpoint_modules.items():
            best_modules[_name] = _copy.deepcopy(agent._get_internal_value(_module))
        _ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(_ckpt_dir, exist_ok=True)
        torch.save(best_modules, os.path.join(_ckpt_dir, "best_agent.pt"))
        if _improved:
            print(
                f"    [BEST AGENT] frozen eval completion {current_completion * 100:.1f}% "
                f"(prev best {prev * 100:.1f}%, rolling return {current_rolling_return:.3f}) "
                f"— saved best_agent.pt @ iter {iteration}"
            )
        else:
            print(
                f"    [BEST AGENT] completion tied at {current_completion * 100:.1f}% but "
                f"rolling return improved to {current_rolling_return:.3f} "
                f"(prev {prev_return:.3f} at last save) — re-saved "
                f"best_agent.pt @ iter {iteration} (B1 tie-break)"
            )
        # Lock SKRL's reward-based heuristic so it never overwrites our save
        agent.checkpoint_best_modules["reward"] = float("inf")
        agent.checkpoint_best_modules["saved"] = True
        agent.checkpoint_best_modules["modules"] = {}

    # 5b. Exp-1: return-sum checkpoint + early-stop. Uses the training episode
    # return (rolling-100 mean from SKRL's deque) — the metric that cliffs at
    # collapse, unlike frozen-eval completion which stays 0 through the plateau.
    # State persists on the function object (created lazily on first use).
    _es = frozen_eval_snapshot
    if not hasattr(_es, "_best_return"):
        _es._best_return = -float("inf")
        _es._es_max = -float("inf")
        _es._es_below = 0
    return _es
