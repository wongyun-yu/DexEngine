"""One batch of the clip-balanced judge loop: assign clips, reset, step until every env finished one episode."""

from __future__ import annotations

import collections
import os
import sys
import threading
import time
import traceback

import torch

from dexengine.rl.judge.rollouts_types import BatchResult

WATCHDOG_S = 300.0  # stall watchdog threshold [s]: a batch step loop without progress this long prints stacks


class _EvalStallWatchdog:
    """Eval hang instrument. When the batch step loop makes no progress for ``threshold`` seconds, a daemon thread
    periodically prints the main thread's Python stack, the recent step durations and the thread count. It uses no
    signals or ptrace, so Kit's faulthandler interception and a blocked py-spy do not matter. If the main thread spins
    in C while holding the GIL this thread cannot run either and stays silent; that silence itself says "C spin with
    the GIL held". threshold <= 0 = off."""

    def __init__(self, tag: str, threshold: float):
        self.tag = tag
        self.threshold = float(threshold)
        self.last_t = time.monotonic()
        self.step = -1
        self.batch = -1
        self.dts = collections.deque(maxlen=50)
        self._stop = threading.Event()
        self._main_ident = threading.main_thread().ident
        self._thread = threading.Thread(target=self._run, name="eval_stall_watchdog", daemon=True)

    def start(self):
        if self.threshold > 0:
            self._thread.start()
        return self

    def tick(self, step: int, batch: int):
        now = time.monotonic()
        self.dts.append(now - self.last_t)
        self.last_t = now
        self.step = step
        self.batch = batch

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(min(30.0, self.threshold / 2)):
            stalled = time.monotonic() - self.last_t
            if stalled < self.threshold:
                continue
            dts = list(self.dts)
            mean_dt = sum(dts) / len(dts) if dts else float("nan")
            max_dt = max(dts) if dts else float("nan")
            frames = sys._current_frames()
            main_stack = (
                "".join(traceback.format_stack(frames[self._main_ident]))
                if self._main_ident in frames
                else "  (no main frame)"
            )
            print(
                f"[EVAL_WATCHDOG] {self.tag}: stalled {stalled:.0f}s (batch {self.batch} step {self.step}) "
                f"| last {len(dts)} steps dt mean {mean_dt:.3f}s max {max_dt:.3f}s "
                f"| threads {threading.active_count()} pid {os.getpid()}\n"
                f"[EVAL_WATCHDOG] main thread Python stack:\n{main_stack}",
                flush=True,
            )
            # wait one threshold before the next print (no repeated output)
            if self._stop.wait(self.threshold):
                return


def _park_probe_cut(park, done_mask, batch_idx: int, step: int, t0: float) -> None:
    """One cut of the policy-free PARK probe: the judge's own at_cut lines ([EVAL_PARK] …), then a marker with the
    cut step and the elapsed time."""
    park.at_cut(done_mask)
    print(
        f"[PARK_PROBE] batch {batch_idx}: env {int(done_mask.numel())} all parked · cut step {step} · "
        f"{time.monotonic() - t0:.1f} s",
        flush=True,
    )


def _run_single_batch(
    wrapped_env,
    policy_fn,
    traj_mgr,
    clip_assignment: torch.Tensor,
    batch_idx: int,
    batch_size: int,
    per_clip_recorder=None,
    *,
    arm_pen,
    clip_names: list[str] | None = None,
    traj_recorder=None,
) -> BatchResult:
    """Run one batch: assign clips, reset, step until all envs complete 1 episode."""
    raw_env = wrapped_env.env.unwrapped
    device = raw_env.device
    n_envs = wrapped_env.num_envs

    # Assign clips: for envs beyond batch_size, assign clip 0 (they'll be ignored)
    full_assignment = torch.zeros(n_envs, dtype=torch.long, device=device)
    full_assignment[:batch_size] = clip_assignment
    full_lengths = traj_mgr._loader.get_clip_lengths(full_assignment)

    _park, result = _patch_traj_mgr_reset(
        wrapped_env,
        policy_fn,
        traj_mgr,
        clip_assignment,
        batch_idx,
        batch_size,
        per_clip_recorder,
        arm_pen,
        clip_names,
        traj_recorder,
        device,
        full_assignment,
        full_lengths,
        n_envs,
        raw_env,
    )
    # trajectory candidates = success ∧ real batch ∧ this batch PARK-valid (gates C, D: Park.invalid)
    if traj_recorder is not None:
        if _park.probe:  # the policy-free probe stops the batch at its last cut (traj_record.mark_probe_cut)
            traj_recorder.mark_probe_cut(_park.step)
        traj_recorder.end_batch(park_invalid=_park.invalid)
    result.park_e = _park.e_rows
    result.park_excl_a = dict(_park.excl_a)

    return result


def _patch_traj_mgr_reset(
    wrapped_env,
    policy_fn,
    traj_mgr,
    clip_assignment,
    batch_idx,
    batch_size,
    per_clip_recorder,
    arm_pen,
    clip_names,
    traj_recorder,
    device,
    full_assignment,
    full_lengths,
    n_envs,
    raw_env,
):
    """Monkey-patch traj_mgr.reset() so it uses our balanced assignment instead of sampling random clips."""
    original_reset = _install_balanced_reset(wrapped_env, traj_mgr, device, full_assignment, full_lengths, n_envs)
    obs, _ = wrapped_env.reset()

    _stabilize_and_fingerprint(wrapped_env, policy_fn, traj_mgr, batch_idx, batch_size, n_envs, obs)

    # Restore original reset for auto-resets during stepping
    traj_mgr.reset = original_reset

    # Track which envs have completed their first episode
    done_mask = torch.zeros(n_envs, dtype=torch.bool, device=device)
    if batch_size < n_envs:
        done_mask[batch_size:] = True

    ep_len_tracker = torch.zeros(n_envs, device=device, dtype=torch.long)
    result = BatchResult(batch_idx=batch_idx, batch_size=batch_size)

    # Per-clip recorder: begin batch, record initial frame
    if per_clip_recorder is not None:
        per_clip_recorder.begin_batch(clip_assignment, batch_size, env_origins=raw_env.scene.env_origins)
        per_clip_recorder.record_step()
    if traj_recorder is not None:  # s_0 = right after this batch's reset (traj_record time convention)
        traj_recorder.begin_batch(batch_idx, clip_assignment, batch_size)
        # if the recorder moved the robot in begin_batch (approach phase), the first policy step sees the current
        # state; otherwise obs is unchanged
        obs = getattr(traj_recorder, "obs_after_begin", lambda o: o)(obs)

    _park = _run_batch_steps(
        wrapped_env,
        policy_fn,
        clip_assignment,
        batch_idx,
        batch_size,
        per_clip_recorder,
        arm_pen,
        clip_names,
        traj_recorder,
        done_mask,
        ep_len_tracker,
        obs,
        raw_env,
        result,
    )
    return _park, result


def _install_balanced_reset(wrapped_env, traj_mgr, device, full_assignment, full_lengths, n_envs):
    """Swap traj_mgr.reset for the balanced reset; returns the original reset."""
    original_reset = traj_mgr.reset

    def _balanced_reset(env_ids: torch.Tensor) -> None:
        traj_mgr.clip_ids[env_ids] = full_assignment[env_ids]
        traj_mgr.clip_lengths[env_ids] = full_lengths[env_ids]
        traj_mgr.start_frames[env_ids] = 0  # every judge episode starts at frame 0

    traj_mgr.reset = _balanced_reset

    # Stabilization sequence for Batch 1: PhysX GPU pipeline needs the first sim
    # step to fully settle articulation state. Without this, the very first batch
    # shows spurious 0% on specific clips that work fine in later batches.
    #
    # CRITICAL: skrl's SkrlVecEnvWrapper.reset() caches behind a `_reset_once`
    # flag and becomes a NO-OP after the first call. We MUST force-bypass that
    # cache so every batch actually resets all envs to episode_length_buf=0.
    # Without this, frozen_eval_snapshot inherits training's per-env buf values,
    # causing trajectory_done to fire 1-3 steps into the eval loop.
    wrapped_env._reset_once = True
    wrapped_env.reset()
    # Step once with zero actions to let PhysX settle
    zero_act = torch.zeros(n_envs, wrapped_env.action_space.shape[0], device=device)
    wrapped_env.step(zero_act)
    # Force a real reset again so all envs start the eval loop at buf=0
    wrapped_env._reset_once = True
    return original_reset


def _stabilize_and_fingerprint(wrapped_env, policy_fn, traj_mgr, batch_idx, batch_size, n_envs, obs):
    """First-batch stabilisation and the batch-0 eval fingerprint."""
    _print_eval_fingerprint(wrapped_env, traj_mgr, batch_idx, batch_size, n_envs, obs)
    # The policy output on the same obs is part of the fingerprint too: when two paths that share this function score
    # the same checkpoint differently, it separates the env from the policy loading (weights, normalisation).
    try:
        if batch_idx == 0:
            _o2 = obs if not isinstance(obs, dict) else next(iter(obs.values()))
            with torch.no_grad():
                _a0 = policy_fn(_o2)
            _af = _a0.detach().float()
            print(
                "[EVAL FINGERPRINT] act[0,:8]="
                + " ".join(f"{v:+.4f}" for v in _af[0, :8].tolist())
                + f" | act mean={_af.mean().item():.5f} std={_af.std().item():.5f} shape={tuple(_af.shape)}",
                flush=True,
            )
    except Exception as _e2:  # noqa: BLE001 — diagnostics must never break an eval
        print(f"[EVAL FINGERPRINT] act unavailable: {_e2}", flush=True)


def _print_eval_fingerprint(wrapped_env, traj_mgr, batch_idx, batch_size, n_envs, obs):
    """EVAL FINGERPRINT: every evaluation path calls this function, so the values that decide the outcome are printed
    here (batch 0 only) and the logs of two paths can be diffed directly. Print-only: no behavioural change.
    """
    try:
        if batch_idx == 0:
            _o = obs if not isinstance(obs, dict) else next(iter(obs.values()))
            _o = _o.detach().float()
            _dmax = _o.max(dim=0).values  # per-dim max over envs
            _dmin = _o.min(dim=0).values
            _imax = int(_dmax.argmax())
            _imin = int(_dmin.argmin())

            # Resolve a flat obs index to "term_name[local]" from the observation manager, the only authority on
            # the term order.
            _uw = getattr(wrapped_env, "unwrapped", wrapped_env)
            _om = getattr(_uw, "observation_manager", None)
            _spans = []  # [(lo, hi, name), ...]
            if _om is not None:
                _lo = 0
                for _nm, _d in zip(_om.active_terms["policy"], _om.group_obs_term_dim["policy"], strict=True):
                    _w = int(_d[0]) if isinstance(_d, (tuple, list)) else int(_d)
                    _spans.append((_lo, _lo + _w, _nm))
                    _lo += _w

            def _where(i):
                for _lo, _hi, _nm in _spans:
                    if _lo <= i < _hi:
                        return f"{_nm}[{i - _lo}]"
                return "?"

            _fp = [
                f"obs dim={tuple(_o.shape)} mean={_o.mean():.5f} std={_o.std():.5f} "
                f"min={_o.min():.4f} max={_o.max():.4f}",
                "term layout: " + " ".join(f"{_nm}={_lo}:{_hi}" for _lo, _hi, _nm in _spans),
                "obs[0,:8]=" + " ".join(f"{v:+.4f}" for v in _o[0, :8].tolist()),
                # which dim carries the extreme (a divergence between paths can live in one feature)
                f"argmax dim={_imax} ({_where(_imax)}) val={float(_dmax[_imax]):+.4f} | "
                f"argmin dim={_imin} ({_where(_imin)}) val={float(_dmin[_imin]):+.4f}",
                "per-dim max top5: "
                + " ".join(
                    f"d{int(i)}({_where(int(i))})={float(_dmax[int(i)]):+.3f}"
                    for i in _dmax.argsort(descending=True)[:5].tolist()
                ),
            ]
            _tm = getattr(getattr(wrapped_env, "unwrapped", wrapped_env), "termination_manager", None)
            _live = (
                dict(zip(getattr(_tm, "_term_names", []), getattr(_tm, "_term_cfgs", []), strict=False)) if _tm else {}
            )
            for _tn in ("object_tracking", "object_tracking_left"):
                _tc = _live.get(_tn)
                if _tc is None or not hasattr(_tc, "params"):
                    continue
                _pp = _tc.params
                _pr = getattr(traj_mgr, "schedule_progress", float("nan"))
                _ps, _pe = _pp.get("pos_start"), _pp.get("pos_end")
                if _ps is not None and _pe is not None:
                    _fp.append(
                        f"{_tn}: eff={100.0 * (_ps + (_pe - _ps) * _pr):.2f}cm progress={_pr:.3f} "
                        f"lag={_pp.get('lag_window', 0)} grace={_pp.get('grace_period', 0)}"
                    )
            _fp.append(
                f"rsi_fraction={getattr(traj_mgr, 'rsi_fraction', None)} "
                f"start_frames[:4]={traj_mgr.start_frames[:4].tolist()} "
                f"n_envs={n_envs} batch={batch_size}"
            )
            for _l in _fp:
                print(f"[EVAL FINGERPRINT] {_l}")
    except Exception as _e:  # noqa: BLE001 — diagnostics must never break an eval
        print(f"[EVAL FINGERPRINT] unavailable: {_e}")


def _run_batch_steps(
    wrapped_env,
    policy_fn,
    clip_assignment,
    batch_idx,
    batch_size,
    per_clip_recorder,
    arm_pen,
    clip_names,
    traj_recorder,
    done_mask,
    ep_len_tracker,
    obs,
    raw_env,
    result,
):
    """Step the batch until every real env is done (or through the park probe), stashing per-step results."""
    step = 0
    # PARK (DEXENGINE_EVAL_PARK: park finished envs, free-fall gates); off = every method returns at once.
    # Default owner: judge/launch.py resolve_park
    from dexengine.rl.judge.park import probe_steps
    from dexengine.rl.judge.park_batch import Park

    _park = Park(raw_env, batch_idx, batch_size, clip_assignment, clip_names)
    _probe = probe_steps()
    if _probe:  # policy-free probe: every env parks from the reset (zero action); measured at each cut step
        done_mask[:] = True
        _probe_t0 = time.monotonic()
    _wd = _EvalStallWatchdog(
        tag=f"batch{batch_idx}",
        threshold=WATCHDOG_S,
    ).start()
    while (step < _probe[-1]) if _probe else not done_mask[:batch_size].all():
        with torch.no_grad():
            obs_tensor = obs["states"] if isinstance(obs, dict) else obs
            actions = policy_fn(obs_tensor)
        actions = _park.before_step(done_mask, actions)

        next_obs, _, terminated, truncated, _ = wrapped_env.step(actions)
        step += 1
        _park.after_step(step)
        if step in _probe[:-1]:
            _park_probe_cut(_park, done_mask, batch_idx, step, _probe_t0)
        _wd.tick(step, batch_idx)
        ep_len_tracker += 1

        term = terminated.squeeze(-1).bool()
        trunc = truncated.squeeze(-1).bool()
        done = term | trunc

        obs = _record_active_envs(
            clip_assignment,
            batch_size,
            per_clip_recorder,
            arm_pen,
            traj_recorder,
            done_mask,
            ep_len_tracker,
            result,
            done,
            next_obs,
            step,
            term,
            trunc,
            obs,
        )
    _wd.stop()
    if _probe:
        _park_probe_cut(_park, done_mask, batch_idx, step, _probe_t0)
    else:
        _park.at_cut(done_mask)
    return _park


def _record_active_envs(
    clip_assignment,
    batch_size,
    per_clip_recorder,
    arm_pen,
    traj_recorder,
    done_mask,
    ep_len_tracker,
    result,
    done,
    next_obs,
    step,
    term,
    trunc,
    obs,
):
    """Record step for all active envs that need capture."""
    if per_clip_recorder is not None:
        per_clip_recorder.record_step()
    # done_mask before this step's update: the post-step (reset) state of envs that just ended is not counted
    if traj_recorder is not None:
        traj_recorder.after_step(step, term, trunc, done_mask)

    # Process newly completed envs (only count first completion)
    newly_done = done & ~done_mask
    if newly_done.any():
        newly_done_ids = newly_done.nonzero(as_tuple=True)[0]
        for eid in newly_done_ids:
            eid_int = eid.item()
            if eid_int >= batch_size:
                continue

            cid = clip_assignment[eid_int].item()
            is_early = (term[eid] & ~trunc[eid]).item()
            ep_len = ep_len_tracker[eid].item()

            # Per-clip recorder: save or discard this env's trajectory
            if per_clip_recorder is not None:
                per_clip_recorder.process_done(eid_int, terminated=is_early)

            if is_early:
                result.early_terminated += 1
                if cid not in result.clip_counts:
                    result.clip_counts[cid] = (0, 0)
                c, e = result.clip_counts[cid]
                result.clip_counts[cid] = (c, e + 1)
            else:
                result.completed += 1
                if cid not in result.clip_counts:
                    result.clip_counts[cid] = (0, 0)
                c, e = result.clip_counts[cid]
                result.clip_counts[cid] = (c + 1, e)

            result.step_sum += ep_len

        done_mask[newly_done] = True
        ep_len_tracker[newly_done] = 0
    arm_pen.sample(done_mask)
    obs = next_obs
    return obs
